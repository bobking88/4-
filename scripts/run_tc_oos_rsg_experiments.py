"""Train and select target-constrained OOS routing on frozen expert caches."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence, TypeVar

import torch
import torch.nn.functional as F

from hrgv_network import (
    apply_residual_target_verifiers,
    regret_gate_targets,
    weighted_soft_gate_loss,
)
from tc_oos_rsg import (
    TargetRiskProjectionHead,
    apply_target_safe_projection,
    build_projection_evidence,
)


EPSILON_TARGET_GRID = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)


@dataclass(frozen=True)
class SelectedGate:
    state_dict: dict[str, torch.Tensor]
    epoch: int
    stop_metrics: dict[str, float]
    state_sha256: str


@dataclass(frozen=True)
class ProjectionCandidate:
    state_dict: dict[str, torch.Tensor]
    epoch: int
    epsilon_target: float
    stop_metrics: dict[str, float]
    state_sha256: str
    objective: str = "nll"


@dataclass(frozen=True)
class FallbackSelection:
    baseline_metrics: dict[str, float]
    reason: str = "no_safe_candidate"


def _value(record: object, field: str) -> str:
    if isinstance(record, Mapping):
        return str(record.get(field, "")).strip()
    return str(getattr(record, field, "")).strip()


def audit_pipeline_update_sets(
    subsets: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    required = (
        "expert_fit",
        "gate_fit",
        "gate_stop_projector_fit",
        "projector_stop",
        "outer_eval",
    )
    missing = [name for name in required if name not in subsets or not subsets[name]]
    if missing:
        raise ValueError(f"Missing nonempty pipeline subsets: {missing}")

    image_sets: dict[str, set[str]] = {}
    group_sets: dict[str, set[str]] = {}
    for name in required:
        image_values = [_value(record, "image_id") for record in subsets[name]]
        group_values = [_value(record, "split_group_id") for record in subsets[name]]
        if any(not value for value in image_values + group_values):
            raise ValueError(f"{name} contains blank image or group identifiers.")
        if len(image_values) != len(set(image_values)):
            raise ValueError(f"{name} contains duplicate image IDs.")
        image_sets[name] = set(image_values)
        group_sets[name] = set(group_values)

    for index, left in enumerate(required):
        for right in required[index + 1 :]:
            if image_sets[left] & image_sets[right]:
                raise ValueError(f"image overlap between {left} and {right}.")
            if group_sets[left] & group_sets[right]:
                raise ValueError(f"group overlap between {left} and {right}.")

    return {
        "row_counts": {name: len(subsets[name]) for name in required},
        "group_counts": {name: len(group_sets[name]) for name in required},
        "cross_subset_image_overlap_count": 0,
        "cross_subset_group_overlap_count": 0,
        "parameter_update_subsets": {
            "expert": "expert_fit",
            "gate": "gate_fit",
            "projector": "gate_stop_projector_fit",
        },
        "selection_subsets": {
            "expert": "expert_stop",
            "gate": "gate_stop_projector_fit",
            "projector_and_epsilon": "projector_stop",
        },
    }


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_verifier_complete_branches(
    candidate_gate: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    direct = cache["direct"]
    mapped = cache["mapped"]
    ti = cache["ti"]
    metallic = cache["metal"]
    if candidate_gate.shape != (direct.shape[0], 1):
        raise ValueError("candidate_gate must have shape [batch, 1].")
    q0_pre_verifier = 0.5 * direct + 0.5 * mapped
    candidate_pre_verifier = candidate_gate * direct + (1.0 - candidate_gate) * mapped
    return {
        "q0": apply_residual_target_verifiers(q0_pre_verifier, ti, metallic),
        "q_candidate": apply_residual_target_verifiers(
            candidate_pre_verifier, ti, metallic
        ),
    }


def build_projection_cache(
    candidate_gate: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    branches = build_verifier_complete_branches(candidate_gate, cache)
    evidence = build_projection_evidence(
        branches["q0"],
        branches["q_candidate"],
        candidate_gate,
        cache["ti"],
        cache["metal"],
    )
    result = {
        **branches,
        "evidence": evidence,
        "candidate_gate": candidate_gate,
        "ti": cache["ti"],
        "metal": cache["metal"],
    }
    if "labels" in cache:
        result["labels"] = cache["labels"]
    return result


def _rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    count = int(denominator.sum().item())
    if count == 0:
        return math.nan
    return float(numerator[denominator].double().mean().item())


def calculate_probability_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("probabilities must have shape [batch, 4].")
    if labels.shape != (probabilities.shape[0],):
        raise ValueError("labels must have shape [batch].")
    epsilon = torch.finfo(probabilities.dtype).tiny
    predictions = probabilities.argmax(dim=1)
    confusion = torch.bincount(
        labels * 4 + predictions, minlength=16
    ).reshape(4, 4).double()
    f1_denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    target_mask = labels == 0
    ti_mask = labels == 1
    metallic_mask = labels == 3
    return {
        "nll": float(F.nll_loss(probabilities.clamp_min(epsilon).log(), labels)),
        "accuracy": float((predictions == labels).double().mean()),
        "macro_f1": float(
            torch.where(
                f1_denominator > 0,
                2.0 * confusion.diag() / f1_denominator,
                torch.zeros_like(f1_denominator),
            ).mean()
        ),
        "target_recall": _rate(predictions == 0, target_mask),
        "ti_intrusion_to_target": _rate(predictions == 0, ti_mask),
        "metallic_intrusion_to_target": _rate(predictions == 0, metallic_mask),
    }


def train_oos_gate(
    gate_network: torch.nn.Module,
    fit_cache: Mapping[str, torch.Tensor],
    stop_cache: Mapping[str, torch.Tensor],
    *,
    epochs: int = 30,
    seed: int = 20260927,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    auxiliary_regret_weight: float = 0.1,
) -> SelectedGate:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if auxiliary_regret_weight < 0.0:
        raise ValueError("auxiliary_regret_weight must be non-negative.")
    labels = fit_cache["labels"]
    if labels.numel() == 0:
        raise ValueError("fit_cache must be nonempty.")
    torch.manual_seed(seed)
    gate_network.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        gate_network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    targets = regret_gate_targets(
        fit_cache["direct"],
        fit_cache["mapped"],
        labels,
        0.2,
        0.5,
        torch,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    for epoch in range(1, epochs + 1):
        for indices in torch.randperm(labels.shape[0], device=labels.device).split(batch_size):
            gate = torch.sigmoid(gate_network(fit_cache["features"][indices]))
            indexed_cache = {
                key: value[indices]
                for key, value in fit_cache.items()
                if torch.is_tensor(value) and value.shape[0] == labels.shape[0]
            }
            final = build_verifier_complete_branches(gate, indexed_cache)["q_candidate"]
            loss = F.nll_loss(
                final.clamp_min(torch.finfo(final.dtype).tiny).log(),
                indexed_cache["labels"],
            )
            if auxiliary_regret_weight > 0.0:
                loss = loss + auxiliary_regret_weight * weighted_soft_gate_loss(
                    gate,
                    targets["soft_oracle_gate"][indices],
                    targets["gate_gap_weight"][indices],
                    torch,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        gate_network.eval()
        with torch.no_grad():
            stop_gate = torch.sigmoid(gate_network(stop_cache["features"]))
            stop_final = build_verifier_complete_branches(
                stop_gate, stop_cache
            )["q_candidate"]
            metrics = calculate_probability_metrics(stop_final, stop_cache["labels"])
            metrics["mean_gate"] = float(stop_gate.mean())
        if best_metrics is None or metrics["nll"] < best_metrics["nll"]:
            best_state = _clone_state_dict(gate_network)
            best_epoch = epoch
            best_metrics = metrics
        gate_network.train()

    assert best_state is not None and best_metrics is not None
    gate_network.load_state_dict(best_state)
    gate_network.eval()
    return SelectedGate(
        state_dict=best_state,
        epoch=best_epoch,
        stop_metrics=best_metrics,
        state_sha256=state_dict_sha256(best_state),
    )


def _hard_negative_auxiliary_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    terms = []
    for class_id in (1, 3):
        eligible = labels == class_id
        if bool(eligible.any()):
            terms.append(probabilities[eligible, 0].mean())
    if not terms:
        return probabilities[:, 0].sum() * 0.0
    return torch.stack(terms).mean()


def project_with_optional_safety(
    q0: torch.Tensor,
    q_candidate: torch.Tensor,
    raw_route: torch.Tensor,
    *,
    epsilon_target: float,
    apply_safety: bool,
) -> dict[str, torch.Tensor]:
    if apply_safety:
        return apply_target_safe_projection(
            q0, q_candidate, raw_route, epsilon_target
        )
    final = (1.0 - raw_route) * q0 + raw_route * q_candidate
    signed_change = q0[:, 0:1] - final[:, 0:1]
    identity_residual = signed_change - raw_route * (
        q0[:, 0:1] - q_candidate[:, 0:1]
    )
    return {
        "route_cap": torch.ones_like(raw_route),
        "projected_route": raw_route,
        "final_probabilities": final,
        "signed_target_change": signed_change,
        "positive_target_harm": signed_change.clamp_min(0.0),
        "identity_residual": identity_residual,
    }


def _projector_metrics(
    projection: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> dict[str, float]:
    metrics = calculate_probability_metrics(projection["final_probabilities"], labels)
    metrics.update(
        {
            "max_target_harm": float(projection["positive_target_harm"].max()),
            "max_identity_residual": float(projection["identity_residual"].abs().max()),
            "mean_projected_route": float(projection["projected_route"].mean()),
            "cap_activation_rate": float(
                (
                    projection["projected_route"]
                    < projection["raw_route"] - 1e-12
                ).double().mean()
            )
            if "raw_route" in projection
            else math.nan,
        }
    )
    return metrics


def train_projection_head(
    fit_cache: Mapping[str, torch.Tensor],
    stop_cache: Mapping[str, torch.Tensor],
    *,
    epsilon_target: float,
    epochs: int = 30,
    seed: int = 20260927,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    hard_negative_weight: float = 0.0,
    apply_safety: bool = True,
) -> ProjectionCandidate:
    if epsilon_target < 0.0 or not math.isfinite(epsilon_target):
        raise ValueError("epsilon_target must be finite and non-negative.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if hard_negative_weight < 0.0:
        raise ValueError("hard_negative_weight must be non-negative.")
    labels = fit_cache["labels"]
    torch.manual_seed(seed)
    head = TargetRiskProjectionHead().to(fit_cache["evidence"].device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    for epoch in range(1, epochs + 1):
        head.train()
        for indices in torch.randperm(labels.shape[0], device=labels.device).split(batch_size):
            raw_route = head(fit_cache["evidence"][indices])
            projected = project_with_optional_safety(
                fit_cache["q0"][indices],
                fit_cache["q_candidate"][indices],
                raw_route,
                epsilon_target=epsilon_target,
                apply_safety=apply_safety,
            )
            final = projected["final_probabilities"]
            loss = F.nll_loss(
                final.clamp_min(torch.finfo(final.dtype).tiny).log(),
                labels[indices],
            )
            if hard_negative_weight > 0.0:
                loss = loss + hard_negative_weight * _hard_negative_auxiliary_loss(
                    final, labels[indices]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            raw_stop = head(stop_cache["evidence"])
            projected_stop = project_with_optional_safety(
                stop_cache["q0"],
                stop_cache["q_candidate"],
                raw_stop,
                epsilon_target=epsilon_target,
                apply_safety=apply_safety,
            )
            metrics = _projector_metrics(
                {**projected_stop, "raw_route": raw_stop}, stop_cache["labels"]
            )
        if best_metrics is None or metrics["nll"] < best_metrics["nll"]:
            best_state = _clone_state_dict(head)
            best_epoch = epoch
            best_metrics = metrics

    assert best_state is not None and best_metrics is not None
    if not apply_safety:
        objective = "unbounded_nll"
    else:
        objective = "nll_plus_hard_negative" if hard_negative_weight > 0.0 else "nll"
    return ProjectionCandidate(
        state_dict=best_state,
        epoch=best_epoch,
        epsilon_target=float(epsilon_target),
        stop_metrics=best_metrics,
        state_sha256=state_dict_sha256(best_state),
        objective=objective,
    )


def select_projection_candidate(
    candidates: Sequence[ProjectionCandidate],
    baseline_metrics: Mapping[str, float],
    recall_margin: float = 0.01,
) -> ProjectionCandidate | FallbackSelection:
    if recall_margin < 0.0:
        raise ValueError("recall_margin must be non-negative.")
    baseline = {key: float(value) for key, value in baseline_metrics.items()}
    required = ("target_recall", "nll")
    if any(key not in baseline or not math.isfinite(baseline[key]) for key in required):
        raise ValueError("baseline_metrics must contain finite target_recall and nll.")
    feasible = []
    minimum_recall = baseline["target_recall"] - recall_margin
    for candidate in candidates:
        recall = float(candidate.stop_metrics.get("target_recall", math.nan))
        nll = float(candidate.stop_metrics.get("nll", math.nan))
        if math.isfinite(recall) and math.isfinite(nll) and recall >= minimum_recall:
            feasible.append(candidate)
    if not feasible:
        return FallbackSelection(baseline_metrics=baseline)
    return min(
        feasible,
        key=lambda candidate: (
            float(candidate.stop_metrics["nll"]),
            candidate.epsilon_target,
            candidate.epoch,
        ),
    )


ProjectionTrainerResult = TypeVar("ProjectionTrainerResult")


def train_registered_projection_candidates(
    trainer: Callable[[float], ProjectionTrainerResult],
) -> list[ProjectionTrainerResult]:
    return [trainer(epsilon_target) for epsilon_target in EPSILON_TARGET_GRID]


def build_evidence_ablations(
    evidence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if evidence.ndim != 2 or evidence.shape[1] != 18:
        raise ValueError("evidence must have shape [batch, 18].")
    masks = {
        "remove_verifier_probabilities": slice(13, 15),
        "remove_abs_difference": slice(8, 12),
        "remove_uncertainty": slice(15, 18),
    }
    result = {}
    for name, columns in masks.items():
        ablated = evidence.clone()
        ablated[:, columns] = 0.0
        result[name] = ablated
    return result


def _head_raw_route(
    cache: Mapping[str, torch.Tensor],
    candidate: ProjectionCandidate,
) -> torch.Tensor:
    head = TargetRiskProjectionHead().to(cache["evidence"].device)
    head.load_state_dict(candidate.state_dict, strict=True)
    head.eval()
    with torch.no_grad():
        return head(cache["evidence"])


def _method_result(
    probabilities: torch.Tensor,
    q0: torch.Tensor,
    route: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    target_harm = (q0[:, 0] - probabilities[:, 0]).clamp_min(0.0)
    result: dict[str, torch.Tensor | float] = {
        "probabilities": probabilities,
        "max_target_harm": float(target_harm.max()),
    }
    if route is not None:
        result["route"] = route
    return result


def _selected_projection(
    cache: Mapping[str, torch.Tensor],
    selected: ProjectionCandidate | FallbackSelection,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(selected, FallbackSelection):
        zero_route = torch.zeros(
            (cache["q0"].shape[0], 1),
            dtype=cache["q0"].dtype,
            device=cache["q0"].device,
        )
        return cache["q0"], zero_route
    raw_route = _head_raw_route(cache, selected)
    projection = apply_target_safe_projection(
        cache["q0"],
        cache["q_candidate"],
        raw_route,
        selected.epsilon_target,
    )
    return projection["final_probabilities"], projection["projected_route"]


def evaluate_method_matrix(
    cache: Mapping[str, torch.Tensor],
    *,
    selected: ProjectionCandidate | FallbackSelection,
    hard_negative_selected: ProjectionCandidate | FallbackSelection,
    m0_probabilities: torch.Tensor,
    unbounded_selected: ProjectionCandidate | None = None,
) -> dict[str, dict[str, torch.Tensor | float]]:
    """Build M0-M6 from one frozen branch cache without model re-fitting."""
    q0 = cache["q0"]
    q_candidate = cache["q_candidate"]
    if m0_probabilities.shape != q0.shape:
        raise ValueError("m0_probabilities must match the branch posterior shape.")

    route_source = unbounded_selected or (
        selected if isinstance(selected, ProjectionCandidate) else None
    )
    if route_source is not None:
        raw_route = _head_raw_route(cache, route_source)
        m3_probabilities = (1.0 - raw_route) * q0 + raw_route * q_candidate
    else:
        raw_route = torch.zeros(
            (q0.shape[0], 1), dtype=q0.dtype, device=q0.device
        )
        m3_probabilities = q0

    if isinstance(selected, ProjectionCandidate):
        safety_route = _head_raw_route(cache, selected)
        safety_only = apply_target_safe_projection(
            q0,
            q_candidate,
            torch.ones_like(safety_route),
            selected.epsilon_target,
        )
    else:
        safety_route = torch.zeros(
            (q0.shape[0], 1), dtype=q0.dtype, device=q0.device
        )
        safety_only = {
            "final_probabilities": q0,
            "projected_route": safety_route,
        }
    m5_probabilities, m5_route = _selected_projection(cache, selected)
    m6_probabilities, m6_route = _selected_projection(cache, hard_negative_selected)
    return {
        "M0": _method_result(m0_probabilities, q0),
        "M1": _method_result(q0, q0, torch.zeros_like(raw_route)),
        "M2": _method_result(q_candidate, q0, cache["candidate_gate"]),
        "M3": _method_result(m3_probabilities, q0, raw_route),
        "M4": _method_result(
            safety_only["final_probabilities"],
            q0,
            safety_only["projected_route"],
        ),
        "M5": _method_result(m5_probabilities, q0, m5_route),
        "M6": _method_result(m6_probabilities, q0, m6_route),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_selection_lock_payload(
    *,
    manifest_paths: Mapping[str, Path],
    model_paths: Mapping[str, Path],
    selected_epsilon: float,
    selection_reason: str,
    gate_state_sha256: str,
    projector_state_sha256: str,
) -> dict[str, object]:
    if selected_epsilon not in EPSILON_TARGET_GRID:
        raise ValueError("selected_epsilon must come from the registered grid.")
    if not selection_reason:
        raise ValueError("selection_reason must be nonempty.")
    return {
        "protocol_version": "tc_oos_rsg_v1",
        "status": "locked_before_outer_tensor_loading",
        "outer_tensors_loaded": False,
        "selected_epsilon_target": selected_epsilon,
        "selection_reason": selection_reason,
        "gate_state_sha256": gate_state_sha256,
        "projector_state_sha256": projector_state_sha256,
        "manifest_sha256": {
            name: _file_sha256(path) for name, path in sorted(manifest_paths.items())
        },
        "model_file_sha256": {
            name: _file_sha256(path) for name, path in sorted(model_paths.items())
        },
    }


def lock_before_outer_evaluation(
    lock_path: Path,
    lock_payload: Mapping[str, object],
    outer_tensor_loader: Callable[[], ProjectionTrainerResult],
    *,
    event_trace: list[str] | None = None,
) -> ProjectionTrainerResult:
    payload = {
        **lock_payload,
        "status": "locked_before_outer_tensor_loading",
    }
    _write_json(lock_path, payload)
    if event_trace is not None:
        event_trace.append("lock_written")
    if not lock_path.is_file():
        raise RuntimeError("Selection lock was not persisted before outer evaluation.")
    result = outer_tensor_loader()
    if event_trace is not None:
        event_trace.append("outer_tensor_loaded")
    return result


def _candidate_summary(
    candidate: ProjectionCandidate | FallbackSelection,
) -> dict[str, object]:
    if isinstance(candidate, FallbackSelection):
        return {
            "type": "fallback",
            "reason": candidate.reason,
            "baseline_metrics": candidate.baseline_metrics,
        }
    return {
        "type": "learned",
        "epoch": candidate.epoch,
        "epsilon_target": candidate.epsilon_target,
        "stop_metrics": candidate.stop_metrics,
        "state_sha256": candidate.state_sha256,
        "objective": candidate.objective,
    }


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return [dict(row) for row in reader]


def _reset_module_parameters(module: torch.nn.Module) -> None:
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def _save_candidate(
    output_dir: Path,
    candidate: ProjectionCandidate | FallbackSelection,
) -> Path | None:
    _write_json(output_dir / "selection.json", _candidate_summary(candidate))
    if isinstance(candidate, FallbackSelection):
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best_model.pt"
    torch.save(candidate.state_dict, path)
    return path


def _load_frozen_expert(
    expert_dir: Path,
    expert_manifest: Path,
    dataset_root: Path,
    device: torch.device,
    dependencies: Mapping[str, object],
):
    from hrgv_network import HierarchicalRiskGatedVerificationNet
    from mineral_hierarchy import validate_species_role_mapping
    from train_hrgv_mineral_classifier import build_role_matrix
    from train_mineral_classifier import load_manifest_records

    environment = json.loads(
        (expert_dir / "environment.json").read_text(encoding="utf-8")
    )
    records = load_manifest_records(expert_manifest, dataset_root)
    mapping = validate_species_role_mapping(records)
    model = HierarchicalRiskGatedVerificationNet(
        dependencies["models"],
        build_role_matrix(mapping, torch),
        pretrained=False,
        backbone_name=environment.get("backbone", "efficientnet_b0"),
        embedding_dim=int(environment.get("embedding_dim", 128)),
        gate_hidden_dim=int(environment.get("gate_hidden_dim", 128)),
        verifier_mode=environment.get("verifier_mode", "residual"),
        ti_verifier_threshold=float(environment.get("ti_verifier_threshold", 0.5)),
        metallic_verifier_threshold=float(
            environment.get("metallic_verifier_threshold", 0.5)
        ),
        ti_verifier_strength=float(environment.get("ti_verifier_strength", 1.0)),
        metallic_verifier_strength=float(
            environment.get("metallic_verifier_strength", 1.0)
        ),
    ).to(device)
    checkpoint = torch.load(
        expert_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, mapping, records


def _extract_baseline_probabilities(
    baseline_dir: Path,
    records: Sequence[object],
    transform,
    dependencies: Mapping[str, object],
    device: torch.device,
) -> torch.Tensor:
    from train_mineral_classifier import MineralImageDataset, build_model

    environment = json.loads(
        (baseline_dir / "environment.json").read_text(encoding="utf-8")
    )
    model = build_model(
        environment.get("model", "efficientnet_b0"),
        num_classes=4,
        pretrained=False,
        models=dependencies["models"],
        nn=dependencies["nn"],
    ).to(device)
    checkpoint = torch.load(
        baseline_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    dataset = MineralImageDataset(records, transform)
    loader = dependencies["DataLoader"](
        dataset, batch_size=32, shuffle=False, num_workers=0
    )
    chunks = []
    with torch.no_grad():
        for images, _ in loader:
            chunks.append(model(images.to(device)).softmax(dim=1))
    return torch.cat(chunks)


def _write_prediction_csv(
    path: Path,
    records: Sequence[object],
    probabilities: torch.Tensor,
    route: torch.Tensor | None,
    fold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_id",
                "split_group_id",
                "mineral_label",
                "outer_fold",
                "true_class_id",
                "predicted_class_id",
                "route",
                "prob_0",
                "prob_1",
                "prob_2",
                "prob_3",
            ]
        )
        route_values = (
            route.reshape(-1).detach().cpu().tolist()
            if route is not None
            else [""] * len(records)
        )
        for record, probability, route_value in zip(
            records,
            probabilities.detach().cpu(),
            route_values,
            strict=True,
        ):
            true_class_id = _value(record, "four_class_id") or _value(
                record, "class_id"
            )
            writer.writerow(
                [
                    _value(record, "image_id"),
                    _value(record, "split_group_id"),
                    _value(record, "mineral_label"),
                    fold,
                    int(true_class_id),
                    int(probability.argmax()),
                    route_value,
                    *[float(value) for value in probability],
                ]
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run one locked TC-OOS-RSG outer fold."
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=root / "outputs" / "training" / "tc_oos_rsg_manifests_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs" / "training" / "tc_oos_rsg_v1",
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gate-epochs", type=int, default=30)
    parser.add_argument("--projector-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    from run_seen_unseen_gate_study import extract_cache
    from train_mineral_classifier import (
        create_transforms,
        load_manifest_records,
        require_training_dependencies,
    )

    args = parse_args(argv)
    fold_source = args.protocol_dir / f"fold_{args.fold}"
    fold_output = args.output_root / f"fold_{args.fold}"
    experiment_output = fold_output / "tc_projection"
    if experiment_output.exists():
        raise RuntimeError(
            f"TC projection output already exists; inspect before rerun: {experiment_output}"
        )
    experiment_output.mkdir(parents=True)
    manifest_paths = {
        name: fold_source / f"{name}.csv"
        for name in (
            "expert_fit",
            "expert_stop",
            "gate_fit",
            "gate_stop_projector_fit",
            "projector_stop",
            "outer_eval",
        )
    }
    raw_subsets = {
        name: _read_manifest_rows(path) for name, path in manifest_paths.items()
    }
    update_audit = audit_pipeline_update_sets(
        {name: raw_subsets[name] for name in (
            "expert_fit",
            "gate_fit",
            "gate_stop_projector_fit",
            "projector_stop",
            "outer_eval",
        )}
    )

    dependencies = require_training_dependencies()
    cuda_available = torch.cuda.is_available()
    device_name = (
        "cuda" if args.device == "auto" and cuda_available else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_name)
    seed = 20260927 + args.fold
    gate_epochs = 2 if args.smoke_run else args.gate_epochs
    projector_epochs = 2 if args.smoke_run else args.projector_epochs
    expert_dir = fold_output / "expert"
    baseline_dir = fold_output / "baseline"
    expert_manifest = fold_output / "manifests" / "expert_training.csv"
    for required in (
        expert_dir / "selection_lock.json",
        baseline_dir / "selection_lock.json",
        expert_dir / "best_model.pt",
        baseline_dir / "best_model.pt",
        expert_manifest,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    model, mapping, _ = _load_frozen_expert(
        expert_dir, expert_manifest, args.dataset_root, device, dependencies
    )
    _, evaluation_transform = create_transforms(224, dependencies["transforms"])
    inner_records = {
        name: load_manifest_records(manifest_paths[name], args.dataset_root)
        for name in ("gate_fit", "gate_stop_projector_fit", "projector_stop")
    }
    inner_caches = {
        name: extract_cache(
            model,
            records,
            mapping,
            evaluation_transform,
            dependencies,
            device,
            name,
        )
        for name, records in inner_records.items()
    }

    torch.manual_seed(seed)
    gate_network = copy.deepcopy(model.gate_network).to(device)
    _reset_module_parameters(gate_network)
    selected_gate = train_oos_gate(
        gate_network,
        inner_caches["gate_fit"],
        inner_caches["gate_stop_projector_fit"],
        epochs=gate_epochs,
        seed=seed,
        batch_size=args.batch_size,
    )
    gate_dir = experiment_output / "gate"
    gate_dir.mkdir()
    gate_path = gate_dir / "best_model.pt"
    torch.save(selected_gate.state_dict, gate_path)
    _write_json(gate_dir / "selection.json", {
        "epoch": selected_gate.epoch,
        "stop_metrics": selected_gate.stop_metrics,
        "state_sha256": selected_gate.state_sha256,
    })
    gate_network.load_state_dict(selected_gate.state_dict, strict=True)
    gate_network.eval().requires_grad_(False)

    with torch.no_grad():
        projection_fit_gate = torch.sigmoid(
            gate_network(inner_caches["gate_stop_projector_fit"]["features"])
        )
        projection_stop_gate = torch.sigmoid(
            gate_network(inner_caches["projector_stop"]["features"])
        )
    projection_fit = build_projection_cache(
        projection_fit_gate, inner_caches["gate_stop_projector_fit"]
    )
    projection_stop = build_projection_cache(
        projection_stop_gate, inner_caches["projector_stop"]
    )
    q0_metrics = calculate_probability_metrics(
        projection_stop["q0"], projection_stop["labels"]
    )

    def train_grid(
        fit_cache: Mapping[str, torch.Tensor],
        stop_cache: Mapping[str, torch.Tensor],
        hard_negative_weight: float = 0.0,
    ) -> list[ProjectionCandidate]:
        return train_registered_projection_candidates(
            lambda epsilon: train_projection_head(
                fit_cache,
                stop_cache,
                epsilon_target=epsilon,
                epochs=projector_epochs,
                seed=seed,
                batch_size=args.batch_size,
                hard_negative_weight=hard_negative_weight,
            )
        )

    nll_candidates = train_grid(projection_fit, projection_stop)
    hn_candidates = train_grid(projection_fit, projection_stop, 0.1)
    selected = select_projection_candidate(nll_candidates, q0_metrics)
    selected_hn = select_projection_candidate(hn_candidates, q0_metrics)
    unbounded = train_projection_head(
        projection_fit,
        projection_stop,
        epsilon_target=0.0,
        epochs=projector_epochs,
        seed=seed,
        batch_size=args.batch_size,
        apply_safety=False,
    )
    selected_paths: dict[str, Path] = {}
    for name, candidate in (
        ("main", selected),
        ("hard_negative", selected_hn),
        ("unbounded", unbounded),
    ):
        path = _save_candidate(experiment_output / "projectors" / name, candidate)
        if path is not None:
            selected_paths[f"projector_{name}"] = path

    candidate_summaries = {
        "nll": [_candidate_summary(candidate) for candidate in nll_candidates],
        "hard_negative": [
            _candidate_summary(candidate) for candidate in hn_candidates
        ],
    }
    for family, candidates in (("nll", nll_candidates), ("hard_negative", hn_candidates)):
        for candidate in candidates:
            candidate_dir = (
                experiment_output
                / "epsilon_candidates"
                / family
                / f"epsilon_{candidate.epsilon_target:.3f}"
            )
            path = _save_candidate(candidate_dir, candidate)
            assert path is not None
            selected_paths[
                f"{family}_epsilon_{candidate.epsilon_target:.3f}"
            ] = path

    ablation_selections: dict[str, ProjectionCandidate | FallbackSelection] = {}
    fit_ablations = build_evidence_ablations(projection_fit["evidence"])
    stop_ablations = build_evidence_ablations(projection_stop["evidence"])
    for name in fit_ablations:
        ablation_fit = {**projection_fit, "evidence": fit_ablations[name]}
        ablation_stop = {**projection_stop, "evidence": stop_ablations[name]}
        ablation_candidates = train_grid(ablation_fit, ablation_stop)
        selection = select_projection_candidate(ablation_candidates, q0_metrics)
        ablation_selections[name] = selection
        path = _save_candidate(experiment_output / "ablations" / name, selection)
        if path is not None:
            selected_paths[f"ablation_{name}"] = path

    run_config = {
        "protocol_version": "tc_oos_rsg_v1",
        "fold": args.fold,
        "seed": seed,
        "gate_epochs": gate_epochs,
        "projector_epochs": projector_epochs,
        "batch_size": args.batch_size,
        "epsilon_target_grid": list(EPSILON_TARGET_GRID),
        "gate_objective": "final_nll_plus_0.1_weighted_regret",
        "projector_objective": "nll",
        "hard_negative_weight": 0.1,
        "projection_evidence_dim": 18,
        "outer_tensors_loaded_during_selection": False,
        "smoke_run": args.smoke_run,
    }
    config_path = experiment_output / "run_config.json"
    _write_json(config_path, run_config)
    model_paths = {
        "expert": expert_dir / "best_model.pt",
        "expert_lock": expert_dir / "selection_lock.json",
        "baseline": baseline_dir / "best_model.pt",
        "baseline_lock": baseline_dir / "selection_lock.json",
        "gate": gate_path,
        **selected_paths,
        "configuration": config_path,
    }
    selected_epsilon = (
        selected.epsilon_target if isinstance(selected, ProjectionCandidate) else 0.0
    )
    selected_hash = (
        selected.state_sha256 if isinstance(selected, ProjectionCandidate) else "0" * 64
    )
    lock_payload = build_selection_lock_payload(
        manifest_paths=manifest_paths,
        model_paths=model_paths,
        selected_epsilon=selected_epsilon,
        selection_reason=(
            "minimum_safe_nll"
            if isinstance(selected, ProjectionCandidate)
            else selected.reason
        ),
        gate_state_sha256=selected_gate.state_sha256,
        projector_state_sha256=selected_hash,
    )
    lock_payload.update(
        {
            "gate_selection": {
                "epoch": selected_gate.epoch,
                "stop_metrics": selected_gate.stop_metrics,
            },
            "q0_selection_baseline": q0_metrics,
            "main_selection": _candidate_summary(selected),
            "hard_negative_selection": _candidate_summary(selected_hn),
            "unbounded_selection": _candidate_summary(unbounded),
            "epsilon_candidates": candidate_summaries,
            "ablation_selections": {
                name: _candidate_summary(selection)
                for name, selection in ablation_selections.items()
            },
            "update_set_audit": update_audit,
        }
    )
    events: list[str] = []

    def load_outer_tensors():
        outer_records = load_manifest_records(
            manifest_paths["outer_eval"], args.dataset_root
        )
        outer_cache = extract_cache(
            model,
            outer_records,
            mapping,
            evaluation_transform,
            dependencies,
            device,
            "outer_eval_after_lock",
        )
        m0_probabilities = _extract_baseline_probabilities(
            baseline_dir,
            outer_records,
            evaluation_transform,
            dependencies,
            device,
        )
        return outer_records, outer_cache, m0_probabilities

    outer_records, outer_expert_cache, m0_probabilities = lock_before_outer_evaluation(
        experiment_output / "selection_lock.json",
        lock_payload,
        load_outer_tensors,
        event_trace=events,
    )
    with torch.no_grad():
        outer_gate = torch.sigmoid(gate_network(outer_expert_cache["features"]))
    outer_projection = build_projection_cache(outer_gate, outer_expert_cache)
    methods = evaluate_method_matrix(
        outer_projection,
        selected=selected,
        hard_negative_selected=selected_hn,
        unbounded_selected=unbounded,
        m0_probabilities=m0_probabilities,
    )
    method_metrics = {}
    for name, result in methods.items():
        probabilities = result["probabilities"]
        assert isinstance(probabilities, torch.Tensor)
        route = result.get("route")
        assert route is None or isinstance(route, torch.Tensor)
        _write_prediction_csv(
            experiment_output / "outer_predictions" / f"{name}.csv",
            outer_records,
            probabilities,
            route,
            args.fold,
        )
        method_metrics[name] = {
            **calculate_probability_metrics(
                probabilities, outer_projection["labels"]
            ),
            "max_target_harm": result["max_target_harm"],
        }

    outer_ablated_evidence = build_evidence_ablations(outer_projection["evidence"])
    ablation_metrics = {}
    for name, selection in ablation_selections.items():
        ablated_cache = {
            **outer_projection,
            "evidence": outer_ablated_evidence[name],
        }
        probabilities, route = _selected_projection(ablated_cache, selection)
        _write_prediction_csv(
            experiment_output / "ablation_predictions" / f"{name}.csv",
            outer_records,
            probabilities,
            route,
            args.fold,
        )
        ablation_metrics[name] = calculate_probability_metrics(
            probabilities, outer_projection["labels"]
        )

    epsilon_metrics = {}
    for candidate in nll_candidates:
        probabilities, route = _selected_projection(outer_projection, candidate)
        name = f"epsilon_{candidate.epsilon_target:.3f}"
        _write_prediction_csv(
            experiment_output / "epsilon_predictions" / f"{name}.csv",
            outer_records,
            probabilities,
            route,
            args.fold,
        )
        epsilon_metrics[name] = calculate_probability_metrics(
            probabilities, outer_projection["labels"]
        )

    summary = {
        "fold": args.fold,
        "event_trace": events,
        "method_metrics": method_metrics,
        "ablation_metrics": ablation_metrics,
        "epsilon_metrics": epsilon_metrics,
        "selection_lock": str(experiment_output / "selection_lock.json"),
    }
    _write_json(experiment_output / "outer_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
