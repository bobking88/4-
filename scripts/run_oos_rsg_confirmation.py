"""Confirm OOS-RSG routing with independent experts and locked final evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from hrgv_network import HierarchicalRiskGatedVerificationNet
from mineral_hierarchy import validate_species_role_mapping
from run_seen_unseen_gate_study import (
    evaluate_gate,
    extract_cache,
    final_probabilities,
    save_predictions,
    select_earliest_best,
    sha256,
    write_json,
)
from train_hrgv_mineral_classifier import build_role_matrix
from train_mineral_classifier import (
    create_transforms,
    load_manifest_records,
    require_training_dependencies,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERT_GATE_SEEDS = {
    20260924: 20260934,
    20260925: 20260935,
    20260926: 20260936,
}


def audit_confirmation_records(
    expert_records,
    seen_records,
    unseen_records,
    gate_stop_records,
    final_eval_records,
) -> dict[str, object]:
    named = {
        "expert": expert_records,
        "gate_seen": seen_records,
        "gate_unseen": unseen_records,
        "gate_stop": gate_stop_records,
        "final_eval": final_eval_records,
    }
    if any(not records for records in named.values()):
        raise ValueError("Every confirmation manifest must be nonempty.")
    if {record.split for record in expert_records} != {"train", "val"}:
        raise ValueError("Expert manifest must contain train and val rows only.")
    expected_splits = {
        "gate_seen": "train",
        "gate_unseen": "train",
        "gate_stop": "val",
        "final_eval": "test",
    }
    for name, expected in expected_splits.items():
        if {record.split for record in named[name]} != {expected}:
            raise ValueError(f"{name} must contain only {expected} rows.")

    stratum = lambda record: (record.mineral_label, record.four_class_label)
    seen_counts = Counter(map(stratum, seen_records))
    unseen_counts = Counter(map(stratum, unseen_records))
    if seen_counts != unseen_counts:
        raise ValueError("Seen and unseen species-role counts do not match exactly.")

    for name, records in named.items():
        image_ids = [record.image_id for record in records]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError(f"{name} contains duplicate image IDs.")

    expert_fit_ids = {
        record.image_id for record in expert_records if record.split == "train"
    }
    if not {record.image_id for record in seen_records}.issubset(expert_fit_ids):
        raise ValueError("Seen gate supervision must be a subset of expert_fit.")

    group_sets = {
        name: {record.split_group_id for record in records}
        for name, records in named.items()
    }
    unexpected_pairs = (
        ("expert", "gate_unseen"),
        ("expert", "gate_stop"),
        ("expert", "final_eval"),
        ("gate_seen", "gate_unseen"),
        ("gate_seen", "gate_stop"),
        ("gate_seen", "final_eval"),
        ("gate_unseen", "gate_stop"),
        ("gate_unseen", "final_eval"),
        ("gate_stop", "final_eval"),
    )
    overlaps = {
        f"{left}|{right}": len(group_sets[left] & group_sets[right])
        for left, right in unexpected_pairs
        if group_sets[left] & group_sets[right]
    }
    if overlaps:
        raise ValueError(f"Confirmation subsets share duplicate groups: {overlaps}")

    return {
        "counts": {name: len(records) for name, records in named.items()},
        "exact_seen_unseen_strata_match": True,
        "gate_seen_subset_of_expert_fit": True,
        "unexpected_group_overlap": overlaps,
        "final_eval_locked": True,
        "strata": {
            f"{mineral}|{role}": count
            for (mineral, role), count in sorted(unseen_counts.items())
        },
    }


def _load_expert(
    expert_dir: Path,
    expert_records,
    dependencies,
    device: torch.device,
):
    environment_path = expert_dir / "environment.json"
    checkpoint_path = expert_dir / "best_model.pt"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    mapping = validate_species_role_mapping(expert_records)
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, mapping, environment, environment_path, checkpoint_path


def _new_paired_gate_template(model, seed: int):
    torch.manual_seed(seed)
    template = copy.deepcopy(model.gate_network).requires_grad_(True)
    for layer in template.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return template


def _train_gate(
    template,
    train_cache,
    stop_cache,
    seed: int,
    epochs: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, float]]:
    torch.manual_seed(seed)
    gate_network = copy.deepcopy(template)
    optimizer = torch.optim.AdamW(
        gate_network.parameters(), lr=0.001, weight_decay=0.0001
    )
    history: list[dict[str, float]] = []
    states: list[dict[str, torch.Tensor]] = []
    for epoch in range(1, epochs + 1):
        gate_network.train()
        batch_losses: list[float] = []
        permutation = torch.randperm(
            len(train_cache["labels"]), device=train_cache["labels"].device
        )
        for indices in permutation.split(256):
            gate = torch.sigmoid(gate_network(train_cache["features"][indices]))
            batch_cache = {
                key: values[indices] for key, values in train_cache.items()
            }
            final = final_probabilities(gate, batch_cache)
            epsilon = torch.finfo(final.dtype).eps
            loss = F.nll_loss(
                final.clamp_min(epsilon).log(), train_cache["labels"][indices]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        gate_network.eval()
        with torch.no_grad():
            stop_gate = torch.sigmoid(gate_network(stop_cache["features"]))
            row = {
                "epoch": epoch,
                "mean_training_batch_loss": sum(batch_losses) / len(batch_losses),
                **evaluate_gate(stop_gate, stop_cache),
            }
        history.append(row)
        states.append(copy.deepcopy(gate_network.state_dict()))
    selected = select_earliest_best(history, "final_nll_nats")
    selected_state = states[int(selected["epoch"]) - 1]
    return selected_state, history, selected


def _state_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_selected_gate(template, state, cache):
    gate_network = copy.deepcopy(template)
    gate_network.load_state_dict(state)
    gate_network.eval()
    with torch.no_grad():
        gate = torch.sigmoid(gate_network(cache["features"]))
    return gate, evaluate_gate(gate, cache)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked seen/unseen OOS-RSG confirmation on completed experts."
    )
    protocol_dir = ROOT / "outputs/training/oos_rsg_confirmation_manifests_v1"
    parser.add_argument("--protocol-dir", type=Path, default=protocol_dir)
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=ROOT / "outputs/training/oos_rsg_confirmation_v1/experts",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/training/oos_rsg_confirmation_v1/gates",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("Epoch count must be positive.")
    dependencies = require_training_dependencies()
    paths = {
        name: args.protocol_dir / f"{name}.csv"
        for name in ("expert", "gate_seen", "gate_unseen", "gate_stop", "final_eval")
    }
    records = {
        name: load_manifest_records(path, args.dataset_root)
        for name, path in paths.items()
    }
    audit = audit_confirmation_records(
        records["expert"],
        records["gate_seen"],
        records["gate_unseen"],
        records["gate_stop"],
        records["final_eval"],
    )
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_name)
    torch.set_num_threads(4)
    _, evaluation_transform = create_transforms(224, dependencies["transforms"])
    args.output_root.mkdir(parents=True, exist_ok=True)

    root_summary: dict[str, object] = {
        "scope": "three-independent-expert OOS-RSG confirmation",
        "audit": audit,
        "expert_gate_seeds": EXPERT_GATE_SEEDS,
        "primary_endpoint": "final_eval final_nll_nats",
        "gate_objective": "final verifier-aware NLL",
        "gate_selection": "lowest gate_stop final NLL; earliest tie",
        "final_eval_access": "after both seen and unseen gate states are locked",
        "experts": {},
    }

    for expert_seed, gate_seed in EXPERT_GATE_SEEDS.items():
        expert_dir = args.expert_root / f"seed{expert_seed}"
        expert_output = args.output_root / f"seed{expert_seed}"
        completion_marker = expert_output / "expert_summary.json"
        if completion_marker.exists():
            root_summary["experts"][str(expert_seed)] = json.loads(
                completion_marker.read_text(encoding="utf-8")
            )
            print(f"SKIP complete gate confirmation: seed {expert_seed}", flush=True)
            continue
        if expert_output.exists():
            raise RuntimeError(
                f"Incomplete gate output exists; inspect before rerun: {expert_output}"
            )
        expert_output.mkdir(parents=True)

        model, mapping, environment, environment_path, checkpoint_path = _load_expert(
            expert_dir, records["expert"], dependencies, device
        )
        if int(environment.get("seed", -1)) != expert_seed:
            raise ValueError(f"Expert seed mismatch in {environment_path}.")
        frozen_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        training_caches = {
            "seen": extract_cache(
                model,
                records["gate_seen"],
                mapping,
                evaluation_transform,
                dependencies,
                device,
                f"seed{expert_seed}-seen",
            ),
            "unseen": extract_cache(
                model,
                records["gate_unseen"],
                mapping,
                evaluation_transform,
                dependencies,
                device,
                f"seed{expert_seed}-unseen",
            ),
            "gate_stop": extract_cache(
                model,
                records["gate_stop"],
                mapping,
                evaluation_transform,
                dependencies,
                device,
                f"seed{expert_seed}-gate-stop",
            ),
        }
        template = _new_paired_gate_template(model, gate_seed)
        selected_states = {}
        selected_rows = {}
        state_hashes = {}
        for source in ("seen", "unseen"):
            state, history, selected = _train_gate(
                template,
                training_caches[source],
                training_caches["gate_stop"],
                gate_seed,
                args.epochs,
            )
            source_dir = expert_output / source
            source_dir.mkdir()
            state_path = source_dir / "gate_state.pt"
            torch.save(state, state_path)
            write_json(source_dir / "history.json", history)
            write_json(source_dir / "selected_gate_stop_metrics.json", selected)
            selected_states[source] = state
            selected_rows[source] = selected
            state_hashes[source] = _state_hash(state_path)
            print(
                f"LOCK expert={expert_seed} source={source} "
                f"epoch={selected['epoch']} gate_stop_nll={selected['final_nll_nats']:.6f}",
                flush=True,
            )

        selection_lock = {
            "expert_seed": expert_seed,
            "gate_seed": gate_seed,
            "selected_before_final_eval": True,
            "selected_gate_stop_metrics": selected_rows,
            "gate_state_sha256": state_hashes,
            "expert_checkpoint_sha256": sha256(checkpoint_path),
        }
        write_json(expert_output / "selection_lock.json", selection_lock)

        final_cache = extract_cache(
            model,
            records["final_eval"],
            mapping,
            evaluation_transform,
            dependencies,
            device,
            f"seed{expert_seed}-final-eval-after-lock",
        )
        equal_gate = torch.full(
            (len(final_cache["labels"]), 1), 0.5, device=device
        )
        methods: dict[str, object] = {
            "equal": evaluate_gate(equal_gate, final_cache),
            "original_joint_gate": evaluate_gate(
                final_cache["original_gate"], final_cache
            ),
        }
        save_predictions(
            expert_output / "equal_final_predictions.csv",
            records["final_eval"],
            equal_gate,
            final_cache,
        )
        save_predictions(
            expert_output / "original_joint_gate_final_predictions.csv",
            records["final_eval"],
            final_cache["original_gate"],
            final_cache,
        )
        for source in ("seen", "unseen"):
            gate, metrics = _evaluate_selected_gate(
                template, selected_states[source], final_cache
            )
            methods[source] = metrics
            save_predictions(
                expert_output / source / "final_predictions.csv",
                records["final_eval"],
                gate,
                final_cache,
            )
            write_json(expert_output / source / "final_metrics.json", metrics)

        unchanged = all(
            torch.equal(frozen_state[key], value.detach().cpu())
            for key, value in model.state_dict().items()
        )
        gradients_absent = all(parameter.grad is None for parameter in model.parameters())
        if not unchanged or not gradients_absent:
            raise AssertionError("Frozen expert changed during gate confirmation.")
        expert_summary = {
            "expert_seed": expert_seed,
            "gate_seed": gate_seed,
            "selection_lock": selection_lock,
            "final_eval_loaded_after_selection_lock": True,
            "methods": methods,
            "frozen_expert_parameters_and_buffers_unchanged": unchanged,
            "frozen_expert_gradients_absent": gradients_absent,
            "hashes": {
                "expert_environment": sha256(environment_path),
                "expert_checkpoint": sha256(checkpoint_path),
                **{name: sha256(path) for name, path in paths.items()},
                "script": sha256(Path(__file__)),
            },
        }
        write_json(completion_marker, expert_summary)
        root_summary["experts"][str(expert_seed)] = expert_summary
        write_json(args.output_root / "summary.json", root_summary)

    write_json(args.output_root / "summary.json", root_summary)


if __name__ == "__main__":
    main()
