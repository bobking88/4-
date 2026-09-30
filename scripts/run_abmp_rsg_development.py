"""Develop ABMP v2 on the already exposed Fold 0 only."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch
from torch.nn import functional as F

from abmp_rsg import (
    AdaptiveBudgetPolicy, apply_abmp_projection,
    audit_abmp_projection, risk_supervision_targets,
)
from run_tc_oos_rsg_experiments import (
    _file_sha256, _load_frozen_expert, _read_manifest_rows,
    _write_json, _write_prediction_csv, audit_pipeline_update_sets,
    build_projection_cache, calculate_probability_metrics, state_dict_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ((.60, .05, .005), (.70, .10, .010), (.80, .15, .010))


def development_grid() -> list[dict[str, float]]:
    return [
        {"epsilon_max": eps, "tau_p": p, "tau_m": m, "delta": delta, "lambda_cal": offset}
        for eps, (p, m, delta), offset in itertools.product((.02, .04, .08), ANCHORS, (-.5, 0., .5))
    ]


def prepare_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def _forward(model, cache, config, lambda_cal=0.0):
    output = model(cache["evidence"], config["epsilon_max"], lambda_cal)
    projected = apply_abmp_projection(
        cache["q0"], cache["q_candidate"], output["raw_route"], output["epsilon"],
        tau_p=config["tau_p"], tau_m=config["tau_m"], delta=config["delta"],
        mode=config.get("mode", "full"),
    )
    return output, projected


def train_policy(
    fit_cache: Mapping[str, torch.Tensor], stop_cache: Mapping[str, torch.Tensor],
    config: dict, *, epochs: int = 30, seed: int = 20260930,
) -> dict:
    if epochs < 1 or len(fit_cache["labels"]) == 0 or len(stop_cache["labels"]) == 0:
        raise ValueError("Training and stopping sets must be nonempty; epochs positive.")
    torch.manual_seed(seed)
    model = AdaptiveBudgetPolicy().to(fit_cache["evidence"].device)
    initial_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    targets = risk_supervision_targets(fit_cache["q0"], fit_cache["q_candidate"], fit_cache["labels"], config["epsilon_max"])
    alpha = config.get("alpha", .25)
    beta = config.get("beta", .10)
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices in torch.randperm(len(fit_cache["labels"]), device=fit_cache["labels"].device).split(256):
            batch = {key: value[indices] for key, value in fit_cache.items() if torch.is_tensor(value)}
            output, projected = _forward(model, batch, config)
            loss = F.nll_loss(projected["final_probabilities"].clamp_min(torch.finfo(output["epsilon"].dtype).tiny).log(), batch["labels"])
            if alpha:
                loss = loss + alpha * F.binary_cross_entropy_with_logits(output["route_logits"], targets["route_target"][indices])
            mask = targets["budget_mask"][indices]
            if beta and bool(mask.any()):
                loss = loss + beta * F.binary_cross_entropy_with_logits(output["budget_logits"][mask], targets["budget_target"][indices][mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            _, projected = _forward(model, stop_cache, config)
            metrics = calculate_probability_metrics(projected["final_probabilities"], stop_cache["labels"])
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), **metrics})
        if not math.isfinite(metrics["nll"]):
            raise RuntimeError("Nonfinite stopping loss.")
        if best is None or metrics["nll"] < best["stop_metrics"]["nll"]:
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best = {"state_dict": state, "epoch": epoch, "stop_metrics": metrics, "state_sha256": state_dict_sha256(state)}
    return {**best, "config": {**config, "lambda_cal": 0.0}, "history": history, "initial_state_sha256": initial_sha, "seed": seed}


def _ece(probabilities, labels) -> float:
    confidence, predictions = probabilities.max(1)
    result = 0.0
    for index in range(15):
        mask = (confidence > index / 15) & (confidence <= (index + 1) / 15)
        if bool(mask.any()):
            result += float(mask.double().mean() * ((predictions[mask] == labels[mask]).double().mean() - confidence[mask].double().mean()).abs())
    return result


def evaluate_policy(cache: Mapping[str, torch.Tensor], trained: dict, *, lambda_cal: float = 0.0) -> dict:
    model = AdaptiveBudgetPolicy().to(cache["evidence"].device)
    model.load_state_dict(trained["state_dict"])
    model.eval()
    config = trained["config"]
    with torch.no_grad():
        output, projected = _forward(model, cache, config, lambda_cal)
    probabilities = projected["final_probabilities"]
    audit = audit_abmp_projection(projected, cache["q0"], cache["q_candidate"], cache["labels"], config["epsilon_max"], config["delta"])
    harmful = (cache["q0"][:, :1] - cache["q_candidate"][:, :1]) > 1e-12
    anchors = projected["anchor_mask"]
    audit["posterior_conditional_activation_rate"] = float(projected["posterior_active"][harmful].double().mean()) if bool(harmful.any()) else 0.0
    audit["margin_conditional_activation_rate"] = float(projected["margin_active"][anchors].double().mean()) if bool(anchors.any()) else 0.0
    eps = output["epsilon"].flatten()
    for name, quantile in (("p05", .05), ("p50", .50), ("p95", .95)):
        audit[f"epsilon_{name}"] = float(torch.quantile(eps, quantile))
    metrics = calculate_probability_metrics(probabilities, cache["labels"])
    metrics["ece"] = _ece(probabilities, cache["labels"])
    return {"config": {**config, "lambda_cal": lambda_cal}, "metrics": metrics, "audit": audit, "probabilities": probabilities, "projection": projected}


def select_development_candidate(candidates: list[dict], q0_metrics: dict) -> dict | None:
    if not math.isfinite(q0_metrics["target_recall"]):
        return None
    eligible = []
    for item in candidates:
        metrics, audit = item["metrics"], item["audit"]
        if not all(math.isfinite(metrics.get(key, math.nan)) for key in ("nll", "target_recall")):
            continue
        if any(value != 0 for key, value in audit.items() if key.endswith("violations")):
            continue
        if max(audit["posterior_conditional_activation_rate"], audit["margin_conditional_activation_rate"]) < .005:
            continue
        if min(audit["max_difference_from_q0"], audit["max_difference_from_unbounded"]) <= 1e-6:
            continue
        if audit["anchor_count"] < 1 or audit["anchor_retention_rate"] != 1.0:
            continue
        if audit["epsilon_std"] <= 1e-6:
            continue
        if metrics["target_recall"] < q0_metrics["target_recall"] - .01 - 1e-12:
            continue
        eligible.append(item)
    if not eligible:
        return None
    best_nll = min(item["metrics"]["nll"] for item in eligible)
    tied = [item for item in eligible if item["metrics"]["nll"] <= best_nll + 1e-4]
    return min(tied, key=lambda item: (item["config"]["epsilon_max"], item["audit"]["epsilon_mean"], -item["audit"]["anchor_coverage"]))


def load_fold_zero_caches(protocol_dir: Path, source_dir: Path, dataset_root: Path, device: torch.device):
    from run_seen_unseen_gate_study import extract_cache
    from train_mineral_classifier import create_transforms, load_manifest_records, require_training_dependencies

    source = source_dir / "fold_0"
    manifests = {name: protocol_dir / "fold_0" / f"{name}.csv" for name in ("expert_fit", "expert_stop", "gate_fit", "gate_stop_projector_fit", "projector_stop", "outer_eval")}
    audit = audit_pipeline_update_sets({name: _read_manifest_rows(path) for name, path in manifests.items()})
    old_lock = json.loads((source / "tc_projection/selection_lock.json").read_text(encoding="utf-8"))
    for name, path in manifests.items():
        if _file_sha256(path) != old_lock["manifest_sha256"][name]:
            raise ValueError(f"Fold 0 manifest hash changed: {name}")
    expert_path = source / "expert/best_model.pt"
    gate_path = source / "tc_projection/gate/best_model.pt"
    for name, path in (("expert", expert_path), ("gate", gate_path)):
        if _file_sha256(path) != old_lock["model_file_sha256"][name]:
            raise ValueError(f"Frozen weight hash changed: {name}")
    dependencies = require_training_dependencies()
    model, mapping, _ = _load_frozen_expert(source / "expert", source / "manifests/expert_training.csv", dataset_root, device, dependencies)
    gate = copy.deepcopy(model.gate_network)
    gate.load_state_dict(torch.load(gate_path, map_location=device, weights_only=True))
    gate.eval().requires_grad_(False)
    _, transform = create_transforms(224, dependencies["transforms"])
    caches, records = {}, {}
    for name in ("gate_stop_projector_fit", "projector_stop", "outer_eval"):
        records[name] = load_manifest_records(manifests[name], dataset_root)
        cached = extract_cache(model, records[name], mapping, transform, dependencies, device, f"abmp_development_{name}")
        with torch.no_grad():
            candidate_gate = torch.sigmoid(gate(cached["features"]))
        caches[name] = {key: value.detach().cpu() for key, value in build_projection_cache(candidate_gate, cached).items()}
    provenance = {
        "manifest_sha256": {name: _file_sha256(path) for name, path in manifests.items()},
        "expert_sha256": _file_sha256(expert_path), "gate_sha256": _file_sha256(gate_path),
        "group_audit": audit, "source_protocol": "tc_oos_rsg_v1",
    }
    return caches, records, provenance


def _serializable_result(result):
    return {key: result[key] for key in ("config", "metrics", "audit")}


def _save_policy(folder: Path, trained: dict):
    folder.mkdir(parents=True, exist_ok=False)
    torch.save(trained["state_dict"], folder / "best_model.pt")
    _write_json(folder / "training.json", {key: value for key, value in trained.items() if key != "state_dict"})


def _write_routing_diagnostics(path, records, evaluated):
    projection = evaluated["projection"]
    keys = ("epsilon", "raw_route", "projected_route", "posterior_cap", "margin_cap", "anchor_mask", "posterior_active", "margin_active", "final_target_margin", "positive_target_harm")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split_group_id", "class_id", *keys])
        for index, record in enumerate(records):
            writer.writerow([record.image_id, record.split_group_id, record.class_id, *[float(projection[key][index]) for key in keys]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "outputs/training/tc_oos_rsg_manifests_v1")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/training/tc_oos_rsg_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/training/abmp_rsg_v2/development_fold_0")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    prepare_output_directory(args.output_dir)
    started = datetime.now(timezone.utc).isoformat()
    caches, records, provenance = load_fold_zero_caches(args.protocol_dir, args.source_dir, args.dataset_root, torch.device(args.device))
    fit, stop, dev = (caches[name] for name in ("gate_stop_projector_fit", "projector_stop", "outer_eval"))
    q0_metrics = calculate_probability_metrics(dev["q0"], dev["labels"])
    _write_json(args.output_dir / "run_config.json", {"protocol": "abmp_rsg_v2", "purpose": "development_only", "epochs": args.epochs, "grid": development_grid(), "source": provenance, "started": started})
    candidates, trained_models = [], {}
    for index, config in enumerate([c for c in development_grid() if c["lambda_cal"] == 0.]):
        name = f"policy_{index:02d}"
        trained = train_policy(fit, stop, config, epochs=args.epochs)
        trained_models[name] = trained
        _save_policy(args.output_dir / "policies" / name, trained)
        for offset in (-.5, 0., .5):
            evaluated = evaluate_policy(dev, trained, lambda_cal=offset)
            candidates.append({**evaluated, "policy_name": name, "epoch": trained["epoch"], "state_sha256": trained["state_sha256"]})
        print(f"develop {name}: stop_epoch={trained['epoch']} stop_nll={trained['stop_metrics']['nll']:.6f}", flush=True)
    selected = select_development_candidate(candidates, q0_metrics)
    diagnostic = selected or min(candidates, key=lambda c: c["metrics"]["nll"])
    full_config = {**diagnostic["config"]}
    main_trained = trained_models[diagnostic["policy_name"]]
    methods = {"A0": {"metrics": q0_metrics}, "A1": {"metrics": calculate_probability_metrics(dev["q_candidate"], dev["labels"])}}
    _write_prediction_csv(args.output_dir / "predictions/A0.csv", records["outer_eval"], dev["q0"], None, 0)
    _write_prediction_csv(args.output_dir / "predictions/A1.csv", records["outer_eval"], dev["q_candidate"], dev["candidate_gate"], 0)
    variants = {
        "A2": {"mode": "none", "beta": 0.},
        "A4": {"mode": "posterior"},
        "A5": {"mode": "margin", "beta": 0.},
        "A7": {"alpha": 0.},
        "A8": {"beta": 0.},
    }
    for name in ("A2", "A4", "A5", "A6", "A7", "A8"):
        if name == "A6":
            trained = main_trained
        else:
            trained = train_policy(fit, stop, {**full_config, **variants[name]}, epochs=args.epochs)
            _save_policy(args.output_dir / "ablations" / name, trained)
        evaluated = evaluate_policy(dev, trained, lambda_cal=full_config["lambda_cal"])
        methods[name] = _serializable_result(evaluated)
        _write_prediction_csv(args.output_dir / f"predictions/{name}.csv", records["outer_eval"], evaluated["probabilities"], evaluated["projection"]["projected_route"], 0)
        _write_routing_diagnostics(args.output_dir / f"predictions/{name}_routing.csv", records["outer_eval"], evaluated)
        print(f"{name}: nll={evaluated['metrics']['nll']:.6f} recall={evaluated['metrics']['target_recall']:.6f}", flush=True)
    # v1 stays an independently trained historical fixed-budget comparison.
    from tc_oos_rsg import TargetRiskProjectionHead, apply_target_safe_projection
    old = args.source_dir / "fold_0/tc_projection"
    old_head = TargetRiskProjectionHead().eval()
    old_head.load_state_dict(torch.load(old / "projectors/main/best_model.pt", map_location="cpu", weights_only=True))
    old_lock = json.loads((old / "selection_lock.json").read_text(encoding="utf-8"))
    with torch.no_grad():
        raw = old_head(dev["evidence"])
        result = apply_target_safe_projection(dev["q0"], dev["q_candidate"], raw, old_lock["selected_epsilon_target"])
    methods["A3"] = {"metrics": calculate_probability_metrics(result["final_probabilities"], dev["labels"]), "fixed_epsilon": old_lock["selected_epsilon_target"]}
    _write_prediction_csv(args.output_dir / "predictions/A3.csv", records["outer_eval"], result["final_probabilities"], result["projected_route"], 0)
    summary = {
        "protocol": "abmp_rsg_v2", "fold": 0, "purpose": "development_only",
        "status": "DEVELOPMENT_GATE_PASSED" if selected else "DEVELOPMENT_GATE_FAILED",
        "selected": {**_serializable_result(selected), "policy_name": selected["policy_name"], "state_sha256": selected["state_sha256"]} if selected else None,
        "diagnostic_config": full_config, "candidate_count": len(candidates),
        "candidates": [{**_serializable_result(c), "policy_name": c["policy_name"], "epoch": c["epoch"]} for c in candidates],
        "methods": methods, "provenance": provenance,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.output_dir / "development_summary.json", summary)
    if selected:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        lock = {
            "protocol": "abmp_rsg_v2", "status": "development_locked_before_confirmation",
            "selected_config": selected["config"], "epochs": args.epochs,
            "policy_seed": 20260930, "alpha": .25, "beta": .10, "tau_r": .25,
            "confirmed_folds": [1, 2], "development_fold": 0, "code_commit": git_commit,
            "development_summary_sha256": _file_sha256(args.output_dir / "development_summary.json"),
            "spec_sha256": _file_sha256(ROOT / "docs/superpowers/specs/2026-09-30-abmp-rsg-net-v2-design.md"),
            "source": provenance, "locked_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(args.output_dir / "abmp_rsg_v2_lock.json", lock)
    print(summary["status"], flush=True)
    return summary


if __name__ == "__main__":
    main()
