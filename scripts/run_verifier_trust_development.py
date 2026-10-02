"""Registered verification-head experiment on exposed Fold 0 inner tables only."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_abmp_candidate_capacity import (
    BRANCH_NAMES, EXTRACTION_NAMES, build_component_branches, validate_sources,
)
from run_tc_oos_rsg_experiments import (
    _file_sha256, _read_manifest_rows, _write_json, calculate_probability_metrics,
    state_dict_sha256,
)
from tc_oos_rsg import TargetRiskProjectionHead, build_projection_evidence
from train_mineral_classifier import CLASS_LABELS
from verifier_risk import conditional_risk_projection, optimal_global_strength, verification_family


ROOT = Path(__file__).resolve().parents[1]
REPORT_NAME = "基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx"
VERIFIER_CONFIG = {"verifier_mode": "residual", "ti_threshold": .5,
                   "metallic_threshold": .5, "ti_strength": 1., "metallic_strength": 1.}


def load_subset(root: Path, name: str, input_hashes: dict) -> dict:
    if name not in EXTRACTION_NAMES:
        raise ValueError("Only the two approved inner development subsets may be loaded.")
    folder = root / "outputs/theory/abmp_candidate_capacity_v1" / name
    paths = {"component_predictions.csv": folder / "component_predictions.csv",
             "routing_evidence.csv": folder / "routing_evidence.csv",
             "manifest.csv": root / "outputs/training/tc_oos_rsg_manifests_v1/fold_0" / f"{name}.csv"}
    hashes = {f"{name}/{file}": _file_sha256(path) for file, path in paths.items()}
    if any(input_hashes.get(key) != digest for key, digest in hashes.items()):
        raise ValueError(f"Registered input hash mismatch in {name}.")
    records = _read_manifest_rows(paths["manifest.csv"])
    evidence_rows = _read_manifest_rows(paths["routing_evidence.csv"])
    components = _read_manifest_rows(paths["component_predictions.csv"])
    ids = [r["image_id"] for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest identity is empty or duplicated.")
    for row in records:
        label = int(row["four_class_id"])
        if (not 0 <= label < 4 or row["four_class_label"] != CLASS_LABELS[label]
                or row.get("outer_fold") != "0" or row.get("tc_subset") != name):
            raise ValueError("Manifest label or subset identity mismatch.")

    def aligned(rows, mineral=False):
        if len(rows) != len(records):
            raise ValueError("CSV identity row count mismatch.")
        for manifest, row in zip(records, rows, strict=True):
            if (manifest["image_id"] != row["image_id"]
                    or manifest["split_group_id"] != row["split_group_id"]
                    or int(manifest["four_class_id"]) != int(row["true_class_id"])
                    or (mineral and manifest["mineral_label"] != row["mineral_label"])):
                raise ValueError("CSV identity or label alignment mismatch.")

    aligned(evidence_rows)
    if set(r["branch"] for r in components) != set(BRANCH_NAMES):
        raise ValueError("Component branches do not match the frozen audit.")
    branches = {}
    for branch in BRANCH_NAMES:
        rows = [r for r in components if r["branch"] == branch]
        aligned(rows, mineral=True)
        branches[branch] = torch.tensor([[float(r[f"prob_{i}"]) for i in range(4)] for r in rows], dtype=torch.float64)
        if [int(r["predicted_class_id"]) for r in rows] != branches[branch].argmax(1).tolist():
            raise ValueError("Stored predicted classes do not replay.")
    column = lambda field: torch.tensor([[float(r[field])] for r in evidence_rows], dtype=torch.float64)
    cache = {"direct": branches["direct_pre"], "mapped": branches["mapped_pre"],
             "original_gate": column("original_gate"), "ti": column("ti_probability"), "metal": column("metal_probability")}
    gate = column("candidate_gate")
    replay = build_component_branches(cache, gate, VERIFIER_CONFIG)
    residual = max(float((branches[b]-replay[b]).abs().max()) for b in BRANCH_NAMES)
    scale_residual = float((column("target_scale")-replay["target_scale"]).abs().max())
    if max(residual, scale_residual) > 1e-12:
        raise ValueError("Frozen component or verifier replay mismatch.")
    contradiction = torch.relu(1-2*cache["ti"]) + torch.relu(1-2*cache["metal"])
    return {"p": branches["oos_pre"], "contradiction": contradiction,
            "labels": torch.tensor([int(r["four_class_id"]) for r in records]),
            "evidence": build_projection_evidence(branches["fixed_verified"], branches["oos_verified"], gate, cache["ti"], cache["metal"]),
            "records": records, "hashes": hashes, "source_replay_max_abs_residual": max(residual, scale_residual)}


def predict_head(head: torch.nn.Module, data: dict, mode: str) -> dict:
    if mode not in ("A3", "A4"):
        raise ValueError("Unknown learned arm.")
    head.eval()
    with torch.no_grad():
        raw = head(data["evidence"])
        if mode == "A3":
            return {"raw": raw, "gamma": raw,
                    "probabilities": verification_family(data["p"], data["contradiction"], raw)}
        return dict(conditional_risk_projection(data["p"], data["contradiction"], raw), raw=raw)


def fit_head(fit: dict, stop: dict, mode: str, seed: int, budget: dict) -> dict:
    if mode not in ("A3", "A4"):
        raise ValueError("Unknown learned arm.")
    if (not isinstance(budget["epochs"], int) or budget["epochs"] < 1
            or not isinstance(budget["batch_size"], int) or budget["batch_size"] < 1
            or not math.isfinite(budget["learning_rate"]) or budget["learning_rate"] <= 0
            or not math.isfinite(budget["weight_decay"]) or budget["weight_decay"] < 0):
        raise ValueError("Invalid training budget.")
    torch.manual_seed(seed)
    head = TargetRiskProjectionHead(dropout=budget["dropout"]).double()
    initial_sha = state_dict_sha256(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=budget["learning_rate"], weight_decay=budget["weight_decay"])

    def objective(data, raw):
        if mode == "A4":
            return F.binary_cross_entropy(raw, (data["labels"] == 0).double().reshape(-1, 1))
        q = verification_family(data["p"], data["contradiction"], raw)
        return F.nll_loss(q.clamp_min(torch.finfo(q.dtype).tiny).log(), data["labels"])

    head.eval()
    with torch.no_grad():
        initial_fit_objective = float(objective(fit, head(fit["evidence"])))
    history, best = [], None
    for epoch in range(1, budget["epochs"]+1):
        head.train()
        order = torch.randperm(len(fit["labels"]))
        for indices in order.split(budget["batch_size"]):
            batch = {key: fit[key][indices] for key in ("p", "contradiction", "labels", "evidence")}
            optimizer.zero_grad(set_to_none=True)
            loss = objective(batch, head(batch["evidence"]))
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            fit_loss = float(objective(fit, head(fit["evidence"])))
        stop_metrics = calculate_probability_metrics(predict_head(head, stop, mode)["probabilities"], stop["labels"])
        history.append({"epoch": epoch, "fit_objective": fit_loss, "stop_metrics": stop_metrics})
        if best is None or stop_metrics["nll"] < best["nll"]:
            best = {"nll": stop_metrics["nll"], "epoch": epoch, "state": copy.deepcopy(head.state_dict()), "fit_objective": fit_loss}
    head.load_state_dict(best["state"])
    head.eval()
    return {"head": head, "state_dict": best["state"], "history": history,
            "best_epoch": best["epoch"], "initial_sha256": initial_sha,
            "state_sha256": state_dict_sha256(best["state"]),
            "parameter_count": sum(p.numel() for p in head.parameters()),
            "initial_fit_objective": initial_fit_objective, "selected_fit_objective": best["fit_objective"]}


def development_gate(metrics: dict, controls: dict, diagnostics: dict, thresholds: dict) -> dict:
    if set(controls) != {"A0", "A1", "A2"}:
        raise ValueError("All three fixed controls must be present.")
    values = list(controls.values())
    checks = {
        "nll_beyond_best_control": metrics["nll"] <= min(m["nll"] for m in values)-thresholds["nll_improvement"],
        "macro_f1_not_below_best_control": metrics["macro_f1"] >= max(m["macro_f1"] for m in values),
        "target_recall_guard": metrics["target_recall"] >= max(m["target_recall"] for m in values)-thresholds["recall_tolerance"],
    }
    for field in ("ti_intrusion_to_target", "metallic_intrusion_to_target"):
        checks[f"{field}_guard"] = metrics[field] <= min(m[field] for m in values)+thresholds["intrusion_tolerance"]
    for field in ("gamma_std_active", "minimum_fixed_output_distance", "fit_mean_replay_output_distance"):
        checks[field] = diagnostics[field] > thresholds["dynamic_tolerance"]
    return {"passed": all(checks.values()), "checks": checks}


def _write_predictions(path: Path, data: dict, predictions: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split_group_id", "true_class_id", "raw_head", "gamma", "contradiction", "predicted_class_id", "prob_0", "prob_1", "prob_2", "prob_3"])
        for i, row in enumerate(data["records"]):
            p = predictions["probabilities"][i]
            writer.writerow([row["image_id"], row["split_group_id"], int(data["labels"][i]), float(predictions["raw"][i]),
                             float(predictions["gamma"][i]), float(data["contradiction"][i]), int(p.argmax()), *p.tolist()])


def run_experiment(root: Path, protocol_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Experiment output already exists; refusing overwrite.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["protocol"] != "abmp_verifier_trust_development_v1":
        raise ValueError("Unregistered experiment protocol.")
    seeds = protocol.get("seeds", [])
    if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Experiment seeds must be a nonempty unique list of nonnegative integers.")
    audit_path = root / "outputs/theory/abmp_candidate_capacity_v1/audit_summary.json"
    if _file_sha256(audit_path) != protocol["source_audit_sha256"]:
        raise ValueError("Source audit hash mismatch.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    inner = root / "outputs/training/tc_oos_rsg_manifests_v1/fold_0"
    source = root / "outputs/training/tc_oos_rsg_v1/fold_0"
    provenance = validate_sources(inner, source)
    if provenance != audit["provenance"] or provenance["verifier_config"] != VERIFIER_CONFIG:
        raise ValueError("Frozen source provenance mismatch.")
    report = root / "结题" / REPORT_NAME
    if _file_sha256(report) != protocol["formal_report_sha256"]:
        raise ValueError("Formal report hash mismatch.")
    fit, stop = [load_subset(root, name, protocol["input_sha256"]) for name in EXTRACTION_NAMES]
    if ({r["image_id"] for r in fit["records"]} & {r["image_id"] for r in stop["records"]}
            or {r["split_group_id"] for r in fit["records"]} & {r["split_group_id"] for r in stop["records"]}):
        raise ValueError("Development fit/stop identities overlap.")
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "registered_protocol.json", protocol)
    code_paths = [Path(__file__), ROOT / "scripts/verifier_risk.py", ROOT / "scripts/tc_oos_rsg.py",
                  ROOT / "scripts/audit_abmp_candidate_capacity.py", ROOT / "scripts/run_tc_oos_rsg_experiments.py"]
    code_hashes = {path.name: _file_sha256(path) for path in code_paths}
    _write_json(output / "run_status.json", {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        scalar = optimal_global_strength(fit["p"], fit["contradiction"], fit["labels"])
        controls, control_outputs = {}, {}
        for arm, gamma in (("A0", 0.), ("A1", 1.), ("A2", scalar["gamma"])):
            controls[arm], control_outputs[arm] = {}, {}
            for subset, data in (("fit", fit), ("stop", stop)):
                column = torch.full_like(data["contradiction"], gamma)
                q = verification_family(data["p"], data["contradiction"], column)
                controls[arm][subset] = calculate_probability_metrics(q, data["labels"])
                control_outputs[arm][subset] = q
                _write_predictions(output / f"{arm}_{subset}_predictions.csv", data, {"probabilities": q, "raw": column, "gamma": column})
        stop_controls = {arm: results["stop"] for arm, results in controls.items()}
        runs = []
        for seed in protocol["seeds"]:
            for arm in ("A3", "A4"):
                trained = fit_head(fit, stop, arm, seed, protocol["budget"])
                folder = output / f"{arm}_seed{seed}"
                folder.mkdir()
                checkpoint = folder / "best_head.pt"
                torch.save(trained["state_dict"], checkpoint)
                reloaded = TargetRiskProjectionHead(dropout=protocol["budget"]["dropout"]).double()
                reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                predictions = {name: predict_head(reloaded, data, arm) for name, data in (("fit", fit), ("stop", stop))}
                fit_mean = float(predictions["fit"]["gamma"].mean())
                mean_q = verification_family(stop["p"], stop["contradiction"], torch.full_like(stop["contradiction"], fit_mean))
                final_q = predictions["stop"]["probabilities"]
                active_gamma = predictions["stop"]["gamma"][stop["contradiction"] > 0]
                replay = predict_head(trained["head"], stop, arm)["probabilities"]
                diagnostics = {"fit_mean_gamma": fit_mean, "gamma_std_active": float(active_gamma.std(unbiased=False)) if active_gamma.numel() else 0.,
                               "gamma_mean": float(predictions["stop"]["gamma"].mean()),
                               "minimum_fixed_output_distance": min(float((final_q-q["stop"]).abs().max()) for q in control_outputs.values()),
                               "fit_mean_replay_output_distance": float((final_q-mean_q).abs().max()),
                               "checkpoint_replay_max_abs_residual": float((final_q-replay).abs().max()),
                               "target_promotions_from_no_verification": int(((final_q.argmax(1) == 0) & (stop["p"].argmax(1) != 0)).sum())}
                if arm == "A4":
                    diagnostics.update(lower_saturated_count=int(predictions["stop"]["lower_saturated"].sum()),
                                       upper_saturated_count=int(predictions["stop"]["upper_saturated"].sum()))
                metrics = {name: calculate_probability_metrics(prediction["probabilities"], data["labels"])
                           for (name, data), prediction in zip((("fit", fit), ("stop", stop)), predictions.values(), strict=True)}
                result = {"arm": arm, "seed": seed, "best_epoch": trained["best_epoch"], "initial_sha256": trained["initial_sha256"],
                          "state_sha256": trained["state_sha256"], "checkpoint_sha256": _file_sha256(checkpoint), "parameter_count": trained["parameter_count"],
                          "initial_fit_objective": trained["initial_fit_objective"], "selected_fit_objective": trained["selected_fit_objective"],
                          "metrics": metrics, "diagnostics": diagnostics, "fit_mean_replay_stop_metrics": calculate_probability_metrics(mean_q, stop["labels"]),
                          "development_gate": development_gate(metrics["stop"], stop_controls, diagnostics, protocol["thresholds"])}
                _write_json(folder / "history.json", trained["history"])
                _write_json(folder / "result.json", result)
                for subset, data in (("fit", fit), ("stop", stop)):
                    _write_predictions(folder / f"{subset}_predictions.csv", data, predictions[subset])
                column = torch.full_like(stop["contradiction"], fit_mean)
                _write_predictions(folder / "fit_mean_replay_stop_predictions.csv", stop, {"probabilities": mean_q, "raw": column, "gamma": column})
                runs.append(result)
                print(f"{arm} seed={seed} epoch={trained['best_epoch']} stop_NLL={metrics['stop']['nll']:.6f} gate={result['development_gate']['passed']}", flush=True)
        aggregate = {}
        for arm in ("A3", "A4"):
            selected = [run for run in runs if run["arm"] == arm]
            aggregate[arm] = {"stop_metrics": {key: {"mean": statistics.mean(run["metrics"]["stop"][key] for run in selected),
                                                     "sample_sd": statistics.stdev(run["metrics"]["stop"][key] for run in selected) if len(selected) > 1 else None} for key in stop_controls["A0"]},
                              "all_seeds_passed": all(run["development_gate"]["passed"] for run in selected)}
        after = validate_sources(inner, source)
        after_inputs = {key: value for name in EXTRACTION_NAMES for key, value in load_subset(root, name, protocol["input_sha256"])["hashes"].items()}
        unchanged = (after == provenance and after_inputs == protocol["input_sha256"]
                     and _file_sha256(report) == protocol["formal_report_sha256"]
                     and _file_sha256(audit_path) == protocol["source_audit_sha256"]
                     and all(_file_sha256(path) == code_hashes[path.name] for path in code_paths))
        if not unchanged:
            raise ValueError("Source, report or code changed during execution.")
        summary = {"protocol": protocol["protocol"], "status": "DEVELOPMENT_SIGNAL_ONLY" if any(r["all_seeds_passed"] for r in aggregate.values()) else "DEVELOPMENT_GATE_FAILED",
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": _file_sha256(protocol_path),
                   "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "float64", "threads": 1},
                   "source_provenance": provenance, "input_sha256": after_inputs, "code_sha256": code_hashes,
                   "scalar_optimum": scalar, "controls": controls, "runs": runs, "aggregate": aggregate,
                   "source_and_report_unchanged": unchanged, "formal_report_sha256": _file_sha256(report),
                   "outer_manifest_read": False, "outer_images_loaded": False, "independent_confirmation": False,
                   "claim_boundary": "Exposed Fold 0 development; risk bound is not a finite-sample efficacy guarantee. No new method promotion or automatic outer access."}
        _write_json(output / "development_summary.json", summary)
        _write_json(output / "run_status.json", {"status": "COMPLETED", "research_status": summary["status"]})
        return summary
    except Exception as exc:
        _write_json(output / "run_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "auto_retry": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/experiment_protocols/abmp_verifier_trust_development_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training/abmp_verifier_trust_v1/development_fold_0")
    args = parser.parse_args()
    result = run_experiment(args.root.resolve(), args.protocol.resolve(), args.output.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
