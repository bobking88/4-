"""Reload all development checkpoints and independently replay saved tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

from run_verifier_trust_development import (
    ROOT, REPORT_NAME, load_subset, predict_head, development_gate,
)
from run_tc_oos_rsg_experiments import _file_sha256, _write_json, calculate_probability_metrics, state_dict_sha256
from audit_abmp_candidate_capacity import validate_sources
from tc_oos_rsg import TargetRiskProjectionHead
from verifier_risk import optimal_global_strength, verification_family


def verify_table(path: Path, data: dict, predictions: dict) -> float:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(data["records"]):
        raise ValueError(f"Prediction table row count changed: {path}")
    maximum = 0.
    for i, (row, record) in enumerate(zip(rows, data["records"], strict=True)):
        if (row.get("image_id") != record["image_id"] or row.get("split_group_id") != record["split_group_id"]
                or int(row["true_class_id"]) != int(data["labels"][i])
                or int(row["predicted_class_id"]) != int(predictions["probabilities"][i].argmax())):
            raise ValueError(f"Prediction table identity or classification changed: {path}")
        actual = [float(row[name]) for name in ("raw_head", "gamma", "contradiction", "prob_0", "prob_1", "prob_2", "prob_3")]
        expected = [float(predictions["raw"][i]), float(predictions["gamma"][i]), float(data["contradiction"][i]), *predictions["probabilities"][i].tolist()]
        if not all(math.isfinite(value) for value in actual):
            raise ValueError(f"Nonfinite prediction table value: {path}")
        maximum = max(maximum, max(abs(a-b) for a, b in zip(actual, expected, strict=True)))
    if maximum > 1e-12:
        raise ValueError(f"Prediction table does not replay: {path}")
    return maximum


def verify_delivery(root: Path, output: Path) -> dict:
    summary = json.loads((output / "development_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "registered_protocol.json").read_text(encoding="utf-8"))
    protocol_path = root / "docs/experiment_protocols/abmp_verifier_trust_development_v1.json"
    if not protocol_path.is_file():
        protocol_path = root / "protocol.json"
    if (_file_sha256(protocol_path) != summary["protocol_sha256"]
            or json.loads(protocol_path.read_text(encoding="utf-8")) != protocol):
        raise ValueError("Original protocol or saved protocol changed.")
    audit_path = root / "outputs/theory/abmp_candidate_capacity_v1/audit_summary.json"
    if _file_sha256(audit_path) != protocol["source_audit_sha256"]:
        raise ValueError("Source audit changed.")
    source = validate_sources(root / "outputs/training/tc_oos_rsg_manifests_v1/fold_0", root / "outputs/training/tc_oos_rsg_v1/fold_0")
    if source != summary["source_provenance"]:
        raise ValueError("Source provenance changed.")
    if _file_sha256(root / "结题" / REPORT_NAME) != protocol["formal_report_sha256"]:
        raise ValueError("Formal report changed.")
    for name, digest in summary["code_sha256"].items():
        if _file_sha256(ROOT / "scripts" / name) != digest:
            raise ValueError(f"Experiment code changed after training: {name}")
    torch.set_num_threads(1)
    fit = load_subset(root, "gate_stop_projector_fit", protocol["input_sha256"])
    stop = load_subset(root, "projector_stop", protocol["input_sha256"])
    scalar = optimal_global_strength(fit["p"], fit["contradiction"], fit["labels"])
    if scalar != summary["scalar_optimum"]:
        raise ValueError("Scalar optimum does not replay.")
    count, residual, controls = 0, 0., {}
    for arm, gamma in (("A0", 0.), ("A1", 1.), ("A2", scalar["gamma"])):
        controls[arm] = {}
        for subset, data in (("fit", fit), ("stop", stop)):
            column = torch.full_like(data["contradiction"], gamma)
            q = verification_family(data["p"], data["contradiction"], column)
            prediction = {"probabilities": q, "raw": column, "gamma": column}
            residual = max(residual, verify_table(output / f"{arm}_{subset}_predictions.csv", data, prediction))
            if calculate_probability_metrics(q, data["labels"]) != summary["controls"][arm][subset]:
                raise ValueError("Fixed-control metrics changed.")
            controls[arm][subset] = q
            count += 1
    expected_runs = {(arm, seed) for arm in ("A3", "A4") for seed in protocol["seeds"]}
    if len(summary["runs"]) != len(expected_runs) or {(r["arm"], r["seed"]) for r in summary["runs"]} != expected_runs:
        raise ValueError("Missing or duplicate learned run.")
    verified_runs = []
    initial_states = {}
    for run in summary["runs"]:
        arm, seed = run["arm"], run["seed"]
        folder = output / f"{arm}_seed{seed}"
        if json.loads((folder / "result.json").read_text(encoding="utf-8")) != run:
            raise ValueError("Run result and summary differ.")
        path = folder / "best_head.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if _file_sha256(path) != run["checkpoint_sha256"] or state_dict_sha256(state) != run["state_sha256"]:
            raise ValueError("Checkpoint changed.")
        torch.manual_seed(seed)
        head = TargetRiskProjectionHead(dropout=protocol["budget"]["dropout"]).double()
        if state_dict_sha256(head.state_dict()) != run["initial_sha256"]:
            raise ValueError("Seeded initialization does not replay.")
        initial_states.setdefault(seed, run["initial_sha256"])
        if initial_states[seed] != run["initial_sha256"]:
            raise ValueError("Paired heads have different initialization.")
        head.load_state_dict(state)
        predictions = {}
        for subset, data in (("fit", fit), ("stop", stop)):
            prediction = predict_head(head, data, arm)
            residual = max(residual, verify_table(folder / f"{subset}_predictions.csv", data, prediction))
            if calculate_probability_metrics(prediction["probabilities"], data["labels"]) != run["metrics"][subset]:
                raise ValueError("Selected metrics do not replay.")
            predictions[subset] = prediction
            count += 1
        history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
        if len(history) != protocol["budget"]["epochs"] or min(history, key=lambda row: row["stop_metrics"]["nll"])["epoch"] != run["best_epoch"]:
            raise ValueError("Checkpoint did not use earliest minimum stopping NLL.")
        fit_mean = float(predictions["fit"]["gamma"].mean())
        column = torch.full_like(stop["contradiction"], fit_mean)
        q_mean = verification_family(stop["p"], stop["contradiction"], column)
        residual = max(residual, verify_table(folder / "fit_mean_replay_stop_predictions.csv", stop, {"probabilities": q_mean, "raw": column, "gamma": column}))
        if fit_mean != run["diagnostics"]["fit_mean_gamma"] or calculate_probability_metrics(q_mean, stop["labels"]) != run["fit_mean_replay_stop_metrics"]:
            raise ValueError("Fit-mean replay changed.")
        count += 1
        diagnostics = dict(run["diagnostics"])
        active = predictions["stop"]["gamma"][stop["contradiction"] > 0]
        q = predictions["stop"]["probabilities"]
        diagnostics.update(gamma_std_active=float(active.std(unbiased=False)) if active.numel() else 0.,
                           minimum_fixed_output_distance=min(float((q-value["stop"]).abs().max()) for value in controls.values()),
                           fit_mean_replay_output_distance=float((q-q_mean).abs().max()))
        if development_gate(run["metrics"]["stop"], {a: m["stop"] for a, m in summary["controls"].items()}, diagnostics, protocol["thresholds"]) != run["development_gate"]:
            raise ValueError("Development gate does not replay.")
        verified_runs.append({"arm": arm, "seed": seed,
                              "classification_difference_from_original_verification": int((q.argmax(1) != controls["A1"]["stop"].argmax(1)).sum()),
                              "classification_difference_from_no_verification": int((q.argmax(1) != controls["A0"]["stop"].argmax(1)).sum()),
                              "nll_improvement_over_fit_mean_replay": run["fit_mean_replay_stop_metrics"]["nll"]-run["metrics"]["stop"]["nll"]})
    aggregate = {}
    for arm in ("A3", "A4"):
        selected = [run for run in summary["runs"] if run["arm"] == arm]
        aggregate[arm] = {"stop_metrics": {key: {"mean": statistics.mean(run["metrics"]["stop"][key] for run in selected),
                                                 "sample_sd": statistics.stdev(run["metrics"]["stop"][key] for run in selected) if len(selected) > 1 else None}
                                            for key in summary["controls"]["A0"]["stop"]},
                          "all_seeds_passed": all(run["development_gate"]["passed"] for run in selected)}
    status = "DEVELOPMENT_SIGNAL_ONLY" if any(row["all_seeds_passed"] for row in aggregate.values()) else "DEVELOPMENT_GATE_FAILED"
    if aggregate != summary["aggregate"] or status != summary["status"]:
        raise ValueError("Development aggregate or research status does not replay.")
    return {"status": "VERIFIED", "prediction_table_count": count, "max_abs_replay_residual": residual,
            "runs": verified_runs, "fixed_oos_no_verification_true_target_capacity": int(((stop["p"].argmax(1) == 0) & (stop["labels"] == 0)).sum()),
            "fit_count": len(fit["labels"]), "stop_count": len(stop["labels"]),
            "source_and_report_unchanged": True, "outer_manifest_read": False,
            "interpretation": "Delivery and implementation replay only; not an independent efficacy test."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training/abmp_verifier_trust_v1/development_fold_0")
    args = parser.parse_args()
    result = verify_delivery(args.root.resolve(), args.output.resolve())
    _write_json(args.output / "delivery_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
