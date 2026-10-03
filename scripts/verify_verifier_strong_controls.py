"""Independent checkpoint, fit, training-history and prediction-table replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from run_tc_oos_rsg_experiments import _file_sha256, _write_json, calculate_probability_metrics, state_dict_sha256
from run_verifier_strong_controls import (
    ROOT, DEFAULT_OUTPUT, DEFAULT_PROTOCOL, aggregate, diagnostics, source_snapshot, validate_protocol,
)
from tc_oos_rsg import TargetRiskProjectionHead
from verifier_strong_controls import (
    LEARNED_ARMS, dirichlet_probabilities, fit_dirichlet, fit_global_temperature,
    fit_head, predict_head, temperature_probabilities,
)


def verify_table(path: Path, data: dict, prediction: dict) -> float:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(data["records"]):
        raise ValueError("Prediction table count mismatch.")
    maximum = 0.
    for i, (row, record) in enumerate(zip(rows, data["records"], strict=True)):
        q = prediction["probabilities"][i]
        if (row["image_id"] != record["image_id"] or row["split_group_id"] != record["split_group_id"]
                or int(row["true_class_id"]) != int(data["labels"][i]) or int(row["predicted_class_id"]) != int(q.argmax())):
            raise ValueError("Prediction table identity mismatch.")
        values = [float(row[f"prob_{k}"]) for k in range(4)]
        expected = q.tolist()
        for name, key in (("raw_head", "raw"), ("temperature", "temperature")):
            if key in prediction:
                values.append(float(row[name]))
                expected.append(float(prediction[key][i]))
            elif row[name] != "":
                raise ValueError("Unexpected auxiliary prediction.")
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite saved prediction.")
        maximum = max(maximum, max(abs(a - b) for a, b in zip(values, expected, strict=True)))
    if maximum > 1e-12:
        raise ValueError("Prediction table does not replay.")
    return maximum


def verify_delivery(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    summary = json.loads((output / "development_summary.json").read_text(encoding="utf-8"))
    if (_file_sha256(protocol_path) != summary["protocol_sha256"]
            or protocol != json.loads((output / "registered_protocol.json").read_text(encoding="utf-8"))):
        raise ValueError("Registered protocol changed.")
    for name, digest in summary["code_sha256"].items():
        if _file_sha256(ROOT / "scripts" / name) != digest:
            raise ValueError("Experiment code changed.")
    artifacts = {str(p.relative_to(output)).replace("\\", "/") for p in output.rglob("*")
                 if p.is_file() and p.name not in ("development_summary.json", "run_status.json", "delivery_verification.json")}
    if artifacts != set(summary["artifact_sha256"]):
        raise ValueError("Experiment artifact set changed.")
    for name, digest in summary["artifact_sha256"].items():
        if _file_sha256(output / name) != digest:
            raise ValueError(f"Experiment artifact changed: {name}")
    if json.loads((output / "run_status.json").read_text(encoding="utf-8"))["status"] != "COMPLETED":
        raise ValueError("Experiment is not completed.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit, stop, snapshot = source_snapshot(root, protocol)
    if snapshot != summary["source_snapshot"]:
        raise ValueError("Source snapshot changed.")
    global_fit = fit_global_temperature(fit["p"], fit["labels"], protocol["global_temperature"]["beta_bounds"])
    dirichlet = fit_dirichlet(fit["p"], fit["labels"], protocol["dirichlet"]["regularization"])
    serial = {k: v.tolist() if torch.is_tensor(v) else v for k, v in dirichlet.items()}
    if json.loads((output / "global_fits.json").read_text(encoding="utf-8")) != {"T0": global_fit, "D0": serial}:
        raise ValueError("Global calibration fits do not replay.")
    count, residual = 0, 0.
    for arm in ("T0", "D0"):
        for subset, data in (("fit", fit), ("stop", stop)):
            q = (temperature_probabilities(data["p"], torch.full_like(data["contradiction"], global_fit["temperature"]))
                 if arm == "T0" else dirichlet_probabilities(data["p"], dirichlet["weight"], dirichlet["bias"]))
            residual = max(residual, verify_table(output / f"{arm}_{subset}_predictions.csv", data, {"probabilities": q}))
            result = summary["global_controls"][arm]
            if calculate_probability_metrics(q, data["labels"]) != result["metrics"][subset] or diagnostics(q, data) != result["diagnostics"][subset]:
                raise ValueError("Global control metrics do not replay.")
            count += 1
    expected = {(arm, seed) for arm in LEARNED_ARMS for seed in protocol["seeds"]}
    if len(summary["runs"]) != len(expected) or {(r["arm"], r["seed"]) for r in summary["runs"]} != expected:
        raise ValueError("Missing or duplicate control runs.")
    replayed = 0
    for run in summary["runs"]:
        arm, seed = run["arm"], run["seed"]
        folder = output / f"{arm}_seed{seed}"
        if run != json.loads((folder / "result.json").read_text(encoding="utf-8")):
            raise ValueError("Result and summary differ.")
        checkpoint = folder / "best_head.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if _file_sha256(checkpoint) != run["checkpoint_sha256"] or state_dict_sha256(state) != run["state_sha256"]:
            raise ValueError("Checkpoint mismatch.")
        trained = fit_head(fit, stop, arm, seed, protocol["budget"])
        for key in ("initial_sha256", "state_sha256", "parameter_count", "best_epoch", "initial_fit_objective", "selected_fit_objective"):
            if trained[key] != run[key]:
                raise ValueError(f"Deterministic training does not replay: {key}")
        if trained["history"] != json.loads((folder / "history.json").read_text(encoding="utf-8")):
            raise ValueError("Training history and batch gradients do not replay.")
        replayed += 1
        head = TargetRiskProjectionHead(dropout=protocol["budget"]["dropout"]).double()
        head.load_state_dict(state)
        for subset, data in (("fit", fit), ("stop", stop)):
            prediction = predict_head(head, data, arm)
            residual = max(residual, verify_table(folder / f"{subset}_predictions.csv", data, prediction))
            if calculate_probability_metrics(prediction["probabilities"], data["labels"]) != run["metrics"][subset]:
                raise ValueError("Learned metrics do not replay.")
            expected_diag = dict(diagnostics(prediction["probabilities"], data), checkpoint_replay_max_abs_residual=0.)
            if arm != "M0":
                expected_diag.update(lower_saturated_count=int(prediction["lower_saturated"].sum()), upper_saturated_count=int(prediction["upper_saturated"].sum()))
            if expected_diag != run["diagnostics"][subset]:
                raise ValueError("Selected diagnostics do not replay.")
            count += 1
    if aggregate(summary["runs"]) != summary["aggregate"]:
        raise ValueError("Aggregate metrics do not replay.")
    original = json.loads((root / protocol["reference_output"] / "development_summary.json").read_text(encoding="utf-8"))
    if summary["retained_original_controls"] != {"A0_A1_A2": original["controls"], "A3_A4": original["aggregate"]}:
        raise ValueError("Retained original controls changed.")
    for run in summary["runs"]:
        if run["arm"] == "P1":
            old = next(r for r in original["runs"] if r["arm"] == "A4" and r["seed"] == run["seed"])
            if not run["same_as_original_a4_state"] or run["state_sha256"] != old["state_sha256"]:
                raise ValueError("BCE control differs from original A4.")
    if (summary["status"] != "DEVELOPMENT_COMPARISON_ONLY" or not summary["source_and_report_unchanged"]
            or any(summary[k] for k in ("outer_manifest_read", "outer_images_loaded", "independent_confirmation", "automatic_promotion"))):
        raise ValueError("Research scope or claim boundary changed.")
    return {"status": "VERIFIED", "prediction_table_count": count,
            "prediction_row_count": count // 2 * (len(fit["labels"]) + len(stop["labels"])),
            "max_abs_replay_residual": residual, "deterministic_training_replays": replayed,
            "fit_count": len(fit["labels"]), "stop_count": len(stop["labels"]),
            "source_and_report_unchanged": True, "outer_manifest_read": False,
            "interpretation": "Implementation and deterministic reproduction, not independent efficacy confirmation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify_delivery(args.root.resolve(), args.protocol.resolve(), args.output.resolve())
    _write_json(args.output / "delivery_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
