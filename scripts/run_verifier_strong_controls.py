"""Strong controls restricted to already exposed Fold 0 inner development."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import scipy
import torch

from run_tc_oos_rsg_experiments import _file_sha256, _write_json, calculate_probability_metrics, state_dict_sha256
from run_verifier_trust_development import ROOT, REPORT_NAME, load_subset
from tc_oos_rsg import TargetRiskProjectionHead
from verifier_strong_controls import (
    LEARNED_ARMS, dirichlet_probabilities, fit_dirichlet, fit_global_temperature,
    fit_head, predict_head, temperature_probabilities,
)
from verify_verifier_trust_delivery import verify_delivery as verify_original


SUBSETS = ("gate_stop_projector_fit", "projector_stop")
DEFAULT_PROTOCOL = ROOT / "docs/experiment_protocols/abmp_verifier_strong_controls_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/training/abmp_verifier_strong_controls_v1/development_fold_0"


def validate_protocol(protocol: dict) -> None:
    if (protocol.get("protocol") != "abmp_verifier_strong_controls_v1" or protocol.get("fold") != 0
            or set(protocol.get("arms", [])) != {"T0", "D0", *LEARNED_ARMS}
            or protocol.get("subsets") != list(SUBSETS)
            or protocol.get("reference_output") != "outputs/training/abmp_verifier_trust_v1/development_fold_0"):
        raise ValueError("Unapproved experiment scope: Fold 0 inner subsets and five controls only.")
    seeds = protocol.get("seeds", [])
    if not seeds or any(not isinstance(s, int) or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique nonnegative integers.")


def aggregate(runs: list) -> dict:
    result = {}
    for arm in LEARNED_ARMS:
        selected = [r for r in runs if r["arm"] == arm]
        result[arm] = {field: {"mean": statistics.mean(r["metrics"]["stop"][field] for r in selected),
                              "sample_sd": statistics.stdev(r["metrics"]["stop"][field] for r in selected) if len(selected) > 1 else None}
                       for field in selected[0]["metrics"]["stop"]}
    return result


def source_snapshot(root: Path, protocol: dict) -> tuple[dict, dict, dict]:
    reference = root / protocol["reference_output"]
    summary_path = reference / "development_summary.json"
    if _file_sha256(summary_path) != protocol["reference_summary_sha256"]:
        raise ValueError("Original experiment summary changed.")
    original_verification = verify_original(root, reference)
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    if (original["input_sha256"] != protocol["input_sha256"]
            or original["formal_report_sha256"] != protocol["formal_report_sha256"]):
        raise ValueError("Registered sources differ from original experiment.")
    if _file_sha256(root / "结题" / REPORT_NAME) != protocol["formal_report_sha256"]:
        raise ValueError("Formal report changed.")
    fit, stop = [load_subset(root, name, protocol["input_sha256"]) for name in SUBSETS]
    if (len(fit["labels"]) != protocol["counts"]["fit"] or len(stop["labels"]) != protocol["counts"]["stop"]):
        raise ValueError("Registered sample count changed.")
    for key in ("image_id", "split_group_id"):
        if {r[key] for r in fit["records"]} & {r[key] for r in stop["records"]}:
            raise ValueError("Inner fit and stop overlap.")
    return fit, stop, {"reference_summary_sha256": _file_sha256(summary_path),
                       "original_delivery": original_verification,
                       "input_sha256": dict(fit["hashes"], **stop["hashes"]),
                       "report_sha256": _file_sha256(root / "结题" / REPORT_NAME)}


def write_predictions(path: Path, data: dict, result: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split_group_id", "true_class_id", "predicted_class_id", "raw_head", "temperature", *[f"prob_{i}" for i in range(4)]])
        for i, record in enumerate(data["records"]):
            q = result["probabilities"][i]
            writer.writerow([record["image_id"], record["split_group_id"], int(data["labels"][i]), int(q.argmax()),
                             float(result["raw"][i]) if "raw" in result else "",
                             float(result["temperature"][i]) if "temperature" in result else "", *q.tolist()])


def diagnostics(q: torch.Tensor, data: dict) -> dict:
    base = data["p"].argmax(1)
    predicted = q.argmax(1)
    return {"classification_changes_from_base": int((predicted != base).sum()),
            "target_promotions_from_base": int(((predicted == 0) & (base != 0)).sum()),
            "true_target_promotions_from_base": int(((predicted == 0) & (base != 0) & (data["labels"] == 0)).sum())}


def run_experiment(root: Path, protocol_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Output exists; refusing to overwrite any experiment.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit, stop, before = source_snapshot(root, protocol)
    code_paths = [Path(__file__), ROOT / "scripts/verifier_strong_controls.py"]
    code_hashes = {p.name: _file_sha256(p) for p in code_paths}
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "registered_protocol.json", protocol)
    _write_json(output / "run_status.json", {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        global_fit = fit_global_temperature(fit["p"], fit["labels"], protocol["global_temperature"]["beta_bounds"])
        dirichlet = fit_dirichlet(fit["p"], fit["labels"], protocol["dirichlet"]["regularization"])
        global_controls = {}
        for arm in ("T0", "D0"):
            global_controls[arm] = {"metrics": {}, "diagnostics": {}}
            for name, data in (("fit", fit), ("stop", stop)):
                q = (temperature_probabilities(data["p"], torch.full_like(data["contradiction"], global_fit["temperature"]))
                     if arm == "T0" else dirichlet_probabilities(data["p"], dirichlet["weight"], dirichlet["bias"]))
                write_predictions(output / f"{arm}_{name}_predictions.csv", data, {"probabilities": q})
                global_controls[arm]["metrics"][name] = calculate_probability_metrics(q, data["labels"])
                global_controls[arm]["diagnostics"][name] = diagnostics(q, data)
            print(f"{arm} stop_NLL={global_controls[arm]['metrics']['stop']['nll']:.9f}", flush=True)
        dirichlet_serial = {k: v.tolist() if torch.is_tensor(v) else v for k, v in dirichlet.items()}
        _write_json(output / "global_fits.json", {"T0": global_fit, "D0": dirichlet_serial})
        original = json.loads((root / protocol["reference_output"] / "development_summary.json").read_text(encoding="utf-8"))
        original_a4 = {r["seed"]: r for r in original["runs"] if r["arm"] == "A4"}
        runs = []
        for seed in protocol["seeds"]:
            for arm in LEARNED_ARMS:
                trained = fit_head(fit, stop, arm, seed, protocol["budget"])
                if trained["initial_sha256"] != original_a4[seed]["initial_sha256"]:
                    raise ValueError("Initialization differs from original paired A4 head.")
                folder = output / f"{arm}_seed{seed}"
                folder.mkdir()
                checkpoint = folder / "best_head.pt"
                torch.save(trained["state_dict"], checkpoint)
                reloaded = TargetRiskProjectionHead(dropout=protocol["budget"]["dropout"]).double()
                reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                metrics, diag = {}, {}
                for subset, data in (("fit", fit), ("stop", stop)):
                    prediction = predict_head(reloaded, data, arm)
                    residual = float((prediction["probabilities"] - predict_head(trained["head"], data, arm)["probabilities"]).abs().max())
                    if residual > 1e-12:
                        raise ValueError("Checkpoint does not replay.")
                    write_predictions(folder / f"{subset}_predictions.csv", data, prediction)
                    metrics[subset] = calculate_probability_metrics(prediction["probabilities"], data["labels"])
                    diag[subset] = dict(diagnostics(prediction["probabilities"], data), checkpoint_replay_max_abs_residual=residual)
                    if arm != "M0":
                        diag[subset].update(lower_saturated_count=int(prediction["lower_saturated"].sum()),
                                            upper_saturated_count=int(prediction["upper_saturated"].sum()))
                result = {key: trained[key] for key in ("initial_sha256", "state_sha256", "parameter_count", "best_epoch", "initial_fit_objective", "selected_fit_objective")}
                result.update(arm=arm, seed=seed, checkpoint_sha256=_file_sha256(checkpoint), metrics=metrics, diagnostics=diag)
                if arm == "P1":
                    result["same_as_original_a4_state"] = result["state_sha256"] == original_a4[seed]["state_sha256"]
                    if not result["same_as_original_a4_state"]:
                        raise ValueError("Matched BCE control failed to reproduce original A4 training.")
                _write_json(folder / "result.json", result)
                _write_json(folder / "history.json", trained["history"])
                runs.append(result)
                print(f"{arm} seed={seed} epoch={result['best_epoch']} stop_NLL={metrics['stop']['nll']:.9f}", flush=True)
        _, _, after = source_snapshot(root, protocol)
        if before != after or any(_file_sha256(p) != code_hashes[p.name] for p in code_paths):
            raise ValueError("Source, original delivery, report or experiment code changed during execution.")
        summary = {"protocol": protocol["protocol"], "status": "DEVELOPMENT_COMPARISON_ONLY",
                   "protocol_sha256": _file_sha256(protocol_path), "code_sha256": code_hashes,
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "environment": {"python": platform.python_version(), "torch": torch.__version__, "scipy": scipy.__version__, "dtype": "float64", "device": "cpu", "threads": 1},
                   "source_snapshot": after, "global_controls": global_controls, "runs": runs, "aggregate": aggregate(runs),
                   "retained_original_controls": {"A0_A1_A2": original["controls"], "A3_A4": original["aggregate"]},
                   "source_and_report_unchanged": True, "outer_manifest_read": False, "outer_images_loaded": False,
                   "independent_confirmation": False, "automatic_promotion": False,
                   "claim_boundary": "Reused stopping set and exposed Fold 0 development only. Calibration controls are not new neural architectures; no significance or industrial performance claim."}
        summary["artifact_sha256"] = {str(p.relative_to(output)).replace("\\", "/"): _file_sha256(p)
                                     for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_status.json"}
        _write_json(output / "development_summary.json", summary)
        _write_json(output / "run_status.json", {"status": "COMPLETED", "research_status": summary["status"]})
        return summary
    except Exception as exc:
        _write_json(output / "run_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "auto_retry": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_experiment(args.root.resolve(), args.protocol.resolve(), args.output.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
