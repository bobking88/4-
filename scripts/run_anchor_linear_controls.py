"""Approved posterior-anchor controls; cached Fold 0 inputs, no image reads."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import scipy
import torch
from threadpoolctl import threadpool_limits

import anchor_linear_controls as controls
import frozen_visual_probe as probe
from run_frozen_visual_probe import ROOT, SUBSETS, configure_runtime, snapshot, tensor_digest
from run_tc_oos_rsg_experiments import _file_sha256, _read_manifest_rows, _write_json, calculate_probability_metrics, state_dict_sha256
from run_verifier_strong_controls import write_predictions
from verifier_strong_controls import fit_global_temperature


DEFAULT_PROTOCOL = ROOT / "docs/experiment_protocols/abmp_anchor_linear_controls_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/training/abmp_anchor_linear_controls_v1/development_fold_0"
SOLVER = {"lbfgs_maxiter": 5000, "lbfgs_maxls": 100, "lbfgs_ftol": 1e-15, "lbfgs_gtol": 1e-9,
          "trust_maxiter": 100, "trust_gtol": 1e-9, "stationarity_l2_tolerance": 1e-7, "gap_upper_tolerance": 1e-8}
CODE_NAMES = ("anchor_linear_controls.py", "run_anchor_linear_controls.py", "frozen_visual_probe.py",
              "run_frozen_visual_probe.py", "verifier_strong_controls.py", "run_verifier_trust_development.py",
              "tc_oos_rsg.py", "hrgv_network.py", "run_tc_oos_rsg_experiments.py", "run_verifier_strong_controls.py")
_THREAD_LIMITER = None


def runtime():
    global _THREAD_LIMITER
    configure_runtime()
    _THREAD_LIMITER = threadpool_limits(limits=1)


def validate_protocol(protocol):
    if (protocol.get("protocol") != "abmp_anchor_linear_controls_v1" or protocol.get("fold") != 0
            or protocol.get("approval") != "Human: 批准锚点收敛强对照"
            or protocol.get("subsets") != list(SUBSETS) or protocol.get("counts") != {"fit": 340, "stop": 340}
            or protocol.get("arms") != list(controls.ARMS) or protocol.get("initializations") != list(controls.INITIALIZATIONS)
            or protocol.get("primary_initialization") != "zero"
            or protocol.get("reference_output") != "outputs/training/abmp_frozen_visual_probe_v1/development_fold_0"):
        raise ValueError("Unapproved control scope.")
    if (protocol.get("regularization") != .001 or protocol.get("perturbation_sd") != .001
            or protocol.get("solver") != SOLVER or protocol.get("restart_probability_tolerance") != 1e-6
            or protocol.get("restart_objective_tolerance") != 1e-9):
        raise ValueError("Unregistered control solver or selection setting.")
    if protocol.get("permutation_seeds") != {"fit": 20271002, "stop": 20281002}:
        raise ValueError("Unregistered permutation setting.")
    expected_screen = {"nll_improvement": .005, "new_correct_targets_min": 1, "net_correct_targets_min": 0,
                       "extra_ti_false_targets_max": 1, "extra_metallic_false_targets_max": 1}
    if any(protocol.get("screening", {}).get(k) != v for k, v in expected_screen.items()):
        raise ValueError("Unregistered screening setting.")
    digests = [protocol.get("reference_summary_sha256", ""), protocol.get("reference_protocol_sha256", "")]
    cache_hashes = protocol.get("cache_sha256", {})
    if set(cache_hashes) != {"frozen_features.pt", "preprocessing.pt"}:
        raise ValueError("Unapproved cache scope.")
    digests.extend(cache_hashes.values())
    if any(len(d) != 64 or any(c not in "0123456789abcdef" for c in d) for d in digests):
        raise ValueError("Invalid SHA256 registration.")


def prepare(root, protocol):
    reference = root / protocol["reference_output"]
    summary_path = reference / "development_summary.json"
    if _file_sha256(summary_path) != protocol["reference_summary_sha256"]:
        raise ValueError("Frozen visual probe summary changed.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original_path = reference / "registered_protocol.json"
    if _file_sha256(original_path) != protocol["reference_protocol_sha256"]:
        raise ValueError("Frozen visual probe protocol changed.")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    data, source = snapshot(root, original)
    if source != summary["source_snapshot"]:
        raise ValueError("Original sources or report changed.")
    for name, digest in summary["code_sha256"].items():
        if _file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Frozen probe code changed.")
    for name, digest in protocol["cache_sha256"].items():
        if digest != summary["artifact_sha256"][name] or _file_sha256(reference / name) != digest:
            raise ValueError("Frozen visual cache changed.")
    cache = torch.load(reference / "frozen_features.pt", map_location="cpu", weights_only=True)
    preprocessing = torch.load(reference / "preprocessing.pt", map_location="cpu", weights_only=True)
    audit_path = reference / "image_feature_audit.json"
    if _file_sha256(audit_path) != summary["artifact_sha256"]["image_feature_audit.json"]:
        raise ValueError("Frozen feature identity audit changed.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    offset = 0
    for name, subset, key in zip(SUBSETS, data, ("fit", "stop"), strict=True):
        values = cache[key]
        if values.shape != (340, 1280):
            raise ValueError("Frozen feature matrix dimensions changed.")
        for record, feature in zip(subset["records"], values, strict=True):
            row = audit[offset]
            if (row["subset"] != name or row["image_id"] != record["image_id"]
                    or row["relative_path"] != record["relative_path"] or row["feature_sha256"] != tensor_digest(feature)):
                raise ValueError("Cached visual row identity changed.")
            offset += 1
    if offset != len(audit):
        raise ValueError("Unexpected frozen audit rows.")
    representation = controls.fit_representation(data[0]["evidence"], data[0]["p"], data[0]["contradiction"])
    blocks = []
    for subset, key in zip(data, ("fit", "stop"), strict=True):
        h = probe.apply_preprocessing(subset["evidence"], cache[key].double(), preprocessing)["H"]
        blocks.append(controls.representations(subset["evidence"], subset["p"], subset["contradiction"], h, representation))
    strong = root / original["strong_control_summary"]
    global_path = strong.parent / "global_fits.json"
    strong_summary = json.loads(strong.read_text(encoding="utf-8"))
    if _file_sha256(global_path) != strong_summary["artifact_sha256"]["global_fits.json"]:
        raise ValueError("Strong temperature fit changed.")
    temperature = fit_global_temperature(data[0]["p"], data[0]["labels"], [1e-6, 100.])
    if temperature != json.loads(global_path.read_text(encoding="utf-8"))["T0"]:
        raise ValueError("Fit-only anchor temperature did not exactly replay.")
    anchors = [subset["p"].log()/temperature["temperature"] for subset in data]
    return data, blocks, anchors, representation, temperature, dict(source, cache_sha256=protocol["cache_sha256"])


def screen(eh, probability_controls, transition, base):
    if set(probability_controls) != {"T0", "P", "E", "EH_permuted"}:
        raise ValueError("All registered strongest-control screens are required.")
    information = eh["nll"] <= min(r["nll"] for r in probability_controls.values())-.005
    checks = {"new_correct_target": transition["new_correct_targets"] >= 1,
              "net_correct_targets": transition["net_correct_targets"] >= 0}
    for k, name in (("1", "ti_guard"), ("3", "metallic_guard")):
        checks[name] = transition["false_target_count_by_class"][k] <= base["false_target_count_by_class"][k]+1
    return {"information_signal": information, "role_usable_signal": information and all(checks.values()), "role_checks": checks}


def result_row(trained, arm, data, xs, anchors):
    row = {key: trained[key] for key in trained if key not in ("theta", "history")}
    row.update(arm=arm, state_sha256=state_dict_sha256({"theta": trained["theta"]}), metrics={}, transitions={})
    for name, subset, x, anchor in zip(("fit", "stop"), data, xs, anchors, strict=True):
        q = controls.probabilities(x, anchor, trained["theta"])
        row["metrics"][name] = calculate_probability_metrics(q, subset["labels"])
        row["transitions"][name] = probe.transition_counts(q, anchor.softmax(1), subset["labels"])
    return row


def restart_checks(fits, xs, anchors, protocol):
    reference = fits[0]
    checks = []
    for other in fits[1:]:
        coefficient_distance = float((reference["theta"]-other["theta"]).norm())
        coefficient_bound = (reference["gradient_l2"]+other["gradient_l2"])/protocol["regularization"]
        output_distance = max(float((controls.probabilities(x,a,reference["theta"])-controls.probabilities(x,a,other["theta"])).abs().max()) for x,a in zip(xs,anchors,strict=True))
        objective_distance = abs(reference["regularized_fit_objective"]-other["regularized_fit_objective"])
        passed = (coefficient_distance <= coefficient_bound+1e-10 and output_distance <= protocol["restart_probability_tolerance"]
                  and objective_distance <= protocol["restart_objective_tolerance"])
        checks.append({"initialization": other["initialization"], "coefficient_l2_distance": coefficient_distance,
                       "strong_convexity_distance_bound": coefficient_bound, "probability_max_abs_distance": output_distance,
                       "objective_abs_distance": objective_distance, "passed": passed})
    return checks


def check_summary_fields(summary, baseline, primary, restarts):
    screening = screen(primary["EH"]["metrics"]["stop"], {"T0":baseline["metrics"]["stop"],**{arm:primary[arm]["metrics"]["stop"] for arm in ("P","E","EH_permuted")}},
                       primary["EH"]["transitions"]["stop"], baseline["transitions"]["stop"])
    expected = {"baseline":baseline,"primary":primary,"restart_checks":restarts,"screening":screening,
                "numerical_stability_passed":all(row["passed"] for rows in restarts.values() for row in rows)}
    if any(summary.get(key) != value for key,value in expected.items()):
        raise ValueError("Aggregate summary did not exactly replay.")


def run_experiment(root, protocol_path, output):
    if output.exists():
        raise FileExistsError("Output exists; refusing to overwrite experiment.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    runtime()
    data, blocks, anchors, representation, temperature, source = prepare(root, protocol)
    code_hashes = {name: _file_sha256(root/"scripts"/name) for name in CODE_NAMES}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, output/"registered_protocol.json")
    _write_json(output/"run_status.json", {"status": "RUNNING", "auto_retry": False})
    try:
        torch.save(representation, output/"representation.pt")
        baseline = {"metrics": {}, "transitions": {}}
        for name, subset, anchor in zip(("fit", "stop"), data, anchors, strict=True):
            q = anchor.softmax(1)
            baseline["metrics"][name] = calculate_probability_metrics(q, subset["labels"])
            baseline["transitions"][name] = probe.transition_counts(q, q, subset["labels"])
            write_predictions(output/f"T0_{name}_predictions.csv", subset, {"probabilities": q})
        runs, primary, restarts = [], {}, {}
        for arm in controls.ARMS:
            xs = [controls.make_input(b, arm, protocol["permutation_seeds"][name]) for b, name in zip(blocks, ("fit", "stop"), strict=True)]
            fits = []
            for initialization in controls.INITIALIZATIONS:
                trained = controls.fit(xs[0], anchors[0], data[0]["labels"], protocol["regularization"], initialization, protocol["solver"])
                row = result_row(trained, arm, data, xs, anchors)
                folder = output/f"{arm}_{initialization}"
                folder.mkdir()
                torch.save({"theta": trained["theta"]}, folder/"coefficients.pt")
                _write_json(folder/"result.json", row)
                _write_json(folder/"history.json", trained["history"])
                for name, subset, x, anchor in zip(("fit", "stop"), data, xs, anchors, strict=True):
                    write_predictions(folder/f"{name}_predictions.csv", subset, {"probabilities": controls.probabilities(x, anchor, trained["theta"])})
                fits.append(trained)
                runs.append(row)
                if initialization == "zero":
                    primary[arm] = row
                print(f"{arm} {initialization} grad={trained['gradient_l2']:.3e} gap<={trained['gap_upper_bound']:.3e} stop_NLL={row['metrics']['stop']['nll']:.9f}", flush=True)
            restarts[arm] = restart_checks(fits,xs,anchors,protocol)
        _, _, _, _, _, after = prepare(root, protocol)
        if source != after or any(_file_sha256(root/"scripts"/name) != digest for name,digest in code_hashes.items()):
            raise ValueError("Code, sources, cached features or report changed.")
        screening = screen(primary["EH"]["metrics"]["stop"], {"T0": baseline["metrics"]["stop"], **{arm:primary[arm]["metrics"]["stop"] for arm in ("P","E","EH_permuted")}},
                           primary["EH"]["transitions"]["stop"], baseline["transitions"]["stop"])
        numerical_stability = all(r["passed"] for checks in restarts.values() for r in checks)
        summary = {"protocol": protocol["protocol"], "status": "DEVELOPMENT_CONVEX_CONTROL_ONLY",
                   "protocol_sha256": _file_sha256(protocol_path), "code_sha256": code_hashes, "source_snapshot": source,
                   "temperature_anchor": temperature, "baseline": baseline, "primary": primary, "runs": runs,
                   "restart_checks": restarts, "numerical_stability_passed": numerical_stability, "screening": screening,
                   "environment": {"python": platform.python_version(), "torch": torch.__version__, "scipy": scipy.__version__, "device": "cpu", "dtype": "float64", "threads": 1},
                   "new_images_loaded": False, "backbone_updated": False, "outer_manifest_read": False, "formal_report_unchanged": True,
                   "independent_confirmation": False, "automatic_promotion": False,
                   "claim_boundary": "Strong convex control, not a new neural architecture. Numerical restarts are not statistical seed replicates. Exposed development stopping data only; no MI, significance or industrial claim."}
        summary["artifact_sha256"] = {str(path.relative_to(output)).replace("\\","/"): _file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run_status.json"}
        _write_json(output/"development_summary.json", summary)
        _write_json(output/"run_status.json", {"status": "COMPLETED", "research_status": summary["status"], "numerical_stability_passed": numerical_stability})
        return summary
    except Exception as exc:
        _write_json(output/"run_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "auto_retry": False})
        raise


def check_prediction(path, subset, expected):
    rows = _read_manifest_rows(path)
    if len(rows) != len(expected):
        raise ValueError("Prediction row count mismatch.")
    for row, record, label, q in zip(rows, subset["records"], subset["labels"], expected, strict=True):
        if (row["image_id"] != record["image_id"] or row["split_group_id"] != record["split_group_id"]
                or int(row["true_class_id"]) != int(label) or int(row["predicted_class_id"]) != int(q.argmax())
                or [float(row[f"prob_{i}"]) for i in range(4)] != q.tolist()):
            raise ValueError("Prediction identity or probability replay mismatch.")
    return len(rows)


def verify_delivery(root, output):
    runtime()
    protocol_path = output/"registered_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    summary = json.loads((output/"development_summary.json").read_text(encoding="utf-8"))
    if [(row["arm"],row["initialization"]) for row in summary["runs"]] != [(arm,initialization) for arm in controls.ARMS for initialization in controls.INITIALIZATIONS]:
        raise ValueError("Unregistered solution replay roster.")
    if _file_sha256(protocol_path) != summary["protocol_sha256"]:
        raise ValueError("Registered protocol changed.")
    if any(_file_sha256(root/"scripts"/name) != digest for name,digest in summary["code_sha256"].items()):
        raise ValueError("Registered code changed.")
    if any(_file_sha256(output/name) != digest for name,digest in summary["artifact_sha256"].items()):
        raise ValueError("Registered artifact changed.")
    data, blocks, anchors, state, temperature, source = prepare(root, protocol)
    if source != summary["source_snapshot"] or temperature != summary["temperature_anchor"]:
        raise ValueError("Source or anchor replay mismatch.")
    saved_state = torch.load(output/"representation.pt", map_location="cpu", weights_only=True)
    if set(saved_state) != set(state) or any(not torch.equal(state[k],saved_state[k]) for k in state):
        raise ValueError("Fit-only representation preprocessing replay mismatch.")
    tables, rows = 0, 0
    baseline = {"metrics":{},"transitions":{}}
    primary, fitted, restarts = {}, {}, {}
    for name,subset,anchor in zip(("fit","stop"),data,anchors,strict=True):
        q = anchor.softmax(1)
        baseline["metrics"][name] = calculate_probability_metrics(q,subset["labels"])
        baseline["transitions"][name] = probe.transition_counts(q,q,subset["labels"])
        rows += check_prediction(output/f"T0_{name}_predictions.csv",subset,q)
        tables += 1
    for registered in summary["runs"]:
        arm, initialization = registered["arm"], registered["initialization"]
        xs = [controls.make_input(b,arm,protocol["permutation_seeds"][name]) for b,name in zip(blocks,("fit","stop"),strict=True)]
        trained = controls.fit(xs[0],anchors[0],data[0]["labels"],protocol["regularization"],initialization,protocol["solver"])
        folder = output/f"{arm}_{initialization}"
        coefficients = torch.load(folder/"coefficients.pt",map_location="cpu",weights_only=True)
        if not torch.equal(trained["theta"],coefficients["theta"]) or result_row(trained,arm,data,xs,anchors) != registered:
            raise ValueError("Convex solution/result did not exactly replay.")
        if json.loads((folder/"result.json").read_text(encoding="utf-8")) != registered:
            raise ValueError("Per-run saved result and summary disagree.")
        fitted.setdefault(arm,[]).append(trained)
        if initialization == "zero":
            primary[arm] = result_row(trained,arm,data,xs,anchors)
        if len(fitted[arm]) == len(controls.INITIALIZATIONS):
            restarts[arm] = restart_checks(fitted[arm],xs,anchors,protocol)
        if trained["history"] != json.loads((folder/"history.json").read_text(encoding="utf-8")):
            raise ValueError("Solver history did not exactly replay.")
        for name,subset,x,anchor in zip(("fit","stop"),data,xs,anchors,strict=True):
            rows += check_prediction(folder/f"{name}_predictions.csv",subset,controls.probabilities(x,anchor,trained["theta"]))
            tables += 1
    check_summary_fields(summary,baseline,primary,restarts)
    _,_,_,_,_,after = prepare(root,protocol)
    if after != source or any(_file_sha256(root/"scripts"/name) != digest for name,digest in summary["code_sha256"].items()):
        raise ValueError("Source, report or code changed during replay.")
    return {"status": "EXACT_REPLAY_VERIFIED", "solution_replays": len(summary["runs"]), "prediction_tables": tables,
            "prediction_rows": rows, "prediction_max_abs_residual": 0.0, "source_and_report_unchanged": True,
            "new_images_loaded": False, "outer_manifest_read": False, "numerical_stability_passed": summary["numerical_stability_passed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_delivery(ROOT,args.output)
        path = args.output/"delivery_verification.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != result:
                raise ValueError("Saved delivery verification mismatch.")
        else:
            _write_json(path,result)
        print(json.dumps(result, ensure_ascii=True))
    else:
        run_experiment(ROOT,args.protocol,args.output)
