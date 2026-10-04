"""Approved frozen-feature feasibility audit on exactly two exposed inner subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import models, transforms

import frozen_visual_probe as probe
from run_tc_oos_rsg_experiments import _file_sha256, _write_json, calculate_probability_metrics, state_dict_sha256
from run_verifier_trust_development import ROOT, REPORT_NAME, load_subset
from verifier_strong_controls import fit_global_temperature, temperature_probabilities


SUBSETS = ("gate_stop_projector_fit", "projector_stop")
DEFAULT_PROTOCOL = ROOT / "docs/experiment_protocols/abmp_frozen_visual_probe_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/training/abmp_frozen_visual_probe_v1/development_fold_0"
BUDGET = {"epochs": 30, "batch_size": 256, "learning_rate": .001, "weight_decay": .0001, "dropout": .1, "hidden_dim": 64}


def validate_protocol(protocol: dict) -> None:
    if (protocol.get("protocol") != "abmp_frozen_visual_probe_v1" or protocol.get("fold") != 0
            or protocol.get("approval") != "Human: 批准冻结视觉特征探查"
            or protocol.get("subsets") != list(SUBSETS) or protocol.get("counts") != {"fit": 340, "stop": 340}
            or protocol.get("arms") != list(probe.ARMS) or protocol.get("architectures") != list(probe.ARCHITECTURES)
            or protocol.get("seeds") != [20261002, 20261003, 20261004]):
        raise ValueError("Unapproved experiment scope.")
    if protocol.get("budget") != BUDGET:
        raise ValueError("Unregistered training budget.")
    digests = [protocol["strong_control_summary_sha256"], protocol["formal_report_sha256"],
               protocol["feature_extractor"]["checkpoint_sha256"], *protocol["input_sha256"].values()]
    if any(len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) for digest in digests):
        raise ValueError("Registered SHA256 must be 64 lowercase hexadecimal characters.")
    feature = protocol.get("feature_extractor", {})
    preprocessing = protocol.get("preprocessing", {})
    if (feature.get("checkpoint") != "outputs/training/tc_oos_rsg_v1/fold_0/expert/best_model.pt"
            or feature.get("backbone") != "efficientnet_b0" or feature.get("raw_dimension") != 1280
            or feature.get("batch_size") != 16 or feature.get("device") != "cuda"
            or preprocessing.get("visual_pca_components") != 64 or preprocessing.get("input_dimension") != 82):
        raise ValueError("Unapproved feature scope.")


def configure_runtime() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def snapshot(root: Path, protocol: dict) -> tuple[list, dict]:
    data = [load_subset(root, name, protocol["input_sha256"]) for name in SUBSETS]
    for key in ("image_id", "split_group_id"):
        if {r[key] for r in data[0]["records"]} & {r[key] for r in data[1]["records"]}:
            raise ValueError("Fit and stop identities overlap.")
    if any(len(d["records"]) != 340 for d in data):
        raise ValueError("Approved inner sample counts changed.")
    paths = {"checkpoint": root / protocol["feature_extractor"]["checkpoint"],
             "strong_control_summary": root / protocol["strong_control_summary"],
             "formal_report": root / "结题" / REPORT_NAME}
    hashes = {key: _file_sha256(path) for key, path in paths.items()}
    expected = {"checkpoint": protocol["feature_extractor"]["checkpoint_sha256"],
                "strong_control_summary": protocol["strong_control_summary_sha256"],
                "formal_report": protocol["formal_report_sha256"]}
    if hashes != expected:
        raise ValueError("Registered source or report changed.")
    return data, dict(hashes, input_sha256=dict(data[0]["hashes"], **data[1]["hashes"]))


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def build_extractor(root: Path, protocol: dict) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError("Registered CUDA extraction unavailable; no silent device change.")
    model = models.efficientnet_b0(weights=None)
    checkpoint = torch.load(root / protocol["feature_extractor"]["checkpoint"], map_location="cpu", weights_only=True)
    model.features.load_state_dict(probe.extract_feature_state(checkpoint), strict=True)
    return probe.freeze_extractor(torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten(1)).cuda())


def extract_features(model: torch.nn.Module, data: list, dataset_root: Path) -> tuple[list, list]:
    transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    arrays, audit = [], []
    all_paths = set()
    for subset, item in zip(SUBSETS, data, strict=True):
        chunks = []
        for start in range(0, len(item["records"]), 16):
            records = item["records"][start:start+16]
            images, image_audit = [], []
            for record in records:
                path = probe.scoped_image_path(dataset_root, record["relative_path"])
                if path in all_paths:
                    raise ValueError("Manifest aliases the same physical image path twice.")
                all_paths.add(path)
                before = _file_sha256(path)
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
                if _file_sha256(path) != before:
                    raise ValueError("Image changed while being read.")
                image_audit.append({"subset": subset, "image_id": record["image_id"], "split_group_id": record["split_group_id"],
                                    "relative_path": record["relative_path"], "image_sha256": before})
            with torch.inference_mode():
                features = model(torch.stack(images).cuda()).cpu()
            if features.shape != (len(records), 1280) or not bool(torch.isfinite(features).all()):
                raise ValueError("Frozen features violate the registered contract.")
            chunks.append(features)
            for row, feature in zip(image_audit, features, strict=True):
                audit.append(dict(row, feature_sha256=tensor_digest(feature)))
            print(f"extract {subset}: {min(start+16,340)}/340", flush=True)
        arrays.append(torch.cat(chunks))
    return arrays, audit


def write_predictions(path: Path, data: dict, q: torch.Tensor, temperature: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split_group_id", "true_class_id", "predicted_class_id", "temperature", "prob_0", "prob_1", "prob_2", "prob_3"])
        for i, record in enumerate(data["records"]):
            writer.writerow([record["image_id"], record["split_group_id"], int(data["labels"][i]), int(q[i].argmax()), temperature, *q[i].tolist()])


def aggregate(runs: list) -> dict:
    result = {}
    for architecture in probe.ARCHITECTURES:
        result[architecture] = {}
        for arm in probe.ARMS:
            selected = [r for r in runs if r["architecture"] == architecture and r["arm"] == arm]
            result[architecture][arm] = {}
            for version in ("raw", "calibrated"):
                result[architecture][arm][version] = {field: {"mean": statistics.mean(r["metrics"]["stop"][version][field] for r in selected),
                                                            "sample_sd": statistics.stdev(r["metrics"]["stop"][version][field] for r in selected)}
                                                     for field in selected[0]["metrics"]["stop"][version]}
    return result


def screen(runs: list, data: dict) -> dict:
    base = probe.transition_counts(data["p"], data["p"], data["labels"])
    result = {}
    for architecture in probe.ARCHITECTURES:
        family = [r for r in runs if r["architecture"] == architecture]
        e = {r["seed"]: r for r in family if r["arm"] == "E"}
        full = [r for r in family if r["arm"] == "EH"]
        shuffled = [r for r in family if r["arm"] == "EH_permuted"]
        deltas = [r["metrics"]["stop"]["raw"]["nll"] - e[r["seed"]]["metrics"]["stop"]["raw"]["nll"] for r in full]
        permutation_delta = statistics.mean(r["metrics"]["stop"]["raw"]["nll"] for r in full) - statistics.mean(r["metrics"]["stop"]["raw"]["nll"] for r in shuffled)
        information = all(d <= -.005 for d in deltas) and permutation_delta <= -.005
        guards = [r["transitions"]["stop"]["raw"]["new_correct_targets"] >= 1
                  and r["transitions"]["stop"]["raw"]["net_correct_targets"] >= 0
                  and all(r["transitions"]["stop"]["raw"]["false_target_count_by_class"][str(k)] <= base["false_target_count_by_class"][str(k)]+1 for k in (1, 3)) for r in full]
        result[architecture] = {"EH_minus_E_raw_nll_per_seed": deltas,
                                "mean_EH_minus_permuted_raw_nll": permutation_delta,
                                "information_signal": information, "role_guards_per_seed": guards,
                                "role_usable_signal": information and all(guards), "automatic_promotion": False}
    return result


def run_experiment(root: Path, protocol_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Output exists; no experiment overwrites.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    configure_runtime()
    data, before = snapshot(root, protocol)
    code_hashes = {name: _file_sha256(root / "scripts" / name) for name in ("frozen_visual_probe.py", "run_frozen_visual_probe.py")}
    protocol_sha = _file_sha256(protocol_path)
    output.mkdir(parents=True)
    shutil.copyfile(protocol_path, output / "registered_protocol.json")
    _write_json(output / "run_status.json", {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        model = build_extractor(root, protocol)
        frozen_state = state_dict_sha256(model.state_dict())
        arrays, image_audit = extract_features(model, data, Path(protocol["dataset_root"]))
        if frozen_state != state_dict_sha256(model.state_dict()):
            raise ValueError("Frozen extractor state changed.")
        torch.save({"fit": arrays[0], "stop": arrays[1]}, output / "frozen_features.pt")
        _write_json(output / "image_feature_audit.json", image_audit)
        del model
        torch.cuda.empty_cache()
        preprocessing = probe.fit_preprocessing(data[0]["evidence"], arrays[0].double(), 64)
        torch.save(preprocessing, output / "preprocessing.pt")
        blocks = [probe.apply_preprocessing(d["evidence"], h.double(), preprocessing) for d, h in zip(data, arrays, strict=True)]
        runs = []
        for architecture in probe.ARCHITECTURES:
            for seed in protocol["seeds"]:
                for arm in probe.ARMS:
                    xs = [probe.make_input(b, arm, seed+offset) for b, offset in zip(blocks, (100000, 200000), strict=True)]
                    trained = probe.fit_classifier(xs[0], data[0]["labels"], xs[1], data[1]["labels"], architecture, seed, protocol["budget"])
                    folder = output / f"{architecture}_{arm}_seed{seed}"
                    folder.mkdir()
                    torch.save(trained["state_dict"], folder / "best_head.pt")
                    raw = [probe.predict_classifier(trained["head"], x) for x in xs]
                    calibration = fit_global_temperature(raw[0], data[0]["labels"], [1e-6, 100.])
                    metrics, transitions = {}, {}
                    for name, item, q in zip(("fit", "stop"), data, raw, strict=True):
                        calibrated = temperature_probabilities(q, torch.full((len(q), 1), calibration["temperature"], dtype=torch.float64))
                        metrics[name], transitions[name] = {}, {}
                        for version, predictions, temperature in (("raw", q, 1.), ("calibrated", calibrated, calibration["temperature"])):
                            metrics[name][version] = calculate_probability_metrics(predictions, item["labels"])
                            transitions[name][version] = probe.transition_counts(predictions, item["p"], item["labels"])
                            write_predictions(folder / f"{name}_{version}_predictions.csv", item, predictions, temperature)
                    result = {key: trained[key] for key in ("initial_sha256", "state_sha256", "best_epoch", "parameter_count")}
                    result.update(architecture=architecture, arm=arm, seed=seed, metrics=metrics, transitions=transitions, calibration=calibration,
                                  active_input_dimension=18 if arm == "E" else 64 if arm == "H" else 82)
                    _write_json(folder / "history.json", trained["history"])
                    _write_json(folder / "result.json", result)
                    runs.append(result)
                    print(f"probe {architecture}/{arm}/{seed} epoch={result['best_epoch']} raw_NLL={metrics['stop']['raw']['nll']:.6f} calibrated_NLL={metrics['stop']['calibrated']['nll']:.6f} new_target={transitions['stop']['raw']['new_correct_targets']} lost_target={transitions['stop']['raw']['lost_correct_targets']}", flush=True)
        _, after = snapshot(root, protocol)
        if before != after or _file_sha256(protocol_path) != protocol_sha or any(_file_sha256(root / "scripts" / name) != digest for name, digest in code_hashes.items()):
            raise ValueError("Sources or registered code changed during experiment.")
        strong = json.loads((root / protocol["strong_control_summary"]).read_text(encoding="utf-8"))
        summary = {"protocol": protocol["protocol"], "status": "FEASIBILITY_PROBE_ONLY",
                   "completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_sha,
                   "code_sha256": code_hashes, "source_snapshot": after, "frozen_extractor_state_sha256": frozen_state,
                   "image_count": len(image_audit), "raw_feature_shapes": [list(a.shape) for a in arrays],
                   "pca_retained_fit_variance": preprocessing["retained_variance"],
                   "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                                   "gpu": torch.cuda.get_device_name(), "head_device": "cpu", "head_dtype": "float64", "threads": 1},
                   "runs": runs, "aggregate": aggregate(runs), "screening": screen(runs, data[1]),
                   "retained_global_controls": strong["global_controls"], "retained_learned_controls": strong["aggregate"],
                   "outer_manifest_read": False, "outer_images_loaded": False, "backbone_updated": False,
                   "formal_report_unchanged": True, "independent_confirmation": False, "automatic_promotion": False,
                   "claim_boundary": protocol["limitations"]}
        summary["artifact_sha256"] = {str(p.relative_to(output)).replace("\\", "/"): _file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_status.json"}
        _write_json(output / "development_summary.json", summary)
        _write_json(output / "run_status.json", {"status": "COMPLETED", "research_status": summary["status"]})
        return summary
    except Exception as exc:
        _write_json(output / "run_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "auto_retry": False})
        raise


def verify_delivery(root: Path, output: Path, replay_images: bool = False) -> dict:
    configure_runtime()
    protocol_path = output / "registered_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    summary = json.loads((output / "development_summary.json").read_text(encoding="utf-8"))
    if _file_sha256(protocol_path) != summary["protocol_sha256"]:
        raise ValueError("Protocol bytes changed.")
    for name, digest in summary["code_sha256"].items():
        if _file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Registered code changed.")
    for name, digest in summary["artifact_sha256"].items():
        if _file_sha256(output / name) != digest:
            raise ValueError(f"Artifact changed: {name}")
    data, current = snapshot(root, protocol)
    if current != summary["source_snapshot"]:
        raise ValueError("Source snapshot changed.")
    cache = torch.load(output / "frozen_features.pt", map_location="cpu", weights_only=True)
    arrays = [cache["fit"], cache["stop"]]
    audit = json.loads((output / "image_feature_audit.json").read_text(encoding="utf-8"))
    index = 0
    for subset, d, array in zip(SUBSETS, data, arrays, strict=True):
        if array.shape != (340, 1280) or array.dtype != torch.float32 or not bool(torch.isfinite(array).all()):
            raise ValueError("Feature cache contract changed.")
        for record, feature in zip(d["records"], array, strict=True):
            row = audit[index]
            if row["image_id"] != record["image_id"] or row["subset"] != subset or row["relative_path"] != record["relative_path"] or row["feature_sha256"] != tensor_digest(feature):
                raise ValueError("Feature row identity/digest mismatch.")
            index += 1
    if index != len(audit) or index != 680:
        raise ValueError("Feature row count mismatch.")
    feature_residual = None
    if replay_images:
        model = build_extractor(root, protocol)
        initial_state = state_dict_sha256(model.state_dict())
        replay, replay_audit = extract_features(model, data, Path(protocol["dataset_root"]))
        feature_residual = max(float((a-b).abs().max()) for a, b in zip(arrays, replay, strict=True))
        if feature_residual != 0 or replay_audit != audit or initial_state != state_dict_sha256(model.state_dict()) or initial_state != summary["frozen_extractor_state_sha256"]:
            raise ValueError("Full frozen-feature extraction did not exactly replay.")
        del model
        torch.cuda.empty_cache()
    state = torch.load(output / "preprocessing.pt", weights_only=True)
    recomputed = probe.fit_preprocessing(data[0]["evidence"], arrays[0].double(), 64)
    for key, value in state.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, recomputed[key], rtol=0, atol=0)
        elif value != recomputed[key]:
            raise ValueError("Fit-only preprocessing did not replay.")
    blocks = [probe.apply_preprocessing(d["evidence"], h.double(), state) for d, h in zip(data, arrays, strict=True)]
    tables, rows, maximum = 0, 0, 0.
    for result in summary["runs"]:
        architecture, arm, seed = result["architecture"], result["arm"], result["seed"]
        folder = output / f"{architecture}_{arm}_seed{seed}"
        xs = [probe.make_input(b, arm, seed+offset) for b, offset in zip(blocks, (100000, 200000), strict=True)]
        trained = probe.fit_classifier(xs[0], data[0]["labels"], xs[1], data[1]["labels"], architecture, seed, protocol["budget"])
        if (trained["state_sha256"] != result["state_sha256"] or trained["history"] != json.loads((folder / "history.json").read_text(encoding="utf-8"))
                or trained["initial_sha256"] != result["initial_sha256"] or trained["best_epoch"] != result["best_epoch"]):
            raise ValueError("Deterministic head training did not replay.")
        saved = torch.load(folder / "best_head.pt", weights_only=True)
        if state_dict_sha256(saved) != result["state_sha256"]:
            raise ValueError("Selected head checkpoint changed.")
        trained["head"].load_state_dict(saved, strict=True)
        raw = [probe.predict_classifier(trained["head"], x) for x in xs]
        calibration = fit_global_temperature(raw[0], data[0]["labels"], [1e-6, 100.])
        if calibration != result["calibration"]:
            raise ValueError("Fit-only calibration did not replay.")
        for name, d, q in zip(("fit", "stop"), data, raw, strict=True):
            calibrated = temperature_probabilities(q, torch.full((len(q), 1), calibration["temperature"], dtype=torch.float64))
            for version, predictions in (("raw", q), ("calibrated", calibrated)):
                if calculate_probability_metrics(predictions, d["labels"]) != result["metrics"][name][version] or probe.transition_counts(predictions, d["p"], d["labels"]) != result["transitions"][name][version]:
                    raise ValueError("Result metrics did not replay.")
                with (folder / f"{name}_{version}_predictions.csv").open(encoding="utf-8", newline="") as handle:
                    stored = list(csv.DictReader(handle))
                if len(stored) != 340:
                    raise ValueError("Prediction table row count changed.")
                for i, (row, record) in enumerate(zip(stored, d["records"], strict=True)):
                    if row["image_id"] != record["image_id"] or row["split_group_id"] != record["split_group_id"] or int(row["true_class_id"]) != int(d["labels"][i]) or int(row["predicted_class_id"]) != int(predictions[i].argmax()):
                        raise ValueError("Prediction table identity changed.")
                    maximum = max(maximum, max(abs(float(row[f"prob_{k}"])-float(predictions[i,k])) for k in range(4)))
                tables += 1
                rows += len(stored)
        print(f"verify {architecture}/{arm}/{seed}", flush=True)
    if maximum != 0 or aggregate(summary["runs"]) != summary["aggregate"] or screen(summary["runs"], data[1]) != summary["screening"]:
        raise ValueError("Aggregate or probability replay failed.")
    result = {"status": "VERIFIED", "head_training_replays": len(summary["runs"]), "prediction_tables": tables,
              "prediction_rows": rows, "max_prediction_residual": maximum, "full_image_replay": replay_images,
              "feature_replay_max_abs_residual": feature_residual, "source_and_report_unchanged": True,
              "outer_manifest_read": False, "independent_confirmation": False}
    _write_json(output / "delivery_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replay-images", action="store_true")
    args = parser.parse_args()
    if args.replay_images and not args.verify:
        parser.error("--replay-images requires --verify")
    result = verify_delivery(args.root.resolve(), args.output.resolve(), args.replay_images) if args.verify else run_experiment(args.root.resolve(), args.protocol.resolve(), args.output.resolve())
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
