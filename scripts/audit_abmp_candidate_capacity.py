"""Audit frozen Fold 0 candidate geometry, without outer-fold access or training."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from hrgv_network import apply_residual_target_verifiers
from run_tc_oos_rsg_experiments import (
    _file_sha256, _load_frozen_expert, _read_manifest_rows, _write_json,
    calculate_probability_metrics,
)
from tc_oos_rsg import _validate_probability_matrix, _validate_unit_column


ROOT = Path(__file__).resolve().parents[1]
INNER_NAMES = ("expert_fit", "expert_stop", "gate_fit", "gate_stop_projector_fit", "projector_stop")
EXTRACTION_NAMES = ("gate_stop_projector_fit", "projector_stop")
BRANCH_NAMES = tuple(f"{name}_{state}" for name in ("direct", "mapped", "fixed", "oos", "original") for state in ("pre", "verified"))


def target_segment_interval(first, second, lower=0.0, upper=1.0, minimum_margin=1e-9):
    """Intersect affine target-margin halfspaces; index zero is the target role."""
    _validate_probability_matrix(first, "first")
    _validate_probability_matrix(second, "second")
    if first.shape != second.shape or first.device != second.device or first.dtype != second.dtype:
        raise ValueError("Segment posteriors must share shape, device and dtype.")
    if not math.isfinite(minimum_margin) or minimum_margin < 0:
        raise ValueError("minimum_margin must be finite and non-negative.")
    first, second = first.double(), second.double()
    def bound_column(value, name):
        if not torch.is_tensor(value):
            value = torch.full_like(first[:, :1], float(value))
        _validate_unit_column(value, name, first)
        return value.double().clone()
    lo, hi = bound_column(lower, "lower"), bound_column(upper, "upper")
    if bool((lo > hi).any()):
        raise ValueError("lower must not exceed upper.")
    intercept = second[:, :1] - second[:, 1:]
    slope = (first[:, :1] - first[:, 1:]) - intercept
    constant_bad = ((slope == 0) & (intercept < minimum_margin)).any(1, keepdim=True)
    # Use a safe denominator before masking so constant directions stay finite.
    crossings = (minimum_margin - intercept) / torch.where(slope == 0, torch.ones_like(slope), slope)
    # Infinite roots only certify emptiness or impose no restriction on [0, 1].
    # Keep ordinary finite roots unchanged for faithful probability-table replay.
    crossings = torch.nan_to_num(crossings, posinf=2.0, neginf=-1.0)
    for index in range(slope.shape[1]):
        direction, crossing = slope[:, index:index+1], crossings[:, index:index+1]
        lo = torch.where(direction > 0, torch.maximum(lo, crossing), lo)
        hi = torch.where(direction < 0, torch.minimum(hi, crossing), hi)
    feasible = (~constant_bad) & (lo <= hi)
    return {"lower": lo, "upper": hi, "feasible": feasible}


def _verifier_parameters(config):
    if config.get("verifier_mode", "residual") != "residual":
        raise ValueError("This audit requires residual verification.")
    values = {
        "ti_threshold": float(config.get("ti_threshold", .5)),
        "metallic_threshold": float(config.get("metallic_threshold", .5)),
        "ti_strength": float(config.get("ti_strength", 1.0)),
        "metallic_strength": float(config.get("metallic_strength", 1.0)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Verifier parameters must be finite.")
    if any(not 0 < values[name] <= 1 for name in ("ti_threshold", "metallic_threshold")):
        raise ValueError("Verifier thresholds must lie in (0, 1].")
    if min(values["ti_strength"], values["metallic_strength"]) < 0:
        raise ValueError("Verifier strengths must be non-negative.")
    return values


def build_component_branches(cache, candidate_gate, verifier_config):
    direct, mapped = cache["direct"], cache["mapped"]
    _validate_probability_matrix(direct, "direct")
    _validate_probability_matrix(mapped, "mapped")
    if direct.shape != mapped.shape or direct.shape[1] != 4:
        raise ValueError("Component posteriors must have matching [batch, 4] shapes.")
    if direct.device != mapped.device or direct.dtype != mapped.dtype:
        raise ValueError("Component posteriors must share dtype and device.")
    for name, value in (("candidate_gate", candidate_gate), ("original_gate", cache["original_gate"]), ("ti", cache["ti"]), ("metal", cache["metal"])):
        _validate_unit_column(value, name, direct)
    parameters = _verifier_parameters(verifier_config)
    direct = direct.double() / direct.double().sum(1, keepdim=True)
    mapped = mapped.double() / mapped.double().sum(1, keepdim=True)
    gate, original = candidate_gate.double(), cache["original_gate"].double()
    ti, metal = cache["ti"].double(), cache["metal"].double()
    scale = torch.exp(-parameters["ti_strength"] * torch.relu((parameters["ti_threshold"] - ti) / parameters["ti_threshold"])
                      -parameters["metallic_strength"] * torch.relu((parameters["metallic_threshold"] - metal) / parameters["metallic_threshold"]))
    if not bool((scale > 0).all()):
        raise ValueError("Positive verifier scale underflowed; geometry assumptions do not hold.")
    branches = {"direct_pre": direct, "mapped_pre": mapped, "fixed_pre": .5*(direct+mapped),
                "oos_pre": gate*direct+(1-gate)*mapped, "original_pre": original*direct+(1-original)*mapped}
    for name in ("direct", "mapped", "fixed", "oos", "original"):
        branches[f"{name}_verified"] = apply_residual_target_verifiers(branches[f"{name}_pre"], ti, metal, **parameters)
    zd = 1 - (1-scale)*direct[:, :1]
    zm = 1 - (1-scale)*mapped[:, :1]
    branches.update({"target_scale": scale,
                     "effective_candidate_gate": gate*zd/(gate*zd+(1-gate)*zm),
                     "effective_fixed_gate": zd/(zd+zm)})
    return branches


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _geometry(cache, branches, minimum_margin=1e-9):
    intervals = {
        "full_pre": target_segment_interval(branches["direct_pre"], branches["mapped_pre"], minimum_margin=minimum_margin),
        "full_verified": target_segment_interval(branches["direct_verified"], branches["mapped_verified"], minimum_margin=minimum_margin),
        "restricted_pre": target_segment_interval(branches["oos_pre"], branches["fixed_pre"], minimum_margin=minimum_margin),
        "restricted_verified": target_segment_interval(branches["oos_verified"], branches["fixed_verified"], minimum_margin=minimum_margin),
    }
    labels = cache["labels"]
    fixed_target = branches["fixed_verified"].argmax(1) == 0
    capacity = {}
    for name, interval in intervals.items():
        mask = interval["feasible"].flatten()
        state = "verified" if name.endswith("verified") else "pre"
        endpoint_target = (branches[f"direct_{state}"].argmax(1) == 0) | (branches[f"mapped_{state}"].argmax(1) == 0)
        target_count = int((labels == 0).sum())
        capacity[name] = {"feasible_count": int(mask.sum()),
                          "true_target_feasible": int((mask & (labels == 0)).sum()),
                          "true_target_feasible_rate": int((mask & (labels == 0)).sum()) / target_count if target_count else None,
                          "missed_fixed_target_feasible": int((mask & (labels == 0) & ~fixed_target).sum()),
                          "non_target_feasible": int((mask & (labels != 0)).sum()),
                          "target_feasible_with_both_expert_endpoints_non_target": int((mask & (labels == 0) & ~endpoint_target).sum())}
    full = intervals["full_verified"]["feasible"].flatten()
    restricted = intervals["restricted_verified"]["feasible"].flatten()
    return intervals, capacity, int((restricted & ~full).sum())


def analyze_component_cache(cache, candidate_gate, verifier_config):
    branches = build_component_branches(cache, candidate_gate, verifier_config)
    labels = cache["labels"]
    if labels.shape != (branches["fixed_pre"].shape[0],) or labels.dtype != torch.long or labels.device != branches["fixed_pre"].device or bool(((labels < 0) | (labels > 3)).any()):
        raise ValueError("labels must be aligned integer role IDs in [0, 3].")
    metrics = {name: calculate_probability_metrics(branches[name], labels) for name in BRANCH_NAMES}
    effects = {}
    for name in ("direct", "mapped", "fixed", "oos", "original"):
        pre = branches[f"{name}_pre"].argmax(1) == 0
        post = branches[f"{name}_verified"].argmax(1) == 0
        effects[name] = {"true_target_removed": int((pre & ~post & (labels == 0)).sum()),
                         "false_target_removed": int((pre & ~post & (labels != 0)).sum()),
                         "target_promotions": int((~pre & post).sum())}
    intervals, capacity, violations = _geometry(cache, branches)
    effective = branches["effective_candidate_gate"]
    replay = effective*branches["direct_verified"] + (1-effective)*branches["mapped_verified"]
    loss_by_verification = intervals["full_pre"]["feasible"].flatten() & ~intervals["full_verified"]["feasible"].flatten()
    pre_disagreement = branches["direct_pre"].argmax(1) != branches["mapped_pre"].argmax(1)
    post_disagreement = branches["direct_verified"].argmax(1) != branches["mapped_verified"].argmax(1)
    classes = {str(index): int((labels == index).sum()) for index in range(4)}
    result = {"sample_count": len(labels), "class_counts": classes, "metrics": metrics,
              "verification_effects": effects, "segment_capacity": capacity,
              "nested_segment_violation_count": violations,
              "verifier_commutation_max_abs_residual": float((replay-branches["oos_verified"]).abs().max()),
              "full_segment_target_capacity_removed_by_verification": {
                  "true_target": int((loss_by_verification & (labels == 0)).sum()),
                  "non_target": int((loss_by_verification & (labels != 0)).sum())},
              "expert_argmax_disagreement": {"pre": int(pre_disagreement.sum()), "verified": int(post_disagreement.sum()),
                                              "true_target_pre": int((pre_disagreement & (labels == 0)).sum()),
                                              "true_target_verified": int((post_disagreement & (labels == 0)).sum())},
              "target_scale": {"min": float(branches["target_scale"].min()), "mean": float(branches["target_scale"].mean()), "max": float(branches["target_scale"].max())},
              "undefined_metric_policy": "Undefined class rates are null; macro F1 uses the fixed four-role convention with zero for undefined per-class F1.",
              "capacity_interpretation": "Label-aware attainable bounds, not a learned classifier or deployable performance.",
              "numeric_policy": "Validate source simplex then renormalise in float64; target feasibility requires margin >= 1e-9."}
    return _json_finite(result)


def validate_sources(protocol_dir: Path, source_dir: Path):
    lock_path = source_dir / "tc_projection/selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("protocol_version") != "tc_oos_rsg_v1" or lock.get("status") != "locked_before_outer_tensor_loading":
        raise ValueError("Source is not a registered TC-OOS-RSG selection lock.")
    manifest_hashes, rows = {}, {}
    for name in INNER_NAMES:
        path = protocol_dir / f"{name}.csv"
        manifest_hashes[name] = _file_sha256(path)
        if manifest_hashes[name] != lock["manifest_sha256"][name]:
            raise ValueError(f"Inner manifest hash changed: {name}")
        rows[name] = _read_manifest_rows(path)
        if not rows[name] or any(not row.get("image_id") or not row.get("split_group_id") for row in rows[name]):
            raise ValueError(f"Empty or unidentified inner manifest: {name}")
    model_paths = {"expert": source_dir / "expert/best_model.pt", "expert_lock": source_dir / "expert/selection_lock.json",
                   "gate": source_dir / "tc_projection/gate/best_model.pt", "configuration": source_dir / "tc_projection/run_config.json"}
    hashes = {name: _file_sha256(path) for name, path in model_paths.items()}
    for name in hashes:
        if hashes[name] != lock["model_file_sha256"][name]:
            raise ValueError(f"Frozen source hash changed: {name}")
    expert_lock = json.loads(model_paths["expert_lock"].read_text(encoding="utf-8"))
    training_path = source_dir / "manifests/expert_training.csv"
    training_hash = _file_sha256(training_path)
    if training_hash != expert_lock["manifest_sha256"]["training_manifest"]:
        raise ValueError("Expert training manifest hash changed.")
    configuration_path = source_dir / "expert/run_config.json"
    if _file_sha256(configuration_path) != expert_lock["configuration_sha256"] or hashes["expert"] != expert_lock["checkpoint_sha256"]:
        raise ValueError("Frozen expert configuration or checkpoint changed.")
    identity = lambda row: (row["image_id"], row["split_group_id"], row.get("mindat_photo_id", ""))
    training_rows = _read_manifest_rows(training_path)
    expected_training = rows["expert_fit"] + rows["expert_stop"]
    if len(training_rows) != len(expected_training) or {identity(row) for row in training_rows} != {identity(row) for row in expected_training}:
        raise ValueError("Expert training identities do not equal its registered fit/stop union.")
    overlaps = {"cross_subset_group_overlap_count": 0, "cross_subset_image_overlap_count": 0, "cross_subset_photo_overlap_count": 0}
    for index, name in enumerate(INNER_NAMES):
        ids = [row["image_id"] for row in rows[name]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate image identities in {name}")
        for other in INNER_NAMES[index+1:]:
            for field, result_key in (("split_group_id", "cross_subset_group_overlap_count"), ("image_id", "cross_subset_image_overlap_count"), ("mindat_photo_id", "cross_subset_photo_overlap_count")):
                left = {row.get(field, "") for row in rows[name]} - {""}
                right = {row.get(field, "") for row in rows[other]} - {""}
                overlaps[result_key] += len(left & right)
    if any(overlaps.values()):
        raise ValueError(f"Inner subset identity overlap: {overlaps}")
    environment_path = source_dir / "expert/environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    for field in ("backbone", "verifier_mode"):
        if field in configuration and environment.get(field) != configuration[field]:
            raise ValueError(f"Source environment/configuration mismatch: {field}")
    verifier_config = {"verifier_mode": environment.get("verifier_mode", "residual"),
                       "ti_threshold": environment.get("ti_verifier_threshold", .5),
                       "metallic_threshold": environment.get("metallic_verifier_threshold", .5),
                       "ti_strength": environment.get("ti_verifier_strength", 1.),
                       "metallic_strength": environment.get("metallic_verifier_strength", 1.)}
    _verifier_parameters(verifier_config)
    return {"manifest_sha256": manifest_hashes, "source_sha256": hashes,
            "source_lock_sha256": _file_sha256(lock_path), "expert_training_manifest_sha256": training_hash,
            "expert_environment_sha256": _file_sha256(environment_path),
            "expert_environment_hash_binding": "Recorded at audit time; environment was not hashed by the original selection lock.",
            "verifier_config": verifier_config, "group_audit": overlaps,
            "outer_manifest_read": False, "outer_images_loaded": False,
            "inner_row_counts": {name: len(value) for name, value in rows.items()}}


def _write_subset_outputs(folder, records, cache, candidate_gate, config):
    folder.mkdir(parents=True, exist_ok=False)
    branches = build_component_branches(cache, candidate_gate, config)
    intervals, _, _ = _geometry(cache, branches)
    with (folder / "component_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split_group_id", "mineral_label", "true_class_id", "branch", "predicted_class_id", "prob_0", "prob_1", "prob_2", "prob_3"])
        for name in BRANCH_NAMES:
            for record, probabilities in zip(records, branches[name], strict=True):
                writer.writerow([record.image_id, record.split_group_id, record.mineral_label, record.class_id, name, int(probabilities.argmax()), *probabilities.tolist()])
    with (folder / "routing_evidence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split_group_id", "true_class_id", "candidate_gate", "original_gate", "ti_probability", "metal_probability", "target_scale", "effective_candidate_gate", "effective_fixed_gate", *[f"{name}_{key}" for name in intervals for key in ("feasible", "lower", "upper")]])
        for index, record in enumerate(records):
            values = [float(value[index]) for value in (candidate_gate, cache["original_gate"], cache["ti"], cache["metal"], branches["target_scale"], branches["effective_candidate_gate"], branches["effective_fixed_gate"])]
            geometry = [bool(interval[key][index]) if key == "feasible" else float(interval[key][index]) for interval in intervals.values() for key in ("feasible", "lower", "upper")]
            writer.writerow([record.image_id, record.split_group_id, record.class_id, *values, *geometry])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=ROOT / "outputs/training/tc_oos_rsg_manifests_v1")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/training/tc_oos_rsg_v1")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/theory/abmp_candidate_capacity_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {args.output_dir}")
    protocol, source = args.protocol_dir / "fold_0", args.source_dir / "fold_0"
    provenance = validate_sources(protocol, source)
    from run_seen_unseen_gate_study import extract_cache
    from train_mineral_classifier import create_transforms, load_manifest_records, require_training_dependencies
    dependencies = require_training_dependencies()
    device = torch.device(args.device)
    model, mapping, _ = _load_frozen_expert(source / "expert", source / "manifests/expert_training.csv", args.dataset_root, device, dependencies)
    gate = copy.deepcopy(model.gate_network)
    gate.load_state_dict(torch.load(source / "tc_projection/gate/best_model.pt", map_location=device, weights_only=True))
    gate.eval().requires_grad_(False)
    _, transform = create_transforms(224, dependencies["transforms"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "run_status.json", {"status": "running", "outer_images_loaded": False})
    results = {}
    for name in EXTRACTION_NAMES:
        records = load_manifest_records(protocol / f"{name}.csv", args.dataset_root)
        extracted = extract_cache(model, records, mapping, transform, dependencies, device, f"capacity_{name}")
        with torch.no_grad():
            candidate = torch.sigmoid(gate(extracted["features"])).detach().cpu()
        cache = {key: value.detach().cpu() for key, value in extracted.items() if key != "features"}
        results[name] = analyze_component_cache(cache, candidate, provenance["verifier_config"])
        _write_subset_outputs(args.output_dir / name, records, cache, candidate, provenance["verifier_config"])
        del extracted
        print(json.dumps({"subset": name, "capacity": results[name]["segment_capacity"]}, allow_nan=False), flush=True)
    after = validate_sources(protocol, source)
    if after != provenance:
        raise ValueError("Frozen source files changed during the audit.")
    summary = {"protocol": "abmp_candidate_capacity_v1", "status": "diagnostic_complete_no_promotion",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
               "subsets": results, "outer_images_loaded": False, "fold_1_2_accessed": False,
               "weights_modified": False, "selection_lock_created": False}
    _write_json(args.output_dir / "audit_summary.json", summary)
    _write_json(args.output_dir / "run_status.json", {"status": "complete", "outer_images_loaded": False})
    print("Candidate capacity audit completed without training or outer-fold access.", flush=True)


if __name__ == "__main__":
    main()
