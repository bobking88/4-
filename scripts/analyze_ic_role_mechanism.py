"""Pooled fit-only descriptive analysis and saved-state forward replay."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from analyze_lc_role_adaptation import summarize_predictions
from ic_role_audit import audit_bounds
from ic_role_data import file_digest, load_fit_bundle
from ic_role_model import ICRoleModel, UNITS, _digest, objective, reference_identity
from ic_role_training import CHECKPOINTS, SEEDS, _branch_diagnostics, build_fit_matrix
from lc_execution_safety import _CURRENT_GUARD
from lc_role_data import transform_subset


def _same_records(actual, expected):
    if len(actual) != len(expected): raise ValueError("Missing or duplicate OOF identities.")
    for a, e in zip(actual, expected, strict=True):
        if (a["image_id"] != e["image_id"] or a["split_group_id"] != e["split_group_id"]
                or int(a.get("four_class_id", a.get("class_id", -1))) != int(e["class_id"])):
            raise ValueError("OOF identity, order or label mismatch.")


def collect_oof(artifacts: list[dict], assignment: list[dict], *, checkpoint: int = 100) -> dict:
    if type(checkpoint) is not int or checkpoint not in CHECKPOINTS:
        raise ValueError("Only preregistered descriptive checkpoints; primary stays 100.")
    if (not assignment or len({r["image_id"] for r in assignment}) != len(assignment)
            or len({r["split_group_id"] for r in assignment}) != len(assignment)
            or {r["inner_fold"] for r in assignment} != {0, 1, 2}):
        raise ValueError("Invalid unique ordered assignment.")
    expected = {(r["unit"], r["fold"], r["seed"]) for r in build_fit_matrix()}
    keys = [(r["unit"], r["inner_fold"], r["seed"]) for r in artifacts]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Expected each of 108 unit/fold/seed artifacts exactly once.")
    lookup = dict(zip(keys, artifacts, strict=True))
    groups = {}
    n = len(assignment)
    labels = torch.tensor([r["class_id"] for r in assignment], dtype=torch.long)
    records = [dict(image_id=r["image_id"], split_group_id=r["split_group_id"],
                    four_class_id=str(r["class_id"])) for r in assignment]
    common_anchor = None
    for unit in UNITS:
        for seed in SEEDS:
            logq, anchor = torch.empty(n, 4, dtype=torch.float64), torch.empty(n, 4, dtype=torch.float64)
            for fold in range(3):
                artifact = lookup[unit, fold, seed]
                indices = [i for i, r in enumerate(assignment) if r["inner_fold"] == fold]
                rows = [assignment[i] for i in indices]
                _same_records(artifact["validation_records"], rows)
                if not torch.equal(artifact["validation_labels"], labels[indices]):
                    raise ValueError("OOF validation labels mismatch.")
                point = artifact["checkpoints"].get(checkpoint)
                if point is None or point["completed_updates"] != checkpoint:
                    raise ValueError("Exact requested completed-update checkpoint missing.")
                if point["validation_logq"].shape != (len(indices), 4) or artifact["validation_anchor"].shape != (len(indices), 4):
                    raise ValueError("OOF prediction shape mismatch.")
                logq[indices], anchor[indices] = point["validation_logq"], artifact["validation_anchor"]
            if common_anchor is None: common_anchor = anchor.clone()
            elif not torch.equal(common_anchor, anchor): raise ValueError("Shared fold T0 changed across units or seeds.")
            packet = dict(logq=logq, labels=labels.clone(), records=records, anchor_logq=anchor,
                          unit=unit, seed=seed, checkpoint=checkpoint)
            packet["metrics"] = summarize_predictions(packet, logq, anchor)
            groups[f"{unit}|{seed}"] = packet
    return dict(groups=groups, primary_checkpoint=100, requested_checkpoint=checkpoint,
                anchor_metrics=summarize_predictions(dict(labels=labels, records=records), common_anchor, common_anchor),
                scope="pooled fit-only OOF; no best epoch, no independent confirmation")


def compare_units(left: dict, right: dict) -> dict:
    if (left["records"] != right["records"] or not torch.equal(left["labels"], right["labels"])):
        raise ValueError("Paired units must have identical ordered rows and labels.")
    lm = summarize_predictions(left, left["logq"], right["logq"])
    rm = summarize_predictions(right, right["logq"], right["logq"])
    lp, rp, labels = left["logq"].argmax(1), right["logq"].argmax(1), left["labels"]
    rows = [dict(image_id=r["image_id"], true_class=int(labels[i]), left_prediction=int(lp[i]),
                 right_prediction=int(rp[i]), changed=bool(lp[i] != rp[i]),
                 left_correct=bool(lp[i] == labels[i]), right_correct=bool(rp[i] == labels[i]))
            for i, r in enumerate(left["records"])]
    return dict(left_metrics=lm, right_metrics=rm, nll_difference=lm["nll"]-rm["nll"],
                **{k: lm[k] for k in ("new_correct_target_count", "lost_correct_target_count", "net_correct_target_count")},
                row_changes=rows, comparison_scope="paired rows, descriptive only; seeds/folds not independent replicates")


def _scoped(output, relative):
    path = (output/relative).resolve()
    if path == output or not path.is_relative_to(output): raise ValueError("Artifact path escape.")
    return path


def _layout(named, prefix=""):
    rows, offset = [], 0
    for name, parameter in named:
        rows.append(dict(name=prefix+name, shape=list(parameter.shape), offset=offset, numel=parameter.numel()))
        offset += parameter.numel()
    return rows, offset


def _finite_tensor(value, shape):
    return (torch.is_tensor(value) and tuple(value.shape) == tuple(shape) and value.dtype == torch.float64
            and value.device.type == "cpu" and not value.requires_grad and bool(torch.isfinite(value).all()))


def _same_diagnostics(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("Missing branch diagnostic schema.")
    for key, value in expected.items():
        if torch.is_tensor(value):
            if not _finite_tensor(actual[key], value.shape) or not torch.equal(actual[key], value):
                raise ValueError("Branch ablation forward replay mismatch.")
        elif actual[key] != value: raise ValueError("Branch diagnostic metadata mismatch.")


def _checkpoint_jacobians(point, model, train):
    saved = point.get("train_jacobians")
    if not isinstance(saved, dict) or set(saved) != {"1", "2", "3"}:
        raise ValueError("Missing checkpoint scalar Jacobians.")
    ids = [r["image_id"] for r in train["records"]]
    for j in range(3):
        views = saved[str(j+1)]
        if set(views) != {"main", "aux"}: raise ValueError("Missing scalar views.")
        for view, block in views.items():
            if not model.adapters:
                if block.get("status") != "NOT_APPLICABLE" or block.get("jacobian") is not None:
                    raise ValueError("Head-only checkpoint Jacobian must be inapplicable.")
                continue
            layout, width = _layout(model.adapters[j].named_parameters(), f"adapters.{j}.")
            if (block.get("status") != "COMPLETE" or block.get("layout") != layout
                    or block.get("row_ids") != ids or block.get("role") != j+1 or block.get("view") != view
                    or not _finite_tensor(block.get("jacobian"), (len(ids), width))):
                raise ValueError("Checkpoint scalar Jacobian schema, finiteness or identity mismatch.")


def _verify_trajectory(trajectory, train, validation):
    if (trajectory["train_records"] != train["records"]
            or trajectory["validation_records"] != validation["records"]):
        raise ValueError("Saved trajectory input identities changed.")
    expected_points = [s for s in CHECKPOINTS if s <= trajectory["optimizer_updates"]]
    if sorted(trajectory["checkpoints"]) != expected_points:
        raise ValueError("Missing trajectory checkpoint.")
    expected_gradients = list(range(min(5, trajectory["optimizer_updates"])))
    if [r["pre_update_index"] for r in trajectory["gradient_vectors"]] != expected_gradients:
        raise ValueError("Missing first-five full gradient provenance.")
    origin_model = ICRoleModel(trajectory["unit"], seed=trajectory["seed"])
    layout, width = _layout(origin_model.named_current())
    for gradients in trajectory["gradient_vectors"]:
        if (gradients.get("layout") != layout or any(not _finite_tensor(gradients.get(k), (width,)) for k in ("data", "regularization", "total"))
                or not torch.allclose(gradients["total"], gradients["data"]+gradients["regularization"], atol=1e-10, rtol=1e-10)):
            raise ValueError("Gradient provenance changed.")
    initial = ICRoleModel(trajectory["unit"], seed=trajectory["seed"])
    initial.load_state_dict(trajectory["checkpoints"][0]["state_dict"])
    with torch.no_grad():
        initial_g = initial(train["h"], train["e"], train["log_anchor"])["g"]
    kernels = trajectory["initial_kernels"]
    if len(kernels) != 3: raise ValueError("Missing initial scalar kernel blocks.")
    kernel_summary = []
    for j, kernel in enumerate(kernels):
        if not initial.adapters:
            if kernel["status"] != "NOT_APPLICABLE": raise ValueError("Head-only kernel must be inapplicable.")
            kernel_summary.append(dict(role=j+1, status="NOT_APPLICABLE"))
            continue
        ja, jm, ka, km, G = (kernel[k] for k in ("Jaux", "Jmain", "Kaux", "Kmain", "G"))
        equations = ((ka, ja @ ja.T), (km, jm @ jm.T), (km, G @ ka @ G),
                     (G, torch.diag(initial_g[:, j])), (jm, initial_g[:, j, None]*ja))
        if (kernel["row_ids"] != [r["image_id"] for r in train["records"]]
                or any(not torch.allclose(a, b, atol=1e-10, rtol=1e-10) for a, b in equations)):
            raise ValueError("Saved scalar kernel identity mismatch.")
        eigen_aux, eigen_main = torch.linalg.eigvalsh(ka), torch.linalg.eigvalsh(km)
        if (not torch.equal(eigen_aux, kernel["aux_eigenvalues"])
                or not torch.equal(eigen_main, kernel["main_eigenvalues"])
                or kernel["numeric_rank_threshold"] != 1e-10):
            raise ValueError("Saved kernel spectrum changed.")
        kernel_summary.append(dict(role=j+1, aux_numeric_rank=int((eigen_aux > 1e-10).sum()),
                                   main_numeric_rank=int((eigen_main > 1e-10).sum()),
                                   main_trace=float(km.trace()), aux_trace=float(ka.trace()),
                                   max_gram_residual=float((km-G @ ka @ G).abs().max()),
                                   scope="initial scalar branch block only"))
    maxima = dict(main_log_drift=0., auxiliary_log_drift=0.)
    for step, point in trajectory["checkpoints"].items():
        model = ICRoleModel(trajectory["unit"], seed=trajectory["seed"])
        model.load_state_dict(point["state_dict"], strict=True)
        identity = reference_identity(model)
        if (identity != point["identity"] or not identity["references_match_initial"]
                or not identity["references_frozen"] or not identity["references_unaliased"]
                or _digest(model.named_current()) != point["current_sha256"] or point["completed_updates"] != step):
            raise ValueError("Saved state or reference identity mismatch.")
        _checkpoint_jacobians(point, model, train)
        if set(point.get("gradient_norms", {})) != {"data", "regularization", "total"} or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in point["gradient_norms"].values()):
            raise ValueError("Invalid checkpoint gradient norms.")
        with torch.no_grad():
            for name, batch in (("train", train), ("validation", validation)):
                result = model(batch["h"], batch["e"], batch["log_anchor"], auxiliary=True)
                if not torch.equal(result["logq"], point[name+"_logq"]):
                    raise ValueError("Forward replay mismatch.")
                if step == 0 and not torch.allclose(result["logq"], batch["log_anchor"], atol=1e-12, rtol=0):
                    raise ValueError("Initial anchor identity mismatch.")
                bounds = audit_bounds(model, batch, result)
                if bounds["violation_count"] or bounds != point["bounds"][name]:
                    raise ValueError("Saved or replayed stability certificate mismatch.")
                _same_diagnostics(point.get("branch_diagnostics", {}).get(name), _branch_diagnostics(model, batch, result))
                losses = {k: float(v) for k, v in objective(model, result, batch["labels"]).items()}
                if point.get(name+"_losses") != losses:
                    raise ValueError("Saved objective forward replay mismatch.")
                for row in bounds["rows"]:
                    maxima["main_log_drift"] = max(maxima["main_log_drift"], row["actual_main_logq_drift"])
                    maxima["auxiliary_log_drift"] = max(maxima["auxiliary_log_drift"], row["actual_aux_logq_drift"])
    return dict(checkpoint_count=len(expected_points), theory_violation_count=0,
                identity=trajectory["checkpoints"][0]["identity"], maximum_actual_drifts=maxima,
                kernels=kernel_summary, costs=trajectory["costs"],
                checkpoints=[dict(step=s, gradient_norms=p["gradient_norms"],
                    parameter_norm=p["parameter_norm"], origin_distance=p["origin_distance"],
                    L_F=p["bounds"]["train"]["L_F"],
                    protected_fraction=p["bounds"]["validation"]["protected_winner_fraction"],
                    removed_branch_max_log_drift=float(p["branch_diagnostics"]["validation"]["removed_branch_max_log_drift"].max()),
                    max_residual_by_check=p["bounds"]["train"]["max_residual_by_check"])
                    for s, p in trajectory["checkpoints"].items()])


def _execution_envelope(output, manifest, *, provisional):
    try:
        state = json.loads((output/"stage_state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Missing terminal supervision state.") from error
    expected_status = "RUNNING" if provisional else "MECHANISM_COMPLETE"
    if state.get("status") != expected_status or state.get("stage") != manifest["stage"] or state.get("alive") is not False:
        raise ValueError("Invalid terminal/supervision status; no delivery acceptance.")
    elapsed = state.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 <= elapsed < 7200:
        raise ValueError("Cumulative budget exhausted or unavailable.")
    if not provisional:
        power = state.get("power", {})
        if not power.get("power_request_acquired") or not power.get("power_request_released"):
            raise ValueError("Terminal power request/release evidence missing.")
    jobs = state.get("jobs", [])
    if manifest["stage"] == "fit-mechanism":
        if state.get("trajectory_count") != 108 or state.get("temperature_count") != 3 or len(jobs) != (111 if provisional else 112):
            raise ValueError("Incomplete terminal fit/temperature/supervision count.")
    for job in jobs:
        power = job.get("supervision", {})
        if (job.get("status") != "COMPLETE" or job.get("alive") is not False
                or not power.get("power_request_acquired") or not power.get("power_request_released")):
            raise ValueError("Failed worker supervision or power release.")
    return state


def verify_mechanism_delivery(root: Path, output: Path) -> dict:
    return _verify_mechanism(root, output, provisional=False)


def _verify_active_mechanism(root: Path, output: Path) -> dict:
    guard = _CURRENT_GUARD.get()
    if guard is None:
        raise ValueError("Provisional analysis requires live supervision.")
    guard.check()
    output = Path(output).resolve()
    claim = output/"runs"/"analysis"/"task_started.json"
    if not claim.exists(): raise ValueError("Provisional analysis supervision claim missing.")
    return _verify_mechanism(root, output, provisional=True)


def _verify_mechanism(root: Path, output: Path, *, provisional: bool) -> dict:
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root): raise ValueError("Delivery scope escape.")
    manifest = json.loads((output/"artifact_manifest.json").read_text(encoding="utf-8"))
    _execution_envelope(output, manifest, provisional=provisional)
    stage = manifest["stage"]
    if stage not in ("synthetic", "fit-mechanism"): raise ValueError("Unrecognized delivery type.")
    for relative, digest in manifest["artifacts"].items():
        if file_digest(_scoped(output, relative)) != digest: raise ValueError("Artifact hash mismatch.")
    artifacts, diagnostics = [], []
    pooled = None
    if stage == "synthetic":
        if manifest["scope"] != "synthetic_saved_processed_inputs": raise ValueError("Synthetic input scope mismatch.")
        paths = manifest["trajectories"]
        for relative in paths:
            if relative not in manifest["artifacts"]: raise ValueError("Unpinned synthetic artifact.")
            packet = torch.load(_scoped(output, relative), map_location="cpu", weights_only=True)
            diagnostics.append(_verify_trajectory(packet["trajectory"], packet["train"], packet["validation"]))
    else:
        from run_ic_role_mechanism import validate_protocol
        protocol = manifest["protocol"]
        validate_protocol(root, protocol, stage)
        if manifest["trajectory_count"] != 108 or manifest["temperature_count"] != 3:
            raise ValueError("Incomplete real mechanism count.")
        raw, assignment = load_fit_bundle(root)
        expected = build_fit_matrix()
        jobs = [j for j in manifest["jobs"] if j["kind"] == "trajectory"]
        if [(j["unit"], j["fold"], j["seed"]) for j in jobs] != [(j["unit"], j["fold"], j["seed"]) for j in expected]:
            raise ValueError("Real fixed job matrix mismatch.")
        for job in jobs:
            relative = f"runs/{job['job_id']}/result.pt"
            if relative not in manifest["artifacts"]: raise ValueError("Unpinned real trajectory.")
            trajectory = torch.load(_scoped(output, relative), map_location="cpu", weights_only=True)
            prepared_relative = trajectory["prepared_relative"]
            if prepared_relative not in manifest["artifacts"]: raise ValueError("Unpinned preprocessing state.")
            prepared = torch.load(_scoped(output, prepared_relative), map_location="cpu", weights_only=True)
            if trajectory["optimizer_updates"] != 400 or prepared["inner_fold"] != job["fold"]:
                raise ValueError("Real update or fold identity changed.")
            train_indices = torch.tensor([i for i, r in enumerate(assignment) if r["inner_fold"] != job["fold"]])
            val_indices = torch.tensor([i for i, r in enumerate(assignment) if r["inner_fold"] == job["fold"]])
            state = prepared["state"]
            if state["train_ids"] != [raw["records"][i]["image_id"] for i in train_indices]:
                raise ValueError("Saved preprocessing training identities changed.")
            train, val = transform_subset(raw, state, train_indices), transform_subset(raw, state, val_indices)
            diagnostics.append(_verify_trajectory(trajectory, train, val))
            artifacts.append(dict(trajectory, validation_labels=val["labels"], validation_anchor=val["log_anchor"]))
        pooled = collect_oof(artifacts, assignment)
    comparisons = []
    if pooled is not None:
        pairs = [("C-S-R", "C-S-U"), ("C-S-R-noaux", "C-S-U-noaux"),
                 ("C-S-R", "HC-S"), ("C-S-R", "FC-S"), ("C-S-U", "Z-S-U"),
                 ("C-S-R", "C-O-R"), ("Z-S-R", "Z-O-R")]
        for left, right in pairs:
            for seed in SEEDS:
                comparisons.append(dict(left=left, right=right, seed=seed,
                    **compare_units(pooled["groups"][f"{left}|{seed}"], pooled["groups"][f"{right}|{seed}"])))
    return dict(stage=stage, trajectory_count=len(diagnostics), checkpoint_count=sum(d["checkpoint_count"] for d in diagnostics),
                theory_violation_count=0, diagnostics=diagnostics, comparisons=comparisons,
                metrics=[] if pooled is None else [dict(unit=r["unit"], seed=r["seed"], **r["metrics"]) for r in pooled["groups"].values()],
                primary_checkpoint=100, independent_confirmation=False, delivery_verified=not provisional,
                scope="synthetic implementation only" if stage == "synthetic" else "exposed fit-only exploration; no industrial result")


def write_mechanism_report(evidence: dict, output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, allow_nan=False, indent=2)
    rows = [dict(unit=r["unit"], seed=r["seed"], accuracy=r["accuracy"], macro_f1=r["macro_f1"],
                 nll=r["nll"], target_recall=r["target_recall"], ti_false_target_count=r["ti_false_target_count"],
                 metallic_false_target_count=r["metallic_false_target_count"]) for r in evidence["metrics"]]
    with (output/"metrics.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["unit", "seed", "scope"])
        writer.writeheader()
        writer.writerows(rows)
    prose = ("# IC-RFA-B v1 实施与描述性机制证据\n\n"
             "仅fit探索；无独立确认。固定100步为主描述点，固定400预算不是收敛证明。\n\n"
             "中心化、L2-SP、瓶颈适配与Gram同余已有前例，不宣称这些数学工具首次提出。"
             "主要增量对照必须是C-S-R/C-S-U；合成通过不是矿物召回改善。\n\n"
             f"类型：{evidence['stage']}；轨迹：{evidence['trajectory_count']}；"
             f"重放检查点：{evidence['checkpoint_count']}；界违规：{evidence['theory_violation_count']}。\n\n"
             "三种子和重复内折不作为独立样本显著性检验；不评价stop、不预测品位、不更改正式报告。\n")
    (output/"mechanism_report.md").write_text(prose, encoding="utf-8")
    return dict(markdown=str(output/"mechanism_report.md"), json=str(output/"evidence.json"),
                csv=str(output/"metrics.csv"), scope=evidence["scope"])
