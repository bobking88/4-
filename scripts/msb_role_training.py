"""Fixed MSB trajectories; no selection, stop evaluation or preprocessing."""
from __future__ import annotations

import copy
import hashlib
import json
import torch
from threadpoolctl import threadpool_limits
from msb_execution_safety import msb_stage_elapsed_seconds as stage_elapsed_seconds
from msb_role_data import validate_batch
from msb_role_model import CONVEX_UNITS, NEURAL_UNITS, MSBRoleModel, reference_digest
from msb_role_controls import GAP_TOLERANCE, control_forward, make_convex_control, objective, solve_convex
from msb_role_audit import audit_forward_contract, gradient_components

SEEDS = (20261002, 20261003, 20261004)
CHECKPOINTS = (0, 1, 2, 5, 20, 50, 100, 200, 400)
PRIMARY = 100


def build_fit_matrix() -> list[dict]:
    jobs = []
    for unit in NEURAL_UNITS:
        for fold in range(3):
            for seed in SEEDS:
                jobs.append(dict(kind="trajectory", unit=unit, fold=fold, seed=seed,
                                 job_id=f"{unit}_fold{fold}_seed{seed}", max_updates=400,
                                 checkpoints=list(CHECKPOINTS), primary_checkpoint=PRIMARY))
    for unit in CONVEX_UNITS:
        for fold in range(3):
            jobs.append(dict(kind="convex", unit=unit, fold=fold, seed=None,
                             job_id=f"{unit}_fold{fold}_convex"))
    return jobs


def _splits(train, validation):
    validate_batch(train); validate_batch(validation)
    for field in ("image_id", "split_group_id"):
        if {r[field] for r in train["records"]} & {r[field] for r in validation["records"]}:
            raise ValueError("MSB training/validation group overlap.")


def _input_digest(batch):
    digest = hashlib.sha256()
    for key in ("h", "e", "log_anchor", "labels"):
        digest.update(key.encode("ascii"))
        digest.update(batch[key].detach().contiguous().numpy().tobytes())
    rows = [{key: r.get(key) for key in ("image_id", "split_group_id", "four_class_id", "tc_subset")}
            for r in batch["records"]]
    digest.update(json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _detach(value):
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, dict): return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, list): return [_detach(item) for item in value]
    if isinstance(value, tuple): return tuple(_detach(item) for item in value)
    return copy.deepcopy(value)


def _forward(model, batch, **kwargs):
    return model(batch["h"], batch["e"], batch["log_anchor"], **kwargs)


def _checkpoint(model, train, validation, completed):
    stage_elapsed_seconds()
    result = _forward(model, train)
    gradients = gradient_components(model, result, train["labels"])
    audit_forward_contract(train, result)
    with torch.no_grad():
        val_result = _forward(model, validation)
        audit_forward_contract(validation, val_result)
        off = []
        if model.unit == "MAIN" and completed == PRIMARY:
            for mask in ((False,True,True),(True,False,True),(True,True,False),(False,False,False)):
                stage_elapsed_seconds()
                forward = _forward(model, validation, role_mask=mask)
                audit_forward_contract(validation, forward)
                off.append(dict(mask=mask, forward=_detach(forward)))
    return dict(completed_updates=completed, model_state=_detach(model.state_dict()),
                reference_digest=reference_digest(model), gradients=_detach(gradients),
                train=_detach(result), validation=_detach(val_result), role_off=off)


def fit_trajectory(train: dict, validation: dict, unit: str, seed: int, *,
                   max_updates: int=400, checkpoints: tuple=CHECKPOINTS) -> dict:
    _splits(train, validation)
    if (unit not in NEURAL_UNITS or seed not in SEEDS or type(max_updates) is not int
            or not 1 <= max_updates <= 400 or not isinstance(checkpoints, tuple) or not checkpoints
            or any(type(c) is not int for c in checkpoints)
            or tuple(sorted(set(checkpoints))) != checkpoints or checkpoints[0] != 0
            or checkpoints[-1] != max_updates):
        raise ValueError("Invalid fixed MSB trajectory.")
    synthetic = all(r["image_id"].startswith("synthetic-")
                    for b in (train,validation) for r in b["records"])
    if (max_updates != 400 or checkpoints != CHECKPOINTS) and not synthetic:
        raise ValueError("Short or altered schedules are synthetic-only.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial_inputs = (_input_digest(train), _input_digest(validation))
    model = MSBRoleModel(unit, seed=seed)
    ref = reference_digest(model)
    optimizer = torch.optim.Adam([p for _, p in model.named_current()], lr=.001,
                                 betas=(.9,.999), eps=1e-8, weight_decay=0.)
    points = []
    with threadpool_limits(limits=1):
        for completed in range(max_updates + 1):
            stage_elapsed_seconds()
            if completed in checkpoints:
                points.append(_checkpoint(model, train, validation, completed))
            if completed == max_updates: break
            optimizer.zero_grad(set_to_none=True)
            result = _forward(model, train)
            loss = objective(model, result, train["labels"])["total"]
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in model.named_current()):
                raise ValueError("NUMERIC_RANGE_FAILURE: training gradient missing or nonfinite")
            optimizer.step()
            if reference_digest(model) != ref:
                raise ValueError("Fixed reference changed during optimization.")
    if initial_inputs != (_input_digest(train), _input_digest(validation)):
        raise ValueError("Frozen prepared inputs changed during trajectory.")
    return dict(unit=unit, seed=seed, complete=True, completed_updates=max_updates,
                primary_checkpoint=PRIMARY, primary_present=PRIMARY in checkpoints,
                checkpoints=points, initial_reference_digest=ref,
                final_reference_digest=reference_digest(model), conditional_on_frozen_expert=True,
                scope="SYNTHETIC" if synthetic else "EXPOSED_FIT_MECHANISM",
                train_inputs_digest=initial_inputs[0], validation_inputs_digest=initial_inputs[1],
                train_records=copy.deepcopy(train["records"]), validation_records=copy.deepcopy(validation["records"]),
                train_labels=train["labels"].clone(), validation_labels=validation["labels"].clone())


def fit_convex_fold(train, validation, unit, *, deadline_check):
    _splits(train, validation)
    if unit not in CONVEX_UNITS:
        raise ValueError("Expected a deterministic convex control.")
    initial_inputs = (_input_digest(train), _input_digest(validation))
    control = make_convex_control(unit, train)
    with threadpool_limits(limits=1):
        solution = solve_convex(control, train, deadline_check=deadline_check)
        if (solution.get("converged") is not True or not solution["gap_upper"] <= GAP_TOLERANCE
                or solution.get("mu") != .001):
            raise ValueError("Unsolved convex control cannot complete.")
        predictions = {}
        for part, batch in (("train",train),("validation",validation)):
            deadline_check()
            with torch.no_grad():
                result = control_forward(solution["model_state"], batch["h"],batch["e"],batch["log_anchor"])
                audit_forward_contract(batch,result)
            predictions[part] = _detach(result)
    if initial_inputs != (_input_digest(train),_input_digest(validation)):
        raise ValueError("Prepared inputs changed during convex fitting.")
    return dict(unit=unit, seed=None, complete=True, solution=_detach(solution), **predictions,
                conditional_on_frozen_expert=True, train_inputs_digest=initial_inputs[0],
                validation_inputs_digest=initial_inputs[1], train_records=copy.deepcopy(train["records"]),
                validation_records=copy.deepcopy(validation["records"]), train_labels=train["labels"].clone(),
                validation_labels=validation["labels"].clone())


def validate_matrix_results(results):
    if not isinstance(results,list) or len(results)!=75:
        raise ValueError("Fixed matrix needs exactly75 results.")
    expected = {row["job_id"]:row for row in build_fit_matrix()}
    seen = set()
    for entry in results:
        schedule, result = entry["schedule"], entry["result"]
        job_id = schedule.get("job_id")
        if job_id in seen or job_id not in expected or schedule != expected[job_id]:
            raise ValueError("Duplicate, changed or unknown fixed job.")
        seen.add(job_id)
        if (result.get("complete") is not True or result.get("unit") != schedule["unit"]
                or result.get("seed") != schedule["seed"]):
            raise ValueError("Incomplete or identity-mismatched fitted result.")
        if schedule["kind"]=="trajectory":
            if (result.get("completed_updates")!=400 or result.get("primary_present") is not True
                    or result.get("primary_checkpoint")!=100
                    or [p["completed_updates"] for p in result.get("checkpoints",[])]!=list(CHECKPOINTS)):
                raise ValueError("Fixed checkpoints/primary point incomplete.")
        else:
            solution=result.get("solution",{})
            if (solution.get("converged") is not True or solution.get("mu")!=.001
                    or not isinstance(solution.get("gap_upper"),(float,int))
                    or not 0 <= solution["gap_upper"] <= GAP_TOLERANCE):
                raise ValueError("Convex certificate incomplete or invalid.")
    return dict(trajectories=63, convex_fits=12, temperature_fits=0)
