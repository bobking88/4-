"""Full-batch trajectories and pooled fit-only OOF selection, never stop selection."""
from __future__ import annotations

import copy
import math

import torch
from threadpoolctl import threadpool_limits

import anchor_linear_controls as controls
from lc_role_adapter import ARMS, NumericRangeFailure, RoleResidualModel, anchor_geometry, objective
from lc_role_data import require_classes
from run_anchor_linear_controls import SOLVER

SEEDS = (20261002, 20261003, 20261004)
CHECKPOINTS = (0, 20, 50, 100, 200, 400)


def _batch(batch: dict) -> None:
    require_classes(batch["labels"])
    anchor_geometry(batch["log_anchor"])
    n = len(batch["labels"])
    if (len(batch["records"]) != n or len({r["image_id"] for r in batch["records"]}) != n
            or any(r.get("tc_subset") == "projector_stop" for r in batch["records"])):
        raise ValueError("Fit/inner-validation identities required; stop records forbidden.")
    for key, width in (("h", 64), ("e", 23), ("P", 4)):
        tensor = batch[key]
        if (tensor.shape != (n, width) or tensor.dtype != torch.float64
                or tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
            raise ValueError("Invalid prepared training batch.")


def _splits(train: dict, validation: dict | None) -> None:
    _batch(train)
    if validation is not None:
        _batch(validation)
        for field in ("image_id", "split_group_id"):
            if {r[field] for r in train["records"]} & {r[field] for r in validation["records"]}:
                raise ValueError("Fit/inner-validation group leakage.")


def _config(config: dict) -> dict:
    values = {key: float(config[key]) for key in ("regularization", "alpha_max", "beta_max")}
    if (not all(math.isfinite(v) for v in values.values()) or values["regularization"] <= 0
            or min(values["alpha_max"], values["beta_max"]) < 0):
        raise ValueError("Invalid candidate configuration.")
    return values


def _trajectory(train, validation, arm, config, seed, checkpoints):
    _splits(train, validation)
    values = _config(config)
    if (seed not in SEEDS or not checkpoints or checkpoints[0] != 0
            or sorted(set(checkpoints)) != list(checkpoints)
            or any(step not in CHECKPOINTS for step in checkpoints)):
        raise ValueError("Unregistered seed or checkpoint trajectory.")
    model = RoleResidualModel(arm, seed=seed, alpha_max=values["alpha_max"], beta_max=values["beta_max"])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
    history, gradients, saved = [], [], {}
    last_update = max(checkpoints)
    parameters = tuple(model.parameters())
    for step in range(last_update+1):
        result = model(train["h"], train["e"], train["log_anchor"], auxiliary=True)
        losses = objective(model, result, train["labels"], values["regularization"])
        row = dict(step=step, **{key: float(value.detach()) for key, value in losses.items()})
        if step in checkpoints:
            with torch.no_grad():
                val = None if validation is None else model(validation["h"], validation["e"], validation["log_anchor"], auxiliary=True)
                val_loss = None if val is None else objective(model, val, validation["labels"], values["regularization"])
            row["validation_nll"] = None if val_loss is None else float(val_loss["nll"])
            saved[step] = dict(state_dict=copy.deepcopy(model.state_dict()),
                               train_logq=result["logq"].detach().clone(),
                               validation_logq=None if val is None else val["logq"].detach().clone(),
                               train_metrics={key: float(value.detach()) for key, value in losses.items()},
                               validation_metrics=None if val_loss is None else {key: float(value) for key, value in val_loss.items()})
        history.append(row)
        if step == last_update:
            break
        if step < 5:
            data_grads = torch.autograd.grad(losses["nll"]+.1*losses["pair"], parameters,
                                             retain_graph=True, allow_unused=True)
            total_grads = torch.autograd.grad(losses["total"], parameters, allow_unused=True)
            for (name, _), d, t in zip(model.named_parameters(), data_grads, total_grads, strict=True):
                gradients.append(dict(step=step+1, parameter=name,
                                      data_gradient_l2=0. if d is None else float(d.norm()),
                                      total_gradient_l2=0. if t is None else float(t.norm())))
            optimizer.zero_grad()
            for parameter, gradient in zip(parameters, total_grads, strict=True):
                parameter.grad = gradient
        else:
            optimizer.zero_grad()
            losses["total"].backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("Nonfinite training gradient.")
        optimizer.step()
    return dict(arm=arm, seed=seed, config=values, checkpoints=saved,
                history=history, gradients=gradients, optimizer_updates=last_update,
                status="OPTIMIZATION_TRUNCATED" if last_update == 400 else "TRAJECTORY_COMPLETE",
                source="fit_only", validation_source="fit_inner" if validation is not None else None)


def fit_neural(train: dict, validation: dict, arm: str, config: dict, seed: int) -> dict:
    checkpoints = tuple(config.get("checkpoints", CHECKPOINTS))
    return _trajectory(train, validation, arm, config, seed, checkpoints)


def refit_neural(fit: dict, arm: str, config: dict, seed: int, updates: int) -> dict:
    if updates not in CHECKPOINTS:
        raise ValueError("Refit update count must be a registered checkpoint.")
    checkpoints = (0,) if updates == 0 else (0, updates)
    result = _trajectory(fit, None, arm, config, seed, checkpoints)
    return dict(result, **result["checkpoints"][updates])


def _linear_prediction(batch, arm, theta):
    x = batch["P" if arm == "P" else "e"]
    design = controls.augment(x)
    logq = torch.log_softmax(batch["log_anchor"]+design@theta, 1)
    if not torch.isfinite(logq).all():
        raise ValueError("Nonfinite convex predictions.")
    if bool((logq.exp() == 0).any()):
        raise NumericRangeFailure(logq)
    return logq


def fit_convex(train: dict, validation: dict | None, arm: str, regularization: float) -> dict:
    if arm not in ("P", "E"):
        raise ValueError("Only the registered probability and E23 convex controls are allowed.")
    _splits(train, validation)
    with threadpool_limits(limits=1):
        fitted = controls.fit(train["P" if arm == "P" else "e"], train["log_anchor"],
                              train["labels"], regularization, "zero", SOLVER)
    return dict(fitted, arm=arm, train_logq=_linear_prediction(train, arm, fitted["theta"]),
                validation_logq=None if validation is None else _linear_prediction(validation, arm, fitted["theta"]),
                source="fit_only")


def select_oof(candidates: list[dict], assignment: list[dict]) -> dict:
    if not candidates or not assignment:
        raise ValueError("Complete fit-only OOF candidates required.")
    ids = [r["image_id"] for r in assignment]
    if len(ids) != len(set(ids)) or {r["inner_fold"] for r in assignment} != {0, 1, 2}:
        raise ValueError("Invalid inner assignment.")
    for fold in range(3):
        val = [r for r in assignment if r["inner_fold"] == fold]
        train = [r for r in assignment if r["inner_fold"] != fold]
        require_classes(torch.tensor([r["class_id"] for r in val]))
        require_classes(torch.tensor([r["class_id"] for r in train]))
        if {r["split_group_id"] for r in val} & {r["split_group_id"] for r in train}:
            raise ValueError("Inner assignment group leakage.")
    arms = {row["arm"] for row in candidates}
    if len(arms) != 1 or not arms <= set(ARMS+("P", "E")):
        raise ValueError("Select each registered arm independently.")
    arm = next(iter(arms))
    seeds = (0,) if arm in ("P", "E") else SEEDS
    by_id = {r["image_id"]: r for r in assignment}
    groups = {}
    for candidate in candidates:
        if candidate.get("source") != "fit_inner_oof" or candidate["seed"] not in seeds or candidate["inner_fold"] not in range(3):
            raise ValueError("Untrusted OOF source, fold, or seed.")
        if arm in ("P", "E"):
            certificate = candidate.get("certificate", {})
            gradient = certificate.get("gradient_l2", float("nan"))
            gap = certificate.get("gap_upper_bound", float("nan"))
            if (certificate.get("converged") is not True or not math.isfinite(gradient)
                    or not math.isfinite(gap) or not 0 <= gradient <= 1e-7 or not 0 <= gap <= 1e-8):
                raise ValueError("Uncertified convex control cannot enter OOF selection.")
        config = _config(candidate["config"])
        updates = candidate["updates"]
        if updates not in CHECKPOINTS:
            raise ValueError("Unregistered selection checkpoint.")
        key = (config["alpha_max"], config["beta_max"], config["regularization"], updates)
        fold_ids = candidate["image_ids"]
        expected = {r["image_id"] for r in assignment if r["inner_fold"] == candidate["inner_fold"]}
        if len(fold_ids) != len(set(fold_ids)) or set(fold_ids) != expected:
            raise ValueError("OOF fold has missing, duplicate, or wrong row identity.")
        logq = candidate["logq"]
        anchor_geometry(logq)
        if len(logq) != len(fold_ids):
            raise ValueError("OOF probability count mismatch.")
        group = groups.setdefault(key, dict(config=config, updates=updates, by_seed={}))
        predictions = group["by_seed"].setdefault(candidate["seed"], {})
        for i, image_id in enumerate(fold_ids):
            if image_id in predictions:
                raise ValueError("OOF row predicted more than once per seed and configuration.")
            predictions[image_id] = -float(logq[i, by_id[image_id]["class_id"]])
    scores = []
    for group in groups.values():
        if set(group["by_seed"]) != set(seeds) or any(set(rows) != set(ids) for rows in group["by_seed"].values()):
            raise ValueError("Missing OOF rows or seeds.")
        pooled = {str(seed): math.fsum(group["by_seed"][seed][i] for i in ids)/len(ids) for seed in seeds}
        scores.append(dict(config=group["config"], updates=group["updates"],
                           seed_pooled_nll=pooled, mean_pooled_nll=math.fsum(pooled.values())/len(seeds)))
    def order(row):
        c = row["config"]
        return (row["mean_pooled_nll"], c["alpha_max"]+c["beta_max"], -c["regularization"],
                row["updates"], c["alpha_max"], c["beta_max"], c["regularization"])
    scores.sort(key=order)
    chosen = scores[0]
    return dict(chosen, arm=arm, pooled_row_count=len(ids), scores=scores,
                source="fit_inner_oof", status="OPTIMIZATION_TRUNCATED" if chosen["updates"] == 400 else "FIT_SELECTED")


def matched_budget_candidates(candidates: list[dict], selection: dict) -> list[dict]:
    if selection.get("arm") != "R1" or selection.get("source") != "fit_inner_oof":
        raise ValueError("Matched S1 budget must come from fit-selected R1.")
    budget = tuple(selection["config"][key] for key in ("alpha_max", "beta_max"))
    result = [row for row in candidates if row["arm"] == "S1" and
              tuple(row["config"][key] for key in ("alpha_max", "beta_max")) == budget]
    if not result:
        raise ValueError("Missing preplanned same-budget S1 candidates.")
    return result
