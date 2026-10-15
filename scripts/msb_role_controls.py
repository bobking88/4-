"""MSB losses and deterministic, certified convex controls."""
from __future__ import annotations

import copy
import torch
from torch.nn import functional as F
from msb_role_data import validate_batch
from msb_role_model import CONVEX_UNITS, anchor_margins, decode_role, unit_spec

MAX_ITERATIONS = 200
MAX_EVALUATIONS = 4201
GAP_TOLERANCE = 1e-8
MU = .001


def _labels(labels, n):
    if (not torch.is_tensor(labels) or labels.dtype != torch.long
            or labels.device.type != "cpu" or labels.shape != (n,)
            or set(labels.tolist()) != {0, 1, 2, 3}):
        raise ValueError("Four populated CPU long classes required for the objective.")


def pair_loss(logq, labels):
    _labels(labels, len(logq))
    values = []
    for j in range(1, 4):
        selected = (labels == 0) | (labels == j)
        signs = torch.where(labels[selected] == 0, 1., -1.)
        margin = logq[selected, 0] - logq[selected, j]
        values.append(F.softplus(-signs * margin).mean())
    return torch.stack(values).mean()


def objective(model, result: dict, labels: torch.Tensor) -> dict:
    _labels(labels, len(result["logq"]))
    nll = F.nll_loss(result["logq"], labels)
    pair = pair_loss(result["logq"], labels)
    weighted_pair = model.settings["pair_weight"] * pair
    relative = .01 * (result["d"] / result["R_ref"]).square().mean()
    sp = .001 / 2 * sum((p - model.initial_parameter(name)).square().sum()
                       for name, p in model.named_current())
    data = nll + weighted_pair
    regularization = relative + sp
    total = data + regularization
    if not torch.isfinite(total):
        raise ValueError("NUMERIC_RANGE_FAILURE: objective nonfinite")
    return dict(nll=nll, pair_raw=pair, pair_weighted=weighted_pair,
                relative_penalty=relative, sp=sp, data=data,
                regularization=regularization, total=total)


def competition_features(log_anchor: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    anchor_margins(log_anchor)
    if (counts.shape != (4,) or counts.dtype != torch.float64 or counts.device.type != "cpu"
            or counts.requires_grad or not torch.isfinite(counts).all() or bool((counts <= 0).any())):
        raise ValueError("Positive frozen training-group counts required.")
    order = torch.argsort(log_anchor, dim=1, descending=True, stable=True)
    rank = torch.empty_like(log_anchor)
    rank.scatter_(1, order, torch.arange(4, dtype=torch.float64).expand(len(order), 4))
    gap = log_anchor[:, :, None] - log_anchor[:, None, :]
    rank_gap = rank[:, None, :] - rank[:, :, None]
    log_counts = counts.log()
    count_gap = (log_counts[:, None] - log_counts[None, :]).expand(len(order), 4, 4)
    return torch.stack([gap, rank_gap, count_gap, gap], dim=-1)


def make_convex_control(unit: str, batch: dict) -> dict:
    settings = unit_spec(unit)
    if unit not in CONVEX_UNITS:
        raise ValueError("A convex MSB control is required.")
    validate_batch(batch)
    return dict(unit=unit, coordinates=torch.zeros(settings["parameters"], dtype=torch.float64),
                counts=torch.bincount(batch["labels"], minlength=4).to(torch.float64), mu=MU)


def _coordinates(control, coordinates):
    unit = control.get("unit")
    if unit not in CONVEX_UNITS:
        raise ValueError("Unknown convex control identity.")
    if (not torch.is_tensor(coordinates) or coordinates.shape != (unit_spec(unit)["parameters"],)
            or coordinates.dtype != torch.float64 or coordinates.device.type != "cpu"
            or not torch.isfinite(coordinates).all()):
        raise ValueError("Finite float64 convex coordinates required.")
    if unit == "BIAS3-CVX" and bool((coordinates.abs() >= 1).any()):
        raise ValueError("Bias coordinates must stay strictly inside (-1,1).")


def _prediction(control, coordinates, h, e, log_anchor):
    _coordinates(control, coordinates)
    _, radius = anchor_margins(log_anchor)
    for value, width in ((h, 64), (e, 23)):
        if (value.shape != (len(log_anchor), width) or value.dtype != torch.float64
                or value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("Frozen float64 control inputs required.")
    unit = control["unit"]
    if unit == "LIN-UNB":
        s = F.linear(torch.cat([h, e], 1), coordinates[:261].reshape(3, 87), coordinates[261:])
        d = radius * s
    elif unit == "BIAS3-CVX":
        d = radius * coordinates
    else:
        offsets = torch.cat([torch.zeros(1, dtype=torch.float64), coordinates[:3]])
        residual = offsets.expand(len(log_anchor), 4)
        if unit == "REPAIR4-COMP":
            features = competition_features(log_anchor, control["counts"])
            residual = residual + (features.sum(2) / 3) @ coordinates[3:]
        d = residual[:, :1] - residual[:, 1:]
    result = decode_role(log_anchor, d)
    result.update(unit=unit, s="NOT_APPLICABLE", c="NOT_APPLICABLE",
                  saturation_count=0, role_effect="NOT_APPLICABLE")
    return result


def control_forward(control, h, e, log_anchor):
    return _prediction(control, control["coordinates"], h, e, log_anchor)


def convex_objective(control, coordinates: torch.Tensor, batch: dict) -> torch.Tensor:
    result = _prediction(control, coordinates, batch["h"], batch["e"], batch["log_anchor"])
    _labels(batch["labels"], len(result["logq"]))
    loss = F.nll_loss(result["logq"], batch["labels"])
    if control["unit"] in ("LIN-UNB", "BIAS3-CVX"):
        loss = loss + .1 * pair_loss(result["logq"], batch["labels"])
        loss = loss + .01 * (result["d"] / result["R_ref"]).square().mean()
    penalty_coordinates = coordinates.atanh() if control["unit"] == "BIAS3-CVX" else coordinates
    loss = loss + .001 / 2 * penalty_coordinates.square().sum()
    if not torch.isfinite(loss):
        raise ValueError("NUMERIC_RANGE_FAILURE: convex objective nonfinite")
    return loss


def solve_convex(control, batch: dict, *, deadline_check) -> dict:
    _coordinates(control, control["coordinates"])
    x = torch.zeros_like(control["coordinates"])
    evaluations = 0
    trace = []

    def evaluate(value):
        nonlocal evaluations
        deadline_check()
        if evaluations >= MAX_EVALUATIONS:
            raise RuntimeError("Convex certificate evaluation budget exhausted.")
        evaluations += 1
        return convex_objective(control, value, batch)

    for iteration in range(MAX_ITERATIONS + 1):
        deadline_check()
        current = x.detach().requires_grad_(True)
        loss = evaluate(current)
        gradient = torch.autograd.grad(loss, current, create_graph=True)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError("NUMERIC_RANGE_FAILURE: convex gradient nonfinite")
        gap = float(gradient.detach().square().sum() / (2 * MU))
        trace.append(dict(iteration=iteration, objective=float(loss.detach()), gap_upper=gap))
        if gap <= GAP_TOLERANCE:
            state = copy.deepcopy(control)
            state["coordinates"] = current.detach().clone()
            return dict(unit=control["unit"], coordinates=state["coordinates"],
                        objective=float(loss.detach()), gradient=gradient.detach().clone(), mu=MU,
                        gap_upper=gap, iterations=iteration, evaluations=evaluations,
                        converged=True, model_state=state, trace=trace)
        if iteration >= MAX_ITERATIONS:
            break
        rows = []
        for component in gradient:
            deadline_check()
            rows.append(torch.autograd.grad(component, current, retain_graph=True)[0])
        hessian = torch.stack(rows).detach()
        if not torch.isfinite(hessian).all():
            raise ValueError("NUMERIC_RANGE_FAILURE: convex Hessian nonfinite")
        factor = torch.linalg.cholesky((hessian + hessian.T) / 2)
        direction = -torch.cholesky_solve(gradient.detach()[:, None], factor).flatten()
        slope = float(gradient.detach() @ direction)
        if not slope < 0:
            raise RuntimeError("Convex Newton direction is not descending.")
        accepted = False
        step = 1.
        for _ in range(20):
            deadline_check()
            candidate = current.detach() + step * direction
            if control["unit"] == "BIAS3-CVX" and bool((candidate.abs() >= 1).any()):
                step *= .5
                continue
            candidate_loss = evaluate(candidate)
            if float(candidate_loss.detach()) <= float(loss.detach()) + 1e-4 * step * slope:
                x, accepted = candidate, True
                break
            step *= .5
        if not accepted:
            raise RuntimeError("Convex line search failed before certificate.")
    raise RuntimeError("Convex certificate not attained within the fixed iteration budget.")
