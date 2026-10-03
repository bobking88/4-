"""Matched calibration and supervision controls for frozen inner evidence."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from run_tc_oos_rsg_experiments import calculate_probability_metrics, state_dict_sha256
from tc_oos_rsg import TargetRiskProjectionHead, _validate_probability_matrix
from verifier_risk import conditional_risk_projection


LEARNED_ARMS = ("M0", "P0", "P1")
TEMPERATURE_FLOOR = 1e-6
TEMPERATURE_OFFSET = math.log(math.expm1(1. - TEMPERATURE_FLOOR))


def validate_data(p: torch.Tensor, labels: torch.Tensor | None = None) -> None:
    _validate_probability_matrix(p, "p")
    if p.dtype != torch.float64 or bool((p <= 0).any()):
        raise ValueError("Controls require strictly positive float64 probabilities.")
    if labels is not None and (labels.shape != (len(p),) or labels.dtype != torch.long
                              or labels.device != p.device
                              or bool(((labels < 0) | (labels >= p.shape[1])).any())):
        raise ValueError("Labels must be aligned long class IDs.")


def temperature_probabilities(p: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    validate_data(p)
    if (temperature.shape != p[:, :1].shape or temperature.dtype != p.dtype
            or temperature.device != p.device or not bool(torch.isfinite(temperature).all())
            or bool((temperature <= 0).any())):
        raise ValueError("Temperature must be a positive finite aligned column.")
    return torch.softmax(p.log() / temperature, dim=1)


def fit_global_temperature(p: torch.Tensor, labels: torch.Tensor, bounds: list) -> dict:
    validate_data(p, labels)
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] or not all(math.isfinite(x) for x in bounds):
        raise ValueError("Inverse-temperature bounds must be positive and ordered.")
    logs = p.detach().log()

    def derivative(beta):
        return float(((torch.softmax(beta * logs, 1) * logs).sum(1) - logs[range(len(p)), labels]).mean())

    lo, hi = bounds
    dlo, dhi = derivative(lo), derivative(hi)
    boundary = None
    if dlo >= 0:
        beta, boundary = lo, "lower"
    elif dhi <= 0:
        beta, boundary = hi, "upper"
    else:
        for _ in range(80):
            middle = (lo + hi) / 2
            if derivative(middle) < 0:
                lo = middle
            else:
                hi = middle
        beta = (lo + hi) / 2
    return {"beta": beta, "temperature": 1 / beta, "boundary": boundary,
            "derivative_at_lower": dlo, "derivative_at_upper": dhi,
            "derivative_at_optimum": derivative(beta)}


def dirichlet_probabilities(p: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    validate_data(p)
    k = p.shape[1]
    if (weight.shape != (k, k) or bias.shape != (k,) or weight.dtype != p.dtype or bias.dtype != p.dtype
            or not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all())):
        raise ValueError("Dirichlet parameters must be finite aligned float64 arrays.")
    return torch.softmax(F.linear(p.log(), weight, bias), dim=1)


def fit_dirichlet(p: torch.Tensor, labels: torch.Tensor, regularization: float) -> dict:
    """Fit convex log-probability multinomial regression with identity-centred L2."""
    validate_data(p, labels)
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Dirichlet regularization must be strictly positive.")
    k, logs = p.shape[1], p.detach().log()
    initial = torch.cat([torch.eye(k, dtype=p.dtype).reshape(-1), torch.zeros(k, dtype=p.dtype)])

    def value_gradient(array):
        vector = torch.tensor(array, dtype=p.dtype, requires_grad=True)
        logits = F.linear(logs, vector[:k*k].reshape(k, k), vector[k*k:])
        loss = F.cross_entropy(logits, labels) + regularization / 2 * (vector - initial).square().sum()
        gradient, = torch.autograd.grad(loss, vector)
        return float(loss.detach()), gradient.detach().numpy()

    result = minimize(value_gradient, initial.numpy(), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "maxls": 50, "ftol": 1e-15, "gtol": 1e-9})
    value, gradient = value_gradient(result.x)
    stationary = bool(math.isfinite(value) and np.isfinite(result.x).all() and np.max(np.abs(gradient)) <= 1e-6)
    if not stationary:
        raise ValueError(f"Dirichlet solver did not reach registered stationarity: {result.message}")
    vector = torch.tensor(result.x, dtype=p.dtype)
    return {"weight": vector[:k*k].reshape(k, k), "bias": vector[k*k:],
            "converged": stationary, "solver_success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "gradient_max_abs": float(np.max(np.abs(gradient))), "regularized_fit_objective": value,
            "regularization": regularization, "raw_parameter_count": int(vector.numel())}


def head_scores(head: TargetRiskProjectionHead, evidence: torch.Tensor) -> torch.Tensor:
    if evidence.ndim != 2 or evidence.shape[1] != head.input_dim or not bool(torch.isfinite(evidence).all()):
        raise ValueError("Evidence must be finite and match the head input dimension.")
    return head.network[:-1](evidence)


def prediction(data: dict, score: torch.Tensor, arm: str) -> dict:
    if arm == "M0":
        temperature = F.softplus(score + TEMPERATURE_OFFSET) + TEMPERATURE_FLOOR
        return {"probabilities": temperature_probabilities(data["p"], temperature),
                "raw": score.sigmoid(), "temperature": temperature}
    if arm in ("P0", "P1"):
        raw = score.sigmoid()
        return dict(conditional_risk_projection(data["p"], data["contradiction"], raw), raw=raw)
    raise ValueError("Unknown learned control arm.")


def objective(data: dict, score: torch.Tensor, arm: str) -> tuple[torch.Tensor, dict]:
    result = prediction(data, score, arm)
    target = (data["labels"] == 0).double().reshape(-1, 1)
    if arm == "P1":
        loss = F.binary_cross_entropy(result["raw"], target)
    elif arm == "M0":
        loss = F.cross_entropy(data["p"].log() / result["temperature"], data["labels"])
    else:
        loss = F.nll_loss(result["probabilities"].log(), data["labels"])
    diagnostic = {"saturated_count": None, "target_saturated_count": None, "degenerate_count": None}
    if arm != "M0":
        saturated = result["lower_saturated"] | result["upper_saturated"]
        diagnostic.update(saturated_count=int(saturated.sum()),
                          target_saturated_count=int((saturated & (target == 1)).sum()),
                          degenerate_count=int((data["contradiction"] == 0).sum()))
    return loss, diagnostic


def predict_head(head: TargetRiskProjectionHead, data: dict, arm: str) -> dict:
    head.eval()
    with torch.no_grad():
        return prediction(data, head_scores(head, data["evidence"]), arm)


def fit_head(fit: dict, stop: dict, arm: str, seed: int, budget: dict) -> dict:
    if arm not in LEARNED_ARMS:
        raise ValueError("Unknown learned control arm.")
    for data in (fit, stop):
        validate_data(data["p"], data["labels"])
    if (not isinstance(budget["epochs"], int) or budget["epochs"] < 1
            or not isinstance(budget["batch_size"], int) or budget["batch_size"] < 1
            or not math.isfinite(budget["learning_rate"]) or budget["learning_rate"] <= 0
            or not math.isfinite(budget["weight_decay"]) or budget["weight_decay"] < 0):
        raise ValueError("Invalid training budget.")
    torch.manual_seed(seed)
    head = TargetRiskProjectionHead(dropout=budget["dropout"]).double()
    initial_sha = state_dict_sha256(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=budget["learning_rate"], weight_decay=budget["weight_decay"])
    head.eval()
    with torch.no_grad():
        initial_loss = float(objective(fit, head_scores(head, fit["evidence"]), arm)[0])
    history, best = [], None
    for epoch in range(1, budget["epochs"] + 1):
        head.train()
        batches = []
        for indices in torch.randperm(len(fit["labels"])).split(budget["batch_size"]):
            batch = {key: fit[key][indices] for key in ("p", "contradiction", "labels", "evidence")}
            optimizer.zero_grad(set_to_none=True)
            score = head_scores(head, batch["evidence"])
            score.retain_grad()
            loss, diagnostic = objective(batch, score, arm)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite training objective.")
            loss.backward()
            gradients = torch.cat([p.grad.reshape(-1) for p in head.parameters()])
            if not bool(torch.isfinite(gradients).all()):
                raise ValueError("Nonfinite training gradient.")
            per_sample_gradient = score.grad.detach() * len(indices)
            diagnostic.update(count=len(indices), loss=float(loss.detach()),
                              zero_score_gradient_count=int((per_sample_gradient == 0).sum()),
                              target_zero_score_gradient_count=int(((per_sample_gradient == 0) & (batch["labels"][:, None] == 0)).sum()),
                              score_gradient_max_abs=float(per_sample_gradient.abs().max()),
                              parameter_gradient_norm=float(gradients.norm()))
            batches.append(diagnostic)
            optimizer.step()
        head.eval()
        with torch.no_grad():
            fit_loss = float(objective(fit, head_scores(head, fit["evidence"]), arm)[0])
        metrics = calculate_probability_metrics(predict_head(head, stop, arm)["probabilities"], stop["labels"])
        history.append({"epoch": epoch, "fit_objective": fit_loss, "stop_metrics": metrics, "training_batches": batches})
        if best is None or metrics["nll"] < best["nll"]:
            best = {"nll": metrics["nll"], "epoch": epoch, "state": copy.deepcopy(head.state_dict()), "fit_objective": fit_loss}
    head.load_state_dict(best["state"])
    return {"head": head, "state_dict": best["state"], "initial_sha256": initial_sha,
            "state_sha256": state_dict_sha256(best["state"]), "parameter_count": sum(p.numel() for p in head.parameters()),
            "history": history, "best_epoch": best["epoch"], "initial_fit_objective": initial_loss,
            "selected_fit_objective": best["fit_objective"]}
