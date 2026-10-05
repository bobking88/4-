"""Converged convex controls for an approved frozen-cache development probe."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from verifier_strong_controls import validate_data


ARMS = ("P", "E", "H", "EH", "EH_permuted")
INITIALIZATIONS = ("zero", "perturbed_20261003", "perturbed_20261004")


def finite_matrix(value: torch.Tensor) -> None:
    if (not torch.is_tensor(value) or value.ndim != 2 or len(value) < 1
            or value.dtype != torch.float64 or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())):
        raise ValueError("Expected nonempty finite CPU float64 matrix.")


def augment(x: torch.Tensor) -> torch.Tensor:
    finite_matrix(x)
    return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)


def probabilities(x: torch.Tensor, anchor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    finite_matrix(anchor)
    finite_matrix(theta)
    design = augment(x)
    if anchor.shape != (len(x), 4) or theta.shape != (design.shape[1], 4):
        raise ValueError("Residual and anchor dimensions do not match.")
    return torch.softmax(anchor + design @ theta, 1)


class RidgeObjective:
    def __init__(self, x: torch.Tensor, anchor: torch.Tensor, labels: torch.Tensor, regularization: float):
        finite_matrix(anchor)
        design = augment(x)
        if (anchor.shape != (len(x), 4) or labels.shape != (len(x),)
                or labels.dtype != torch.long or labels.device.type != "cpu"
                or bool(((labels < 0) | (labels >= 4)).any())
                or not math.isfinite(regularization) or regularization <= 0):
            raise ValueError("Invalid convex objective inputs.")
        self.x = design.detach().numpy().copy()
        self.anchor = anchor.detach().numpy().copy()
        self.y = labels.numpy().copy()
        self.regularization = regularization
        self.shape = (design.shape[1], 4)
        self.targets = np.eye(4)[self.y]

    def distribution(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.size != np.prod(self.shape) or not np.isfinite(vector).all():
            raise ValueError("Invalid residual coefficient vector.")
        theta = vector.reshape(self.shape)
        logits = self.anchor + self.x @ theta
        logq = logits - logsumexp(logits, axis=1, keepdims=True)
        return theta, logq, np.exp(logq)

    def value_gradient(self, vector):
        theta, logq, q = self.distribution(vector)
        value = -logq[np.arange(len(q)), self.y].mean() + self.regularization/2*np.square(theta).sum()
        gradient = self.x.T @ (q-self.targets)/len(q) + self.regularization*theta
        return float(value), gradient.reshape(-1)

    def hessian(self, vector):
        _, _, q = self.distribution(vector)
        covariance = q[:, :, None]*np.eye(4)[None, :, :] - q[:, :, None]*q[:, None, :]
        matrix = np.einsum("ni,nj,nkl->ikjl", self.x, self.x, covariance, optimize=True)/len(q)
        size = int(np.prod(self.shape))
        return matrix.reshape(size, size) + self.regularization*np.eye(size)


def fit(x, anchor, labels, regularization, initialization, solver) -> dict:
    objective = RidgeObjective(x, anchor, labels, regularization)
    if initialization not in INITIALIZATIONS:
        raise ValueError("Unknown registered initialization.")
    initial = torch.zeros(objective.shape, dtype=torch.float64)
    if initialization != "zero":
        generator = torch.Generator().manual_seed(int(initialization.split("_")[1]))
        initial = .001*torch.randn(objective.shape, generator=generator, dtype=torch.float64)
    history = []

    def callback(phase):
        def record(vector):
            value, gradient = objective.value_gradient(vector)
            history.append({"phase": phase, "objective": value, "gradient_l2": float(np.linalg.norm(gradient))})
        return record

    first = minimize(objective.value_gradient, initial.numpy().reshape(-1), jac=True, method="L-BFGS-B",
                     callback=callback("lbfgs"), options={"maxiter": solver["lbfgs_maxiter"], "maxls": solver["lbfgs_maxls"],
                     "ftol": solver["lbfgs_ftol"], "gtol": solver["lbfgs_gtol"]})
    second = minimize(objective.value_gradient, first.x, jac=True, hess=objective.hessian, method="trust-exact",
                      callback=callback("trust_exact"), options={"maxiter": solver["trust_maxiter"], "gtol": solver["trust_gtol"]})
    value, gradient = objective.value_gradient(second.x)
    gradient_l2 = float(np.linalg.norm(gradient))
    gap = gradient_l2**2/(2*regularization)
    if (not np.isfinite(second.x).all() or not math.isfinite(value)
            or gradient_l2 > solver["stationarity_l2_tolerance"] or gap > solver["gap_upper_tolerance"]):
        raise ValueError(f"Registered stationarity not reached: norm={gradient_l2}, gap={gap}, {second.message}")
    theta = torch.from_numpy(second.x.reshape(objective.shape).copy())
    _, logq, _ = objective.distribution(second.x)
    fit_nll = float(-logq[np.arange(len(labels)), objective.y].mean())
    identity_value = objective.value_gradient(np.zeros_like(second.x))[0]
    if value > identity_value + gap + 1e-12:
        raise ValueError("Convex fit violates the registered identity-anchor bound.")
    return {"theta": theta, "initialization": initialization, "converged": True,
            "gradient_l2": gradient_l2, "gradient_max_abs": float(np.abs(gradient).max()),
            "gap_upper_bound": gap, "regularized_fit_objective": value, "fit_nll": fit_nll,
            "identity_fit_objective": identity_value, "regularization": regularization,
            "parameter_count": theta.numel(), "history": history,
            "solver_stages": [{"method": method, "success": bool(result.success), "message": str(result.message),
                               "iterations": int(result.nit), "function_evaluations": int(result.nfev)}
                              for method, result in (("L-BFGS-B", first), ("trust-exact", second))]}


def fit_representation(e, p, contradiction) -> dict:
    finite_matrix(e)
    finite_matrix(contradiction)
    validate_data(p)
    if e.shape != (len(p), 18) or contradiction.shape != (len(p), 1):
        raise ValueError("Registered probability representation dimensions required.")
    logs = p.log()
    evidence = torch.cat([e, logs, contradiction], 1)
    return {"P_mean": logs.mean(0), "P_scale": logs.std(0, correction=0).clamp_min(1e-8),
            "E_mean": evidence.mean(0), "E_scale": evidence.std(0, correction=0).clamp_min(1e-8)}


def representations(e, p, contradiction, h, state) -> dict:
    finite_matrix(e)
    finite_matrix(contradiction)
    finite_matrix(h)
    validate_data(p)
    if e.shape != (len(p), 18) or contradiction.shape != (len(p), 1) or h.shape != (len(p), 64):
        raise ValueError("Registered frozen representation dimensions required.")
    for key, size in (("P_mean", 4), ("P_scale", 4), ("E_mean", 23), ("E_scale", 23)):
        if state[key].shape != (size,) or not bool(torch.isfinite(state[key]).all()):
            raise ValueError("Invalid fit-only representation state.")
        if key.endswith("scale") and bool((state[key] < 1e-8).any()):
            raise ValueError("Standardization scales must remain strictly positive.")
    logs = p.log()
    evidence = torch.cat([e, logs, contradiction], 1)
    return {"P": (logs-state["P_mean"])/state["P_scale"],
            "E": (evidence-state["E_mean"])/state["E_scale"], "H": h}


def make_input(blocks, arm, permutation_seed) -> torch.Tensor:
    if arm not in ARMS:
        raise ValueError("Unknown registered control arm.")
    if arm in ("P", "E", "H"):
        return blocks[arm]
    h = blocks["H"]
    if arm == "EH_permuted":
        generator = torch.Generator().manual_seed(permutation_seed)
        h = h[torch.randperm(len(h), generator=generator)]
    return torch.cat([blocks["E"], h], 1)
