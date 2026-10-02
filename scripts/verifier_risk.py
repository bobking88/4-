"""Conditional-risk optima inside a frozen target-verification family."""

from __future__ import annotations

import torch

from tc_oos_rsg import _validate_probability_matrix, _validate_unit_column


def _validate_family(p: torch.Tensor, contradiction: torch.Tensor) -> None:
    _validate_probability_matrix(p, "p")
    if p.shape[1] != 4 or bool(((p[:, 0] <= 0) | (p[:, 0] >= 1)).any()):
        raise ValueError("p requires four classes and a strictly interior target probability.")
    if (not torch.is_tensor(contradiction) or contradiction.shape != p[:, :1].shape
            or contradiction.dtype != p.dtype or contradiction.device != p.device
            or not bool(torch.isfinite(contradiction).all())
            or bool((contradiction < 0).any())):
        raise ValueError("contradiction must be a finite nonnegative aligned column.")


def _validate_probability_column(value: torch.Tensor, name: str, p: torch.Tensor) -> None:
    _validate_unit_column(value, name, p)
    if value.dtype != p.dtype:
        raise ValueError(f"{name} must share the posterior dtype.")


def _reconstruct(p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.cat([target, (1-target) * p[:, 1:] / (1-p[:, :1])], dim=1)


def verification_family(p: torch.Tensor, contradiction: torch.Tensor,
                        gamma: torch.Tensor) -> torch.Tensor:
    """Apply exp(-gamma * L) to target odds, preserving non-target ratios."""
    _validate_family(p, contradiction)
    _validate_probability_column(gamma, "gamma", p)
    target = torch.sigmoid(torch.logit(p[:, :1]) - gamma*contradiction)
    return _reconstruct(p, target)


def conditional_risk_projection(p: torch.Tensor, contradiction: torch.Tensor,
                                target_probability: torch.Tensor) -> dict[str, torch.Tensor]:
    """Project an estimated target posterior onto [q_T(L), p_T]."""
    _validate_family(p, contradiction)
    _validate_probability_column(target_probability, "target_probability", p)
    upper = p[:, :1]
    lower = torch.sigmoid(torch.logit(upper) - contradiction)
    if bool((lower <= 0).any()):
        raise ValueError("Verification interval underflowed; use a representable family.")
    target = torch.maximum(lower, torch.minimum(upper, target_probability))
    tilt = (torch.logit(upper) - torch.logit(target)).clamp_min(0)
    tilt = torch.minimum(tilt, contradiction)
    safe_l = torch.where(contradiction > 0, contradiction, torch.ones_like(contradiction))
    gamma = torch.where(contradiction > 0, tilt/safe_l, torch.zeros_like(tilt))
    return {"probabilities": _reconstruct(p, target), "gamma": gamma,
            "lower": lower, "upper": upper,
            "lower_saturated": target_probability <= lower,
            "upper_saturated": target_probability >= upper}


def optimal_global_strength(p: torch.Tensor, contradiction: torch.Tensor,
                            labels: torch.Tensor) -> dict[str, float]:
    """Minimise fit-only empirical NLL by convex endpoint tests and bisection."""
    _validate_family(p, contradiction)
    if (labels.shape != (p.shape[0],) or labels.dtype != torch.long
            or labels.device != p.device or bool(((labels < 0) | (labels > 3)).any())):
        raise ValueError("labels must be aligned long class IDs in [0, 3].")
    p, contradiction = p.detach().double(), contradiction.detach().double()
    y = (labels == 0).double().reshape(-1, 1)

    def derivative(gamma: float) -> float:
        q = torch.sigmoid(torch.logit(p[:, :1]) - gamma*contradiction)
        return float((contradiction*(y-q)).mean())

    d0, d1 = derivative(0.), derivative(1.)
    if d0 >= 0:
        gamma = 0.
    elif d1 <= 0:
        gamma = 1.
    else:
        lo, hi = 0., 1.
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) < 0:
                lo = mid
            else:
                hi = mid
        gamma = (lo+hi)/2
    return {"gamma": gamma, "derivative_at_zero": d0,
            "derivative_at_one": d1, "derivative_at_optimum": derivative(gamma)}
