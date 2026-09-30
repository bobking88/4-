"""Adaptive risk allocation and analytic target-margin routing protection."""

from __future__ import annotations

import math

import torch
from torch import nn

from tc_oos_rsg import (
    _epsilon_column,
    _validate_probability_matrix,
    _validate_unit_column,
    apply_target_safe_projection,
    build_projection_evidence,
)


class AdaptiveBudgetPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(18, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Dropout(.10), nn.Linear(64, 16), nn.SiLU(),
        )
        self.route_head = nn.Linear(16, 1)
        self.budget_head = nn.Linear(16, 1)

    def forward(
        self, evidence: torch.Tensor, epsilon_max: float, lambda_cal: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if evidence.ndim != 2 or evidence.shape[0] == 0 or evidence.shape[1] != 18:
            raise ValueError("evidence must have shape [nonempty batch, 18].")
        if not evidence.is_floating_point() or not bool(torch.isfinite(evidence).all()):
            raise ValueError("evidence must contain finite floating-point values.")
        if not math.isfinite(epsilon_max) or not 0 < epsilon_max <= 1:
            raise ValueError("epsilon_max must lie in (0, 1].")
        if not math.isfinite(lambda_cal):
            raise ValueError("lambda_cal must be finite.")
        shared = self.trunk(evidence)
        route_logits = self.route_head(shared)
        budget_logits = self.budget_head(shared) - lambda_cal
        budget_fraction = budget_logits.sigmoid()
        return {
            "route_logits": route_logits,
            "budget_logits": budget_logits,
            "raw_route": route_logits.sigmoid(),
            "budget_fraction": budget_fraction,
            "epsilon": epsilon_max * budget_fraction,
        }


def risk_supervision_targets(
    q0: torch.Tensor, q_candidate: torch.Tensor, labels: torch.Tensor,
    epsilon_max: float, tau_r: float = .25, target_index: int = 0,
) -> dict[str, torch.Tensor]:
    _validate_probability_matrix(q0, "q0")
    _validate_probability_matrix(q_candidate, "q_candidate")
    if q0.shape != q_candidate.shape or q0.device != q_candidate.device or q0.dtype != q_candidate.dtype:
        raise ValueError("posteriors must share shape, dtype and device.")
    if labels.shape != (len(q0),) or labels.dtype != torch.long or labels.device != q0.device:
        raise ValueError("labels must be a long [batch] tensor on the posterior device.")
    if bool(((labels < 0) | (labels >= q0.shape[1])).any()):
        raise ValueError("labels must identify posterior columns.")
    if not 0 <= target_index < q0.shape[1]:
        raise ValueError("invalid target_index.")
    if not math.isfinite(epsilon_max) or not 0 < epsilon_max <= 1 or not math.isfinite(tau_r) or tau_r <= 0:
        raise ValueError("epsilon_max and tau_r must be finite and positive.")
    tiny = torch.finfo(q0.dtype).tiny
    with torch.no_grad():
        p0 = q0.gather(1, labels[:, None]).clamp_min(tiny)
        pp = q_candidate.gather(1, labels[:, None]).clamp_min(tiny)
        route_target = ((pp.log() - p0.log()) / tau_r).sigmoid()
        drop = (q0[:, target_index:target_index + 1] - q_candidate[:, target_index:target_index + 1]).clamp_min(0)
        return {
            "route_target": route_target,
            "budget_target": route_target * (drop / epsilon_max).clamp_max(1),
            "budget_mask": drop > 1e-12,
        }


def apply_abmp_projection(
    q0: torch.Tensor, q_candidate: torch.Tensor, raw_route: torch.Tensor,
    epsilon: torch.Tensor | float, *, tau_p: float, tau_m: float, delta: float,
    target_index: int = 0, mode: str = "full",
) -> dict[str, torch.Tensor]:
    if mode not in {"full", "posterior", "margin", "none"}:
        raise ValueError("Unknown projection mode.")
    if not all(math.isfinite(v) for v in (tau_p, tau_m, delta)):
        raise ValueError("anchor thresholds must be finite.")
    if not 0 <= tau_p <= 1 or not 0 < delta <= tau_m <= 1:
        raise ValueError("Require 0 <= tau_p <= 1 and 0 < delta <= tau_m <= 1.")
    posterior = apply_target_safe_projection(q0, q_candidate, raw_route, epsilon, target_index=target_index)
    epsilon_column = _epsilon_column(epsilon, q0)
    target0 = q0[:, target_index:target_index + 1]
    target_phi = q_candidate[:, target_index:target_index + 1]
    other_indices = [k for k in range(q0.shape[1]) if k != target_index]
    other0 = q0[:, other_indices]
    other_phi = q_candidate[:, other_indices]
    initial_margins = target0 - other0
    anchor = (q0.argmax(1, keepdim=True) == target_index) & (target0 >= tau_p) & (initial_margins.min(1, keepdim=True).values >= tau_m)
    delta_direction = (other_phi - other0) - (target_phi - target0)
    shrinking = delta_direction > 0
    denominator = torch.where(shrinking, delta_direction, torch.ones_like(delta_direction))
    pair_caps = torch.where(shrinking, ((initial_margins - delta) / denominator).clamp(0, 1), torch.ones_like(delta_direction))
    margin_cap = torch.where(anchor, pair_caps.min(1, keepdim=True).values, torch.ones_like(raw_route))
    posterior_cap = posterior["route_cap"]
    used_post = posterior_cap if mode in {"full", "posterior"} else torch.ones_like(raw_route)
    used_margin = margin_cap if mode in {"full", "margin"} else torch.ones_like(raw_route)
    route = torch.minimum(raw_route, torch.minimum(used_post, used_margin))
    final = (1 - route) * q0 + route * q_candidate
    signed_change = target0 - final[:, target_index:target_index + 1]
    return {
        "raw_route": raw_route,
        "projected_route": route,
        "epsilon": epsilon_column,
        "posterior_cap": posterior_cap,
        "margin_cap": margin_cap,
        "anchor_mask": anchor,
        "posterior_active": used_post < raw_route - 1e-7,
        "margin_active": used_margin < raw_route - 1e-7,
        "final_probabilities": final,
        "signed_target_change": signed_change,
        "positive_target_harm": signed_change.clamp_min(0),
        "identity_residual": signed_change - route * (target0 - target_phi),
        "final_target_margin": (final[:, target_index:target_index + 1] - final[:, other_indices]).min(1, keepdim=True).values,
    }


def audit_abmp_projection(
    result: dict[str, torch.Tensor], q0: torch.Tensor, q_candidate: torch.Tensor,
    labels: torch.Tensor, epsilon_max: float, delta: float, target_index: int = 0,
) -> dict[str, float | int]:
    tolerance = 1e-6
    final = result["final_probabilities"]
    route = result["projected_route"]
    eps = result["epsilon"]
    anchors = result["anchor_mask"].flatten()
    anchor_count = int(anchors.sum())
    retained = final.argmax(1) == target_index
    tiny = torch.finfo(final.dtype).tiny
    row_index = torch.arange(len(final), device=final.device)
    nll = -final[row_index, labels].clamp_min(tiny).log()
    upper = (1 - route.flatten()) * -q0[row_index, labels].clamp_min(tiny).log() + route.flatten() * -q_candidate[row_index, labels].clamp_min(tiny).log()
    residual = final - ((1 - route) * q0 + route * q_candidate)
    return {
        "rows": len(final),
        "nonfinite_violations": int((~torch.isfinite(final).all(1)).sum()),
        "simplex_violations": int(((final.min(1).values < -tolerance) | ((final.sum(1) - 1).abs() > tolerance)).sum()),
        "budget_range_violations": int(((eps < -tolerance) | (eps > epsilon_max + tolerance) | ~torch.isfinite(eps)).sum()),
        "posterior_violations": int((result["signed_target_change"] > eps + tolerance).sum()),
        "margin_violations": int((anchors & (result["final_target_margin"].flatten() < delta - tolerance)).sum()),
        "decomposition_violations": int((residual.abs().max(1).values > tolerance).sum()),
        "convex_nll_violations": int((nll > upper + tolerance).sum()),
        "max_decomposition_residual": float(residual.abs().max()),
        "max_target_identity_residual": float(result["identity_residual"].abs().max()),
        "anchor_count": anchor_count,
        "anchor_true_target_count": int((anchors & (labels == target_index)).sum()),
        "anchor_false_positive_count": int((anchors & (labels != target_index)).sum()),
        "anchor_retention_rate": float(retained[anchors].double().mean()) if anchor_count else math.nan,
        "anchor_coverage": anchor_count / len(final),
        "posterior_activation_rate": float(result["posterior_active"].double().mean()),
        "margin_activation_rate": float(result["margin_active"].double().mean()),
        "joint_activation_rate": float((result["posterior_active"] & result["margin_active"]).double().mean()),
        "epsilon_mean": float(eps.mean()),
        "epsilon_std": float(eps.std(unbiased=False)),
        "epsilon_min": float(eps.min()),
        "epsilon_max_observed": float(eps.max()),
        "max_difference_from_q0": float((final - q0).abs().max()),
        "max_difference_from_unbounded": float((final - ((1 - result["raw_route"]) * q0 + result["raw_route"] * q_candidate)).abs().max()),
        "exact_fallback_count": int((route == 0).sum()),
    }
