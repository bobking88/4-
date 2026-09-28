from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


_SIMPLEX_TOLERANCE = 1e-6


def _validate_probability_matrix(probabilities: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(probabilities) or probabilities.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, classes].")
    if probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError(f"{name} must contain at least one row and two classes.")
    if not probabilities.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype.")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError(f"{name} must contain only finite values.")
    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise ValueError(f"{name} values must lie in [0, 1].")
    row_sums = probabilities.sum(dim=1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=_SIMPLEX_TOLERANCE,
        rtol=_SIMPLEX_TOLERANCE,
    ):
        raise ValueError(f"{name} rows must sum to one.")


def _validate_unit_column(
    values: torch.Tensor,
    name: str,
    reference: torch.Tensor,
) -> None:
    expected_shape = (reference.shape[0], 1)
    if not torch.is_tensor(values) or tuple(values.shape) != expected_shape:
        raise ValueError(f"{name} must have shape [batch, 1].")
    if not values.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype.")
    if values.device != reference.device:
        raise ValueError(f"{name} must be on the same device as the posteriors.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values.")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError(f"{name} values must lie in [0, 1].")


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(probabilities.dtype).tiny
    return -(
        probabilities * probabilities.clamp_min(epsilon).log()
    ).sum(dim=1, keepdim=True)


def _jensen_shannon_divergence(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.finfo(first.dtype).tiny
    midpoint = 0.5 * (first + second)
    first_kl = (
        first * (first.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())
    ).sum(dim=1, keepdim=True)
    second_kl = (
        second * (second.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())
    ).sum(dim=1, keepdim=True)
    return 0.5 * (first_kl + second_kl)


def build_projection_evidence(
    q0: torch.Tensor,
    q_candidate: torch.Tensor,
    candidate_gate: torch.Tensor,
    ti_target_probability: torch.Tensor,
    metallic_target_probability: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed 18D inference-time evidence vector."""
    _validate_probability_matrix(q0, "q0")
    _validate_probability_matrix(q_candidate, "q_candidate")
    if q0.shape != q_candidate.shape:
        raise ValueError("q0 and q_candidate must have the same shape.")
    if q0.shape[1] != 4:
        raise ValueError("TC-OOS-RSG evidence requires four role classes.")
    if q0.device != q_candidate.device or q0.dtype != q_candidate.dtype:
        raise ValueError("q0 and q_candidate must share dtype and device.")
    _validate_unit_column(candidate_gate, "candidate_gate", q0)
    _validate_unit_column(ti_target_probability, "ti_target_probability", q0)
    _validate_unit_column(
        metallic_target_probability,
        "metallic_target_probability",
        q0,
    )
    columns = (candidate_gate, ti_target_probability, metallic_target_probability)
    if any(column.dtype != q0.dtype for column in columns):
        raise ValueError("Evidence columns must share the posterior dtype.")

    return torch.cat(
        [
            q0,
            q_candidate,
            (q_candidate - q0).abs(),
            candidate_gate,
            ti_target_probability,
            metallic_target_probability,
            _entropy(q0),
            _entropy(q_candidate),
            _jensen_shannon_divergence(q0, q_candidate),
        ],
        dim=1,
    )


class TargetRiskProjectionHead(nn.Module):
    """Predict a raw route from compact, inference-available risk evidence."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dims: tuple[int, int] = (64, 16),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim < 1 or len(hidden_dims) != 2 or any(width < 1 for width in hidden_dims):
            raise ValueError("input_dim and both hidden dimensions must be positive.")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        first_hidden, second_hidden = hidden_dims
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.SiLU(),
            nn.Linear(second_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 2 or evidence.shape[1] != self.input_dim:
            raise ValueError(
                f"evidence must have shape [batch, {self.input_dim}]."
            )
        if not bool(torch.isfinite(evidence).all()):
            raise ValueError("evidence must contain only finite values.")
        return self.network(evidence)


def _epsilon_column(
    epsilon_target: Any,
    reference: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.as_tensor(
        epsilon_target,
        dtype=reference.dtype,
        device=reference.device,
    )
    if epsilon.ndim == 0:
        epsilon = epsilon.reshape(1, 1).expand(reference.shape[0], 1)
    elif tuple(epsilon.shape) != (reference.shape[0], 1):
        raise ValueError("epsilon_target must be a scalar or have shape [batch, 1].")
    if not bool(torch.isfinite(epsilon).all()) or bool((epsilon < 0.0).any()):
        raise ValueError("epsilon_target must contain finite non-negative values.")
    return epsilon


def apply_target_safe_projection(
    q0: torch.Tensor,
    q_candidate: torch.Tensor,
    raw_route: torch.Tensor,
    epsilon_target: Any,
    target_index: int = 0,
    eta: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Project candidate routing onto a per-image target-posterior budget."""
    _validate_probability_matrix(q0, "q0")
    _validate_probability_matrix(q_candidate, "q_candidate")
    if q0.shape != q_candidate.shape:
        raise ValueError("q0 and q_candidate must have the same shape.")
    if q0.device != q_candidate.device or q0.dtype != q_candidate.dtype:
        raise ValueError("q0 and q_candidate must share dtype and device.")
    _validate_unit_column(raw_route, "raw_route", q0)
    if raw_route.dtype != q0.dtype:
        raise ValueError("raw_route must share the posterior dtype.")
    if not isinstance(target_index, int) or not 0 <= target_index < q0.shape[1]:
        raise ValueError("target_index must identify a posterior column.")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive.")

    epsilon = _epsilon_column(epsilon_target, q0)
    fallback_target = q0[:, target_index : target_index + 1]
    candidate_target = q_candidate[:, target_index : target_index + 1]
    target_drop = (fallback_target - candidate_target).clamp_min(0.0)
    has_positive_drop = target_drop > eta
    safe_denominator = torch.where(
        has_positive_drop,
        target_drop,
        torch.ones_like(target_drop),
    )
    positive_drop_cap = (epsilon / safe_denominator).clamp(min=0.0, max=1.0)
    route_cap = torch.where(
        has_positive_drop,
        positive_drop_cap,
        torch.ones_like(target_drop),
    )
    projected_route = torch.minimum(raw_route, route_cap)
    final_probabilities = (
        (1.0 - projected_route) * q0 + projected_route * q_candidate
    )

    signed_target_change = (
        fallback_target
        - final_probabilities[:, target_index : target_index + 1]
    )
    identity_rhs = projected_route * (fallback_target - candidate_target)
    identity_residual = signed_target_change - identity_rhs
    return {
        "route_cap": route_cap,
        "projected_route": projected_route,
        "final_probabilities": final_probabilities,
        "signed_target_change": signed_target_change,
        "positive_target_harm": signed_target_change.clamp_min(0.0),
        "identity_residual": identity_residual,
    }
