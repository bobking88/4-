"""LC-RFA-B feature adapters and shared bounded probability correction."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from lc_role_data import require_classes

ARMS = ("H0", "F0", "S0", "S1", "R0", "R1", "U1")


class NumericRangeFailure(ValueError):
    def __init__(self, logq: torch.Tensor):
        self.logq = logq.detach().clone()
        super().__init__("NUMERIC_RANGE_FAILURE: finite logq underflows in float64 probabilities.")


def _matrix(value: torch.Tensor, columns: int) -> None:
    if (not torch.is_tensor(value) or value.ndim != 2 or len(value) == 0
            or value.shape[1] != columns or value.dtype != torch.float64
            or value.device.type != "cpu" or not torch.isfinite(value).all()):
        raise ValueError(f"Expected nonempty finite CPU float64 matrix with {columns} columns.")


def _budget(alpha_max: float, beta_max: float) -> None:
    if not all(math.isfinite(v) and v >= 0 for v in (alpha_max, beta_max)):
        raise ValueError("Budget maxima must be finite and nonnegative.")


def anchor_geometry(log_anchor: torch.Tensor) -> dict:
    _matrix(log_anchor, 4)
    if not torch.allclose(torch.logsumexp(log_anchor, 1), torch.zeros(len(log_anchor), dtype=torch.float64), atol=1e-12, rtol=0):
        raise ValueError("Anchor must be a normalized log probability distribution.")
    if bool((log_anchor.exp() == 0).any()):
        raise NumericRangeFailure(log_anchor)
    log_non_target = torch.logsumexp(log_anchor[:, 1:], 1, keepdim=True)
    g = 4 * torch.exp(log_anchor[:, :1] + log_anchor[:, 1:])
    return dict(logw=log_anchor[:, 1:] - log_non_target,
                logit_t=log_anchor[:, :1] - log_non_target,
                g=g, b=g.sum(1, keepdim=True))


def bounded_log_probabilities(log_anchor: torch.Tensor, scores: torch.Tensor,
                              b: torch.Tensor, *, alpha_max: float, beta_max: float,
                              bounded: bool = True) -> dict:
    geometry = anchor_geometry(log_anchor)
    _matrix(scores, 4)
    _matrix(b, 1)
    _budget(alpha_max, beta_max)
    if len(scores) != len(log_anchor) or len(b) != len(scores) or bool(((b < 0) | (b > 1+1e-12)).any()):
        raise ValueError("Scores and uncertainty budgets must align and b must be in [0,1].")
    alpha, beta = alpha_max * b, beta_max * b
    u = alpha * torch.tanh(scores[:, :1]) if bounded else scores[:, :1]
    v = beta / 2 * torch.tanh(scores[:, 1:]) if bounded else scores[:, 1:]
    odds = geometry["logit_t"] + u
    logq = torch.cat([F.logsigmoid(odds),
                      F.logsigmoid(-odds) + torch.log_softmax(geometry["logw"] + v, 1)], 1)
    if not torch.isfinite(logq).all():
        raise ValueError("Nonfinite log probabilities.")
    if bool((logq.exp() == 0).any()):
        raise NumericRangeFailure(logq)
    return dict(logq=logq, u=u, v=v, alpha=alpha, beta=beta,
                logw=geometry["logw"], logit_t=geometry["logit_t"])


class BottleneckAdapter(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down = nn.Linear(64, 4, dtype=torch.float64)
            self.up = nn.Linear(4, 64, bias=False, dtype=torch.float64)
            nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.up(F.silu(self.down(h)))) / math.sqrt(64)


class RoleResidualModel(nn.Module):
    def __init__(self, arm: str, *, seed: int, alpha_max: float, beta_max: float):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("Unknown LC-RFA arm.")
        _budget(alpha_max, beta_max)
        self.arm, self.seed = arm, seed
        self.alpha_max, self.beta_max = alpha_max, beta_max
        self.adapters = nn.ModuleList([] if arm in ("H0", "F0") else
                                      [BottleneckAdapter(seed+200+j) for j in range(3)])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            width = 34 if arm == "F0" else 16
            self.head = nn.Sequential(nn.Linear(87, width, dtype=torch.float64), nn.SiLU(),
                                      nn.Linear(width, 4, dtype=torch.float64))
            nn.init.zeros_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)

    def readout(self, h: torch.Tensor, e: torch.Tensor, log_anchor: torch.Tensor,
                b: torch.Tensor) -> dict:
        scores = self.head(torch.cat([h, e], 1))
        output = bounded_log_probabilities(log_anchor, scores, b, alpha_max=self.alpha_max,
                                          beta_max=self.beta_max, bounded=self.arm != "U1")
        return dict(output, scores=scores)

    def forward(self, h: torch.Tensor, e: torch.Tensor, log_anchor: torch.Tensor,
                *, auxiliary: bool = False) -> dict:
        _matrix(h, 64)
        _matrix(e, 23)
        if len(h) != len(e) or len(h) != len(log_anchor):
            raise ValueError("Aligned inference-only inputs required.")
        geometry = anchor_geometry(log_anchor)
        b = geometry["b"]
        if not len(self.adapters):
            delta = torch.zeros(len(h), 3, 64, dtype=h.dtype)
            g = torch.zeros_like(geometry["g"])
        else:
            delta = torch.stack([branch(h) for branch in self.adapters], 1)
            g = b.expand(-1, 3) / 3 if self.arm in ("S0", "S1") else geometry["g"]
        adapted_h = h + (g[:, :, None] * delta).sum(1)
        result = dict(self.readout(adapted_h, e, log_anchor, b),
                      g=g, b=b, delta=delta, adapted_h=adapted_h)
        if auxiliary and self.arm in ("S1", "R1", "U1"):
            result["aux_logq"] = torch.stack([self.readout(h+delta[:, j], e, log_anchor, b)["logq"]
                                               for j in range(3)], 1)
        return result


def pair_loss(logq: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    require_classes(labels)
    if logq.ndim == 2:
        _matrix(logq, 4)
        views = logq[:, None, :].expand(-1, 3, -1)
    elif (logq.ndim == 3 and logq.shape[1:] == (3, 4) and logq.dtype == torch.float64
          and logq.device.type == "cpu" and torch.isfinite(logq).all()):
        views = logq
    else:
        raise ValueError("Expected four-class main or three auxiliary log probability views.")
    if len(views) != len(labels):
        raise ValueError("Pair loss labels must align.")
    losses = []
    for j in range(1, 4):
        mask = (labels == 0) | (labels == j)
        pair = views[mask, j-1][:, [0, j]]
        pair = pair - torch.logsumexp(pair, 1, keepdim=True)
        losses.append(-pair[torch.arange(len(pair)), (labels[mask] != 0).long()].mean())
    return torch.stack(losses).mean()


def objective(model: RoleResidualModel, result: dict, labels: torch.Tensor,
              regularization: float) -> dict:
    require_classes(labels)
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Positive finite explicit regularization required.")
    logq = result["logq"]
    _matrix(logq, 4)
    if len(logq) != len(labels):
        raise ValueError("Objective labels must align.")
    nll = -logq[torch.arange(len(labels)), labels].mean()
    pair = torch.zeros((), dtype=logq.dtype)
    if model.arm == "F0":
        pair = pair_loss(logq, labels)
    elif model.arm in ("S1", "R1", "U1"):
        pair = pair_loss(result["aux_logq"], labels)
    l2 = regularization / 2 * sum(p.square().sum() for p in model.parameters())
    total = nll + .10 * pair + l2
    if not torch.isfinite(total):
        raise ValueError("Nonfinite objective.")
    return dict(total=total, nll=nll, pair=pair, l2=l2)
