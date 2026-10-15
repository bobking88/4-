"""Centered role evidence and matched ordinary heads, all in float64."""
from __future__ import annotations

import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

ROLE_UNITS = ("MAIN", "ADD", "FIX1", "UNB-SCALE", "NOAUX")
NEURAL_UNITS = ("MAIN", "SHARED-BIL", "CONCAT-MLP", "ADD", "FIX1", "UNB-SCALE", "NOAUX")
CONVEX_UNITS = ("LIN-UNB", "BIAS3-CVX", "CLASSWISE4", "REPAIR4-COMP")


def unit_spec(unit: str) -> dict:
    counts = dict(zip(NEURAL_UNITS, (1083, 1107, 1095, 1083, 1083, 1083, 1083)))
    counts.update({"LIN-UNB": 264, "BIAS3-CVX": 3, "CLASSWISE4": 3, "REPAIR4-COMP": 7})
    if unit not in counts:
        raise ValueError("Unknown MSB unit.")
    return dict(unit=unit, parameters=counts[unit], rank=4, gamma=2., tau=.25,
                pair_weight=0. if unit == "NOAUX" else .1,
                relative_weight=.01, sp_weight=.001,
                convex=unit in CONVEX_UNITS)


def _finite(value, message):
    if not torch.isfinite(value).all():
        raise ValueError("NUMERIC_RANGE_FAILURE: " + message)
    return value


def anchor_margins(log_anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (not torch.is_tensor(log_anchor) or log_anchor.ndim != 2
            or log_anchor.shape[1] != 4 or not len(log_anchor)
            or log_anchor.dtype != torch.float64 or log_anchor.device.type != "cpu"
            or log_anchor.requires_grad):
        raise ValueError("Frozen CPU float64 four-class log anchor required.")
    _finite(log_anchor, "anchor nonfinite")
    if (not torch.allclose(torch.logsumexp(log_anchor, 1), torch.zeros(len(log_anchor), dtype=torch.float64),
                           atol=1e-10, rtol=0.) or bool((log_anchor.exp() == 0).any())):
        raise ValueError("NUMERIC_RANGE_FAILURE: anchor normalization or underflow")
    m = log_anchor[:, :1] - log_anchor[:, 1:]
    return _finite(m, "margin overflow"), _finite(2 * m.abs() + .25, "radius overflow")


def decode_role(log_anchor: torch.Tensor, d: torch.Tensor) -> dict:
    m, radius = anchor_margins(log_anchor)
    if d.shape != m.shape or d.dtype != torch.float64 or d.device.type != "cpu":
        raise ValueError("Three float64 correction coordinates required.")
    _finite(d, "correction nonfinite")
    scores = _finite(torch.cat([log_anchor[:, :1], log_anchor[:, 1:] - d], 1), "score overflow")
    logq = _finite(torch.log_softmax(scores, 1), "normalizer overflow")
    if bool((logq.exp() == 0).any()):
        raise ValueError("NUMERIC_RANGE_FAILURE: output probability underflow")
    return dict(logq=logq, scores=scores, m=m, R_ref=radius, d=d)


def _draw(shape, generator, fan_in):
    return (2 * torch.rand(shape, generator=generator, dtype=torch.float64) - 1) / math.sqrt(fan_in)


class _Referenced(nn.Module):
    def snapshot(self):
        for name, parameter in list(self.named_parameters(recurse=False)):
            self.register_buffer(name + "0", parameter.detach().clone())


class RoleBranch(_Referenced):
    def __init__(self, seed: int, *, additive: bool = False):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.U = nn.Parameter(_draw((4, 64), generator, 64))
        self.u = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.V = nn.Parameter(_draw((4, 23), generator, 23))
        self.v = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.a = nn.Parameter(_draw((4,), generator, 4))
        self.zeta = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.additive = additive
        self.snapshot()

    def _score(self, h, e, suffix):
        p = lambda name: getattr(self, name + suffix)
        left = _finite(F.linear(h, p("U"), p("u")), "visual projection overflow").tanh()
        right = _finite(F.linear(e, p("V"), p("v")), "evidence projection overflow").tanh()
        interaction = left + right if self.additive else left * right
        return _finite(interaction @ p("a") / 2 + p("zeta"), "branch score overflow")

    def score(self, h, e):
        return self._score(h, e, "")

    def fixed_score(self, h, e):
        return self._score(h, e, "0")


class _SharedHead(_Referenced):
    def __init__(self, seed):
        super().__init__()
        branches = [RoleBranch(seed + 200 + j) for j in range(3)]
        for name in ("U", "u", "V", "v"):
            self.register_parameter(name, nn.Parameter(torch.cat([getattr(b, name).detach() for b in branches]).clone()))
        matrix = torch.zeros(3, 12, dtype=torch.float64)
        for j, b in enumerate(branches):
            matrix[j, j*4:(j+1)*4] = b.a.detach()
        self.A = nn.Parameter(matrix)
        self.zeta = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.snapshot()

    def score(self, h, e, suffix=""):
        p = lambda name: getattr(self, name + suffix)
        left = _finite(F.linear(h, p("U"), p("u")), "shared visual overflow").tanh()
        right = _finite(F.linear(e, p("V"), p("v")), "shared evidence overflow").tanh()
        return _finite(F.linear(left * right, p("A")) / 2 + p("zeta"), "shared score overflow")


class _ConcatHead(_Referenced):
    def __init__(self, seed, *, linear=False):
        super().__init__()
        generator = torch.Generator().manual_seed(seed + 100)
        self.linear = linear
        self.W = nn.Parameter(_draw((3 if linear else 12, 87), generator, 87))
        self.b = nn.Parameter(torch.zeros(3 if linear else 12, dtype=torch.float64))
        if not linear:
            self.A = nn.Parameter(_draw((3, 12), generator, 12))
            self.zeta = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.snapshot()

    def score(self, h, e, suffix=""):
        p = lambda name: getattr(self, name + suffix)
        x = torch.cat([h, e], 1)
        first = _finite(F.linear(x, p("W"), p("b")), "concat projection overflow")
        if self.linear:
            return first
        return _finite(F.linear(first.tanh(), p("A"), p("zeta")), "concat score overflow")


class MSBRoleModel(nn.Module):
    def __init__(self, unit: str, *, seed: int):
        super().__init__()
        self.settings = unit_spec(unit)
        if unit not in NEURAL_UNITS + ("LIN-UNB",) or type(seed) is not int:
            raise ValueError("Expected a centered neural or linear MSB model.")
        self.unit, self.seed = unit, seed
        if unit in ROLE_UNITS:
            self.branches = nn.ModuleList([RoleBranch(seed + 200 + j, additive=unit == "ADD") for j in range(3)])
        elif unit == "SHARED-BIL":
            self.head = _SharedHead(seed)
        else:
            self.head = _ConcatHead(seed, linear=unit == "LIN-UNB")

    def named_current(self):
        return [(name, p) for name, p in self.named_parameters() if p.requires_grad]

    def initial_parameter(self, name):
        parent, _, field = name.rpartition(".")
        return getattr(self.get_submodule(parent), field + "0")

    def forward(self, h, e, log_anchor, *, role_mask=(True, True, True)):
        m, radius = anchor_margins(log_anchor)
        for value, width in ((h, 64), (e, 23)):
            if (value.shape != (len(log_anchor), width) or value.dtype != torch.float64
                    or value.device.type != "cpu" or value.requires_grad):
                raise ValueError("Frozen float64 h64/e23 required.")
            _finite(value, "input nonfinite")
        if (not isinstance(role_mask, tuple) or len(role_mask) != 3
                or any(type(x) is not bool for x in role_mask)
                or (self.unit not in ROLE_UNITS and role_mask != (True, True, True))):
            raise ValueError("Invalid same-state role mask.")
        if self.unit in ROLE_UNITS:
            s = torch.stack([b.score(h, e) - b.fixed_score(h, e) for b in self.branches], 1)
        elif self.unit == "LIN-UNB":
            s = F.linear(torch.cat([h, e], 1), self.head.W - self.head.W0, self.head.b - self.head.b0)
        else:
            s = self.head.score(h, e) - self.head.score(h, e, "0")
        _finite(s, "centered score overflow")
        unbounded = self.unit in ("UNB-SCALE", "LIN-UNB")
        c = s if unbounded else s.tanh()
        output_radius = torch.ones_like(radius) if self.unit == "FIX1" else radius
        d = _finite(output_radius * c, "scaled correction overflow")
        d = d * torch.tensor(role_mask, dtype=torch.float64)
        result = decode_role(log_anchor, d)
        result.update(s=s, c="NOT_APPLICABLE" if unbounded else c,
                      saturation_count=0 if unbounded else int((c.abs() == 1).sum()), unit=self.unit)
        return result


def reference_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_buffers():
        digest.update(name.encode("utf-8"))
        digest.update(str((value.shape, value.dtype)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
