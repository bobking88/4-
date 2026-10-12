"""Identity-centered bounded role adapters and their matched controls."""
from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from lc_role_adapter import anchor_geometry, bounded_log_probabilities, pair_loss
from lc_role_data import require_classes

UNITS = ("Z-O-U", "Z-O-R", "Z-S-U", "Z-S-R", "C-O-U", "C-O-R",
         "C-S-U", "C-S-R", "HC-S", "FC-S", "C-S-U-noaux", "C-S-R-noaux")


def unit_spec(unit: str) -> dict:
    if unit not in UNITS:
        raise ValueError("Unknown identity-centered control unit.")
    if unit in ("HC-S", "FC-S"):
        return dict(parameterization="C", penalty="S", routing="none",
                    width=34 if unit == "FC-S" else 16, adapters=False, pair_weight=.1)
    family, penalty, routing = unit.split("-")[:3]
    return dict(parameterization=family, penalty=penalty, routing=routing,
                width=16, adapters=True, pair_weight=0. if unit.endswith("noaux") else .1)


def _matrix(value: torch.Tensor, width: int) -> None:
    if (not torch.is_tensor(value) or value.ndim != 2 or not len(value)
            or value.shape[1] != width or value.dtype != torch.float64
            or value.device.type != "cpu" or not torch.isfinite(value).all()):
        raise ValueError(f"Expected nonempty finite CPU float64 matrix [n,{width}].")


def _digest(named) -> str:
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Adapter(nn.Module):
    def __init__(self, seed: int, *, zero_upper: bool):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down = nn.Linear(64, 4, dtype=torch.float64)
            self.up = nn.Linear(4, 64, bias=False, dtype=torch.float64)
            if zero_upper:
                nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.up(F.silu(self.down(h))))


class ICRoleModel(nn.Module):
    def __init__(self, unit: str, *, seed: int):
        super().__init__()
        if type(seed) is not int:
            raise ValueError("Expected an integer initialization seed.")
        self.unit, self.seed, self.configuration = unit, seed, unit_spec(unit)
        config = self.configuration
        centered = config["parameterization"] == "C"
        self.adapters = nn.ModuleList([Adapter(seed+200+j, zero_upper=not centered)
                                       for j in range(3)] if config["adapters"] else [])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            self.head = nn.Sequential(nn.Linear(87, config["width"], dtype=torch.float64),
                                      nn.SiLU(), nn.Linear(config["width"], 4, dtype=torch.float64))
            if not centered:
                nn.init.zeros_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)
        self.adapter_refs = copy.deepcopy(self.adapters) if centered else nn.ModuleList()
        self.head_ref = copy.deepcopy(self.head) if centered else None
        for _, parameter in self.named_function_reference():
            parameter.requires_grad_(False)
        origin = (torch.cat([p.detach().flatten() for _, p in self.named_current()]).clone()
                  if not centered and config["penalty"] == "S" else torch.empty(0, dtype=torch.float64))
        self.register_buffer("regularization_origin", origin)
        self.initial_current_sha256 = _digest(self.named_current())
        self.initial_reference_sha256 = _digest(self.named_function_reference())
        self.initial_snapshot_sha256 = _digest([("origin", self.regularization_origin)])

    def named_current(self):
        for name, parameter in self.adapters.named_parameters():
            yield "adapters."+name, parameter
        for name, parameter in self.head.named_parameters():
            yield "head."+name, parameter

    def named_function_reference(self):
        for name, parameter in self.adapter_refs.named_parameters():
            yield "adapters."+name, parameter
        if self.head_ref is not None:
            for name, parameter in self.head_ref.named_parameters():
                yield "head."+name, parameter

    def forward(self, h: torch.Tensor, e: torch.Tensor, log_anchor: torch.Tensor,
                *, auxiliary: bool = False, adaptation: bool = True) -> dict:
        _matrix(h, 64)
        _matrix(e, 23)
        geometry = anchor_geometry(log_anchor)
        if len(h) != len(e) or len(h) != len(log_anchor):
            raise ValueError("Inference input rows must align.")
        if type(auxiliary) is not bool or type(adaptation) is not bool:
            raise ValueError("Expected boolean inference options.")
        if any(p.requires_grad for _, p in self.named_function_reference()):
            raise ValueError("Function references must remain frozen.")
        b = geometry["b"]
        centered = self.configuration["parameterization"] == "C"
        if self.adapters:
            current = torch.stack([branch(h) for branch in self.adapters], 1)
            if centered:
                reference = torch.stack([branch(h) for branch in self.adapter_refs], 1)
                delta = (current-reference)/16
            else:
                delta = current/math.sqrt(64)
            g = b.expand(-1, 3)/3 if self.configuration["routing"] == "U" else geometry["g"]
        else:
            delta = h.new_zeros(len(h), 3, 64)
            g = h.new_zeros(len(h), 3)
        adapted_h = h+(g[:, :, None]*delta).sum(1) if adaptation else h
        # The frozen reference sees original h, never the adapted input.
        baseline = self.head_ref(torch.cat([h, e], 1)) if centered else h.new_zeros(len(h), 4)

        def readout(visual):
            scores = self.head(torch.cat([visual, e], 1))-baseline
            probability = bounded_log_probabilities(log_anchor, scores, b,
                alpha_max=.5, beta_max=.5, bounded=True)
            return dict(probability, scores=scores)

        result = dict(readout(adapted_h), g=g, b=b, delta=delta, adapted_h=adapted_h,
                      adaptation_enabled=adaptation)
        if auxiliary:
            result["aux_logq"] = (torch.stack([readout(h+delta[:, j])["logq"] for j in range(3)], 1)
                                  if self.adapters else result["logq"][:, None, :].expand(-1, 3, -1))
        return result


def reference_identity(model: ICRoleModel) -> dict:
    current = list(model.named_current())
    fixed = list(model.named_function_reference())
    current_pointers = {p.untyped_storage().data_ptr() for _, p in current}
    fixed_pointers = [p.untyped_storage().data_ptr() for _, p in fixed]
    snapshot_count = model.regularization_origin.numel()
    unaliased = not (current_pointers & set(fixed_pointers))
    unaliased = unaliased and len(fixed_pointers) == len(set(fixed_pointers))
    if snapshot_count:
        unaliased = unaliased and model.regularization_origin.untyped_storage().data_ptr() not in current_pointers
    reference_sha = _digest(fixed)
    snapshot_sha = _digest([("origin", model.regularization_origin)])
    return dict(unit=model.unit, seed=model.seed, torch_version=str(torch.__version__),
                trainable_count=sum(p.numel() for _, p in current if p.requires_grad),
                fixed_function_count=sum(p.numel() for _, p in fixed),
                regularization_snapshot_count=snapshot_count,
                total_model_scalars=sum(p.numel() for _, p in current+fixed)+snapshot_count,
                references_unaliased=unaliased,
                references_frozen=all(not p.requires_grad for _, p in fixed),
                current_initial_sha256=model.initial_current_sha256,
                reference_initial_sha256=model.initial_reference_sha256,
                reference_current_sha256=reference_sha,
                snapshot_current_sha256=snapshot_sha,
                references_match_initial=(reference_sha == model.initial_reference_sha256
                                           and snapshot_sha == model.initial_snapshot_sha256))


def objective(model: ICRoleModel, result: dict, labels: torch.Tensor) -> dict:
    require_classes(labels)
    logq = result["logq"]
    _matrix(logq, 4)
    if len(labels) != len(logq):
        raise ValueError("Objective labels and predictions must align.")
    nll = -logq[torch.arange(len(labels)), labels].mean()
    pair = logq.new_zeros(())
    if model.configuration["pair_weight"]:
        pair = pair_loss(result["aux_logq"] if model.adapters else logq, labels)
    current = torch.cat([p.flatten() for _, p in model.named_current()])
    if model.configuration["penalty"] == "O":
        distance = current
    else:
        origin = (torch.cat([p.flatten() for _, p in model.named_function_reference()])
                  if model.head_ref is not None else model.regularization_origin)
        distance = current-origin
    regularization = .001/2*distance.square().sum()
    data = nll+model.configuration["pair_weight"]*pair
    total = data+regularization
    if not torch.isfinite(total):
        raise ValueError("Nonfinite centered-model objective.")
    return dict(nll=nll, pair=pair, data=data, regularization=regularization, total=total)
