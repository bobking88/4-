"""Fixed fit-only trajectories; no model selection, refit or stop interface."""
from __future__ import annotations

import copy
import time

import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from ic_role_audit import audit_bounds, audit_initial_kernel, scalar_branch_jacobian
from ic_role_model import ICRoleModel, UNITS, _digest, objective, reference_identity
from lc_role_adapter import anchor_geometry, bounded_log_probabilities
from lc_role_data import require_classes

SEEDS = (20261002, 20261003, 20261004)
CHECKPOINTS = (0, 1, 2, 5, 20, 50, 100, 200, 400)


def _batch(batch: dict) -> None:
    if not isinstance(batch, dict):
        raise ValueError("Prepared fit or inner-validation batch is required.")
    require_classes(batch["labels"])
    anchor_geometry(batch["log_anchor"])
    n = len(batch["labels"])
    for key, width in (("h", 64), ("e", 23), ("log_anchor", 4)):
        value = batch[key]
        if (value.shape != (n, width) or value.dtype != torch.float64
                or value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("Expected aligned frozen finite CPU float64 inputs.")
    records = batch["records"]
    if (len(records) != n or len({r["image_id"] for r in records}) != n
            or any(not r.get("split_group_id") or r.get("tc_subset") == "projector_stop"
                   or int(r["four_class_id"]) != int(batch["labels"][i]) for i, r in enumerate(records))):
        raise ValueError("Fit identities required; stop records and conflicting labels forbidden.")


def _splits(train, validation):
    _batch(train)
    _batch(validation)
    for field in ("image_id", "split_group_id"):
        if {r[field] for r in train["records"]} & {r[field] for r in validation["records"]}:
            raise ValueError("Fit/inner-validation group leakage.")


def gradient_components(model: ICRoleModel, result: dict, labels: torch.Tensor) -> dict:
    named = list(model.named_current())
    losses = objective(model, result, labels)
    parameters = tuple(p for _, p in named)
    vectors = {}
    for key in ("data", "regularization", "total"):
        gradients = torch.autograd.grad(losses[key], parameters, retain_graph=key != "total",
                                        allow_unused=True)
        vectors[key] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                 for p, g in zip(parameters, gradients, strict=True)])
    if (any(not torch.isfinite(v).all() for v in vectors.values())
            or not torch.allclose(vectors["total"], vectors["data"]+vectors["regularization"],
                                  atol=1e-10, rtol=1e-10)):
        raise ValueError("Nonfinite or inconsistent gradient provenance.")
    layout, offset, norms = [], 0, {}
    for name, parameter in named:
        count = parameter.numel()
        layout.append(dict(name=name, shape=list(parameter.shape), offset=offset, numel=count))
        norms[name] = {k: float(v[offset:offset+count].norm()) for k, v in vectors.items()}
        offset += count
    return dict(vectors, layout=layout, block_norms=norms,
                norms={k: float(v.norm()) for k, v in vectors.items()},
                losses={k: float(v.detach()) for k, v in losses.items()})


@torch.no_grad()
def _branch_diagnostics(model, batch, result):
    h, e, delta, gates = batch["h"], batch["e"], result["delta"], result["g"]
    off = model(h, e, batch["log_anchor"], adaptation=False)["logq"]
    removed = []
    baseline = (model.head_ref(torch.cat([h, e], 1)) if model.head_ref is not None
                else h.new_zeros(len(h), 4))
    for j in range(3):
        kept = gates.clone()
        kept[:, j] = 0
        visual = h+(kept[:, :, None]*delta).sum(1)
        score = model.head(torch.cat([visual, e], 1))-baseline
        removed.append(bounded_log_probabilities(batch["log_anchor"], score, result["b"],
                        alpha_max=.5, beta_max=.5, bounded=True)["logq"])
    cosine = torch.stack([F.cosine_similarity(delta[:, j], delta[:, k], dim=1, eps=1e-12)
                          for j, k in ((0, 1), (0, 2), (1, 2))], 1)
    return dict(delta_norms=delta.norm(dim=2).detach().clone(), branch_cosines=cosine,
                gates=gates.detach().clone(), off_logq=off,
                removed_branch_logq=torch.stack(removed, 1),
                removed_branch_max_log_drift=torch.stack(removed, 1).sub(result["logq"][:, None]).abs().amax(2),
                ablation_scope="same state inference; no retraining")


def _checkpoint(model, train, validation, result, gradients, completed, initial_vector):
    identity = reference_identity(model)
    if not all(identity[k] for k in ("references_unaliased", "references_frozen", "references_match_initial")):
        raise ValueError("Reference identity changed during trajectory.")
    with torch.no_grad():
        val = model(validation["h"], validation["e"], validation["log_anchor"], auxiliary=True)
        bounds = {"train": audit_bounds(model, train, result),
                  "validation": audit_bounds(model, validation, val)}
        if any(a["violation_count"] for a in bounds.values()):
            raise ValueError("Trajectory certificate violation.")
        diagnostics = {"train": _branch_diagnostics(model, train, result),
                       "validation": _branch_diagnostics(model, validation, val)}
        current = torch.cat([p.flatten() for _, p in model.named_current()])
        distance = float((current-initial_vector).norm())
        val_losses = {k: float(v) for k, v in objective(model, val, validation["labels"]).items()}
    jacobians = {str(role): {view: scalar_branch_jacobian(model, train, role, view)
                           for view in ("main", "aux")} for role in (1, 2, 3)}
    return dict(completed_updates=completed, state_dict=copy.deepcopy(model.state_dict()),
                train_logq=result["logq"].detach().clone(), validation_logq=val["logq"].clone(),
                identity=identity, current_sha256=_digest(model.named_current()),
                gradient_norms=gradients["norms"], gradient_block_norms=gradients["block_norms"],
                train_losses=gradients["losses"], validation_losses=val_losses,
                parameter_norm=float(current.norm()), origin_distance=distance,
                branch_diagnostics=diagnostics, bounds=bounds, train_jacobians=jacobians,
                jacobian_scope="actual checkpoint scalar blocks; no post-init kernel identity claim")


def fit_trajectory(train: dict, validation: dict, unit: str, seed: int, *, max_updates: int = 400) -> dict:
    _splits(train, validation)
    if seed not in SEEDS or type(max_updates) is not int or not 0 <= max_updates <= 400:
        raise ValueError("Unregistered seed or update count.")
    torch.set_num_threads(1)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        model = ICRoleModel(unit, seed=seed)
        initial_vector = torch.cat([p.detach().flatten() for _, p in model.named_current()]).clone()
        initialization_seconds = time.perf_counter()-start
        kernel_start = time.perf_counter()
        initial_kernels = [audit_initial_kernel(model, train, role) for role in (1, 2, 3)]
        if any(a.get("passed") is False for a in initial_kernels):
            raise ValueError("Initial-kernel audit failure.")
        kernel_seconds = time.perf_counter()-kernel_start
        optimizer = torch.optim.Adam([p for _, p in model.named_current()], lr=.001,
                                     betas=(.9, .999), eps=1e-8, weight_decay=0)
        saved, history, vectors, update_times, checkpoint_times = {}, [], [], [], []
        for completed in range(max_updates+1):
            update_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            result = model(train["h"], train["e"], train["log_anchor"], auxiliary=True)
            gradients = gradient_components(model, result, train["labels"])
            history.append(dict(completed_updates=completed, **gradients["losses"]))
            if completed in CHECKPOINTS:
                checkpoint_start = time.perf_counter()
                saved[completed] = _checkpoint(model, train, validation, result, gradients, completed, initial_vector)
                checkpoint_times.append(time.perf_counter()-checkpoint_start)
            if completed == max_updates:
                break
            if completed < 5:
                vectors.append(dict(pre_update_index=completed,
                                    **{k: gradients[k].clone() for k in ("data", "regularization", "total")},
                                    layout=copy.deepcopy(gradients["layout"])))
            for (name, parameter), row in zip(model.named_current(), gradients["layout"], strict=True):
                if row["name"] != name:
                    raise ValueError("Gradient layout and optimizer parameters differ.")
                parameter.grad = gradients["total"][row["offset"]:row["offset"]+row["numel"]].view_as(parameter).clone()
            optimizer.step()
            if any(not torch.isfinite(p).all() for _, p in model.named_current()):
                raise ValueError("Nonfinite optimizer state.")
            update_times.append(time.perf_counter()-update_start-(checkpoint_times[-1] if completed in CHECKPOINTS else 0))
    return dict(unit=unit, seed=seed, optimizer_updates=max_updates, primary_checkpoint=100,
                checkpoints=saved, history=history, gradient_vectors=vectors, initial_kernels=initial_kernels,
                train_records=copy.deepcopy(train["records"]), validation_records=copy.deepcopy(validation["records"]),
                costs=dict(initialization_seconds=initialization_seconds, initial_kernel_seconds=kernel_seconds,
                           update_seconds=update_times, checkpoint_seconds=checkpoint_times,
                           total_seconds=time.perf_counter()-start),
                scope="fixed fit-only trajectory; truncated at 400, not convergence or independent confirmation")


def build_fit_matrix() -> list[dict]:
    return [dict(unit=unit, fold=fold, seed=seed, max_updates=400, primary_checkpoint=100,
                 stage="inner-fit-trajectory", job_id=f"{unit}_fold{fold}_seed{seed}")
            for unit in UNITS for fold in range(3) for seed in SEEDS]
