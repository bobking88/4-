"""Reproducible feasibility probes, not a proposed production architecture."""

from __future__ import annotations

import copy
from pathlib import Path, PureWindowsPath

import torch
import torch.nn.functional as F
from torch import nn

from run_tc_oos_rsg_experiments import calculate_probability_metrics, state_dict_sha256


ARMS = ("E", "H", "EH", "EH_permuted")
ARCHITECTURES = ("linear", "mlp")


def finite_matrix(value: torch.Tensor) -> None:
    if value.ndim != 2 or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise ValueError("Expected finite float64 feature matrix.")


def fit_preprocessing(e: torch.Tensor, h: torch.Tensor, components: int) -> dict:
    finite_matrix(e)
    finite_matrix(h)
    if len(e) != len(h) or len(e) < 2 or not 1 <= components <= min(len(h)-1, h.shape[1]):
        raise ValueError("Aligned fit-only features and a feasible PCA rank are required.")
    mean_h = h.mean(0)
    _, singular, vh = torch.linalg.svd(h - mean_h, full_matrices=False)
    basis = vh[:components].T.contiguous()
    # Fix the otherwise arbitrary sign of each principal direction.
    pivots = basis.abs().argmax(0)
    signs = basis[pivots, torch.arange(components)].sign()
    basis = basis * signs
    projected = (h - mean_h) @ basis
    return {"e_mean": e.mean(0), "e_scale": e.std(0, correction=0).clamp_min(1e-8),
            "h_mean": mean_h, "basis": basis, "projected_mean": projected.mean(0),
            "projected_scale": projected.std(0, correction=0).clamp_min(1e-8),
            "retained_variance": float(singular[:components].square().sum() / singular.square().sum().clamp_min(1e-30))}


def apply_preprocessing(e: torch.Tensor, h: torch.Tensor, state: dict) -> dict:
    finite_matrix(e)
    finite_matrix(h)
    if len(e) != len(h) or e.shape[1] != len(state["e_mean"]) or h.shape[1] != len(state["h_mean"]):
        raise ValueError("Features do not match the fit-only preprocessing state.")
    return {"E": (e - state["e_mean"]) / state["e_scale"],
            "H": ((h - state["h_mean"]) @ state["basis"] - state["projected_mean"]) / state["projected_scale"]}


def make_input(blocks: dict, arm: str, permutation_seed: int) -> torch.Tensor:
    if arm not in ARMS:
        raise ValueError("Unknown probe arm.")
    e, h = blocks["E"], blocks["H"]
    finite_matrix(e)
    finite_matrix(h)
    if len(e) != len(h):
        raise ValueError("Feature blocks must be aligned.")
    if arm == "E":
        h = torch.zeros_like(h)
    elif arm == "H":
        e = torch.zeros_like(e)
    elif arm == "EH_permuted":
        generator = torch.Generator().manual_seed(permutation_seed)
        h = h[torch.randperm(len(h), generator=generator)]
    return torch.cat([e, h], 1)


def transition_counts(q: torch.Tensor, p: torch.Tensor, labels: torch.Tensor) -> dict:
    new, old = q.argmax(1), p.argmax(1)
    promoted, removed = (new == 0) & (old != 0), (new != 0) & (old == 0)
    gained = int((promoted & (labels == 0)).sum())
    lost = int((removed & (labels == 0)).sum())
    return {"classification_changes": int((new != old).sum()), "new_correct_targets": gained,
            "lost_correct_targets": lost, "net_correct_targets": gained-lost,
            "correct_target_count": int(((labels == 0) & (new == 0)).sum()),
            "target_count": int((labels == 0).sum()),
            "new_false_target_by_class": {str(k): int((promoted & (labels == k)).sum()) for k in (1, 2, 3)},
            "removed_false_target_by_class": {str(k): int((removed & (labels == k)).sum()) for k in (1, 2, 3)},
            "false_target_count_by_class": {str(k): int(((new == 0) & (labels == k)).sum()) for k in (1, 2, 3)}}


def freeze_extractor(model: nn.Module) -> nn.Module:
    return model.eval().requires_grad_(False)


def extract_feature_state(checkpoint: dict) -> dict:
    state = {key.removeprefix("features."): value for key, value in checkpoint["model_state_dict"].items()
             if key.startswith("features.")}
    if not state:
        raise ValueError("Checkpoint has no registered visual feature weights.")
    return state


def scoped_image_path(dataset_root: Path, relative: str) -> Path:
    root = dataset_root.resolve()
    if Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError("Absolute image path is outside the scoped manifest contract.")
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path escapes the scoped dataset root.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_classifier(dimension: int, architecture: str, budget: dict) -> nn.Module:
    if architecture == "linear":
        head = nn.Sequential(nn.Linear(dimension, 4))
    elif architecture == "mlp":
        head = nn.Sequential(nn.Linear(dimension, budget["hidden_dim"]), nn.SiLU(),
                             nn.Dropout(budget["dropout"]), nn.Linear(budget["hidden_dim"], 4))
    else:
        raise ValueError("Only registered simple heads may be built.")
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head.double()


def predict_classifier(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    head.eval()
    with torch.no_grad():
        return torch.softmax(head(features), 1)


def fit_classifier(x_fit: torch.Tensor, y_fit: torch.Tensor, x_stop: torch.Tensor,
                   y_stop: torch.Tensor, architecture: str, seed: int, budget: dict) -> dict:
    for x, y in ((x_fit, y_fit), (x_stop, y_stop)):
        finite_matrix(x)
        if y.shape != (len(x),) or y.dtype != torch.long or bool(((y < 0) | (y >= 4)).any()):
            raise ValueError("Aligned four-class integer labels are required.")
    if x_fit.shape[1] != x_stop.shape[1] or budget["epochs"] < 1 or budget["batch_size"] < 1:
        raise ValueError("Invalid dimensions or training budget.")
    torch.manual_seed(seed)
    head = build_classifier(x_fit.shape[1], architecture, budget)
    initial_sha = state_dict_sha256(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=budget["learning_rate"], weight_decay=budget["weight_decay"])
    history, best = [], None
    for epoch in range(1, budget["epochs"]+1):
        head.train()
        for index in torch.randperm(len(y_fit)).split(budget["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x_fit[index]), y_fit[index])
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite probe training loss.")
            loss.backward()
            if any(not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
                raise ValueError("Nonfinite probe training gradient.")
            optimizer.step()
        fit_metrics = calculate_probability_metrics(predict_classifier(head, x_fit), y_fit)
        stop_metrics = calculate_probability_metrics(predict_classifier(head, x_stop), y_stop)
        history.append({"epoch": epoch, "fit_nll": fit_metrics["nll"], "stop_nll": stop_metrics["nll"], "stop_metrics": stop_metrics})
        if best is None or stop_metrics["nll"] < best["nll"]:
            best = {"nll": stop_metrics["nll"], "epoch": epoch, "state": copy.deepcopy(head.state_dict())}
    head.load_state_dict(best["state"])
    return {"head": head, "state_dict": best["state"], "initial_sha256": initial_sha,
            "state_sha256": state_dict_sha256(best["state"]), "history": history,
            "best_epoch": best["epoch"], "parameter_count": sum(p.numel() for p in head.parameters())}
