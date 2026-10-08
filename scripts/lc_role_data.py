"""Fit-only preparation for the approved cached LC-RFA development scope."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath

import torch
from sklearn.model_selection import StratifiedGroupKFold

from anchor_linear_controls import fit_representation, representations
from frozen_visual_probe import apply_preprocessing, fit_preprocessing as fit_visual
from run_frozen_visual_probe import tensor_digest
from run_verifier_trust_development import load_subset
from verifier_strong_controls import fit_global_temperature, validate_data

SUBSETS = ("gate_stop_projector_fit", "projector_stop")


def scoped_file(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError("Expected a workspace-relative source path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes the workspace.")
    return path


def checked_file(root: Path, path: str, expected: str) -> Path:
    result = scoped_file(root, path)
    if hashlib.sha256(result.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Source hash mismatch: {path}")
    return result


def require_classes(labels: torch.Tensor) -> None:
    if (labels.ndim != 1 or labels.dtype != torch.long or labels.device.type != "cpu"
            or set(labels.tolist()) != {0, 1, 2, 3}):
        raise ValueError("Every training and validation fold must contain all four classes.")


def load_cache_subset(root: Path, protocol: dict, subset: str) -> dict:
    if subset not in SUBSETS or protocol.get("outer_fold") != 0:
        raise ValueError("Only the registered Fold 0 subsets are permitted.")
    path = checked_file(root, protocol["cache_path"], protocol["cache_sha256"])
    audit_path = checked_file(root, protocol["audit_path"], protocol["audit_sha256"])
    source = load_subset(root, subset, protocol["input_hashes"])
    records = source["records"]
    if (len(records) != 340 or len({r["image_id"] for r in records}) != 340
            or len({r["split_group_id"] for r in records}) != 340):
        raise ValueError("Expected 340 unique image and split-group identities.")
    # Deserialization sees the container, but only the requested subset is indexed.
    cache = torch.load(path, map_location="cpu", weights_only=True)
    H = cache["fit" if subset == SUBSETS[0] else "stop"]
    if H.shape != (340, 1280) or H.dtype != torch.float32 or not torch.isfinite(H).all():
        raise ValueError("Original finite float32 feature cache contract mismatch.")
    audit = [r for r in json.loads(audit_path.read_text(encoding="utf-8")) if r["subset"] == subset]
    if len(audit) != 340:
        raise ValueError("Feature audit count mismatch.")
    for record, row, feature, label in zip(records, audit, H, source["labels"], strict=True):
        if (any(record[key] != row[key] for key in ("image_id", "relative_path", "split_group_id"))
                or int(record["four_class_id"]) != int(label)
                or ("four_class_id" in row and int(row["four_class_id"]) != int(label))
                or row["feature_sha256"] != tensor_digest(feature)):
            raise ValueError("Feature row identity, label, or original-dtype digest mismatch.")
    require_classes(source["labels"])
    validate_data(source["p"], source["labels"])
    return dict(source, H=H.to(dtype=torch.float64), hashes=dict(source["hashes"],
                feature_cache=protocol["cache_sha256"], feature_audit=protocol["audit_sha256"]))


def make_inner_assignment(raw: dict) -> list[dict]:
    labels, records = raw["labels"], raw["records"]
    require_classes(labels)
    if len(records) != len(labels) or len({r["image_id"] for r in records}) != len(records):
        raise ValueError("Invalid or duplicated row identities.")
    groups = [r["split_group_id"] for r in records]
    if len(set(groups)) < 3:
        raise ValueError("Insufficient distinct split groups.")
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261006)
    folds = [-1] * len(labels)
    for fold, (train, validation) in enumerate(splitter.split(torch.zeros(len(labels)), labels, groups)):
        require_classes(labels[train])
        require_classes(labels[validation])
        if {groups[i] for i in train} & {groups[i] for i in validation}:
            raise ValueError("Inner split group leakage.")
        for i in validation:
            if folds[i] != -1:
                raise ValueError("Duplicate inner validation row.")
            folds[i] = fold
    if -1 in folds:
        raise ValueError("Incomplete inner assignment.")
    return [dict(image_id=r["image_id"], split_group_id=r["split_group_id"],
                 inner_fold=folds[i], class_id=int(labels[i])) for i, r in enumerate(records)]


def _indices(raw: dict, indices: torch.Tensor | None) -> torch.Tensor:
    if indices is None:
        return torch.arange(len(raw["p"]))
    if (indices.ndim != 1 or indices.dtype != torch.long or indices.device.type != "cpu"
            or not len(indices) or len(set(indices.tolist())) != len(indices)
            or bool((indices < 0).any()) or bool((indices >= len(raw["p"])).any())):
        raise ValueError("Indices must be unique in-range CPU long values.")
    return indices


def fit_preprocessing(raw: dict, train_indices: torch.Tensor) -> dict:
    indices = _indices(raw, train_indices)
    p, labels = raw["p"][indices], raw["labels"][indices]
    require_classes(labels)
    representation = fit_representation(raw["evidence"][indices], p, raw["contradiction"][indices])
    e23 = torch.cat([raw["evidence"][indices], p.log(), raw["contradiction"][indices]], 1)
    visual = fit_visual(e23, raw["H"][indices], 64)
    temperature = fit_global_temperature(p, labels, [1e-6, 100.])
    return dict(visual, representation=representation, temperature=temperature,
                train_ids=[raw["records"][i]["image_id"] for i in indices.tolist()],
                conditional_on_frozen_expert=True, components=64)


def transform_subset(raw: dict, state: dict, indices: torch.Tensor | None = None) -> dict:
    indices = _indices(raw, indices)
    p, H = raw["p"][indices], raw["H"][indices]
    e18, L = raw["evidence"][indices], raw["contradiction"][indices]
    validate_data(p)
    e23 = torch.cat([e18, p.log(), L], 1)
    visual = apply_preprocessing(e23, H, state)
    blocks = representations(e18, p, L, visual["H"], state["representation"])
    log_anchor = torch.log_softmax(p.log() * state["temperature"]["beta"], 1)
    if not torch.isfinite(log_anchor).all() or bool((log_anchor.exp() == 0).any()):
        raise ValueError("NUMERIC_RANGE_FAILURE: temperature anchor underflow.")
    return dict(h=visual["H"], e=visual["E"], P=blocks["P"], log_anchor=log_anchor,
                labels=raw["labels"][indices], records=[raw["records"][i] for i in indices.tolist()])
