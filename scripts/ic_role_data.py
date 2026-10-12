"""Isolated fit-only inputs and training-fold preparation for IC-RFA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from lc_role_data import fit_preprocessing, require_classes, transform_subset
from verifier_strong_controls import validate_data

FIT_PATH = "outputs/training/lc_role_adaptation_v2/development_fold_0/fit_raw.pt"
ASSIGNMENT_PATH = "outputs/training/lc_role_adaptation_v2/development_fold_0/inner_assignment.json"
FIT_SHA = "d1edbe1f87977ce0a91639bc8cb455708d4c2e17d53340c48d11d2e8b2a568b1"
ASSIGNMENT_SHA = "e603ba0b603bc903f82649dd19edaebb69f7e171be0dec2092d69048c5469927"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_raw(raw: dict) -> int:
    if not isinstance(raw, dict):
        raise ValueError("Expected a RawFit dictionary.")
    required = {"p", "H", "evidence", "contradiction", "labels", "records", "hashes"}
    if not required.issubset(raw):
        raise ValueError("Incomplete RawFit fields.")
    labels = raw["labels"]
    if not torch.is_tensor(labels):
        raise ValueError("Expected CPU long labels.")
    require_classes(labels)
    n = len(labels)
    for key, width in (("p", 4), ("H", 1280), ("evidence", 18), ("contradiction", 1)):
        value = raw[key]
        if (not torch.is_tensor(value) or value.shape != (n, width)
                or value.dtype != torch.float64 or value.device.type != "cpu"
                or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f"Expected frozen finite CPU float64 {key}[n,{width}].")
    validate_data(raw["p"], labels)
    if not isinstance(raw["hashes"], dict) or not isinstance(raw["records"], list):
        raise ValueError("Invalid RawFit identity metadata.")
    records = raw["records"]
    if len(records) != n:
        raise ValueError("Row identity count mismatch.")
    for i, record in enumerate(records):
        if (not isinstance(record, dict)
                or any(not isinstance(record.get(key), str) or not record[key]
                       for key in ("image_id", "split_group_id", "relative_path"))
                or str(record.get("four_class_id")) != str(int(labels[i]))):
            raise ValueError("Row identity or label mismatch.")
    for key in ("image_id", "split_group_id"):
        if len({r[key] for r in records}) != n:
            raise ValueError(f"Duplicate {key} in the unique fit-group scope.")
    return n


def validate_assignment(raw: dict, assignment: list[dict]) -> list[dict]:
    n = _validate_raw(raw)
    if not isinstance(assignment, list) or len(assignment) != n:
        raise ValueError("Assignment must cover every fit row exactly once.")
    for i, row in enumerate(assignment):
        record = raw["records"][i]
        if (not isinstance(row, dict)
                or row.get("image_id") != record["image_id"]
                or row.get("split_group_id") != record["split_group_id"]
                or type(row.get("class_id")) is not int
                or row["class_id"] != int(raw["labels"][i])
                or type(row.get("inner_fold")) is not int
                or row["inner_fold"] not in (0, 1, 2)):
            raise ValueError("Ordered assignment identity, label or fold mismatch.")
    for fold in range(3):
        validation = torch.tensor([i for i, row in enumerate(assignment)
                                   if row["inner_fold"] == fold], dtype=torch.long)
        train = torch.tensor([i for i, row in enumerate(assignment)
                              if row["inner_fold"] != fold], dtype=torch.long)
        require_classes(raw["labels"][train])
        require_classes(raw["labels"][validation])
        if ({raw["records"][i]["split_group_id"] for i in train.tolist()}
                & {raw["records"][i]["split_group_id"] for i in validation.tolist()}):
            raise ValueError("Assignment split-group leakage.")
    return assignment


def load_fit_bundle(root: Path) -> tuple[dict, list[dict]]:
    root = root.resolve()
    paths = [(root / relative).resolve() for relative in (FIT_PATH, ASSIGNMENT_PATH)]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Fixed fit input path escapes the workspace.")
    # Validate both artifacts before deserializing either input container.
    for path, expected in zip(paths, (FIT_SHA, ASSIGNMENT_SHA), strict=True):
        if file_digest(path) != expected:
            raise ValueError(f"Input hash mismatch: {path.name}")
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"rows"}:
        raise ValueError("Assignment container must contain its rows list.")
    raw = torch.load(paths[0], map_location="cpu", weights_only=True)
    n = _validate_raw(raw)
    if n != 340 or torch.bincount(raw["labels"], minlength=4).tolist() != [66, 130, 97, 47]:
        raise ValueError("Registered 340-group fit class counts do not match.")
    rows = validate_assignment(raw, payload["rows"])
    return dict(raw, hashes=dict(raw["hashes"], fit_bundle=FIT_SHA,
                                inner_assignment=ASSIGNMENT_SHA)), rows


def prepare_inner_fold(raw: dict, assignment: list[dict], fold: int) -> dict:
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("Expected one of the three existing inner folds.")
    rows = validate_assignment(raw, assignment)
    train_indices = torch.tensor([i for i, r in enumerate(rows) if r["inner_fold"] != fold],
                                 dtype=torch.long)
    validation_indices = torch.tensor([i for i, r in enumerate(rows) if r["inner_fold"] == fold],
                                      dtype=torch.long)
    state = fit_preprocessing(raw, train_indices)
    expected = [raw["records"][i]["image_id"] for i in train_indices.tolist()]
    if state.get("train_ids") != expected or state.get("components") != 64:
        raise ValueError("Preprocessing training row identity mismatch.")
    return dict(state=state, train=transform_subset(raw, state, train_indices),
                validation=transform_subset(raw, state, validation_indices),
                train_indices=train_indices, validation_indices=validation_indices,
                inner_fold=fold, conditional_on_frozen_expert=True)
