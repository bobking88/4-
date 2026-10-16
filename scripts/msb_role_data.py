"""Registered prepared-only MSB inputs; no raw loading or preprocessing."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import torch

PREPARED_BASE = "outputs/training/ic_role_mechanism_v2/development_fold_0/runs"
COUNTS = [66, 130, 97, 47]
ROW_FIELDS = ("image_id", "split_group_id", "four_class_id", "tc_subset")
ENTRY_FIELDS = {"fold", "relative_path", "sha256", "registered_train_ids_digest", "registered_validation_ids_digest"}


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_batch(batch: dict) -> int:
    if not isinstance(batch, dict):
        raise ValueError("Expected prepared batch.")
    y = batch.get("labels")
    if (not torch.is_tensor(y) or y.dtype != torch.long or y.device.type != "cpu"
            or y.ndim != 1 or not len(y) or set(y.tolist()) != {0, 1, 2, 3}):
        raise ValueError("Four populated classes and CPU long labels required.")
    n = len(y)
    for key, width in (("h", 64), ("e", 23), ("log_anchor", 4)):
        value = batch.get(key)
        if (not torch.is_tensor(value) or value.shape != (n, width)
                or value.dtype != torch.float64 or value.device.type != "cpu"
                or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f"Frozen finite CPU float64 {key} required.")
    logp = batch["log_anchor"]
    if (not torch.allclose(torch.logsumexp(logp, 1), torch.zeros(n, dtype=torch.float64),
                           atol=1e-10, rtol=0) or bool((logp.exp() == 0).any())):
        raise ValueError("NUMERIC_RANGE_FAILURE: invalid anchor.")
    records = batch.get("records")
    if not isinstance(records, list) or len(records) != n:
        raise ValueError("Prepared row identities missing.")
    for i, row in enumerate(records):
        if (not isinstance(row, dict) or type(row.get("tc_subset")) is not str
                or row["tc_subset"] != "projector_fit"
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in ("image_id", "split_group_id"))
                or str(row.get("four_class_id")) != str(int(y[i]))):
            raise ValueError("Row identity, label or fit scope mismatch.")
    for key in ("image_id", "split_group_id"):
        if len({r[key] for r in records}) != n:
            raise ValueError("Duplicate prepared row identity.")
    return n


def validate_prepared_fold(data: dict, fold: int) -> dict:
    if type(fold) is not int or fold not in (0, 1, 2) or data.get("inner_fold") != fold:
        raise ValueError("Prepared fold identity mismatch.")
    nt = validate_batch(data.get("train"))
    nv = validate_batch(data.get("validation"))
    n = nt + nv
    if n != 340:
        raise ValueError("Registered fold must cover 340 fit groups.")
    train, val = data["train"], data["validation"]
    indices = []
    for key, count in (("train_indices", nt), ("validation_indices", nv)):
        value = data.get(key)
        if (not torch.is_tensor(value) or value.dtype != torch.long or value.device.type != "cpu"
                or value.shape != (count,) or len(set(value.tolist())) != count
                or bool((value < 0).any()) or bool((value >= n).any())):
            raise ValueError("Invalid registered indices.")
        indices.extend(value.tolist())
    if sorted(indices) != list(range(n)):
        raise ValueError("Training/validation indices overlap or omit rows.")
    for key in ("image_id", "split_group_id"):
        if {r[key] for r in train["records"]} & {r[key] for r in val["records"]}:
            raise ValueError("Prepared group leakage.")
    state = data.get("state")
    if (not isinstance(state, dict) or state.get("components") != 64
            or state.get("train_ids") != [r["image_id"] for r in train["records"]]):
        raise ValueError("Prepared state train_ids mismatch.")
    if torch.bincount(torch.cat([train["labels"], val["labels"]]), minlength=4).tolist() != COUNTS:
        raise ValueError("Registered class counts mismatch.")
    return data


def validate_fold_collection(folds: list[dict]) -> dict:
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("Three prepared folds required.")
    summaries = []
    for fold, item in enumerate(folds):
        entry = dict(fold=fold, relative_path=f"{PREPARED_BASE}/prepare_fold{fold}/result.pt", sha256="0"*64)
        for part in ("train", "validation"):
            entry[f"registered_{part}_ids_digest"] = identity_digest([r["image_id"] for r in item[part]["records"]])
        summaries.append(prepared_identity_summary(item, fold, entry))
    return validate_identity_collection(summaries)


def prepared_identity_summary(data: dict, fold: int, entry: dict) -> dict:
    validate_prepared_fold(data, fold)
    result = dict(schema="msb_fold_identity_v1", fold=fold, prepared_entry=dict(entry),
                  entry_sha256=canonical_digest(entry), counts=COUNTS.copy())
    for part in ("train", "validation"):
        result[part+"_indices"] = data[part+"_indices"].tolist()
        result[part+"_records"] = [{k: str(r[k]) for k in ROW_FIELDS} for r in data[part]["records"]]
        result[part+"_ids_digest"] = identity_digest([r["image_id"] for r in result[part+"_records"]])
    _validate_identity_summary(result, fold)
    return result


def _validate_identity_summary(summary: dict, fold: int) -> None:
    required = {"schema", "fold", "prepared_entry", "entry_sha256", "counts",
                "train_indices", "validation_indices", "train_records", "validation_records",
                "train_ids_digest", "validation_ids_digest"}
    if (not isinstance(summary, dict) or set(summary) != required
            or summary["schema"] != "msb_fold_identity_v1" or type(summary["fold"]) is not int
            or summary["fold"] != fold or summary["counts"] != COUNTS):
        raise ValueError("Preflight summary schema or fold mismatch.")
    entry = summary["prepared_entry"]
    if (not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS or type(entry["fold"]) is not int
            or entry["fold"] != fold or entry["relative_path"] != f"{PREPARED_BASE}/prepare_fold{fold}/result.pt"
            or any(type(entry[k]) is not str or not re.fullmatch(r"[a-f0-9]{64}", entry[k])
                   for k in ENTRY_FIELDS if k.endswith("digest") or k == "sha256")
            or summary["entry_sha256"] != canonical_digest(entry)):
        raise ValueError("Preflight prepared registration mismatch.")
    all_indices, all_rows = [], []
    for part in ("train", "validation"):
        indices, rows = summary[part+"_indices"], summary[part+"_records"]
        if (not isinstance(indices, list) or not isinstance(rows, list) or not rows
                or len(indices) != len(rows) or any(type(i) is not int or not 0 <= i < 340 for i in indices)):
            raise ValueError("Preflight indices invalid.")
        for row in rows:
            if (not isinstance(row, dict) or set(row) != set(ROW_FIELDS)
                    or any(type(row[k]) is not str or not row[k] for k in ROW_FIELDS)
                    or row["four_class_id"] not in ("0", "1", "2", "3") or row["tc_subset"] != "projector_fit"):
                raise ValueError("Preflight row identity or fit scope invalid.")
        digest = identity_digest([r["image_id"] for r in rows])
        if digest != summary[part+"_ids_digest"] or digest != entry[f"registered_{part}_ids_digest"]:
            raise ValueError("Preflight ordered identities differ from registration.")
        all_indices.extend(indices); all_rows.extend(rows)
    if sorted(all_indices) != list(range(340)):
        raise ValueError("Preflight single-fold indices overlap or omit rows.")
    for field in ("image_id", "split_group_id"):
        if len({r[field] for r in all_rows}) != 340:
            raise ValueError("Preflight single-fold identity duplicated.")
    if [sum(r["four_class_id"] == str(k) for r in all_rows) for k in range(4)] != COUNTS:
        raise ValueError("Preflight class counts mismatch.")


def validate_identity_collection(summaries: list[dict]) -> dict:
    if not isinstance(summaries, list) or len(summaries) != 3:
        raise ValueError("Three preflight summaries required.")
    by_index = {}
    for fold, item in enumerate(summaries):
        _validate_identity_summary(item, fold)
        for index, row in zip(item["validation_indices"], item["validation_records"]):
            if index in by_index:
                raise ValueError("OOF index duplicated.")
            by_index[index] = row
    if sorted(by_index) != list(range(340)):
        raise ValueError("OOF rows incomplete.")
    for field in ("image_id", "split_group_id"):
        if len({row[field] for row in by_index.values()}) != 340:
            raise ValueError("OOF identity duplicated.")
    for item in summaries:
        for part in ("train", "validation"):
            for index, row in zip(item[part + "_indices"], item[part+"_records"]):
                if any(row[k] != by_index[index][k] for k in ROW_FIELDS):
                    raise ValueError("Cross-fold row identity/order mismatch.")
    canonical = [[i, *(by_index[i][k] for k in ROW_FIELDS)] for i in range(340)]
    return dict(groups=340, counts=COUNTS.copy(), canonical_identity_sha256=canonical_digest(canonical))


def load_prepared_fold(root: Path, entry: dict, authorization: dict) -> dict:
    if (authorization.get("human_approved") is not True
            or authorization.get("scope") != "msb_rre_v1_fit_only_once"):
        raise ValueError("New MSB authorization required before data access.")
    fold = entry.get("fold")
    if type(fold) is not int or fold not in (0, 1, 2):
        raise ValueError("Invalid prepared fold registration.")
    relative = f"{PREPARED_BASE}/prepare_fold{fold}/result.pt"
    if entry.get("relative_path") != relative:
        raise ValueError("Only registered prepared input paths are allowed.")
    root = root.resolve()
    original = root / relative
    path = original.resolve()
    if not path.is_relative_to(root) or any(p.is_symlink() for p in [original, *original.parents] if p != root):
        raise ValueError("Prepared input path escaped or is a symlink.")
    if file_digest(path) != entry.get("sha256"):
        raise ValueError("Prepared input hash mismatch before deserialization.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    validate_prepared_fold(data, fold)
    for part in ("train", "validation"):
        ids = [r["image_id"] for r in data[part]["records"]]
        if identity_digest(ids) != entry.get(f"registered_{part}_ids_digest"):
            raise ValueError("Registered ordered identity digest mismatch.")
    return data
