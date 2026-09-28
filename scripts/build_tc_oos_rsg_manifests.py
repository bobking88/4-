"""Build deterministic, group-isolated TC-OOS-RSG development folds."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable


TC_PROTOCOL_VERSION = "tc_oos_rsg_v1"
INNER_SUBSET_FRACTIONS = {
    "expert_fit": 0.60,
    "expert_stop": 0.10,
    "gate_fit": 0.10,
    "gate_stop_projector_fit": 0.10,
    "projector_stop": 0.10,
}
INNER_SUBSETS = tuple(INNER_SUBSET_FRACTIONS)


def _stable_hash(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}|{group_id}".encode("utf-8")).hexdigest()


def _validate_and_group(
    rows: Iterable[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("At least one development row is required.")
    image_ids = [row.get("image_id", "").strip() for row in materialized]
    if "" in image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("image_id values must be present and unique.")
    if any(row.get("confirmation_subset") == "final_eval" for row in materialized):
        raise ValueError("Spent final_eval rows are not eligible for TC-OOS-RSG.")
    if any(row.get("split") not in (None, "", "train") for row in materialized):
        raise ValueError("Only original training rows are eligible.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in materialized:
        group_id = row.get("split_group_id", "").strip()
        if not group_id:
            raise ValueError("Every row must have a split_group_id.")
        if not row.get("mineral_label", "").strip():
            raise ValueError("Every row must have a mineral_label.")
        if not row.get("four_class_label", "").strip():
            raise ValueError("Every row must have a four_class_label.")
        grouped[group_id].append(row)

    for group_id, members in grouped.items():
        labels = {
            (row["mineral_label"], row["four_class_label"])
            for row in members
        }
        if len(labels) != 1:
            raise ValueError(
                f"split_group_id {group_id} contains multiple mineral-role labels."
            )
    return dict(grouped)


def _stratified_groups(
    groups: dict[str, list[dict[str, str]]],
) -> dict[tuple[str, str], list[str]]:
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        first = members[0]
        strata[(first["mineral_label"], first["four_class_label"])].append(group_id)
    return dict(strata)


def build_outer_folds(
    rows: list[dict[str, str]],
    seed: int = 20260927,
    fold_count: int = 3,
) -> list[dict[str, str]]:
    """Assign every intact development group to exactly one outer fold."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least two.")
    groups = _validate_and_group(rows)
    strata = _stratified_groups(groups)
    assignments: dict[str, int] = {}
    total_rows = [0 for _ in range(fold_count)]

    for _, group_ids in sorted(strata.items()):
        stratum_rows = [0 for _ in range(fold_count)]
        ordered = sorted(
            group_ids,
            key=lambda group_id: (
                -len(groups[group_id]),
                _stable_hash(seed, group_id),
                group_id,
            ),
        )
        for group_id in ordered:
            fold = min(
                range(fold_count),
                key=lambda candidate: (
                    stratum_rows[candidate],
                    total_rows[candidate],
                    candidate,
                ),
            )
            assignments[group_id] = fold
            group_size = len(groups[group_id])
            stratum_rows[fold] += group_size
            total_rows[fold] += group_size

    partitioned = []
    for group_id, members in groups.items():
        fold = str(assignments[group_id])
        for row in members:
            partitioned.append({
                **row,
                "outer_fold": fold,
                "tc_subset": "outer_eval",
                "tc_protocol_version": TC_PROTOCOL_VERSION,
            })
    return sorted(partitioned, key=lambda row: row["image_id"])


def _allocation_error(
    counts: dict[str, int],
    targets: dict[str, float],
    selected_subset: str,
    group_size: int,
) -> float:
    return sum(
        (
            counts[name]
            + (group_size if name == selected_subset else 0)
            - targets[name]
        ) ** 2
        for name in INNER_SUBSETS
    )


def build_inner_subsets(
    outer_train_rows: list[dict[str, str]],
    seed: int,
) -> list[dict[str, str]]:
    """Split one outer-training pool into five disjoint functional subsets."""
    groups = _validate_and_group(outer_train_rows)
    if any("outer_fold" not in row for members in groups.values() for row in members):
        raise ValueError("outer_train_rows must retain their outer_fold assignment.")
    strata = _stratified_groups(groups)
    assignments: dict[str, str] = {}
    global_rows = Counter({name: 0 for name in INNER_SUBSETS})

    for _, group_ids in sorted(strata.items()):
        total = sum(len(groups[group_id]) for group_id in group_ids)
        targets = {
            name: fraction * total
            for name, fraction in INNER_SUBSET_FRACTIONS.items()
        }
        counts = {name: 0 for name in INNER_SUBSETS}
        ordered = sorted(
            group_ids,
            key=lambda group_id: (
                -len(groups[group_id]),
                _stable_hash(seed, group_id),
                group_id,
            ),
        )
        for group_id in ordered:
            group_size = len(groups[group_id])
            subset = min(
                INNER_SUBSETS,
                key=lambda name: (
                    _allocation_error(counts, targets, name, group_size),
                    global_rows[name],
                    INNER_SUBSETS.index(name),
                ),
            )
            assignments[group_id] = subset
            counts[subset] += group_size
            global_rows[subset] += group_size

    partitioned = []
    for group_id, members in groups.items():
        for row in members:
            partitioned.append({
                **row,
                "tc_subset": assignments[group_id],
                "tc_protocol_version": TC_PROTOCOL_VERSION,
            })
    return sorted(partitioned, key=lambda row: row["image_id"])


def audit_tc_manifests(
    partitioned_rows: list[dict[str, str]],
) -> dict[str, object]:
    """Reject duplicate IDs, spent rows, mixed groups, and subset leakage."""
    groups = _validate_and_group(partitioned_rows)
    rows = [row for members in groups.values() for row in members]
    required = ("outer_fold", "tc_subset", "tc_protocol_version")
    for row in rows:
        missing = [field for field in required if not row.get(field, "").strip()]
        if missing:
            raise ValueError(f"TC manifest row is missing fields: {missing}")
        if row["tc_protocol_version"] != TC_PROTOCOL_VERSION:
            raise ValueError("Unexpected tc_protocol_version.")

    for group_id, members in groups.items():
        folds = {row["outer_fold"] for row in members}
        if len(folds) != 1:
            raise ValueError(f"split_group_id {group_id} spans multiple outer_fold values.")
        subsets = {row["tc_subset"] for row in members}
        if len(subsets) != 1:
            raise ValueError(f"split_group_id {group_id} spans multiple tc_subset values.")

    return {
        "row_count": len(rows),
        "group_count": len(groups),
        "outer_fold_counts": dict(sorted(Counter(row["outer_fold"] for row in rows).items())),
        "tc_subset_counts": dict(sorted(Counter(row["tc_subset"] for row in rows).items())),
        "role_counts": dict(sorted(Counter(row["four_class_label"] for row in rows).items())),
        "mineral_counts": dict(sorted(Counter(row["mineral_label"] for row in rows).items())),
        "cross_subset_group_overlap_count": 0,
        "unique_image_ids": True,
        "spent_final_eval_rows": 0,
        "protocol_version": TC_PROTOCOL_VERSION,
    }
