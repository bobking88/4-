"""Build deterministic, group-isolated TC-OOS-RSG development folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Sequence


TC_PROTOCOL_VERSION = "tc_oos_rsg_v1"
INNER_SUBSET_FRACTIONS = {
    "expert_fit": 0.60,
    "expert_stop": 0.10,
    "gate_fit": 0.10,
    "gate_stop_projector_fit": 0.10,
    "projector_stop": 0.10,
}
INNER_SUBSETS = tuple(INNER_SUBSET_FRACTIONS)
EPSILON_TARGET_GRID = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["image_id"]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register deterministic TC-OOS-RSG outer and inner manifests."
    )
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-document", type=Path, required=True)
    parser.add_argument("--outer-seed", type=int, default=20260927)
    parser.add_argument("--fold-count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite an existing TC-OOS-RSG protocol directory.")
    if not args.protocol_document.is_file():
        raise ValueError("The pre-registered protocol document does not exist.")

    source_bytes = args.partition.read_bytes()
    with args.partition.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Source partition has no header.")
        source_fields = list(reader.fieldnames)
        source_rows = list(reader)
    spent_rows = [
        row for row in source_rows if row.get("confirmation_subset") == "final_eval"
    ]
    development_rows = [
        row for row in source_rows if row.get("confirmation_subset") != "final_eval"
    ]
    spent_ids = {row["image_id"] for row in spent_rows}
    development_ids = {row["image_id"] for row in development_rows}
    if spent_ids & development_ids:
        raise ValueError("Spent final_eval image IDs overlap the development pool.")

    outer_partition = build_outer_folds(
        development_rows,
        seed=args.outer_seed,
        fold_count=args.fold_count,
    )
    extended_fields = [
        *source_fields,
        "outer_fold",
        "tc_subset",
        "tc_protocol_version",
    ]
    args.output_dir.mkdir(parents=True)
    partition_path = args.output_dir / "partition.csv"
    _write_csv(partition_path, outer_partition, extended_fields)

    manifest_paths = [partition_path]
    manifest_counts = {"partition.csv": len(outer_partition)}
    fold_audits: dict[str, object] = {}
    for fold in range(args.fold_count):
        fold_name = f"fold_{fold}"
        eval_rows = [
            {
                **row,
                "outer_fold": str(fold),
                "tc_subset": "outer_eval",
            }
            for row in outer_partition
            if row["outer_fold"] == str(fold)
        ]
        train_pool = [
            row for row in outer_partition if row["outer_fold"] != str(fold)
        ]
        inner_rows = build_inner_subsets(train_pool, seed=args.outer_seed + fold)
        inner_rows = [{**row, "outer_fold": str(fold)} for row in inner_rows]
        combined = [*eval_rows, *inner_rows]
        if len(combined) != len(outer_partition):
            raise AssertionError("Each fold must cover the development pool exactly once.")
        fold_audits[fold_name] = audit_tc_manifests(combined)

        by_subset = {
            "outer_eval": eval_rows,
            **{
                subset: [row for row in inner_rows if row["tc_subset"] == subset]
                for subset in INNER_SUBSETS
            },
        }
        for subset, subset_rows in by_subset.items():
            if not subset_rows:
                raise ValueError(f"{fold_name}/{subset} is empty.")
            path = args.output_dir / fold_name / f"{subset}.csv"
            _write_csv(path, subset_rows, extended_fields)
            manifest_paths.append(path)
            manifest_counts[path.relative_to(args.output_dir).as_posix()] = len(subset_rows)

    manifest_hashes = {
        path.relative_to(args.output_dir).as_posix(): _sha256(path)
        for path in manifest_paths
    }
    root_audit = audit_tc_manifests(outer_partition)
    audit = {
        "protocol_version": TC_PROTOCOL_VERSION,
        "root_partition": root_audit,
        "folds": fold_audits,
        "development_row_count": len(development_rows),
        "spent_final_eval_row_count": len(spent_rows),
        "spent_final_eval_ids_absent": not bool(
            spent_ids & {row["image_id"] for row in outer_partition}
        ),
        "all_manifests_hashed": len(manifest_hashes) == 1 + args.fold_count * 6,
    }
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    spec_path = (
        project_root
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-09-27-tc-oos-rsg-design.md"
    )
    registration: dict[str, object] = {
        "protocol_version": TC_PROTOCOL_VERSION,
        "status": "pre_registered_before_model_execution",
        "source_partition": str(args.partition),
        "source_partition_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "protocol_document": str(args.protocol_document),
        "protocol_document_sha256": _sha256(args.protocol_document),
        "design_specification": str(spec_path),
        "design_specification_sha256": _sha256(spec_path),
        "builder_script_sha256": _sha256(Path(__file__)),
        "outer_seed": args.outer_seed,
        "inner_seeds": {
            f"fold_{fold}": args.outer_seed + fold
            for fold in range(args.fold_count)
        },
        "fold_count": args.fold_count,
        "inner_subset_fractions": INNER_SUBSET_FRACTIONS,
        "epsilon_target_grid": EPSILON_TARGET_GRID,
        "selection_rule": [
            "target recall decrease versus q0 must be no more than 0.01",
            "among feasible candidates choose minimum NLL",
            "break NLL ties with smaller epsilon_target",
            "if no candidate is feasible select explicit q0 fallback",
        ],
        "primary_metrics": [
            "NLL",
            "target_recall",
            "ti_intrusion_to_target",
            "metallic_intrusion_to_target",
        ],
        "secondary_metrics": ["accuracy", "macro_f1", "ECE", "complexity"],
        "bootstrap": {
            "iterations": 10000,
            "unit": "split_group_id",
            "paired": True,
            "seed": 20260927,
        },
        "promotion_criteria": [
            "upper endpoint of paired NLL difference 95% CI is below zero",
            "one-sided target-recall noninferiority lower bound is at least -0.01",
            "zero deterministic target-posterior safety violations",
            "at least one hard-negative intrusion improves and the other worsens by no more than 0.01",
            "at least two folds have favorable NLL and no single fold drives the result",
        ],
        "titanomagnetite_count_definitions": {
            "raw_collected": 35,
            "formal_dataset_after_audit": 23,
            "new_development_pool": 13,
        },
        "spent_subset": "final_eval",
        "development_row_count": len(development_rows),
        "spent_final_eval_row_count": len(spent_rows),
        "manifest_row_counts": manifest_counts,
        "manifest_sha256": manifest_hashes,
    }
    registration_path = args.output_dir / "registered_protocol.json"
    registration_path.write_text(
        json.dumps(registration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.partition.read_bytes() != source_bytes:
        raise AssertionError("Source partition changed during registration.")
    print(json.dumps(registration, ensure_ascii=False, indent=2))
    return registration


if __name__ == "__main__":
    main()
