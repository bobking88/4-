"""Build a five-way, group-isolated protocol for OOS-RSG confirmation.

Only rows from the original training split are eligible.  The final evaluation
subset is locked before any new expert or gate is trained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SUBSET_FRACTIONS = {
    "final_eval": 0.15,
    "expert_stop": 0.10,
    "gate_fit": 0.10,
    "gate_stop": 0.10,
}
HELD_SUBSETS = tuple(SUBSET_FRACTIONS)


def _hash_order(seed: int, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(), value


def _group_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("split_group_id", "")
        if not group_id:
            raise ValueError("Missing duplicate group.")
        groups[group_id].append(row)
    return dict(groups)


def partition_training_rows(
    rows: list[dict[str, str]], seed: int = 20260922
) -> list[dict[str, str]]:
    """Assign intact duplicate groups within each mineral-role stratum."""
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("Only nonempty original training rows are allowed.")
    image_ids = [row.get("image_id", "") for row in rows]
    if "" in image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Image IDs must be present and unique.")

    groups = _group_rows(rows)
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        labels = {
            (row.get("mineral_label", ""), row.get("four_class_label", ""))
            for row in members
        }
        if "" in {value for pair in labels for value in pair}:
            raise ValueError("Missing mineral or role label.")
        if len(labels) != 1:
            raise ValueError("Mixed-label group requires explicit adjudication.")
        strata[next(iter(labels))].append(group_id)

    assignment: dict[str, str] = {}
    for label, group_ids in sorted(strata.items()):
        minimum_groups = len(HELD_SUBSETS) + 1
        if len(group_ids) < minimum_groups:
            raise ValueError(
                f"Insufficient groups for {label}: need at least {minimum_groups}."
            )
        ordered = sorted(group_ids, key=lambda value: _hash_order(seed, value))
        counts = {
            name: max(1, round(fraction * len(ordered)))
            for name, fraction in SUBSET_FRACTIONS.items()
        }
        if sum(counts.values()) >= len(ordered):
            raise ValueError(f"Held-out allocations exhaust stratum {label}.")
        offset = 0
        for subset in HELD_SUBSETS:
            next_offset = offset + counts[subset]
            for group_id in ordered[offset:next_offset]:
                assignment[group_id] = subset
            offset = next_offset
        for group_id in ordered[offset:]:
            assignment[group_id] = "expert_fit"

    return [
        {**row, "confirmation_subset": assignment[row["split_group_id"]]}
        for row in sorted(rows, key=lambda item: item["image_id"])
    ]


def _select_exact_group_subset(
    rows: list[dict[str, str]], target_count: int, seed: int
) -> list[dict[str, str]]:
    """Select intact groups whose image counts sum exactly to target_count."""
    if target_count <= 0:
        raise ValueError("Target count must be positive.")
    groups = _group_rows(rows)
    ordered = sorted(groups, key=lambda value: _hash_order(seed, value))
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id in ordered:
        size = len(groups[group_id])
        for total, selected in sorted(reachable.items(), reverse=True):
            candidate = total + size
            if candidate <= target_count and candidate not in reachable:
                reachable[candidate] = (*selected, group_id)
    if target_count not in reachable:
        raise ValueError(
            f"No intact-group subset matches the requested {target_count} images."
        )
    selected_groups = set(reachable[target_count])
    return sorted(
        [row for group_id in selected_groups for row in groups[group_id]],
        key=lambda item: item["image_id"],
    )


def _with_split(rows: Iterable[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [{**row, "split": split} for row in rows]


def build_confirmation_manifests(
    partition: list[dict[str, str]], seen_seed: int = 20260923
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    """Create trainer manifests and enforce all functional isolation contracts."""
    expected = {"expert_fit", "expert_stop", "gate_fit", "gate_stop", "final_eval"}
    observed = {row.get("confirmation_subset") for row in partition}
    if observed != expected:
        raise ValueError(f"Unexpected confirmation subsets: {sorted(observed)}")

    parts = {
        name: [row for row in partition if row["confirmation_subset"] == name]
        for name in expected
    }
    stratum = lambda row: (row["mineral_label"], row["four_class_label"])
    unseen_counts = Counter(map(stratum, parts["gate_fit"]))
    fit_by_stratum: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in parts["expert_fit"]:
        fit_by_stratum[stratum(row)].append(row)

    gate_seen: list[dict[str, str]] = []
    for label, target_count in sorted(unseen_counts.items()):
        gate_seen.extend(
            _select_exact_group_subset(
                fit_by_stratum[label],
                target_count,
                seen_seed + int(hashlib.sha256(repr(label).encode()).hexdigest()[:8], 16),
            )
        )
    gate_seen.sort(key=lambda row: row["image_id"])
    if Counter(map(stratum, gate_seen)) != unseen_counts:
        raise AssertionError("Seen and unseen supervision strata do not match exactly.")

    manifests = {
        "expert": _with_split(parts["expert_fit"], "train")
        + _with_split(parts["expert_stop"], "val"),
        "gate_seen": _with_split(gate_seen, "train"),
        "gate_unseen": _with_split(parts["gate_fit"], "train"),
        "gate_stop": _with_split(parts["gate_stop"], "val"),
        "final_eval": _with_split(parts["final_eval"], "test"),
    }
    group_sets = {
        name: {row["split_group_id"] for row in records}
        for name, records in manifests.items()
    }
    unexpected_pairs = (
        ("expert", "gate_unseen"),
        ("expert", "gate_stop"),
        ("expert", "final_eval"),
        ("gate_seen", "gate_unseen"),
        ("gate_seen", "gate_stop"),
        ("gate_seen", "final_eval"),
        ("gate_unseen", "gate_stop"),
        ("gate_unseen", "final_eval"),
        ("gate_stop", "final_eval"),
    )
    overlaps = {
        f"{left}|{right}": len(group_sets[left] & group_sets[right])
        for left, right in unexpected_pairs
        if group_sets[left] & group_sets[right]
    }
    if overlaps:
        raise AssertionError(f"Functional subsets share duplicate groups: {overlaps}")
    if not group_sets["gate_seen"].issubset(group_sets["expert"]):
        raise AssertionError("Seen gate supervision must come from expert-fit groups.")

    audit: dict[str, object] = {
        "counts": {name: len(records) for name, records in manifests.items()},
        "partition_counts": dict(Counter(row["confirmation_subset"] for row in partition)),
        "exact_seen_unseen_strata_match": True,
        "gate_seen_subset_of_expert": True,
        "cross_function_group_overlap": overlaps,
        "final_eval_locked": True,
        "original_validation_and_test_omitted": True,
        "selection_contract": {
            "expert_stop": "select expert epoch only",
            "gate_stop": "select gate epoch only",
            "final_eval": "read only after expert and gate selections are locked",
        },
        "strata": {
            f"{mineral}|{role}": count
            for (mineral, role), count in sorted(unseen_counts.items())
        },
    }
    return manifests, audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build locked manifests for multi-expert OOS-RSG confirmation."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition-seed", type=int, default=20260922)
    parser.add_argument("--seen-seed", type=int, default=20260923)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite an existing confirmation protocol.")
    source_bytes = args.manifest.read_bytes()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header.")
        fields = list(reader.fieldnames)
        source_rows = list(reader)
    training_rows = [row for row in source_rows if row.get("split") == "train"]
    partition = partition_training_rows(training_rows, seed=args.partition_seed)
    manifests, audit = build_confirmation_manifests(partition, seen_seed=args.seen_seed)

    args.output_dir.mkdir(parents=True)
    partition_path = args.output_dir / "partition.csv"
    with partition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "confirmation_subset"])
        writer.writeheader()
        writer.writerows(partition)

    output_hashes: dict[str, str] = {"partition": _sha256(partition_path)}
    for name, rows in manifests.items():
        path = args.output_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["image_id"]))
        output_hashes[name] = _sha256(path)

    if args.manifest.read_bytes() != source_bytes:
        raise AssertionError("Source manifest changed during protocol construction.")
    audit.update(
        {
            "partition_seed": args.partition_seed,
            "seen_seed": args.seen_seed,
            "source_manifest": str(args.manifest),
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "script_sha256": _sha256(Path(__file__)),
            "output_sha256": output_hashes,
        }
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
