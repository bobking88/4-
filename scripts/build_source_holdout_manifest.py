from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from train_mineral_classifier import CLASS_LABELS


def normalize_source_group(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.replace("©", " ")).strip().casefold()
    if not normalized:
        raise ValueError("Every strict source-holdout row must have a photographer.")
    return normalized


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _source_components(rows):
    image_ids = [row["image_id"] for row in rows]
    union = UnionFind(image_ids)
    by_source = defaultdict(list)
    by_duplicate_group = defaultdict(list)
    for row in rows:
        by_source[normalize_source_group(row.get("photographer_or_credit", ""))].append(row["image_id"])
        by_duplicate_group[row["split_group_id"]].append(row["image_id"])
    for groups in (by_source, by_duplicate_group):
        for image_group in groups.values():
            for image_id in image_group[1:]:
                union.union(image_group[0], image_id)
    components = defaultdict(list)
    for row in rows:
        components[union.find(row["image_id"])].append(row)
    return list(components.values())


def allocate_source_groups(rows, target_test_ratio: float = 0.15, seed: int = 20260813):
    if not 0.0 < target_test_ratio < 0.5:
        raise ValueError("Target test ratio must lie between zero and one half.")
    components = _source_components(rows)
    totals = Counter(row["four_class_label"] for row in rows)
    targets = {
        "train": {role: totals[role] * (1 - 2 * target_test_ratio) for role in CLASS_LABELS},
        "val": {role: totals[role] * target_test_ratio for role in CLASS_LABELS},
        "test": {role: totals[role] * target_test_ratio for role in CLASS_LABELS},
    }
    assigned_counts = {split: Counter() for split in ("train", "val", "test")}
    rng = random.Random(seed)
    tie_breakers = {id(component): rng.random() for component in components}
    components.sort(key=lambda component: (-len(component), tie_breakers[id(component)]))
    assignments = {}
    split_order = {"train": 0, "val": 1, "test": 2}
    for component in components:
        component_counts = Counter(row["four_class_label"] for row in component)
        candidates = []
        for split in ("train", "val", "test"):
            score = 0.0
            for role in CLASS_LABELS:
                target = max(targets[split][role], 1.0)
                before = (assigned_counts[split][role] - target) ** 2 / target
                after = (assigned_counts[split][role] + component_counts[role] - target) ** 2 / target
                score += after - before
            candidates.append((score, split_order[split], split))
        selected = min(candidates)[2]
        assigned_counts[selected].update(component_counts)
        for row in component:
            assignments[row["image_id"]] = selected
    return [{**row, "split": assignments[row["image_id"]]} for row in rows]


def validate_source_holdout(rows, minimum_test_per_role: int = 30):
    photographer_splits = defaultdict(set)
    duplicate_splits = defaultdict(set)
    split_counts = Counter()
    split_role_counts = {split: Counter() for split in ("train", "val", "test")}
    for row in rows:
        split = row["split"]
        photographer_splits[normalize_source_group(row["photographer_or_credit"])].add(split)
        duplicate_splits[row["split_group_id"]].add(split)
        split_counts[split] += 1
        split_role_counts[split][row["four_class_label"]] += 1
    audit = {
        "row_count": len(rows),
        "photographer_count": len(photographer_splits),
        "photographer_cross_split_count": sum(len(splits) > 1 for splits in photographer_splits.values()),
        "split_group_cross_split_count": sum(len(splits) > 1 for splits in duplicate_splits.values()),
        "split_counts": dict(split_counts),
        "split_role_counts": {split: dict(counts) for split, counts in split_role_counts.items()},
    }
    failures = []
    if audit["photographer_cross_split_count"]:
        failures.append("photographer overlap")
    if audit["split_group_cross_split_count"]:
        failures.append("duplicate-group overlap")
    for split in ("train", "val", "test"):
        for role in CLASS_LABELS:
            required = minimum_test_per_role if split == "test" else 1
            if split_role_counts[split][role] < required:
                failures.append(f"{split}/{role} has fewer than {required} rows")
    audit["accepted"] = not failures
    audit["failures"] = failures
    return audit


def _read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field for field in rows[0] if field not in {"photographer_or_credit", "locality"}]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a strict photographer-held-out manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main():
    args = parse_args()
    final_rows = _read_csv(args.manifest)
    metadata = {row["image_id"]: row for row in _read_csv(args.audit_manifest)}
    enriched = []
    for row in final_rows:
        source = metadata.get(row["image_id"], {})
        photographer = source.get("photographer_or_credit", "").strip()
        if photographer:
            enriched.append({**row, "photographer_or_credit": photographer, "locality": source.get("locality", "")})
    assigned = allocate_source_groups(enriched, args.test_ratio, args.seed)
    audit = validate_source_holdout(assigned)
    manifest_path = args.output_dir / "photographer_holdout_manifest.csv"
    _write_csv(manifest_path, assigned)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    payload = {
        **audit,
        "source_field": "photographer_or_credit",
        "missing_photographer_rows_excluded": len(final_rows) - len(enriched),
        "target_test_ratio": args.test_ratio,
        "seed": args.seed,
        "manifest_sha256": manifest_hash,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "photographer_holdout_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not audit["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
