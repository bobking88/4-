from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_DECISIONS = {"keep", "exclude", "needs_expert"}


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return reader.fieldnames, list(reader)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cross_split_group_count(rows: list[dict[str, str]], field: str) -> int:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(field, "").strip()
        if value:
            groups[value].add(row["split"])
    return sum(len(splits) > 1 for splits in groups.values())


def apply_review_decisions(
    manifest_path: Path,
    review_queue_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    manifest_fields, manifest_rows = _read_rows(manifest_path)
    review_fields, review_rows = _read_rows(review_queue_path)
    required_manifest = {
        "image_id", "four_class_label", "mineral_label", "mindat_photo_id",
        "split_group_id", "split",
    }
    required_review = {"image_id", "review_decision", "review_reason"}
    if not required_manifest.issubset(manifest_fields):
        raise ValueError("Manifest is missing required fields.")
    if not required_review.issubset(review_fields):
        raise ValueError("Review queue is missing required fields.")

    manifest_ids = {row["image_id"] for row in manifest_rows}
    decisions: dict[str, dict[str, str]] = {}
    for row in review_rows:
        image_id = row["image_id"].strip()
        decision = row["review_decision"].strip().lower()
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Invalid review decision for {image_id}: {decision}")
        if image_id not in manifest_ids:
            raise ValueError(f"Review queue references unknown image_id: {image_id}")
        if image_id in decisions:
            raise ValueError(f"Duplicate review decision for image_id: {image_id}")
        decisions[image_id] = row

    final_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    expert_rows: list[dict[str, str]] = []
    enriched_fields = [*manifest_fields, "review_decision", "review_reason", "expert_note", "reviewer", "review_date"]
    for manifest_row in manifest_rows:
        review = decisions.get(manifest_row["image_id"])
        decision = review["review_decision"].strip().lower() if review else "not_sampled"
        if decision in {"keep", "not_sampled"}:
            final_rows.append(manifest_row)
            continue
        enriched = {
            **manifest_row,
            "review_decision": decision,
            "review_reason": review.get("review_reason", ""),
            "expert_note": review.get("expert_note", ""),
            "reviewer": review.get("reviewer", ""),
            "review_date": review.get("review_date", ""),
        }
        if decision == "exclude":
            excluded_rows.append(enriched)
        else:
            expert_rows.append(enriched)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "dataset_split_manifest_v1_0.csv", manifest_fields, final_rows)
    _write_rows(output_dir / "excluded_after_review.csv", enriched_fields, excluded_rows)
    _write_rows(output_dir / "needs_expert_queue.csv", enriched_fields, expert_rows)

    split_counts = Counter(row["split"] for row in final_rows)
    class_counts = Counter(row["four_class_label"] for row in final_rows)
    summary: dict[str, object] = {
        "source_manifest": str(manifest_path),
        "review_queue": str(review_queue_path),
        "original_images": len(manifest_rows),
        "reviewed_images": len(review_rows),
        "kept_reviewed": sum(row["review_decision"].strip().lower() == "keep" for row in review_rows),
        "removed_exclude": len(excluded_rows),
        "quarantined_needs_expert": len(expert_rows),
        "not_sampled_retained": len(manifest_rows) - len(review_rows),
        "final_images": len(final_rows),
        "final_by_split": dict(sorted(split_counts.items())),
        "final_by_class": dict(sorted(class_counts.items())),
        "split_group_leakage_count": _cross_split_group_count(final_rows, "split_group_id"),
        "photo_id_leakage_count": _cross_split_group_count(final_rows, "mindat_photo_id"),
    }
    summary["validation_passed"] = (
        summary["split_group_leakage_count"] == 0
        and summary["photo_id_leakage_count"] == 0
        and len(final_rows) + len(excluded_rows) + len(expert_rows) == len(manifest_rows)
    )
    (output_dir / "dataset_v1_0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply visual review decisions to the fixed split manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = apply_review_decisions(args.manifest, args.review_queue, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
