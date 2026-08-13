from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def build_proxy_ablation_rows(final_rows):
    result = [row for row in final_rows if row["mineral_label"].strip().casefold() != "magnetite"]
    labels = Counter(row["mineral_label"] for row in result)
    if labels["ilmenite"] == 0 or labels["titanomagnetite"] == 0:
        raise ValueError("Proxy ablation must retain ilmenite and titanomagnetite.")
    return result


def summarize_titanomagnetite_provenance(audit_rows, final_rows):
    downloaded = [row for row in audit_rows if row["mineral_label"] == "titanomagnetite"]
    final = [row for row in final_rows if row["mineral_label"] == "titanomagnetite"]
    def exclusion_status(row):
        return row.get("exclusion_reason", "") or row.get("quality_status", "")

    conflicts = [
        row for row in downloaded
        if exclusion_status(row) in {
            "exclude_exact_label_conflict", "exclude_near_label_conflict"
        }
    ]
    return {
        "downloaded": len(downloaded),
        "excluded_exact_label_conflict": sum(
            exclusion_status(row) == "exclude_exact_label_conflict" for row in downloaded
        ),
        "excluded_near_label_conflict": sum(
            exclusion_status(row) == "exclude_near_label_conflict" for row in downloaded
        ),
        "excluded_cross_label_conflict": len(conflicts),
        "final": len(final),
    }


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a manifest without ordinary magnetite proxy images.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--quality-issues", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    final_rows = _read_csv(args.manifest)
    audit_rows = _read_csv(args.audit_manifest)
    issues_by_image = {row["image_id"]: row for row in _read_csv(args.quality_issues)}
    provenance_rows = [
        {**row, "quality_status": issues_by_image.get(row["image_id"], {}).get("quality_status", "")}
        for row in audit_rows
    ]
    result = build_proxy_ablation_rows(final_rows)
    output_path = args.output_dir / "no_magnetite_proxy_manifest.csv"
    _write_csv(output_path, result)
    before_species = Counter(row["mineral_label"] for row in final_rows)
    after_species = Counter(row["mineral_label"] for row in result)
    payload = {
        "interpretation": "Sensitivity to removal of ordinary magnetite proxy images; not a direct titanomagnetite performance claim.",
        "original_rows": len(final_rows),
        "ablation_rows": len(result),
        "removed_magnetite_rows": before_species["magnetite"],
        "species_counts_before": dict(sorted(before_species.items())),
        "species_counts_after": dict(sorted(after_species.items())),
        "role_counts_before": dict(Counter(row["four_class_label"] for row in final_rows)),
        "role_counts_after": dict(Counter(row["four_class_label"] for row in result)),
        "titanomagnetite_provenance": summarize_titanomagnetite_provenance(provenance_rows, final_rows),
        "manifest_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    with (args.output_dir / "proxy_ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
