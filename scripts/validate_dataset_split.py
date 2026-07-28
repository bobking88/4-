from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


AUDIT_DIR = Path(r"D:\成信工科研\人工智能选矿\数据集\dataset_audit")
SUMMARY_PATH = AUDIT_DIR / "dataset_audit_summary.json"
MASTER_PATH = AUDIT_DIR / "dataset_master_manifest.csv"
SPLIT_PATH = AUDIT_DIR / "dataset_split_manifest.csv"
OUTPUT_PATH = AUDIT_DIR / "dataset_split_validation.json"


def multi_split_group_count(frame: pd.DataFrame, column: str) -> int:
    valid = frame.loc[frame[column].notna() & frame[column].astype(str).ne("")]
    if valid.empty:
        return 0
    return int((valid.groupby(column)["split"].nunique() > 1).sum())


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    split = pd.read_csv(SPLIT_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    eligible_master = master.loc[master["split"].isin(["train", "val", "test"])].copy()

    expected_classes = {
        "target_mineral",
        "ti_bearing_negative",
        "gangue_negative",
        "metallic_hard_negative",
    }
    required_core = [
        "mineral_label",
        "mindat_photo_id",
        "detail_page_url",
        "page_title",
        "screening_decision",
    ]

    validation = {
        "candidate_row_count": len(master),
        "candidate_row_count_matches_summary": len(master) == summary["candidate_images"],
        "eligible_row_count": len(split),
        "eligible_row_count_matches_summary": len(split) == summary["eligible_images"],
        "accepted_split_values_only": set(split["split"]) == {"train", "val", "test"},
        "four_class_labels_match_plan": set(split["four_class_label"]) == expected_classes,
        "split_group_leakage_count": multi_split_group_count(
            eligible_master, "split_group_id"
        ),
        "photo_id_leakage_count": multi_split_group_count(
            eligible_master, "mindat_photo_id"
        ),
        "sha256_leakage_count": multi_split_group_count(eligible_master, "sha256"),
        "near_duplicate_group_leakage_count": multi_split_group_count(
            eligible_master, "near_duplicate_group_id"
        ),
        "core_metadata_missing_cells": int(
            sum(
                (eligible_master[column].astype(str).str.strip() == "").sum()
                for column in required_core
            )
        ),
    }
    validation["passed"] = all(
        [
            validation["candidate_row_count_matches_summary"],
            validation["eligible_row_count_matches_summary"],
            validation["accepted_split_values_only"],
            validation["four_class_labels_match_plan"],
            validation["split_group_leakage_count"] == 0,
            validation["photo_id_leakage_count"] == 0,
            validation["sha256_leakage_count"] == 0,
            validation["near_duplicate_group_leakage_count"] == 0,
            validation["core_metadata_missing_cells"] == 0,
        ]
    )
    OUTPUT_PATH.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
