from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_review_decisions import apply_review_decisions


MANIFEST_FIELDS = [
    "image_id",
    "relative_path",
    "mineral_label",
    "four_class_label",
    "four_class_id",
    "mindat_photo_id",
    "split_group_id",
    "split",
]


class ApplyReviewDecisionsTests(unittest.TestCase):
    def test_excludes_and_quarantines_reviewed_rows_without_reshuffling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            review = root / "review.csv"
            output = root / "output"
            manifest_rows = [
                ["VTM-1", "a.jpg", "magnetite", "target_mineral", "0", "1", "DG-1", "train"],
                ["VTM-2", "b.jpg", "pyrite", "metallic_hard_negative", "3", "2", "DG-2", "test"],
                ["VTM-3", "c.jpg", "rutile", "ti_bearing_negative", "1", "3", "DG-3", "val"],
            ]
            with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(MANIFEST_FIELDS)
                writer.writerows(manifest_rows)
            with review.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_id", "review_decision", "review_reason"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"image_id": "VTM-1", "review_decision": "exclude", "review_reason": "wrong_label"},
                        {"image_id": "VTM-2", "review_decision": "needs_expert", "review_reason": "uncertain_primary_mineral"},
                    ]
                )

            summary = apply_review_decisions(manifest, review, output)

            self.assertEqual(summary["final_images"], 1)
            self.assertEqual(summary["removed_exclude"], 1)
            self.assertEqual(summary["quarantined_needs_expert"], 1)
            with (output / "dataset_split_manifest_v1_0.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual(final_rows[0]["image_id"], "VTM-3")
            self.assertEqual(final_rows[0]["split"], "val")
            self.assertTrue((output / "excluded_after_review.csv").exists())
            self.assertTrue((output / "needs_expert_queue.csv").exists())

    def test_rejects_unknown_review_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            review = root / "review.csv"
            with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(MANIFEST_FIELDS)
                writer.writerow(["VTM-1", "a.jpg", "magnetite", "target_mineral", "0", "1", "DG-1", "train"])
            review.write_text(
                "image_id,review_decision,review_reason\nVTM-1,maybe,\n",
                encoding="utf-8-sig",
            )

            with self.assertRaisesRegex(ValueError, "Invalid review decision"):
                apply_review_decisions(manifest, review, root / "output")


if __name__ == "__main__":
    unittest.main()
