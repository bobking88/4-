from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_mineral_classifier import ManifestRecord


class PredictionRowTests(unittest.TestCase):
    def test_build_prediction_rows_marks_errors_with_true_to_predicted_pair(self) -> None:
        from analyze_misclassifications import build_prediction_rows

        records = [
            ManifestRecord(
                image_id="VTM-000001",
                image_path=Path("one.jpg"),
                mineral_label="magnetite",
                four_class_label="target_mineral",
                class_id=0,
                mindat_photo_id="1001",
                split_group_id="G-1",
                split="test",
            ),
            ManifestRecord(
                image_id="VTM-000002",
                image_path=Path("two.jpg"),
                mineral_label="pyrite",
                four_class_label="metallic_hard_negative",
                class_id=3,
                mindat_photo_id="1002",
                split_group_id="G-2",
                split="test",
            ),
        ]

        rows = build_prediction_rows(records, [0, 1], [0.91, 0.62])

        self.assertEqual(rows[0]["is_correct"], "true")
        self.assertEqual(rows[0]["error_pair"], "")
        self.assertEqual(rows[1]["is_correct"], "false")
        self.assertEqual(
            rows[1]["error_pair"],
            "metallic_hard_negative__as__ti_bearing_negative",
        )

    def test_build_prediction_rows_rejects_mismatched_lengths(self) -> None:
        from analyze_misclassifications import build_prediction_rows

        record = ManifestRecord(
            image_id="VTM-000001",
            image_path=Path("one.jpg"),
            mineral_label="magnetite",
            four_class_label="target_mineral",
            class_id=0,
            mindat_photo_id="1001",
            split_group_id="G-1",
            split="test",
        )

        with self.assertRaisesRegex(ValueError, "length"):
            build_prediction_rows([record], [], [])


if __name__ == "__main__":
    unittest.main()
