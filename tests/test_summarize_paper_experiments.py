from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class SummarizePaperExperimentsTests(unittest.TestCase):
    def test_prediction_metrics_report_role_and_species_recall(self) -> None:
        from summarize_paper_experiments import calculate_prediction_metrics

        rows = [
            {"mineral_label": "ilmenite", "true_label": "target_mineral", "predicted_label": "target_mineral"},
            {"mineral_label": "titanomagnetite", "true_label": "target_mineral", "predicted_label": "gangue_negative"},
            {"mineral_label": "quartz", "true_label": "gangue_negative", "predicted_label": "gangue_negative"},
            {"mineral_label": "pyrite", "true_label": "metallic_hard_negative", "predicted_label": "target_mineral"},
        ]

        summary = calculate_prediction_metrics(rows)

        self.assertEqual(summary["row_count"], 4)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["target_recall"], 0.5)
        self.assertAlmostEqual(summary["species_role_recall"]["ilmenite"], 1.0)
        self.assertAlmostEqual(summary["species_role_recall"]["titanomagnetite"], 0.0)

    def test_common_subset_requires_identical_image_ids(self) -> None:
        from summarize_paper_experiments import common_subset_rows

        full = [{"image_id": "1", "mineral_label": "ilmenite"}, {"image_id": "2", "mineral_label": "magnetite"}]
        ablation = [{"image_id": "1", "mineral_label": "ilmenite"}]

        full_common, ablation_common = common_subset_rows(full, ablation)

        self.assertEqual([row["image_id"] for row in full_common], ["1"])
        self.assertEqual(full_common, ablation_common)


if __name__ == "__main__":
    unittest.main()
