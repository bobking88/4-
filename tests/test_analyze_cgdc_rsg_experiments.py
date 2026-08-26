from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class CGDCCalibrationAnalysisTests(unittest.TestCase):
    def test_calibration_summary_reports_brier_and_ece(self) -> None:
        from analyze_cgdc_rsg_experiments import summarize_calibration

        summary = summarize_calibration(
            [
                {
                    "true_label": "target_mineral",
                    "role_probability_target_mineral": "0.90",
                    "role_probability_ti_bearing_negative": "0.10",
                    "role_probability_gangue_negative": "0.00",
                    "role_probability_metallic_hard_negative": "0.00",
                },
                {
                    "true_label": "ti_bearing_negative",
                    "role_probability_target_mineral": "0.80",
                    "role_probability_ti_bearing_negative": "0.20",
                    "role_probability_gangue_negative": "0.00",
                    "role_probability_metallic_hard_negative": "0.00",
                },
            ],
            bins=2,
        )

        self.assertAlmostEqual(summary["brier_score"], 0.65)
        self.assertAlmostEqual(summary["expected_calibration_error"], 0.35)

    def test_paired_calibration_bootstrap_reports_brier_and_ece_intervals(self) -> None:
        from analyze_cgdc_rsg_experiments import paired_calibration_bootstrap

        def row(image_id: str, group_id: str, true_label: str, target_probability: float):
            return {
                "image_id": image_id,
                "split_group_id": group_id,
                "true_label": true_label,
                "role_probability_target_mineral": str(target_probability),
                "role_probability_ti_bearing_negative": str(1.0 - target_probability),
                "role_probability_gangue_negative": "0.0",
                "role_probability_metallic_hard_negative": "0.0",
            }

        reference = {
            "20260727": [row("a", "g1", "target_mineral", 0.55), row("b", "g2", "ti_bearing_negative", 0.55)],
            "20260728": [row("a", "g1", "target_mineral", 0.55), row("b", "g2", "ti_bearing_negative", 0.55)],
            "20260729": [row("a", "g1", "target_mineral", 0.55), row("b", "g2", "ti_bearing_negative", 0.55)],
        }
        comparison = {
            seed: [row("a", "g1", "target_mineral", 0.90), row("b", "g2", "ti_bearing_negative", 0.10)]
            for seed in reference
        }

        result = paired_calibration_bootstrap(reference, comparison, replicates=50, rng_seed=7)

        self.assertEqual(set(result), {"brier_score", "expected_calibration_error"})
        self.assertLess(result["brier_score"]["difference"], 0.0)
        self.assertIn("ci_low", result["expected_calibration_error"])


if __name__ == "__main__":
    unittest.main()
