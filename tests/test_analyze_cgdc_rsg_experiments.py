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


if __name__ == "__main__":
    unittest.main()
