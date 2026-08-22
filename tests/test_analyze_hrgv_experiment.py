from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class HRGVExperimentAnalysisTests(unittest.TestCase):
    def test_summary_uses_mean_and_sample_standard_deviation(self) -> None:
        from analyze_hrgv_experiment import summarize_metric_rows

        summary = summarize_metric_rows(
            [
                {"macro_f1": 0.70, "target_recall": 0.60},
                {"macro_f1": 0.75, "target_recall": 0.70},
                {"macro_f1": 0.80, "target_recall": 0.80},
            ]
        )

        self.assertAlmostEqual(summary["macro_f1"]["mean"], 0.75)
        self.assertAlmostEqual(summary["macro_f1"]["sample_std"], 0.05)
        self.assertAlmostEqual(summary["target_recall"]["mean"], 0.70)
        self.assertEqual(summary["macro_f1"]["values"], [0.70, 0.75, 0.80])

    def test_gate_summary_is_grouped_by_true_role(self) -> None:
        from analyze_hrgv_experiment import summarize_gate_by_role

        rows = [
            {"true_label": "target_mineral", "gate": "0.20"},
            {"true_label": "target_mineral", "gate": "0.40"},
            {"true_label": "ti_bearing_negative", "gate": "0.80"},
        ]

        summary = summarize_gate_by_role(rows)

        self.assertEqual(summary["target_mineral"]["count"], 2)
        self.assertAlmostEqual(summary["target_mineral"]["mean"], 0.30)
        self.assertAlmostEqual(summary["ti_bearing_negative"]["mean"], 0.80)

    def test_verifier_auc_uses_only_target_and_its_selected_negative(self) -> None:
        from analyze_hrgv_experiment import calculate_verifier_auc

        rows = [
            {
                "true_label": "target_mineral",
                "ti_target_probability": "0.90",
                "metallic_target_probability": "0.80",
            },
            {
                "true_label": "target_mineral",
                "ti_target_probability": "0.70",
                "metallic_target_probability": "0.70",
            },
            {
                "true_label": "ti_bearing_negative",
                "ti_target_probability": "0.20",
                "metallic_target_probability": "0.99",
            },
            {
                "true_label": "metallic_hard_negative",
                "ti_target_probability": "0.99",
                "metallic_target_probability": "0.10",
            },
            {
                "true_label": "gangue_negative",
                "ti_target_probability": "0.50",
                "metallic_target_probability": "0.50",
            },
        ]

        ti = calculate_verifier_auc(
            rows, "ti_bearing_negative", "ti_target_probability"
        )
        metallic = calculate_verifier_auc(
            rows, "metallic_hard_negative", "metallic_target_probability"
        )

        self.assertAlmostEqual(ti["roc_auc"], 1.0)
        self.assertEqual(ti["eligible_count"], 3)
        self.assertAlmostEqual(metallic["roc_auc"], 1.0)
        self.assertEqual(metallic["eligible_count"], 3)

    def test_configuration_delta_orients_lower_intrusion_as_improvement(self) -> None:
        from analyze_hrgv_experiment import compare_summaries

        baseline = {
            "macro_f1": {"mean": 0.70},
            "target_recall": {"mean": 0.75},
            "ti_to_target_intrusion_rate": {"mean": 0.12},
            "metallic_to_target_intrusion_rate": {"mean": 0.15},
        }
        comparison = {
            "macro_f1": {"mean": 0.72},
            "target_recall": {"mean": 0.73},
            "ti_to_target_intrusion_rate": {"mean": 0.09},
            "metallic_to_target_intrusion_rate": {"mean": 0.10},
        }

        result = compare_summaries(baseline, comparison)

        self.assertAlmostEqual(result["macro_f1"]["difference"], 0.02)
        self.assertAlmostEqual(result["macro_f1"]["oriented_improvement"], 0.02)
        self.assertAlmostEqual(
            result["ti_to_target_intrusion_rate"]["difference"], -0.03
        )
        self.assertAlmostEqual(
            result["ti_to_target_intrusion_rate"]["oriented_improvement"], 0.03
        )


if __name__ == "__main__":
    unittest.main()
