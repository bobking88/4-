from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TargetProxyMetricsTests(unittest.TestCase):
    def test_target_proxy_metrics_count_misses_and_high_risk_intrusions(self) -> None:
        from analyze_target_proxy_metrics import calculate_target_proxy_metrics

        rows = [
            {"true_label": "target_mineral", "predicted_label": "target_mineral"},
            {"true_label": "target_mineral", "predicted_label": "ti_bearing_negative"},
            {"true_label": "ti_bearing_negative", "predicted_label": "target_mineral"},
            {"true_label": "metallic_hard_negative", "predicted_label": "target_mineral"},
            {"true_label": "gangue_negative", "predicted_label": "gangue_negative"},
        ]

        result = calculate_target_proxy_metrics(rows)

        self.assertEqual(result["target_support"], 2)
        self.assertEqual(result["target_true_positive"], 1)
        self.assertEqual(result["target_false_negative"], 1)
        self.assertEqual(result["target_false_positive"], 2)
        self.assertAlmostEqual(result["target_precision"], 1 / 3)
        self.assertAlmostEqual(result["target_recall"], 1 / 2)
        self.assertAlmostEqual(result["target_f1"], 0.4)
        self.assertAlmostEqual(result["target_miss_rate"], 1 / 2)
        self.assertAlmostEqual(result["ti_bearing_intrusion_rate"], 1.0)
        self.assertAlmostEqual(result["metallic_intrusion_rate"], 1.0)
        self.assertAlmostEqual(result["gangue_intrusion_rate"], 0.0)

    def test_target_proxy_metrics_reject_rows_missing_required_columns(self) -> None:
        from analyze_target_proxy_metrics import calculate_target_proxy_metrics

        with self.assertRaisesRegex(ValueError, "predicted_label"):
            calculate_target_proxy_metrics([{"true_label": "target_mineral"}])

    def test_summary_ignores_run_name_and_count_fields(self) -> None:
        from analyze_target_proxy_metrics import summarize_metric_runs

        summary = summarize_metric_runs(
            [
                {"run_name": "seed_a", "sample_count": 10, "target_precision": 0.7},
                {"run_name": "seed_b", "sample_count": 10, "target_precision": 0.9},
            ]
        )

        self.assertEqual(set(summary), {"target_precision"})
        self.assertAlmostEqual(summary["target_precision"]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
