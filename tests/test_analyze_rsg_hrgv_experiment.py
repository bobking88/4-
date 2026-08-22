from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGRoutingMetricTests(unittest.TestCase):
    def test_routing_metrics_capture_complementarity_and_regret(self) -> None:
        from analyze_rsg_hrgv_experiment import calculate_routing_metrics

        rows = [
            {
                "true_label": "target_mineral",
                "predicted_label": "target_mineral",
                "direct_predicted_label": "target_mineral",
                "mapped_predicted_label": "gangue_negative",
                "gate_selection_correct": "1",
                "routing_regret_nll": "0.10",
            },
            {
                "true_label": "ti_bearing_negative",
                "predicted_label": "ti_bearing_negative",
                "direct_predicted_label": "target_mineral",
                "mapped_predicted_label": "ti_bearing_negative",
                "gate_selection_correct": "1",
                "routing_regret_nll": "0.20",
            },
            {
                "true_label": "gangue_negative",
                "predicted_label": "gangue_negative",
                "direct_predicted_label": "gangue_negative",
                "mapped_predicted_label": "gangue_negative",
                "gate_selection_correct": "1",
                "routing_regret_nll": "0.00",
            },
            {
                "true_label": "metallic_hard_negative",
                "predicted_label": "target_mineral",
                "direct_predicted_label": "target_mineral",
                "mapped_predicted_label": "target_mineral",
                "gate_selection_correct": "0",
                "routing_regret_nll": "0.30",
            },
        ]

        metrics = calculate_routing_metrics(rows)

        self.assertAlmostEqual(metrics["direct_accuracy"], 0.50)
        self.assertAlmostEqual(metrics["mapped_accuracy"], 0.50)
        self.assertAlmostEqual(metrics["fused_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["oracle_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["expert_prediction_disagreement_rate"], 0.50)
        self.assertEqual(metrics["one_right_one_wrong_count"], 2)
        self.assertAlmostEqual(metrics["one_right_gate_selection_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_routing_regret_nll"], 0.15)
        self.assertAlmostEqual(metrics["complementarity_recovery"], 1.0)

    def test_complementarity_recovery_is_none_without_oracle_headroom(self) -> None:
        from analyze_rsg_hrgv_experiment import calculate_routing_metrics

        rows = [
            {
                "true_label": "target_mineral",
                "predicted_label": "target_mineral",
                "direct_predicted_label": "target_mineral",
                "mapped_predicted_label": "target_mineral",
                "gate_selection_correct": "1",
                "routing_regret_nll": "0",
            }
        ]

        self.assertIsNone(calculate_routing_metrics(rows)["complementarity_recovery"])

    def test_formal_seed_validation_rejects_missing_and_duplicate_seeds(self) -> None:
        from analyze_rsg_hrgv_experiment import validate_formal_seeds

        validate_formal_seeds(["20260727", "20260728", "20260729"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_formal_seeds(["20260727", "20260727", "20260729"])
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_formal_seeds(["20260727", "20260728"])


if __name__ == "__main__":
    unittest.main()
