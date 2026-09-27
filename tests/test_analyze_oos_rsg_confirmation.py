from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_oos_rsg_confirmation import (  # noqa: E402
    align_prediction_rows,
    assess_promotion,
    calculate_metrics,
)


def row(image_id, group, true_id, predicted_id, probabilities):
    return {
        "image_id": image_id,
        "split_group_id": group,
        "mineral_label": f"m{true_id}",
        "true_class_id": str(true_id),
        "predicted_class_id": str(predicted_id),
        **{f"prob_{index}": str(value) for index, value in enumerate(probabilities)},
    }


class OOSRSGConfirmationAnalysisTests(unittest.TestCase):
    def test_metrics_include_nll_and_role_risks(self) -> None:
        rows = [
            row("a", "ga", 0, 0, (0.8, 0.1, 0.05, 0.05)),
            row("b", "gb", 1, 0, (0.6, 0.2, 0.1, 0.1)),
            row("c", "gc", 2, 2, (0.1, 0.1, 0.7, 0.1)),
            row("d", "gd", 3, 3, (0.1, 0.1, 0.1, 0.7)),
        ]

        metrics = calculate_metrics(rows)

        self.assertGreater(metrics["final_nll_nats"], 0.0)
        self.assertEqual(metrics["target_recall"], 1.0)
        self.assertEqual(metrics["ti_intrusion"], 1.0)
        self.assertEqual(metrics["metal_intrusion"], 0.0)

    def test_alignment_rejects_group_or_label_changes(self) -> None:
        baseline = [row("a", "ga", 0, 0, (0.8, 0.1, 0.05, 0.05))]
        comparison = [row("a", "other", 0, 0, (0.8, 0.1, 0.05, 0.05))]

        with self.assertRaises(ValueError):
            align_prediction_rows(baseline, comparison)

    def test_promotion_requires_target_recall_harm_guardrail(self) -> None:
        passing = {
            "final_nll_nats": {"difference": -0.02, "ci_high": -0.001},
            "macro_f1": {"difference": -0.004},
            "target_recall": {"difference": -0.009},
            "ti_intrusion": {"difference": 0.009},
            "metal_intrusion": {"difference": 0.009},
            "nll_favorable_seed_count": 3,
        }
        self.assertTrue(assess_promotion(passing)["promote_to_main_method"])

        failing = {**passing, "target_recall": {"difference": -0.011}}
        decision = assess_promotion(failing)
        self.assertFalse(decision["promote_to_main_method"])
        self.assertFalse(decision["criteria"]["target_recall_guardrail"])


if __name__ == "__main__":
    unittest.main()
