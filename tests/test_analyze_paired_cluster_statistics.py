from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def prediction_row(
    image_id: str,
    group_id: str,
    true_label: str,
    predicted_label: str,
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "split_group_id": group_id,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "mineral_label": "synthetic",
    }


class PairedClusterStatisticsTests(unittest.TestCase):
    def test_alignment_preserves_paired_predictions(self) -> None:
        from analyze_paired_cluster_statistics import align_prediction_rows

        baseline = [prediction_row("a", "g1", "target_mineral", "gangue_negative")]
        comparison = [prediction_row("a", "g1", "target_mineral", "target_mineral")]

        pairs = align_prediction_rows(baseline, comparison)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["baseline_prediction"], "gangue_negative")
        self.assertEqual(pairs[0]["comparison_prediction"], "target_mineral")

    def test_alignment_rejects_true_label_mismatch(self) -> None:
        from analyze_paired_cluster_statistics import align_prediction_rows

        baseline = [prediction_row("a", "g1", "target_mineral", "target_mineral")]
        comparison = [prediction_row("a", "g1", "gangue_negative", "gangue_negative")]

        with self.assertRaisesRegex(ValueError, "true label"):
            align_prediction_rows(baseline, comparison)

    def test_cluster_resampling_keeps_groups_intact(self) -> None:
        from analyze_paired_cluster_statistics import resample_clusters

        pairs = [
            {"image_id": "a", "split_group_id": "g1"},
            {"image_id": "b", "split_group_id": "g1"},
            {"image_id": "c", "split_group_id": "g2"},
        ]

        sampled = resample_clusters(pairs, random.Random(17))
        g1_count = sum(row["split_group_id"] == "g1" for row in sampled)
        g2_count = sum(row["split_group_id"] == "g2" for row in sampled)

        self.assertEqual(g1_count % 2, 0)
        self.assertEqual(g2_count % 1, 0)

    def test_metrics_include_target_and_hard_negative_risks(self) -> None:
        from analyze_paired_cluster_statistics import calculate_metrics

        rows = [
            {"true_label": "target_mineral", "prediction": "target_mineral"},
            {"true_label": "target_mineral", "prediction": "gangue_negative"},
            {"true_label": "ti_bearing_negative", "prediction": "target_mineral"},
            {"true_label": "metallic_hard_negative", "prediction": "metallic_hard_negative"},
            {"true_label": "gangue_negative", "prediction": "gangue_negative"},
        ]

        metrics = calculate_metrics(rows)

        self.assertAlmostEqual(metrics["target_recall"], 0.5)
        self.assertAlmostEqual(metrics["target_miss_rate"], 0.5)
        self.assertAlmostEqual(metrics["ti_to_target_intrusion"], 1.0)
        self.assertAlmostEqual(metrics["metallic_to_target_intrusion"], 0.0)

    def test_confusion_metrics_match_row_metrics(self) -> None:
        import numpy as np

        from analyze_paired_cluster_statistics import calculate_metrics, confusion_to_metrics
        from train_mineral_classifier import CLASS_LABELS

        rows = [
            {"true_label": "target_mineral", "prediction": "target_mineral"},
            {"true_label": "target_mineral", "prediction": "gangue_negative"},
            {"true_label": "ti_bearing_negative", "prediction": "target_mineral"},
            {"true_label": "metallic_hard_negative", "prediction": "metallic_hard_negative"},
            {"true_label": "gangue_negative", "prediction": "gangue_negative"},
        ]
        label_index = {label: index for index, label in enumerate(CLASS_LABELS)}
        confusion = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=np.int64)
        for row in rows:
            confusion[label_index[row["true_label"]], label_index[row["prediction"]]] += 1

        expected = calculate_metrics(rows)
        actual = confusion_to_metrics(confusion)

        for metric in expected:
            self.assertAlmostEqual(actual[metric], expected[metric])

    def test_exact_mcnemar_uses_discordant_pairs(self) -> None:
        from analyze_paired_cluster_statistics import exact_mcnemar

        pairs = [
            {"true_label": "target_mineral", "baseline_prediction": "target_mineral", "comparison_prediction": "gangue_negative"},
            {"true_label": "gangue_negative", "baseline_prediction": "target_mineral", "comparison_prediction": "gangue_negative"},
            {"true_label": "gangue_negative", "baseline_prediction": "target_mineral", "comparison_prediction": "gangue_negative"},
            {"true_label": "target_mineral", "baseline_prediction": "target_mineral", "comparison_prediction": "target_mineral"},
        ]

        result = exact_mcnemar(pairs)

        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["comparison_only_correct"], 2)
        self.assertAlmostEqual(result["p_value_two_sided_exact"], 1.0)

    def test_two_stage_bootstrap_is_reproducible(self) -> None:
        from analyze_paired_cluster_statistics import paired_two_stage_bootstrap

        run = [
            {
                "image_id": "a",
                "split_group_id": "g1",
                "true_label": "target_mineral",
                "baseline_prediction": "gangue_negative",
                "comparison_prediction": "target_mineral",
            },
            {
                "image_id": "b",
                "split_group_id": "g2",
                "true_label": "gangue_negative",
                "baseline_prediction": "gangue_negative",
                "comparison_prediction": "gangue_negative",
            },
            {
                "image_id": "c",
                "split_group_id": "g3",
                "true_label": "metallic_hard_negative",
                "baseline_prediction": "target_mineral",
                "comparison_prediction": "metallic_hard_negative",
            },
            {
                "image_id": "d",
                "split_group_id": "g4",
                "true_label": "ti_bearing_negative",
                "baseline_prediction": "target_mineral",
                "comparison_prediction": "ti_bearing_negative",
            },
        ]

        first = paired_two_stage_bootstrap({"1": run, "2": run}, 100, 20260819)
        second = paired_two_stage_bootstrap({"1": run, "2": run}, 100, 20260819)

        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(len(first["replicates"]), 100)
        self.assertGreater(first["summary"]["accuracy"]["difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
