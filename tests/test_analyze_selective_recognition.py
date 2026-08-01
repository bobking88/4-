from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class SelectiveRecognitionTests(unittest.TestCase):
    def test_threshold_defers_low_confidence_errors(self) -> None:
        from analyze_selective_recognition import calculate_selective_metrics

        rows = [
            {"true_label": "target_mineral", "predicted_label": "target_mineral", "confidence": "0.95"},
            {"true_label": "ti_bearing_negative", "predicted_label": "target_mineral", "confidence": "0.55"},
        ]

        values = calculate_selective_metrics(rows, thresholds=(0.0, 0.8))

        self.assertEqual(values[0]["coverage"], 1.0)
        self.assertEqual(values[0]["risk"], 0.5)
        self.assertEqual(values[1]["coverage"], 0.5)
        self.assertEqual(values[1]["risk"], 0.0)

    def test_retained_metrics_include_target_miss_and_intrusion_rates(self) -> None:
        from analyze_selective_recognition import calculate_selective_metrics

        rows = [
            {"true_label": "target_mineral", "predicted_label": "ti_bearing_negative", "confidence": "0.95"},
            {"true_label": "target_mineral", "predicted_label": "target_mineral", "confidence": "0.92"},
            {"true_label": "ti_bearing_negative", "predicted_label": "target_mineral", "confidence": "0.90"},
            {"true_label": "metallic_hard_negative", "predicted_label": "target_mineral", "confidence": "0.89"},
        ]

        value = calculate_selective_metrics(rows, thresholds=(0.8,))[0]

        self.assertEqual(value["target_proxy_miss_rate"], 0.5)
        self.assertEqual(value["titanium_interference_intrusion_rate"], 1.0)
        self.assertEqual(value["metallic_hard_negative_intrusion_rate"], 1.0)

    def test_undefined_retained_rate_is_none(self) -> None:
        from analyze_selective_recognition import calculate_selective_metrics

        rows = [
            {"true_label": "target_mineral", "predicted_label": "target_mineral", "confidence": "0.95"},
        ]

        value = calculate_selective_metrics(rows, thresholds=(0.8,))[0]

        self.assertIsNone(value["titanium_interference_intrusion_rate"])
        self.assertIsNone(value["metallic_hard_negative_intrusion_rate"])

    def test_aggregate_threshold_metrics_keeps_seed_values_and_summary(self) -> None:
        from analyze_selective_recognition import aggregate_threshold_metrics

        result = aggregate_threshold_metrics(
            {
                "seed_a": [{"threshold": 0.0, "coverage": 1.0, "risk": 0.2}],
                "seed_b": [{"threshold": 0.0, "coverage": 0.8, "risk": 0.4}],
                "seed_c": [{"threshold": 0.0, "coverage": 0.6, "risk": None}],
            }
        )

        threshold = result[0]
        self.assertEqual(threshold["seed_values"]["seed_a"]["coverage"], 1.0)
        self.assertAlmostEqual(threshold["mean"]["coverage"], 0.8)
        self.assertAlmostEqual(threshold["sample_std"]["coverage"], 0.2)
        self.assertAlmostEqual(threshold["mean"]["risk"], 0.3)
        self.assertAlmostEqual(threshold["sample_std"]["risk"], 0.1414213562)

    def test_input_resolver_rejects_duplicate_files_for_expected_seed(self) -> None:
        from analyze_selective_recognition import EXPECTED_RUN_NAMES, _resolve_input_paths

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_name in EXPECTED_RUN_NAMES:
                run_dir = root / run_name
                run_dir.mkdir()
                (run_dir / "test_predictions.csv").touch()
            (root / EXPECTED_RUN_NAMES[0] / "alternate_predictions.csv").touch()

            with self.assertRaisesRegex(ValueError, "duplicate"):
                _resolve_input_paths(str(root / "*" / "*.csv"))


if __name__ == "__main__":
    unittest.main()
