from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@dataclass(frozen=True)
class FakeRecord:
    image_id: str
    mindat_photo_id: str
    mineral_label: str
    four_class_label: str
    class_id: int
    split_group_id: str
    image_path: Path


class HRGVTrainingConfigurationTests(unittest.TestCase):
    def test_cli_defaults_match_the_network_specification(self) -> None:
        from train_hrgv_mineral_classifier import parse_args, validate_args

        args = parse_args(
            [
                "--manifest",
                "split.csv",
                "--dataset-root",
                "dataset",
                "--output-dir",
                "output",
            ]
        )
        validate_args(args)

        self.assertEqual(args.lambda_direct, 0.25)
        self.assertEqual(args.lambda_species, 0.50)
        self.assertEqual(args.lambda_consistency, 0.10)
        self.assertEqual(args.lambda_verifier, 0.50)
        self.assertEqual(args.lambda_contrast, 0.10)
        self.assertEqual(args.lambda_gate_regret, 0.0)
        self.assertEqual(args.gate_regret_temperature, 0.20)
        self.assertEqual(args.gate_gap_temperature, 0.50)
        self.assertFalse(args.disable_gate_regret)
        self.assertFalse(args.hard_gate_target)
        self.assertFalse(args.unweighted_gate_regret)
        self.assertFalse(args.detach_gate_features)
        self.assertFalse(args.couple_gate_features)
        self.assertEqual(args.gate_hidden_dim, 128)
        self.assertIsNone(args.fixed_gate)
        self.assertFalse(args.disable_verifiers)
        self.assertEqual(args.verifier_mode, "residual")
        self.assertEqual(args.ti_verifier_threshold, 0.5)
        self.assertEqual(args.metallic_verifier_threshold, 0.5)
        self.assertEqual(args.ti_verifier_strength, 1.0)
        self.assertEqual(args.metallic_verifier_strength, 1.0)
        self.assertFalse(args.couple_verifier_features)

    def test_cli_can_restore_coupled_verifier_gradients_for_ablation(self) -> None:
        from train_hrgv_mineral_classifier import parse_args, validate_args

        args = parse_args(
            [
                "--manifest",
                "split.csv",
                "--dataset-root",
                "dataset",
                "--output-dir",
                "output",
                "--couple-verifier-features",
            ]
        )
        validate_args(args)

        self.assertTrue(args.couple_verifier_features)

    def test_validation_rejects_negative_weights_and_invalid_fixed_gate(self) -> None:
        from train_hrgv_mineral_classifier import parse_args, validate_args

        common = [
            "--manifest",
            "split.csv",
            "--dataset-root",
            "dataset",
            "--output-dir",
            "output",
        ]
        negative = parse_args([*common, "--lambda-verifier", "-0.1"])
        invalid_gate = parse_args([*common, "--fixed-gate", "1.1"])
        invalid_regret_weight = parse_args([*common, "--lambda-gate-regret", "-0.1"])
        invalid_temperature = parse_args([*common, "--gate-regret-temperature", "0"])

        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_args(negative)
        with self.assertRaisesRegex(ValueError, "fixed gate"):
            validate_args(invalid_gate)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_args(invalid_regret_weight)
        with self.assertRaisesRegex(ValueError, "Gate temperatures"):
            validate_args(invalid_temperature)

    def test_validation_rejects_invalid_residual_parameters(self) -> None:
        from train_hrgv_mineral_classifier import parse_args, validate_args

        common = [
            "--manifest",
            "split.csv",
            "--dataset-root",
            "dataset",
            "--output-dir",
            "output",
        ]
        invalid_threshold = parse_args([*common, "--ti-verifier-threshold", "0"])
        invalid_strength = parse_args([*common, "--metallic-verifier-strength", "-1"])

        with self.assertRaisesRegex(ValueError, "threshold"):
            validate_args(invalid_threshold)
        with self.assertRaisesRegex(ValueError, "strength"):
            validate_args(invalid_strength)


class HRGVArtifactTests(unittest.TestCase):
    def test_risk_metrics_measure_target_recall_and_both_intrusions(self) -> None:
        from train_hrgv_mineral_classifier import calculate_hrgv_risk_metrics

        targets = [0, 0, 1, 1, 2, 3, 3]
        predictions = [0, 1, 0, 1, 2, 0, 3]

        metrics = calculate_hrgv_risk_metrics(targets, predictions)

        self.assertAlmostEqual(metrics["target_recall"], 0.5)
        self.assertAlmostEqual(metrics["target_miss_rate"], 0.5)
        self.assertAlmostEqual(metrics["ti_to_target_intrusion_rate"], 0.5)
        self.assertAlmostEqual(metrics["metallic_to_target_intrusion_rate"], 0.5)

    def test_gate_summary_reports_routing_accuracy_regret_and_disagreement_subset(self) -> None:
        from train_hrgv_mineral_classifier import summarize_gate_routing

        metrics = summarize_gate_routing(
            gate_selection_correct=[True, False, True, True],
            routing_regrets_nll=[0.00, 0.40, 0.10, 0.20],
            weighted_gate_errors=[0.00, 0.30, 0.05, 0.10],
            one_right_one_wrong=[True, True, False, False],
        )

        self.assertAlmostEqual(metrics["gate_selection_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["one_right_gate_selection_accuracy"], 0.50)
        self.assertAlmostEqual(metrics["mean_routing_regret_nll"], 0.175)
        self.assertAlmostEqual(metrics["mean_weighted_gate_error"], 0.1125)

    def test_prediction_rows_include_all_hrgv_diagnostics(self) -> None:
        from train_hrgv_mineral_classifier import HRGV_PREDICTION_FIELDS, build_hrgv_prediction_rows

        records = [
            FakeRecord(
                image_id="VTM-000001",
                mindat_photo_id="123",
                mineral_label="magnetite",
                four_class_label="target_mineral",
                class_id=0,
                split_group_id="DG-1",
                image_path=Path("one.jpg"),
            )
        ]
        rows = build_hrgv_prediction_rows(
            records=records,
            final_prediction_ids=[0],
            confidences=[0.81],
            direct_prediction_ids=[1],
            mapped_prediction_ids=[0],
            gates=[0.35],
            ti_target_probabilities=[0.90],
            metallic_target_probabilities=[0.80],
            expert_js_divergences=[0.12],
            direct_true_probabilities=[0.22],
            mapped_true_probabilities=[0.61],
            fused_true_probabilities=[0.4735],
            hard_oracle_gates=[0.0],
            soft_oracle_gates=[0.03],
            gate_gap_weights=[0.88],
            gate_selection_correct=[True],
            routing_regrets_nll=[0.2540],
            weighted_gate_errors=[0.1365],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), set(HRGV_PREDICTION_FIELDS))
        self.assertEqual(rows[0]["predicted_label"], "target_mineral")
        self.assertEqual(rows[0]["direct_predicted_label"], "ti_bearing_negative")
        self.assertEqual(rows[0]["mapped_predicted_label"], "target_mineral")
        self.assertEqual(rows[0]["gate"], "0.350000")
        self.assertEqual(rows[0]["ti_target_probability"], "0.900000")
        self.assertEqual(rows[0]["metallic_target_probability"], "0.800000")
        self.assertEqual(rows[0]["expert_js_divergence"], "0.120000")
        self.assertEqual(rows[0]["direct_true_probability"], "0.220000")
        self.assertEqual(rows[0]["mapped_true_probability"], "0.610000")
        self.assertEqual(rows[0]["fused_true_probability"], "0.473500")
        self.assertEqual(rows[0]["hard_oracle_gate"], "0.000000")
        self.assertEqual(rows[0]["soft_oracle_gate"], "0.030000")
        self.assertEqual(rows[0]["gate_gap_weight"], "0.880000")
        self.assertEqual(rows[0]["gate_selection_correct"], "1")
        self.assertEqual(rows[0]["routing_regret_nll"], "0.254000")
        self.assertEqual(rows[0]["weighted_gate_error"], "0.136500")


if __name__ == "__main__":
    unittest.main()
