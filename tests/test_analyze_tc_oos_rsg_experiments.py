from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _row(
    image_id: str,
    group_id: str,
    fold: int,
    true_id: int,
    probabilities: tuple[float, float, float, float],
) -> dict[str, str]:
    predicted_id = max(range(4), key=probabilities.__getitem__)
    return {
        "image_id": image_id,
        "split_group_id": group_id,
        "mineral_label": f"species-{true_id}",
        "outer_fold": str(fold),
        "true_class_id": str(true_id),
        "predicted_class_id": str(predicted_id),
        **{
            f"prob_{index}": f"{probability:.12f}"
            for index, probability in enumerate(probabilities)
        },
    }


class TCOOSRSGAnalysisTests(unittest.TestCase):
    def test_role_metrics_include_nll_and_business_risks(self) -> None:
        from analyze_tc_oos_rsg_experiments import calculate_role_metrics

        rows = [
            _row("t1", "gt", 0, 0, (0.8, 0.1, 0.05, 0.05)),
            _row("n1", "gti", 0, 1, (0.6, 0.2, 0.1, 0.1)),
            _row("g1", "gg", 0, 2, (0.1, 0.1, 0.7, 0.1)),
            _row("m1", "gm", 0, 3, (0.55, 0.1, 0.1, 0.25)),
        ]

        metrics = calculate_role_metrics(rows)

        expected_nll = -sum(math.log(value) for value in (0.8, 0.2, 0.7, 0.25)) / 4
        self.assertAlmostEqual(metrics["nll"], expected_nll)
        self.assertEqual(metrics["target_recall"], 1.0)
        self.assertEqual(metrics["ti_intrusion_to_target"], 1.0)
        self.assertEqual(metrics["metallic_intrusion_to_target"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.5)

    def test_bootstrap_is_group_paired_and_nll_ci_favors_method(self) -> None:
        from analyze_tc_oos_rsg_experiments import cluster_paired_bootstrap

        baseline = []
        method = []
        for fold in range(3):
            for true_id in range(4):
                for group_index in range(3):
                    group_id = f"f{fold}-c{true_id}-g{group_index}"
                    for member in range(2):
                        image_id = f"{group_id}-i{member}"
                        baseline_probs = [0.1, 0.1, 0.1, 0.1]
                        method_probs = [0.05, 0.05, 0.05, 0.05]
                        baseline_probs[true_id] = 0.7
                        method_probs[true_id] = 0.85
                        baseline.append(
                            _row(image_id, group_id, fold, true_id, tuple(baseline_probs))
                        )
                        method.append(
                            _row(image_id, group_id, fold, true_id, tuple(method_probs))
                        )

        result = cluster_paired_bootstrap(
            baseline, method, iterations=200, seed=19
        )

        self.assertEqual(result["audit"]["sampling_unit"], "split_group_id")
        self.assertEqual(result["audit"]["group_count"], 36)
        self.assertEqual(result["audit"]["paired_image_count"], 72)
        self.assertLess(result["confidence_intervals"]["nll"]["ci_high"], 0.0)
        self.assertGreaterEqual(
            result["confidence_intervals"]["target_recall"][
                "one_sided_95_lower"
            ],
            0.0,
        )
        self.assertTrue(
            all(
                values["nll"] < 0.0
                for values in result["fold_differences"].values()
            )
        )

        identical = cluster_paired_bootstrap(
            baseline, copy.deepcopy(baseline), iterations=100, seed=23
        )
        for interval in identical["confidence_intervals"].values():
            self.assertAlmostEqual(interval["difference"], 0.0)
            self.assertAlmostEqual(interval["ci_low"], 0.0)
            self.assertAlmostEqual(interval["ci_high"], 0.0)

    def test_bootstrap_rejects_identity_or_fold_mismatch(self) -> None:
        from analyze_tc_oos_rsg_experiments import cluster_paired_bootstrap

        baseline = [
            _row("x", "group-a", 0, 0, (0.8, 0.1, 0.05, 0.05))
        ]
        for field, changed_value in (
            ("split_group_id", "group-b"),
            ("true_class_id", "1"),
            ("outer_fold", "1"),
        ):
            method = copy.deepcopy(baseline)
            method[0][field] = changed_value
            with self.assertRaisesRegex(ValueError, "mismatch"):
                cluster_paired_bootstrap(baseline, method, iterations=5, seed=1)

    def test_theory_invariants_verify_projection_and_target_bound(self) -> None:
        from analyze_tc_oos_rsg_experiments import verify_theory_invariants

        q0 = (0.60, 0.20, 0.10, 0.10)
        q_phi = (0.20, 0.50, 0.20, 0.10)
        rho = 0.025
        q_tc = tuple((1.0 - rho) * first + rho * second for first, second in zip(q0, q_phi))
        rows = [
            {
                "image_id": "safe-1",
                "split_group_id": "g-safe",
                "outer_fold": 0,
                "true_class_id": 0,
                "q0_probabilities": q0,
                "qphi_probabilities": q_phi,
                "qtc_probabilities": q_tc,
                "route": rho,
                "epsilon_target": 0.01,
            }
        ]

        result = verify_theory_invariants(rows)

        self.assertEqual(result["simplex_violation_count"], 0)
        self.assertEqual(result["target_safety_violation_count"], 0)
        self.assertEqual(result["decomposition_violation_count"], 0)
        self.assertEqual(result["convex_nll_bound_violation_count"], 0)
        self.assertEqual(result["cross_fold_group_overlap_count"], 0)
        self.assertLessEqual(result["max_target_harm"], 0.01 + 1e-12)

    def test_promotion_requires_all_theory_and_empirical_criteria(self) -> None:
        from analyze_tc_oos_rsg_experiments import assess_tc_promotion

        summary = {
            "confidence_intervals": {
                "nll": {"difference": -0.05, "ci_low": -0.08, "ci_high": -0.01},
                "target_recall": {
                    "difference": 0.0,
                    "ci_low": -0.02,
                    "ci_high": 0.02,
                    "one_sided_95_lower": -0.005,
                },
                "ti_intrusion_to_target": {
                    "difference": -0.02,
                    "ci_low": -0.04,
                    "ci_high": -0.001,
                },
                "metallic_intrusion_to_target": {
                    "difference": 0.005,
                    "ci_low": -0.01,
                    "ci_high": 0.009,
                },
            },
            "fold_differences": {
                "0": {"nll": -0.03},
                "1": {"nll": -0.02},
                "2": {"nll": 0.001},
            },
            "invariants": {
                "target_safety_violation_count": 0,
                "simplex_violation_count": 0,
                "decomposition_violation_count": 0,
                "convex_nll_bound_violation_count": 0,
                "nonfinite_or_negative_count": 0,
                "cross_fold_group_overlap_count": 0,
            },
        }

        decision = assess_tc_promotion(summary)

        self.assertTrue(decision["promote_to_main_method"])
        self.assertEqual(decision["decision"], "promote_tc_oos_rsg")
        self.assertTrue(all(decision["criteria"].values()))

        failed = copy.deepcopy(summary)
        failed["invariants"]["target_safety_violation_count"] = 1
        fallback = assess_tc_promotion(failed)
        self.assertFalse(fallback["promote_to_main_method"])
        self.assertEqual(fallback["decision"], "retain_q0_fallback")
        self.assertIn("zero_theory_violations", fallback["failed_criteria"])


if __name__ == "__main__":
    unittest.main()
