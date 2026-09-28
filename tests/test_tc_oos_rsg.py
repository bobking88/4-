from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TargetConstrainedProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def _probabilities(self):
        q0 = self.torch.tensor(
            [
                [0.60, 0.20, 0.10, 0.10],
                [0.20, 0.30, 0.30, 0.20],
            ],
            dtype=self.torch.float64,
        )
        candidate = self.torch.tensor(
            [
                [0.30, 0.40, 0.20, 0.10],
                [0.40, 0.20, 0.20, 0.20],
            ],
            dtype=self.torch.float64,
        )
        return q0, candidate

    def test_projection_returns_simplex_probabilities(self) -> None:
        from tc_oos_rsg import apply_target_safe_projection

        q0, candidate = self._probabilities()
        result = apply_target_safe_projection(
            q0,
            candidate,
            self.torch.tensor([[0.90], [0.80]], dtype=self.torch.float64),
            epsilon_target=0.05,
        )

        final = result["final_probabilities"]
        self.assertEqual(tuple(final.shape), (2, 4))
        self.assertTrue(self.torch.isfinite(final).all())
        self.assertTrue((final >= 0.0).all())
        self.assertTrue((final <= 1.0).all())
        self.assertLessEqual(
            float((final.sum(dim=1) - 1.0).abs().max()),
            1e-7,
        )

    def test_target_drop_never_exceeds_epsilon(self) -> None:
        from tc_oos_rsg import apply_target_safe_projection

        q0, candidate = self._probabilities()
        epsilon = 0.05
        result = apply_target_safe_projection(
            q0,
            candidate,
            self.torch.ones((2, 1), dtype=self.torch.float64),
            epsilon_target=epsilon,
        )

        realized_drop = (q0[:, :1] - result["final_probabilities"][:, :1]).clamp_min(0.0)
        self.assertLessEqual(float(realized_drop.max()), epsilon + 1e-12)
        self.assertAlmostEqual(float(result["route_cap"][0, 0]), 1.0 / 6.0, places=12)
        self.assertAlmostEqual(float(realized_drop[0, 0]), epsilon, places=12)

    def test_zero_budget_still_routes_when_candidate_does_not_reduce_target(self) -> None:
        from tc_oos_rsg import apply_target_safe_projection

        q0, candidate = self._probabilities()
        raw_route = self.torch.tensor([[0.90], [0.80]], dtype=self.torch.float64)
        result = apply_target_safe_projection(
            q0,
            candidate,
            raw_route,
            epsilon_target=0.0,
        )

        self.assertAlmostEqual(float(result["projected_route"][0, 0]), 0.0, places=12)
        self.assertAlmostEqual(float(result["route_cap"][1, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(result["projected_route"][1, 0]), 0.8, places=12)
        self.assertAlmostEqual(float(result["final_probabilities"][1, 0]), 0.36, places=12)

    def test_exact_target_change_identity_has_zero_residual(self) -> None:
        from tc_oos_rsg import apply_target_safe_projection

        q0, candidate = self._probabilities()
        result = apply_target_safe_projection(
            q0,
            candidate,
            self.torch.tensor([[0.90], [0.80]], dtype=self.torch.float64),
            epsilon_target=0.05,
        )

        expected_change = result["projected_route"] * (q0[:, :1] - candidate[:, :1])
        self.assertLessEqual(
            float((result["signed_target_change"] - expected_change).abs().max()),
            1e-7,
        )
        self.assertLessEqual(float(result["identity_residual"].abs().max()), 1e-7)
        self.assertTrue(self.torch.allclose(
            result["positive_target_harm"],
            result["signed_target_change"].clamp_min(0.0),
        ))

    def test_projection_head_emits_one_finite_probability_per_row(self) -> None:
        from tc_oos_rsg import TargetRiskProjectionHead, build_projection_evidence

        q0, candidate = self._probabilities()
        gate = self.torch.tensor([[0.65], [0.35]], dtype=self.torch.float64)
        ti = self.torch.tensor([[0.80], [0.25]], dtype=self.torch.float64)
        metallic = self.torch.tensor([[0.70], [0.40]], dtype=self.torch.float64)
        evidence = build_projection_evidence(q0, candidate, gate, ti, metallic)
        head = TargetRiskProjectionHead().to(dtype=self.torch.float64)

        route = head(evidence)

        self.assertEqual(tuple(evidence.shape), (2, 18))
        self.assertTrue(self.torch.equal(evidence[:, :4], q0))
        self.assertTrue(self.torch.equal(evidence[:, 4:8], candidate))
        self.assertTrue(self.torch.equal(evidence[:, 8:12], (candidate - q0).abs()))
        self.assertEqual(tuple(route.shape), (2, 1))
        self.assertTrue(self.torch.isfinite(route).all())
        self.assertTrue(((route > 0.0) & (route < 1.0)).all())

    def test_invalid_probabilities_and_shapes_are_rejected(self) -> None:
        from tc_oos_rsg import apply_target_safe_projection, build_projection_evidence

        q0, candidate = self._probabilities()
        valid_column = self.torch.full((2, 1), 0.5, dtype=self.torch.float64)
        malformed = {
            "non_finite": q0.clone(),
            "negative": q0.clone(),
            "not_simplex": q0.clone(),
        }
        malformed["non_finite"][0, 0] = float("nan")
        malformed["negative"][0, 0] = -0.1
        malformed["negative"][0, 1] = 0.9
        malformed["not_simplex"][0] *= 0.5

        for name, invalid in malformed.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    build_projection_evidence(
                        invalid,
                        candidate,
                        valid_column,
                        valid_column,
                        valid_column,
                    )

        three_class_candidate = candidate[:, :3]
        three_class_candidate = three_class_candidate / three_class_candidate.sum(
            dim=1,
            keepdim=True,
        )
        with self.assertRaisesRegex(ValueError, "same shape"):
            build_projection_evidence(
                q0,
                three_class_candidate,
                valid_column,
                valid_column,
                valid_column,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            build_projection_evidence(
                q0,
                candidate,
                self.torch.full((2,), 0.5, dtype=self.torch.float64),
                valid_column,
                valid_column,
            )
        with self.assertRaisesRegex(ValueError, "raw_route"):
            apply_target_safe_projection(
                q0,
                candidate,
                self.torch.full((2, 1), 1.1, dtype=self.torch.float64),
                epsilon_target=0.01,
            )
        with self.assertRaisesRegex(ValueError, "epsilon_target"):
            apply_target_safe_projection(
                q0,
                candidate,
                valid_column,
                epsilon_target=-0.01,
            )


if __name__ == "__main__":
    unittest.main()
