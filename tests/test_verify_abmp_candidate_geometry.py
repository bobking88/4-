import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_abmp_candidate_geometry import minimum_target_preserving_scale, verify_geometry_properties


class GeometryPropertyTests(unittest.TestCase):
    def test_exact_scale_boundary_preserves_normalised_target_margin(self):
        q = torch.tensor([[.6, .25, .1, .05]], dtype=torch.float64)
        result = minimum_target_preserving_scale(q, margin=.05)
        scale = result["minimum_scale"]
        self.assertTrue(bool(result["feasible"].item()))
        self.assertAlmostEqual(float(scale), (.25+.05*.4)/(.95*.6))
        scaled = q.clone()
        scaled[:, :1] *= scale
        scaled /= scaled.sum(1, keepdim=True)
        self.assertAlmostEqual(float(scaled[0, 0]-scaled[0, 1]), .05, places=12)

    def test_infeasible_nontarget_and_zero_target_have_finite_certificate(self):
        q = torch.tensor([[.1, .7, .1, .1], [0., .8, .1, .1]], dtype=torch.float64)
        result = minimum_target_preserving_scale(q, margin=.01)
        self.assertFalse(bool(result["feasible"].any()))
        self.assertTrue(bool(torch.isfinite(result["minimum_scale"]).all()))

    def test_random_geometry_has_no_constraint_violations(self):
        result = verify_geometry_properties(count=1000, seed=20261002)
        self.assertEqual(result["sample_count"], 1000)
        self.assertEqual(result["purpose"], "structural_formula_check_not_classifier_performance")
        self.assertEqual(result["nested_segment_violations"], 0)
        self.assertEqual(result["target_set_promotions"], 0)
        self.assertEqual(result["scale_margin_violations"], 0)
        self.assertLess(result["commutation_max_abs_residual"], 1e-12)


if __name__ == "__main__":
    unittest.main()
