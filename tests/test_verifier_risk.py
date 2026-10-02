import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import verifier_risk as risk
except ModuleNotFoundError:
    risk = None


class VerifierRiskTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(risk, "verifier risk implementation is missing")
        self.p = torch.tensor([[.6, .2, .1, .1]], dtype=torch.float64)
        self.l = torch.tensor([[math.log(2)]], dtype=torch.float64)

    def test_strength_endpoints_and_non_target_ratios(self):
        identity = risk.verification_family(self.p, self.l, torch.zeros_like(self.l))
        full = risk.verification_family(self.p, self.l, torch.ones_like(self.l))
        torch.testing.assert_close(identity, self.p, atol=1e-15, rtol=0)
        torch.testing.assert_close(full, torch.tensor([[3/7, 2/7, 1/7, 1/7]], dtype=torch.float64))

    def test_projection_clips_both_endpoints_and_keeps_interior(self):
        p, l = self.p.repeat(3, 1), self.l.repeat(3, 1)
        result = risk.conditional_risk_projection(p, l, torch.tensor([[0.], [.5], [1.]], dtype=torch.float64))
        torch.testing.assert_close(result["probabilities"][:, 0], torch.tensor([3/7, .5, .6], dtype=torch.float64))
        torch.testing.assert_close(result["gamma"], torch.tensor([[1.], [math.log(1.5)/math.log(2)], [0.]], dtype=torch.float64))

    def test_zero_contradiction_is_identity_with_finite_gradients(self):
        r = torch.tensor([[.3]], dtype=torch.float64, requires_grad=True)
        out = risk.conditional_risk_projection(self.p, torch.zeros_like(self.l), r)
        torch.testing.assert_close(out["probabilities"], self.p)
        self.assertEqual(float(out["gamma"].detach()), 0.)
        out["probabilities"].square().sum().backward()
        self.assertTrue(bool(torch.isfinite(r.grad).all()))

    def test_free_strength_has_gradient_and_batch_independent_output(self):
        gamma = torch.tensor([[.3], [.7]], dtype=torch.float64, requires_grad=True)
        p, l = self.p.repeat(2, 1), self.l.repeat(2, 1)
        out = risk.verification_family(p, l, gamma)
        torch.testing.assert_close(out[:1], risk.verification_family(p[:1], l[:1], gamma[:1]))
        out[:, 0].sum().backward()
        self.assertTrue(bool((gamma.grad < 0).all()))

    def test_regret_is_nonnegative_and_bounded_by_bernoulli_kl(self):
        torch.manual_seed(8)
        p = torch.softmax(torch.randn(500, 4, dtype=torch.float64), 1)
        l = torch.rand(500, 1, dtype=torch.float64) * 2
        r = torch.rand(500, 1, dtype=torch.float64) * .98 + .01
        rh = torch.rand(500, 1, dtype=torch.float64) * .98 + .01
        q = risk.conditional_risk_projection(p, l, r)["probabilities"][:, :1]
        qh = risk.conditional_risk_projection(p, l, rh)["probabilities"][:, :1]
        bce = lambda a: -r*a.log() - (1-r)*torch.log1p(-a)
        regret = bce(qh) - bce(q)
        bound = r*(r/rh).log() + (1-r)*((1-r)/(1-rh)).log()
        self.assertGreaterEqual(float(regret.min()), -1e-14)
        self.assertLessEqual(float((regret-bound).max()), 1e-14)

    def test_scalar_optimum_finds_interior_and_both_boundaries(self):
        p = self.p.repeat(10, 1)
        l = self.l.repeat(10, 1)
        labels = torch.tensor([0]*5 + [1]*5)
        result = risk.optimal_global_strength(p, l, labels)
        self.assertAlmostEqual(result["gamma"], math.log(1.5)/math.log(2), places=12)
        self.assertAlmostEqual(result["derivative_at_optimum"], 0., places=13)
        self.assertEqual(risk.optimal_global_strength(p, l, torch.zeros(10, dtype=torch.long))["gamma"], 0.)
        self.assertEqual(risk.optimal_global_strength(p, l, torch.ones(10, dtype=torch.long))["gamma"], 1.)
        self.assertEqual(risk.optimal_global_strength(p, torch.zeros_like(l), labels)["gamma"], 0.)

    def test_invalid_simplex_endpoints_dtype_and_strength_rejected(self):
        for p, l, gamma in [
            (torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64), self.l, self.l),
            (self.p * 2, self.l, self.l),
            (self.p, -self.l, self.l),
            (self.p, self.l.float(), self.l),
            (self.p, self.l, torch.tensor([[1.1]], dtype=torch.float64)),
            (self.p, self.l, torch.tensor([[float("nan")]], dtype=torch.float64)),
        ]:
            with self.subTest(p=p, l=l, gamma=gamma), self.assertRaises(ValueError):
                risk.verification_family(p, l, gamma)
        with self.assertRaises(ValueError):
            risk.optimal_global_strength(self.p, self.l, torch.tensor([4]))


if __name__ == "__main__":
    unittest.main()
