import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from abmp_rsg import (
    AdaptiveBudgetPolicy,
    apply_abmp_projection,
    audit_abmp_projection,
    risk_supervision_targets,
)


class ABMPPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260930)
        torch.set_num_threads(1)

    def test_heads_have_gradients_and_bounded_budget(self):
        model = AdaptiveBudgetPolicy()
        output = model(torch.randn(8, 18), epsilon_max=0.04)
        self.assertEqual(output["raw_route"].shape, (8, 1))
        self.assertTrue(bool((output["epsilon"] <= 0.04).all()))
        self.assertTrue(bool((output["epsilon"] >= 0).all()))
        (output["raw_route"].sum() + output["epsilon"].sum()).backward()
        for head in (model.route_head, model.budget_head):
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
            self.assertTrue(bool(torch.isfinite(head.weight.grad).all()))

    def test_eval_is_independent_of_batch_members(self):
        model = AdaptiveBudgetPolicy().eval()
        evidence = torch.randn(9, 18)
        whole = model(evidence, epsilon_max=0.08, lambda_cal=0.5)
        one = model(evidence[3:4], epsilon_max=0.08, lambda_cal=0.5)
        for key in ("epsilon", "raw_route"):
            torch.testing.assert_close(whole[key][3:4], one[key], atol=1e-7, rtol=1e-6)

    def test_calibration_offset_decreases_budget_only(self):
        model = AdaptiveBudgetPolicy().eval()
        evidence = torch.randn(7, 18)
        low = model(evidence, epsilon_max=0.08, lambda_cal=-0.5)
        high = model(evidence, epsilon_max=0.08, lambda_cal=0.5)
        self.assertTrue(bool((high["epsilon"] < low["epsilon"]).all()))
        torch.testing.assert_close(high["raw_route"], low["raw_route"])

    def test_policy_rejects_invalid_evidence_and_parameters(self):
        model = AdaptiveBudgetPolicy()
        for evidence, cap in ((torch.randn(2, 17), .04), (torch.full((2, 18), float("nan")), .04), (torch.randn(2, 18), -.1)):
            with self.assertRaises(ValueError):
                model(evidence, epsilon_max=cap)


class ABMPProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260930)
        torch.set_num_threads(1)
        self.q0 = torch.tensor([[.75, .10, .10, .05], [.40, .45, .10, .05]], dtype=torch.float64)
        self.qphi = torch.tensor([[.25, .60, .10, .05], [.20, .65, .10, .05]], dtype=torch.float64)
        self.route = torch.ones(2, 1, dtype=torch.float64)
        self.eps = torch.full((2, 1), .08, dtype=torch.float64)

    def project(self, **kwargs):
        return apply_abmp_projection(self.q0, self.qphi, self.route, self.eps, tau_p=.70, tau_m=.10, delta=.01, **kwargs)

    def test_posterior_cap_activates_and_preserves_bound(self):
        result = self.project()
        self.assertTrue(bool(result["posterior_active"].all()))
        self.assertTrue(bool((result["positive_target_harm"] <= self.eps + 1e-12).all()))
        torch.testing.assert_close(result["projected_route"], torch.tensor([[.16], [.4]], dtype=torch.float64))

    def test_margin_cap_preserves_target_anchor_and_false_positive(self):
        result = self.project(mode="margin")
        self.assertEqual(result["anchor_mask"].flatten().tolist(), [True, False])
        self.assertTrue(bool(result["margin_active"][0]))
        self.assertAlmostEqual(float(result["final_probabilities"][0, 0] - result["final_probabilities"][0, 1]), .01)
        audit = audit_abmp_projection(result, self.q0, self.qphi, torch.tensor([1, 0]), .08, .01)
        self.assertEqual(audit["anchor_false_positive_count"], 1)
        self.assertEqual(audit["anchor_retention_rate"], 1.0)

    def test_margin_is_independent_on_boundary_target_anchor(self):
        q0 = torch.tensor([[.42, .36, .11, .11]], dtype=torch.float64)
        qp = torch.tensor([[.38, .42, .10, .10]], dtype=torch.float64)
        args = (q0, qp, torch.tensor([[.9]], dtype=torch.float64), .04)
        full = apply_abmp_projection(*args, tau_p=.4, tau_m=.05, delta=.005)
        posterior = apply_abmp_projection(*args, tau_p=.4, tau_m=.05, delta=.005, mode="posterior")
        self.assertEqual(int(full["final_probabilities"].argmax(1)), 0)
        self.assertEqual(int(posterior["final_probabilities"].argmax(1)), 1)
        self.assertLess(float(full["projected_route"]), float(posterior["projected_route"]))

    def test_random_simplexes_satisfy_all_invariants(self):
        count = 10000
        q0 = torch.rand(count, 4, dtype=torch.float64)
        q0 = q0 / q0.sum(1, keepdim=True)
        qp = torch.rand_like(q0)
        qp = qp / qp.sum(1, keepdim=True)
        result = apply_abmp_projection(q0, qp, torch.rand(count, 1, dtype=torch.float64), torch.rand(count, 1, dtype=torch.float64) * .08, tau_p=.6, tau_m=.05, delta=.005)
        audit = audit_abmp_projection(result, q0, qp, torch.randint(4, (count,)), .08, .005)
        for key, value in audit.items():
            if key.endswith("violations"):
                self.assertEqual(value, 0, key)

    def test_zero_drop_zero_probabilities_and_exact_fallback(self):
        q0 = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=torch.float64)
        result = apply_abmp_projection(q0, q0, torch.zeros(2, 1, dtype=torch.float64), torch.zeros(2, 1, dtype=torch.float64), tau_p=.7, tau_m=.1, delta=.01)
        torch.testing.assert_close(result["final_probabilities"], q0)
        self.assertTrue(bool(torch.isfinite(result["posterior_cap"]).all()))

    def test_target_index_is_not_assumed_zero(self):
        order = [1, 2, 0, 3]
        result = apply_abmp_projection(self.q0[:, order], self.qphi[:, order], self.route, self.eps, tau_p=.7, tau_m=.1, delta=.01, target_index=2)
        torch.testing.assert_close(result["final_probabilities"], self.project()["final_probabilities"][:, order])

    def test_invalid_margin_and_budget_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_abmp_projection(self.q0, self.qphi, self.route, self.eps, tau_p=.7, tau_m=.01, delta=.02)
        with self.assertRaises(ValueError):
            apply_abmp_projection(self.q0, self.qphi, self.route, -self.eps, tau_p=.7, tau_m=.1, delta=.01)

    def test_dual_projection_gradients_are_finite(self):
        route = torch.full((2, 1), .9, dtype=torch.float64, requires_grad=True)
        eps = torch.full((2, 1), .03, dtype=torch.float64, requires_grad=True)
        result = apply_abmp_projection(self.q0, self.qphi, route, eps, tau_p=.7, tau_m=.1, delta=.01)
        (-result["final_probabilities"].log().mean()).backward()
        self.assertTrue(bool(torch.isfinite(route.grad).all()))
        self.assertTrue(bool(torch.isfinite(eps.grad).all()))
        self.assertGreater(float(eps.grad.abs().sum()), 0)

    def test_risk_targets_reward_candidate_label_gain(self):
        labels = torch.tensor([1, 0])
        result = risk_supervision_targets(self.q0, self.qphi, labels, .08)
        self.assertGreater(float(result["route_target"][0]), .9)
        self.assertLess(float(result["route_target"][1]), .1)
        self.assertEqual(result["budget_mask"].flatten().tolist(), [True, True])
        self.assertGreater(float(result["budget_target"][0]), float(result["budget_target"][1]))


if __name__ == "__main__":
    unittest.main()
