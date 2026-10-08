import copy
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lc_role_adapter import RoleResidualModel, pair_loss
try:
    import lc_role_audit as audit
except ModuleNotFoundError:
    audit = None


class TheoryCertificateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, "LC-RFA theory audit module missing")
        torch.set_num_threads(1)
        self.gen = torch.Generator().manual_seed(20261006)
        self.model = RoleResidualModel("R1", seed=20261002, alpha_max=.5, beta_max=.5)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if "up.weight" in name or name.startswith("head.2"):
                    p.copy_(.3 * torch.randn(p.shape, dtype=p.dtype, generator=self.gen))
        self.batch = self.synthetic_batch(80)

    def synthetic_batch(self, n):
        logits = 15 * torch.randn(n, 4, generator=self.gen, dtype=torch.float64)
        logits[:4] = torch.tensor([[50, 0, 0, 0], [-50, 0, -10, -20], [0, 0, 0, 0], [0, 0, -40, -50]])
        return dict(h=torch.randn(n, 64, generator=self.gen, dtype=torch.float64),
                    e=torch.randn(n, 23, generator=self.gen, dtype=torch.float64),
                    log_anchor=torch.log_softmax(logits, 1), labels=torch.arange(n) % 4,
                    records=[dict(image_id=f"synthetic-{i}") for i in range(n)])

    def test_valid_nonzero_network_and_one_violation_cannot_average_away(self):
        with torch.no_grad():
            result = self.model(self.batch["h"], self.batch["e"], self.batch["log_anchor"])
            valid = audit.audit_theory(self.model, self.batch, result)
            self.assertEqual(valid["violation_count"], 0)
            broken = copy.deepcopy(result)
            broken["delta"][0, 0] = 10
            report = audit.audit_theory(self.model, self.batch, broken)
        self.assertEqual(report["violation_count"], 1)
        self.assertEqual(report["violating_image_ids"], ["synthetic-0"])
        self.assertGreater(report["max_residual_by_check"]["delta_norm"], 1)

    def test_lipschitz_certificate_uses_actual_matrices_and_b_squared(self):
        with torch.no_grad():
            result = self.model(self.batch["h"], self.batch["e"], self.batch["log_anchor"])
            report = audit.audit_theory(self.model, self.batch, result)
        expected = (1+math.exp(-1))*float(torch.linalg.matrix_norm(self.model.head[2].weight.detach(), 2))*float(torch.linalg.matrix_norm(self.model.head[0].weight[:, :64].detach(), 2))
        self.assertAlmostEqual(report["L_F"], expected, places=13)
        for row, b in zip(report["rows"], result["b"][:, 0]):
            self.assertAlmostEqual(row["eta"], expected*float(b)**2, places=12)
            self.assertLessEqual(row["logq_drift"], row["eta"] + 1e-10*(1+row["eta"]))
            self.assertLessEqual(row["pair_log_odds_drift"], row["eta"] + 1e-10*(1+row["eta"]))
        x = torch.linspace(-50, 50, 10001, dtype=torch.float64, requires_grad=True)
        derivative, = torch.autograd.grad(torch.nn.functional.silu(x).sum(), x)
        self.assertLessEqual(float(derivative.abs().max()), 1+math.exp(-1))
        self.assertFalse(report["auxiliary_b_squared_certificate_applicable"])

    def test_same_weight_plain_replay_not_anchor_and_flat_reconstruction(self):
        with torch.no_grad():
            result = self.model(self.batch["h"], self.batch["e"], self.batch["log_anchor"])
            plain = audit.replay_without_adaptation(self.model, self.batch)
            self.assertGreater(float((plain["logq"] - self.batch["log_anchor"]).abs().max()), .001)
            expected = self.model.readout(self.batch["h"], self.batch["e"], self.batch["log_anchor"], result["b"])
            torch.testing.assert_close(plain["logq"], expected["logq"], atol=0, rtol=0)
            reconstructed = audit.check_flat_reconstruction(result, self.batch["log_anchor"])
            self.assertEqual(reconstructed["violation_count"], 0)
            self.assertLess(reconstructed["max_abs_residual"], 1e-12)

    def test_unique_margin_protection_and_ties(self):
        model = RoleResidualModel("R1", seed=1, alpha_max=0, beta_max=0)
        with torch.no_grad():
            result = model(self.batch["h"], self.batch["e"], self.batch["log_anchor"])
            report = audit.audit_theory(model, self.batch, result)
        self.assertTrue(report["rows"][0]["winner_protected"])
        self.assertFalse(report["rows"][2]["winner_protected"])
        self.assertEqual(report["violation_count"], 0)
        for row in report["rows"]:
            if row["winner_protected"]:
                self.assertFalse(row["winner_changed"])
            if row["target_impossible"]:
                self.assertLess(row["actual_target_margin"], 1e-10)
            self.assertNotIn("correction_guaranteed", row)

    def test_simultaneous_class_branch_routing_and_supervision_relabel(self):
        permutation = torch.tensor([0, 3, 1, 2])
        inverse = permutation.argsort()
        branches = [2, 0, 1]
        relabeled = copy.deepcopy(self.model)
        relabeled.adapters = torch.nn.ModuleList([copy.deepcopy(self.model.adapters[i]) for i in branches])
        with torch.no_grad():
            relabeled.head[2].weight.copy_(self.model.head[2].weight[permutation])
            relabeled.head[2].bias.copy_(self.model.head[2].bias[permutation])
            old = self.model(self.batch["h"], self.batch["e"], self.batch["log_anchor"], auxiliary=True)
            new = relabeled(self.batch["h"], self.batch["e"], self.batch["log_anchor"][:, permutation], auxiliary=True)
        torch.testing.assert_close(new["logq"], old["logq"][:, permutation], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(new["aux_logq"], old["aux_logq"][:, branches][:, :, permutation])
        torch.testing.assert_close(pair_loss(new["aux_logq"], inverse[self.batch["labels"]]), pair_loss(old["aux_logq"], self.batch["labels"]))

    def test_5000_fixed_seed_edge_cases_and_zero_budget(self):
        batch = self.synthetic_batch(5000)
        with torch.no_grad():
            result = self.model(batch["h"], batch["e"], batch["log_anchor"])
            report = audit.audit_theory(self.model, batch, result)
            zero = RoleResidualModel("R1", seed=1, alpha_max=0, beta_max=0)
            zero_result = zero(batch["h"], batch["e"], batch["log_anchor"])
            zero_report = audit.audit_theory(zero, batch, zero_result)
        self.assertEqual(report["row_count"], 5000)
        self.assertEqual(report["violation_count"], 0, report["max_residual_by_check"])
        self.assertEqual(zero_report["violation_count"], 0)
        self.assertTrue(all(r["eta"] == 0 for r in zero_report["rows"]))
        self.assertGreater(report["protected_winner_count"], 0)
        self.assertGreater(report["target_impossible_count"], 0)
        self.assertGreaterEqual(report["loose_certificate_fraction"], 0)

    def test_unbounded_arm_does_not_receive_bounded_probability_claim(self):
        model = RoleResidualModel("U1", seed=5, alpha_max=.5, beta_max=.5)
        result = model(self.batch["h"], self.batch["e"], self.batch["log_anchor"])
        report = audit.audit_theory(model, self.batch, result)
        self.assertFalse(report["bounded_probability_certificate_applicable"])
        self.assertIsNone(report["rows"][0]["eta"])
        self.assertIsNone(report["rows"][0]["target_margin_envelope"])


if __name__ == "__main__":
    unittest.main()
