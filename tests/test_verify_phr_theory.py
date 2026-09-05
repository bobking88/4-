from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class PHRTheoryVerificationTests(unittest.TestCase):
    def test_effective_gate_reconstructs_inactive_base_margin(self):
        import torch
        from hrgv_network import effective_pairwise_gates

        direct = torch.tensor([[4., 0.], [2., -2.]], dtype=torch.float64)
        mapped = torch.tensor([[0., 2.], [2., 2.]], dtype=torch.float64)
        fused = torch.tensor([[1., 1.], [2., 1.]], dtype=torch.float64)
        gate = effective_pairwise_gates(direct, mapped, fused, torch)
        self.assertTrue(torch.allclose(gate * direct + (1-gate) * mapped, fused))
        self.assertEqual(float(gate[1, 0]), .5)

    def test_ti_only_mask_removes_metallic_edge_from_diagnostics(self):
        import torch
        from hrgv_network import restrict_pairwise_diagnostics

        diagnostics = {
            "eligible_mask": torch.tensor([[True, True], [False, True]]),
            "margin_regrets": torch.tensor([[.1, .2], [.3, .4]]),
            "hard_gate_selection_correct": torch.tensor([[True, False], [False, True]]),
        }
        masked = restrict_pairwise_diagnostics(diagnostics, "ti", torch)
        self.assertTrue(masked["eligible_mask"][:, 0].equal(diagnostics["eligible_mask"][:, 0]))
        self.assertFalse(masked["eligible_mask"][:, 1].any())
        self.assertTrue(torch.equal(masked["margin_regrets"][:, 1], torch.zeros(2)))
        self.assertFalse(masked["hard_gate_selection_correct"][:, 1].any())

    def test_seeded_numerical_checks_and_counterexample(self):
        from verify_phr_theory import verify_properties

        result = verify_properties(seed=20260905, samples=512)
        self.assertTrue(result["all_checks_passed"])
        self.assertEqual(result["evidence_type"], "synthetic_numerical_verification")
        self.assertLess(result["max_identity_error"], 1e-10)
        self.assertLess(result["max_projection_error"], 1e-10)
        self.assertGreater(result["counterexample"]["gangue_nll_change"], 0)
        self.assertEqual(result, verify_properties(seed=20260905, samples=512))

    def test_invalid_sample_count_is_rejected(self):
        from verify_phr_theory import verify_properties

        with self.assertRaises(ValueError):
            verify_properties(samples=0)


if __name__ == "__main__":
    unittest.main()
