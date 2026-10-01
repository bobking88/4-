import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_abmp_development import diagnose_geometry


class GeometryTests(unittest.TestCase):
    def test_detects_independent_margin_capacity_and_fragile_target(self):
        q0 = torch.tensor([[.42, .36, .11, .11]], dtype=torch.float64)
        qp = torch.tensor([[.38, .42, .10, .10]], dtype=torch.float64)
        result = diagnose_geometry(q0, qp, torch.tensor([0]), torch.tensor([[.9]], dtype=torch.float64), torch.tensor([[.04]], dtype=torch.float64), {"epsilon_max": .04, "tau_p": .4, "tau_m": .05, "delta": .005})
        self.assertEqual(result["q0_target_to_candidate_nontarget_count"], 1)
        self.assertEqual(result["target_recall_gain_bound"], 0.)
        self.assertGreater(max(r["independent_margin_capacity_count"] for r in result["capacity_by_config"]), 0)

    def test_identical_experts_have_no_routing_capacity(self):
        q0 = torch.tensor([[.7, .1, .1, .1], [.1, .7, .1, .1]], dtype=torch.float64)
        result = diagnose_geometry(q0, q0, torch.tensor([0, 1]), torch.full((2, 1), .5, dtype=torch.float64), torch.full((2, 1), .02, dtype=torch.float64), {"epsilon_max": .04, "tau_p": .4, "tau_m": .05, "delta": .005})
        self.assertEqual(result["q0_candidate_argmax_agreement"], 1.0)
        self.assertEqual(result["fixed_mean_budget_max_probability_difference"], 0.0)
        self.assertEqual(result["accuracy_absolute_change_bound"], 0.0)

    def test_records_endpoint_ties_for_capacity_theorem_assumption(self):
        q0 = torch.tensor([[.4, .4, .1, .1]], dtype=torch.float64)
        qp = torch.tensor([[.7, .1, .1, .1]], dtype=torch.float64)
        result = diagnose_geometry(q0, qp, torch.tensor([0]), torch.tensor([[.5]], dtype=torch.float64), torch.tensor([[.02]], dtype=torch.float64), {"epsilon_max": .04, "tau_p": .4, "tau_m": .05, "delta": .005})
        self.assertEqual(result["q0_argmax_tie_count"], 1)
        self.assertEqual(result["candidate_argmax_tie_count"], 0)
        self.assertFalse(result["unique_endpoint_argmax_assumption_verified"])


if __name__ == "__main__":
    unittest.main()
