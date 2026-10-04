import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from analyze_frozen_visual_probe import capacity_counts
except ModuleNotFoundError:
    capacity_counts = None


class CapacityAnalysisTests(unittest.TestCase):
    def test_new_target_below_quarter_requires_target_mass_increase(self):
        self.assertIsNotNone(capacity_counts, "capacity analysis missing")
        p = torch.tensor([[.2, .5, .2, .1], [.3, .5, .1, .1], [.2, .5, .2, .1]], dtype=torch.float64)
        q = torch.tensor([[.4, .3, .2, .1], [.3, .25, .23, .22], [.4, .3, .2, .1]], dtype=torch.float64)
        result = capacity_counts(q, p, torch.tensor([0, 0, 1]))
        self.assertEqual(result["new_correct_targets"], 2)
        self.assertEqual(result["new_correct_targets_base_below_quarter"], 1)
        self.assertEqual(result["new_false_targets_base_below_quarter"], 1)
        self.assertEqual(result["promotions_without_target_mass_increase"], 1)
        self.assertEqual(result["target_mass_increased_rows"], 2)


if __name__ == "__main__":
    unittest.main()
