import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_abmp_capacity_figure import build_plot_data


class CapacityFigureTests(unittest.TestCase):
    def test_plot_counts_are_audited_counts_not_oracle_accuracy(self):
        subset = {"class_counts": {"0": 66}, "metrics": {"fixed_verified": {"target_recall": 35/66}},
                  "verification_effects": {"fixed": {"true_target_removed": 2, "false_target_removed": 3}},
                  "segment_capacity": {name: {"true_target_feasible": value} for name, value in (("restricted_verified", 35), ("full_verified", 38), ("full_pre", 38))}}
        result = build_plot_data({"subsets": {"gate_stop_projector_fit": subset, "projector_stop": subset}, "outer_images_loaded": False})
        self.assertEqual(result["gate_stop_projector_fit"]["capacity_counts"], [35, 35, 38, 38])
        self.assertEqual(result["gate_stop_projector_fit"]["target_count"], 66)
        self.assertEqual(result["gate_stop_projector_fit"]["fixed_verifier_removals"], [2, 3])

    def test_figure_refuses_outer_evaluation_claim(self):
        with self.assertRaises(ValueError):
            build_plot_data({"outer_images_loaded": True})


if __name__ == "__main__":
    unittest.main()
