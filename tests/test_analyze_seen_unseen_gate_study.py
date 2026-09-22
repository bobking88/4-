import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_seen_unseen_gate_study import summarize_runs


class SeenUnseenAnalysisTests(unittest.TestCase):
    def test_pairs_same_objective_and_seed(self):
        summary = {
            "references": {"equal": {"final_nll_nats": 0.7}},
            "runs": {
                "seen_final_nll_seed1": {
                    "source": "seen", "objective": "final_nll", "seed": 1,
                    "selected": {"epoch": 1, "final_nll_nats": 0.8, "final_macro_f1": 0.6},
                },
                "unseen_final_nll_seed1": {
                    "source": "unseen", "objective": "final_nll", "seed": 1,
                    "selected": {"epoch": 2, "final_nll_nats": 0.75, "final_macro_f1": 0.62},
                },
            },
        }

        result = summarize_runs(summary)

        pair = result["paired_differences"][0]
        self.assertAlmostEqual(pair["unseen_minus_seen_final_nll_nats"], -0.05)
        self.assertAlmostEqual(pair["unseen_minus_seen_final_macro_f1"], 0.02)
        self.assertEqual(result["direction_counts"]["final_nll"]["nll_improved"], 1)

    def test_rejects_missing_pair(self):
        summary = {
            "references": {},
            "runs": {
                "only": {
                    "source": "seen", "objective": "final_nll", "seed": 1,
                    "selected": {"epoch": 1, "final_nll_nats": 0.8},
                }
            },
        }

        with self.assertRaisesRegex(ValueError, "paired run"):
            summarize_runs(summary)


if __name__ == "__main__":
    unittest.main()
