import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_open_set_protocol import evaluate_open_set_scores


class OpenSetProtocolTests(unittest.TestCase):
    def test_perfectly_separated_scores_have_perfect_auc_and_zero_fpr95(self) -> None:
        result = evaluate_open_set_scores([0.95, 0.85, 0.80], [0.30, 0.20, 0.10])

        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["fpr_at_95_tpr"], 0.0)
        self.assertEqual(result["known_count"], 3)
        self.assertEqual(result["unknown_count"], 3)

    def test_scores_require_both_known_and_unknown_examples(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_open_set_scores([0.9], [])

    def test_threshold_retains_at_least_requested_known_fraction(self) -> None:
        result = evaluate_open_set_scores([0.95, 0.90, 0.70, 0.60], [0.75, 0.20], target_tpr=0.75)

        self.assertGreaterEqual(result["known_accept_rate"], 0.75)
        self.assertAlmostEqual(result["threshold"], 0.70)


if __name__ == "__main__":
    unittest.main()
