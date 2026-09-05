from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class PHRAnalysisTests(unittest.TestCase):
    def test_pairwise_summary_keeps_ti_and_metallic_edges_separate(self) -> None:
        from analyze_phr_hrgv_experiments import summarize_pairwise_rows

        rows = [
            {"phr_ti_eligible": "1", "phr_ti_margin_regret": "0.10", "phr_ti_gate_selection_correct": "1", "phr_ti_expert_sign_agreement": "1", "phr_ti_sign_preserved": "1",
             "phr_metallic_eligible": "1", "phr_metallic_margin_regret": "0.30", "phr_metallic_gate_selection_correct": "0", "phr_metallic_expert_sign_agreement": "1", "phr_metallic_sign_preserved": "1"},
            {"phr_ti_eligible": "0", "phr_ti_margin_regret": "0.00", "phr_ti_gate_selection_correct": "0", "phr_ti_expert_sign_agreement": "0", "phr_ti_sign_preserved": "0",
             "phr_metallic_eligible": "1", "phr_metallic_margin_regret": "0.10", "phr_metallic_gate_selection_correct": "1", "phr_metallic_expert_sign_agreement": "0", "phr_metallic_sign_preserved": "0"},
        ]
        summary = summarize_pairwise_rows(rows)

        self.assertEqual(summary["phr_ti_eligible_count"], 1)
        self.assertAlmostEqual(summary["phr_ti_mean_margin_regret"], 0.10)
        self.assertEqual(summary["phr_metallic_eligible_count"], 2)
        self.assertAlmostEqual(summary["phr_metallic_mean_margin_regret"], 0.20)
        self.assertAlmostEqual(summary["phr_metallic_gate_selection_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
