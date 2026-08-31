from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGTheoryReplayAnalysisTests(unittest.TestCase):
    @staticmethod
    def _write_rows(path: Path) -> None:
        fields = (
            "image_id",
            "split_group_id",
            "true_label",
            "gate",
            "direct_true_probability",
            "mapped_true_probability",
            "fused_true_probability",
            "hard_oracle_gate",
            "soft_oracle_gate",
            "routing_regret_nll",
        )
        rows = (
            {
                "image_id": "VTM-1",
                "split_group_id": "DG-1",
                "true_label": "target_mineral",
                "gate": "0.8",
                "direct_true_probability": "0.8",
                "mapped_true_probability": "0.2",
                "fused_true_probability": "0.68",
                "hard_oracle_gate": "1.0",
                "soft_oracle_gate": "0.999023437500",
                "routing_regret_nll": "0.162518929497",
            },
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_summarizes_b1_and_b2_without_numerical_violations(self) -> None:
        from analyze_rsg_theory_replay import analyze_rsg_theory_replay

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixed = root / "fixed"
            holdout = root / "photographer_holdout"
            fixed.mkdir()
            holdout.mkdir()
            self._write_rows(fixed / "seed20260727.csv")
            self._write_rows(holdout / "seed20260730.csv")

            summary = analyze_rsg_theory_replay(root, root / "analysis")

            self.assertEqual(summary["overall"]["sample_count"], 2)
            self.assertEqual(summary["overall"]["b1_violation_count"], 0)
            self.assertEqual(summary["overall"]["b2_violation_count"], 0)
            self.assertTrue((root / "analysis" / "theory_replay_summary.json").is_file())
            payload = json.loads(
                (root / "analysis" / "theory_replay_summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("Theorem B.1", payload["claim_boundary"])
            self.assertEqual(len(payload["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
