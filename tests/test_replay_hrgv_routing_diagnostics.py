from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class HighPrecisionRoutingReplayTests(unittest.TestCase):
    def test_builds_high_precision_rows_without_rounding_to_six_digits(self) -> None:
        from replay_hrgv_routing_diagnostics import build_high_precision_rows

        records = [
            {
                "image_id": "VTM-000001",
                "split_group_id": "DG-1",
                "true_label": "target_mineral",
            }
        ]
        rows = build_high_precision_rows(
            records,
            gates=[0.123456789123],
            direct_true_probabilities=[0.000000123456789],
            mapped_true_probabilities=[0.987654321987],
            fused_true_probabilities=[0.864197532864],
            hard_oracle_gates=[0.0],
            soft_oracle_gates=[0.000001234567],
            routing_regrets_nll=[0.135791357913],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], "VTM-000001")
        self.assertEqual(rows[0]["gate"], "0.123456789123")
        self.assertEqual(rows[0]["direct_true_probability"], "0.000000123456789")
        self.assertEqual(rows[0]["routing_regret_nll"], "0.135791357913")


if __name__ == "__main__":
    unittest.main()
