from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGGateReliabilityAnalysisTests(unittest.TestCase):
    @staticmethod
    def _write_replay(path: Path) -> None:
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
            ("1", "0.80", "0.20", "0.80", "1", "0.9995", "0.162518929497"),
            ("2", "0.20", "0.80", "0.20", "0", "0.0005", "0.162518929497"),
            ("3", "0.90", "0.10", "0.95", "1", "0.99999", "0.045462374077"),
            ("4", "0.10", "0.90", "0.05", "0", "0.00001", "0.045462374077"),
            ("5", "0.60", "0.40", "0.75", "1", "0.8830", "0.087011376990"),
            ("6", "0.40", "0.60", "0.25", "0", "0.1170", "0.087011376990"),
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for image_id, direct, mapped, gate, hard, soft, regret in rows:
                fused = float(gate) * float(direct) + (1.0 - float(gate)) * float(mapped)
                writer.writerow(
                    {
                        "image_id": image_id,
                        "split_group_id": f"DG-{image_id}",
                        "true_label": "target_mineral",
                        "gate": gate,
                        "direct_true_probability": direct,
                        "mapped_true_probability": mapped,
                        "fused_true_probability": f"{fused:.15f}",
                        "hard_oracle_gate": hard,
                        "soft_oracle_gate": soft,
                        "routing_regret_nll": regret,
                    }
                )

    def test_generates_b1_b2_strata_and_keeps_claim_boundary(self) -> None:
        from analyze_rsg_gate_reliability import analyze_rsg_gate_reliability

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay_root = root / "replay"
            output_dir = root / "analysis"
            for protocol, seed in (("fixed", "seed20260727"), ("resnet50_portability", "seed20260728")):
                protocol_dir = replay_root / protocol
                protocol_dir.mkdir(parents=True)
                self._write_replay(protocol_dir / f"{seed}.csv")

            summary = analyze_rsg_gate_reliability(replay_root, output_dir, strata_count=3)

            self.assertEqual(summary["overall"]["sample_count"], 12)
            self.assertLess(summary["overall"]["exact_decomposition_max_abs_residual"], 1e-10)
            self.assertEqual(summary["overall"]["exact_decomposition_violation_count"], 0)
            self.assertEqual(summary["overall"]["b1_local_violation_count"], 0)
            self.assertEqual(summary["overall"]["b2_violation_count"], 0)
            self.assertEqual(set(summary["protocols"]), {"fixed", "resnet50_portability"})
            self.assertTrue((output_dir / "b1_local_bound_strata.csv").is_file())
            self.assertTrue((output_dir / "b2_margin_strata.csv").is_file())
            persisted = json.loads((output_dir / "gate_reliability_summary.json").read_text(encoding="utf-8"))
            self.assertIn("mechanism diagnosis", persisted["claim_boundary"])
            with (output_dir / "b1_local_bound_strata.csv").open(encoding="utf-8-sig") as handle:
                b1_rows = list(csv.DictReader(handle))
            self.assertEqual(len(b1_rows), 6)
            self.assertTrue(all(int(row["b1_local_violation_count"]) == 0 for row in b1_rows))
            self.assertTrue(
                all(float(row["exact_decomposition_max_abs_residual"]) < 1e-10 for row in b1_rows)
            )


if __name__ == "__main__":
    unittest.main()
