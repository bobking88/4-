from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class HierarchicalExperimentAnalysisTests(unittest.TestCase):
    def test_summary_requires_exactly_three_formal_runs(self) -> None:
        from analyze_hierarchical_experiment import summarize_formal_runs

        with self.assertRaisesRegex(ValueError, "exactly three"):
            summarize_formal_runs([{"run_name": "seed_a", "macro_f1": 0.7}])

    def test_load_and_summary_calculates_mean_and_sample_std(self) -> None:
        from analyze_hierarchical_experiment import load_run_metrics, summarize_formal_runs

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            run_dirs = []
            for index, value in enumerate((0.70, 0.75, 0.80), start=1):
                run_dir = base / f"seed{index}"
                run_dir.mkdir()
                (run_dir / "test_metrics.json").write_text(
                    json.dumps(
                        {
                            "macro_f1": value,
                            "accuracy": value + 0.01,
                            "macro_precision": value - 0.01,
                            "macro_recall": value + 0.02,
                            "species_accuracy": value - 0.10,
                            "class_recall": {
                                "target_mineral": value,
                                "ti_bearing_negative": value,
                                "gangue_negative": value,
                                "metallic_hard_negative": value,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                run_dirs.append(run_dir)

            rows = load_run_metrics(run_dirs)
            summary = summarize_formal_runs(rows)

        self.assertEqual([row["run_name"] for row in rows], ["seed1", "seed2", "seed3"])
        self.assertAlmostEqual(summary["macro_f1"]["mean"], 0.75)
        self.assertAlmostEqual(summary["macro_f1"]["sample_std"], 0.05)
        self.assertAlmostEqual(summary["target_mineral_recall"]["mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
