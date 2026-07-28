from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class FigureSummaryTests(unittest.TestCase):
    def test_summarize_values_returns_sample_standard_deviation(self) -> None:
        from generate_paper_figures import summarize_values

        mean, sample_std = summarize_values([0.70, 0.80, 0.90])

        self.assertAlmostEqual(mean, 0.80)
        self.assertAlmostEqual(sample_std, 0.10)

    def test_summarize_values_rejects_fewer_than_two_runs(self) -> None:
        from generate_paper_figures import summarize_values

        with self.assertRaisesRegex(ValueError, "two"):
            summarize_values([0.80])


if __name__ == "__main__":
    unittest.main()
