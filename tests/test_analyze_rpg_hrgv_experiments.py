from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RPGAnalysisTests(unittest.TestCase):
    def test_rpg_analysis_registers_partitioned_ablation_names(self) -> None:
        from analyze_rpg_hrgv_experiments import REQUIRED_CONFIGURATIONS

        self.assertEqual(REQUIRED_CONFIGURATIONS[0], "rsg_complete")
        self.assertEqual(
            set(REQUIRED_CONFIGURATIONS),
            {
                "rsg_complete",
                "rpg_complete",
                "rpg_without_within",
                "rpg_without_between",
                "rpg_total_entropy_only",
            },
        )


if __name__ == "__main__":
    unittest.main()
