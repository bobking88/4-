from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class MRPGAnalysisTests(unittest.TestCase):
    def test_mrpg_analysis_declares_both_complete_baselines(self) -> None:
        from analyze_mrpg_hrgv_experiments import REQUIRED_CONFIGURATIONS

        self.assertEqual(REQUIRED_CONFIGURATIONS[:2], ("rsg_complete", "rpg_complete"))
        self.assertEqual(
            set(REQUIRED_CONFIGURATIONS),
            {
                "rsg_complete",
                "rpg_complete",
                "mrpg_complete",
                "mrpg_unconstrained_between",
                "mrpg_without_between",
            },
        )

    def test_mrpg_analysis_registers_direct_component_ablations(self) -> None:
        from analyze_mrpg_hrgv_experiments import DIRECT_MRPG_ABLATIONS

        self.assertEqual(
            DIRECT_MRPG_ABLATIONS,
            (
                ("mrpg_complete", "mrpg_unconstrained_between"),
                ("mrpg_complete", "mrpg_without_between"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
