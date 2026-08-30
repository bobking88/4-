from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class AnalyzeTargetRecallExtensionTests(unittest.TestCase):
    def test_registered_run_directories_merge_original_and_extension_seed_ranges(self) -> None:
        from analyze_target_recall_extension import (
            EXTENSION_SEEDS,
            FIVE_SEEDS,
            ORIGINAL_SEEDS,
            registered_run_directories,
        )

        run_directories = registered_run_directories(
            Path("rsg_original_runs"),
            Path("mrpg_original_runs"),
            Path("extension_runs"),
        )

        self.assertEqual(ORIGINAL_SEEDS, ("20260727", "20260728", "20260729"))
        self.assertEqual(EXTENSION_SEEDS, ("20260730", "20260731"))
        self.assertEqual(FIVE_SEEDS, ORIGINAL_SEEDS + EXTENSION_SEEDS)
        self.assertEqual(
            run_directories["rsg_complete"]["20260727"],
            Path("rsg_original_runs") / "formal_rsg_complete_seed20260727",
        )
        self.assertEqual(
            run_directories["mrpg_complete"]["20260727"],
            Path("mrpg_original_runs") / "formal_mrpg_complete_seed20260727",
        )
        self.assertEqual(
            run_directories["mrpg_complete"]["20260731"],
            Path("extension_runs") / "extension_mrpg_complete_seed20260731",
        )


if __name__ == "__main__":
    unittest.main()
