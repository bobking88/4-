from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TargetRecallExtensionTests(unittest.TestCase):
    def test_extension_registers_two_new_seeds_and_two_locked_configurations(self) -> None:
        from run_target_recall_extension import EXTENSION_SEEDS, REGISTERED_CONFIGURATIONS

        self.assertEqual(EXTENSION_SEEDS, (20260730, 20260731))
        self.assertEqual(REGISTERED_CONFIGURATIONS, ("rsg_complete", "mrpg_complete"))


if __name__ == "__main__":
    unittest.main()
