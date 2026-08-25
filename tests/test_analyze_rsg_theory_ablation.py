from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGTheoryAblationTests(unittest.TestCase):
    def test_parse_config_roots_preserves_named_roots(self) -> None:
        from analyze_rsg_theory_ablation import parse_config_roots

        roots = parse_config_roots(
            [
                "rsg_complete=C:/runs/controlled",
                "rsg_hard_target=C:/runs/ablation",
            ]
        )

        self.assertEqual(roots["rsg_complete"], Path("C:/runs/controlled"))
        self.assertEqual(roots["rsg_hard_target"], Path("C:/runs/ablation"))

    def test_parse_config_roots_rejects_duplicate_configuration(self) -> None:
        from analyze_rsg_theory_ablation import parse_config_roots

        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_config_roots(["rsg_complete=C:/one", "rsg_complete=C:/two"])


if __name__ == "__main__":
    unittest.main()
