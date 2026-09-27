from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_oos_confirmation_experts import (  # noqa: E402
    CONFIRMATION_EXPERT_SEEDS,
    build_expert_commands,
)


class OOSConfirmationExpertMatrixTests(unittest.TestCase):
    def test_registers_three_independent_expert_seeds(self) -> None:
        commands = build_expert_commands(
            project_root=Path("project"),
            manifest=Path("protocol/expert.csv"),
            dataset_root=Path("dataset"),
            output_root=Path("outputs"),
            python_executable=Path("python.exe"),
            torch_home=Path("torch-cache"),
            device="cuda",
        )

        self.assertEqual(len(commands), 3)
        self.assertEqual({command.seed for command in commands}, set(CONFIRMATION_EXPERT_SEEDS))
        self.assertEqual(len({command.output_dir for command in commands}), 3)
        for command in commands:
            arguments = list(map(str, command.arguments))
            self.assertIn("--validation-only", arguments)
            self.assertEqual(arguments[arguments.index("--epochs") + 1], "30")
            self.assertEqual(arguments[arguments.index("--patience") + 1], "8")
            self.assertEqual(arguments[arguments.index("--batch-size") + 1], "16")
            self.assertEqual(arguments[arguments.index("--lambda-gate-regret") + 1], "0.0")
            self.assertNotIn("--no-pretrained", arguments)


if __name__ == "__main__":
    unittest.main()
