from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class MRPGExperimentMatrixTests(unittest.TestCase):
    def test_formal_mrpg_matrix_has_three_configurations_and_three_seeds(self) -> None:
        from run_mrpg_hrgv_experiments import CONFIGURATION_FLAGS, build_experiment_commands

        commands = build_experiment_commands(
            project_root=PROJECT_ROOT,
            manifest=Path("split.csv"),
            dataset_root=Path("dataset"),
            output_root=Path("outputs"),
            python_executable=Path("python"),
            device="cuda",
            torch_home=Path("torch-cache"),
            stage="formal",
        )

        self.assertEqual(
            set(CONFIGURATION_FLAGS),
            {
                "mrpg_complete",
                "mrpg_unconstrained_between",
                "mrpg_without_between",
            },
        )
        self.assertEqual(len(commands), 9)
        unconstrained = [
            command
            for command in commands
            if command.configuration == "mrpg_unconstrained_between"
        ]
        self.assertEqual(len(unconstrained), 3)
        for command in commands:
            self.assertIn("--enable-mrpg", command.arguments)
            self.assertIn("--detach-gate-features", command.arguments)
            self.assertIn("--couple-verifier-features", command.arguments)
        self.assertIn("unconstrained", unconstrained[0].arguments)


if __name__ == "__main__":
    unittest.main()
