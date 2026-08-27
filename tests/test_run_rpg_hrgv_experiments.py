from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RPGExperimentMatrixTests(unittest.TestCase):
    def test_formal_rpg_matrix_has_four_configurations_and_three_seeds(self) -> None:
        from run_rpg_hrgv_experiments import CONFIGURATION_FLAGS, build_experiment_commands

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

        self.assertEqual(set(CONFIGURATION_FLAGS), {
            "rpg_complete",
            "rpg_without_within",
            "rpg_without_between",
            "rpg_total_entropy_only",
        })
        self.assertEqual(len(commands), 12)
        self.assertEqual(
            {(item.configuration, item.seed) for item in commands},
            {
                (configuration, seed)
                for configuration in CONFIGURATION_FLAGS
                for seed in (20260727, 20260728, 20260729)
            },
        )
        for command in commands:
            self.assertIn("--enable-rpg", command.arguments)
            self.assertIn("--detach-gate-features", command.arguments)
            self.assertIn("--verifier-mode", command.arguments)
            self.assertIn("residual", command.arguments)


if __name__ == "__main__":
    unittest.main()
