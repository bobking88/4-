from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class PHRExperimentMatrixTests(unittest.TestCase):
    def test_screen_registers_the_eight_predeclared_configurations(self) -> None:
        from run_phr_hrgv_experiments import build_experiment_commands

        commands = build_experiment_commands(
            project_root=Path("project"), manifest=Path("manifest.csv"),
            dataset_root=Path("images"), output_root=Path("outputs"),
            python_executable=Path("python.exe"), device="cpu", torch_home=Path("cache"),
            mode="screen",
        )

        self.assertEqual(len(commands), 8)
        self.assertEqual({command.seed for command in commands}, {20260728})
        self.assertEqual(
            {command.configuration for command in commands},
            {
                "rsg_reference", "phr_complete", "phr_fixed_half", "phr_hard_target",
                "phr_unweighted", "phr_coupled_features", "phr_ti_only", "phr_metallic_only",
            },
        )
        flags = {command.configuration: " ".join(command.arguments) for command in commands}
        self.assertIn("--enable-phr", flags["phr_complete"])
        self.assertIn("--fixed-gate 0.5", flags["phr_fixed_half"])
        self.assertIn("--phr-edges ti", flags["phr_ti_only"])
        self.assertIn("--phr-edges metallic", flags["phr_metallic_only"])

    def test_formal_runs_only_reference_and_complete_for_three_seeds(self) -> None:
        from run_phr_hrgv_experiments import build_experiment_commands

        commands = build_experiment_commands(
            project_root=Path("project"), manifest=Path("manifest.csv"),
            dataset_root=Path("images"), output_root=Path("outputs"),
            python_executable=Path("python.exe"), device="cpu", torch_home=Path("cache"),
            mode="formal",
        )

        self.assertEqual(len(commands), 6)
        self.assertEqual({command.configuration for command in commands}, {"rsg_reference", "phr_complete"})
        self.assertEqual({command.seed for command in commands}, {20260727, 20260728, 20260729})


if __name__ == "__main__":
    unittest.main()
