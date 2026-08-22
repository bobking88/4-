from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class HRGVExperimentMatrixTests(unittest.TestCase):
    def test_matrix_contains_five_configurations_and_three_fixed_seeds(self) -> None:
        from run_hrgv_experiments import build_experiment_commands

        commands = build_experiment_commands(
            project_root=Path("project"),
            manifest=Path("split.csv"),
            dataset_root=Path("dataset"),
            output_root=Path("outputs"),
            python_executable=Path("python.exe"),
            device="cuda",
            torch_home=Path("shared-torch-cache"),
        )

        self.assertEqual(len(commands), 15)
        self.assertEqual(
            {command.configuration for command in commands},
            {
                "decoupled_residual",
                "residual_coupled",
                "gate_only",
                "equal_fusion",
                "no_contrast",
            },
        )
        for configuration in {command.configuration for command in commands}:
            seeds = {
                command.seed for command in commands if command.configuration == configuration
            }
            self.assertEqual(seeds, {20260727, 20260728, 20260729})

    def test_every_command_has_unique_output_and_required_paths(self) -> None:
        from run_hrgv_experiments import build_experiment_commands

        commands = build_experiment_commands(
            project_root=Path("project"),
            manifest=Path("split.csv"),
            dataset_root=Path("dataset"),
            output_root=Path("outputs"),
            python_executable=Path("python.exe"),
            device="cuda",
            torch_home=Path("shared-torch-cache"),
        )

        output_directories = {command.output_dir for command in commands}
        self.assertEqual(len(output_directories), len(commands))
        for command in commands:
            joined = " ".join(str(part) for part in command.arguments)
            self.assertIn("scripts/train_hrgv_mineral_classifier.py", joined.replace("\\", "/"))
            self.assertIn("split.csv", joined)
            self.assertIn("dataset", joined)
            self.assertIn("--device cuda", joined)
            self.assertIn(f"--seed {command.seed}", joined)
            self.assertIn("--torch-home shared-torch-cache", joined)

    def test_ablation_flags_match_the_registered_configuration(self) -> None:
        from run_hrgv_experiments import build_experiment_commands

        commands = build_experiment_commands(
            project_root=Path("project"),
            manifest=Path("split.csv"),
            dataset_root=Path("dataset"),
            output_root=Path("outputs"),
            python_executable=Path("python.exe"),
            device="cuda",
            torch_home=Path("shared-torch-cache"),
        )
        representative = {}
        for command in commands:
            representative.setdefault(command.configuration, " ".join(map(str, command.arguments)))

        self.assertNotIn("--couple-verifier-features", representative["decoupled_residual"])
        self.assertIn("--verifier-mode residual", representative["decoupled_residual"])
        self.assertIn("--couple-verifier-features", representative["residual_coupled"])
        self.assertIn("--disable-verifiers", representative["gate_only"])
        self.assertIn("--lambda-verifier 0", representative["gate_only"])
        self.assertIn("--fixed-gate 0.5", representative["equal_fusion"])
        self.assertIn("--couple-verifier-features", representative["equal_fusion"])
        self.assertIn("--lambda-contrast 0", representative["no_contrast"])
        self.assertIn("--couple-verifier-features", representative["no_contrast"])


if __name__ == "__main__":
    unittest.main()
