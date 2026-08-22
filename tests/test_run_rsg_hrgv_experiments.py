from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGHRGVExperimentMatrixTests(unittest.TestCase):
    def build(self, stage: str):
        from run_rsg_hrgv_experiments import build_experiment_commands

        return build_experiment_commands(
            project_root=Path("project"),
            manifest=Path("split.csv"),
            dataset_root=Path("dataset"),
            output_root=Path("outputs"),
            python_executable=Path("python.exe"),
            device="cuda",
            torch_home=Path("shared-torch-cache"),
            stage=stage,
        )

    def test_pilot_matrix_uses_one_seed_and_five_unique_configurations(self) -> None:
        commands = self.build("pilot")

        self.assertEqual(len(commands), 5)
        self.assertEqual({command.seed for command in commands}, {20260728})
        self.assertEqual(
            {command.configuration for command in commands},
            {
                "hrgv_reference",
                "rsg_complete",
                "rsg_hard_target",
                "rsg_unweighted",
                "rsg_coupled_gate",
            },
        )
        self.assertEqual(len({command.output_dir for command in commands}), len(commands))

    def test_formal_matrix_uses_three_registered_seeds(self) -> None:
        commands = self.build("formal")

        self.assertEqual(len(commands), 15)
        for configuration in {command.configuration for command in commands}:
            self.assertEqual(
                {command.seed for command in commands if command.configuration == configuration},
                {20260727, 20260728, 20260729},
            )

    def test_ablation_flags_match_registered_rsg_variants(self) -> None:
        commands = self.build("pilot")
        arguments = {
            command.configuration: " ".join(map(str, command.arguments))
            for command in commands
        }

        self.assertIn("--disable-gate-regret", arguments["hrgv_reference"])
        self.assertNotIn("--detach-gate-features", arguments["hrgv_reference"])
        for configuration in arguments:
            self.assertIn("--couple-verifier-features", arguments[configuration])
        for configuration in ("rsg_complete", "rsg_hard_target", "rsg_unweighted"):
            self.assertIn("--lambda-gate-regret 0.1", arguments[configuration])
            self.assertIn("--detach-gate-features", arguments[configuration])
        self.assertIn("--hard-gate-target", arguments["rsg_hard_target"])
        self.assertIn("--unweighted-gate-regret", arguments["rsg_unweighted"])
        self.assertIn("--lambda-gate-regret 0.1", arguments["rsg_coupled_gate"])
        self.assertIn("--couple-gate-features", arguments["rsg_coupled_gate"])
        self.assertNotIn("--detach-gate-features", arguments["rsg_coupled_gate"])


if __name__ == "__main__":
    unittest.main()
