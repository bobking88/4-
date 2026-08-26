from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class CGDCExperimentMatrixTests(unittest.TestCase):
    def test_formal_matrix_has_five_configurations_and_three_seeds(self) -> None:
        from run_cgdc_rsg_experiments import CONFIGURATION_FLAGS, build_experiment_commands

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
            "rsg_complete",
            "cgdc_complete",
            "cgdc_shared_features",
            "cgdc_unconditional",
            "cgdc_no_decomposition_loss",
        })
        self.assertEqual(len(commands), 15)
        self.assertIn("cgdc_complete", {item.configuration for item in commands})
        complete = next(item for item in commands if item.configuration == "cgdc_complete")
        self.assertIn("--enable-cgdc", complete.arguments)
        self.assertIn("--detach-gate-features", complete.arguments)


if __name__ == "__main__":
    unittest.main()
