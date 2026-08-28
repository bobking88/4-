from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class HRGVArchitectureFigureTests(unittest.TestCase):
    def test_figure_exports_publication_bundle_and_all_required_modules(self) -> None:
        from generate_hrgv_architecture_figure import generate_hrgv_architecture_figure

        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "fig_hrgv_architecture"
            outputs = generate_hrgv_architecture_figure(prefix)

            self.assertEqual(
                set(outputs), {"png", "svg", "pdf", "tiff", "source_description"}
            )
            for path in outputs.values():
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 100, path)

            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "EfficientNet-B0",
                "Direct role expert",
                "Species expert",
                "Mapping matrix A",
                "Reliability gate",
                "Regret-supervised gate",
                "Ti-bearing verifier",
                "Metallic verifier",
                "Neutral-zone residual correction",
                "Final role posterior",
                "Role-aware contrastive head",
                "soft oracle target",
            ):
                self.assertIn(label, svg)

            with Image.open(outputs["png"]) as image:
                self.assertGreaterEqual(image.width, 1800)
                self.assertGreaterEqual(image.height, 1000)

    def test_cgdc_figure_exports_disagreement_calibration_modules(self) -> None:
        from generate_hrgv_architecture_figure import generate_cgdc_architecture_figure

        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_cgdc_architecture_figure(
                Path(temporary) / "fig_cgdc_rsg_hrgv_architecture"
            )

            self.assertTrue(outputs["png"].exists())
            self.assertTrue(outputs["svg"].exists())
            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "Direct residual adapter",
                "Species residual adapter",
                "Jensen-Shannon disagreement",
                "Disagreement-triggered calibrator",
                "tanh-bounded residual",
            ):
                self.assertIn(label, svg)

    def test_rpg_figure_exports_partitioned_uncertainty_modules(self) -> None:
        from generate_hrgv_architecture_figure import generate_rpg_architecture_figure

        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_rpg_architecture_figure(
                Path(temporary) / "fig_rpg_hrgv_architecture"
            )

            for path in outputs.values():
                self.assertTrue(path.exists(), path)
            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "Frozen role partition M",
                "U_between",
                "U_within",
                "RPG regret gate",
                "RSG fusion",
                "Residual hard-negative verifiers",
            ):
                self.assertIn(label, svg)
            source_text = outputs["source_description"].read_text(encoding="utf-8")
            self.assertIn("H(S) = H(R) + H(S|R)", source_text)

    def test_mrpg_figure_contains_capacity_normalization_and_monotonicity(self) -> None:
        from generate_hrgv_architecture_figure import generate_mrpg_architecture_figure

        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_mrpg_architecture_figure(
                Path(temporary) / "fig_mrpg_hrgv_architecture"
            )

            for key in ("png", "svg", "pdf", "tiff", "source_description"):
                self.assertTrue(outputs[key].exists(), key)
            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "Frozen role partition M",
                "Capacity normalization",
                "Monotone M-RPG gate",
                "softplus(beta)",
                "RSG fusion",
            ):
                self.assertIn(label, svg)
            self.assertIn("partial g", outputs["source_text"])


if __name__ == "__main__":
    unittest.main()
