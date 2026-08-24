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


if __name__ == "__main__":
    unittest.main()
