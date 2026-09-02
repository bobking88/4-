from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGExpertComplementarityFigureTests(unittest.TestCase):
    def test_figure_exports_cross_granularity_diagnostics(self) -> None:
        from generate_rsg_expert_complementarity_figure import (
            generate_rsg_expert_complementarity_figure,
        )

        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_rsg_expert_complementarity_figure(
                fixed_summary_json=PROJECT_ROOT
                / "outputs"
                / "business_metrics"
                / "rsg_hrgv"
                / "formal"
                / "rsg_three_seed_summary.json",
                holdout_summary_json=PROJECT_ROOT
                / "outputs"
                / "business_metrics"
                / "rsg_hrgv"
                / "source_holdout"
                / "rsg_three_seed_summary.json",
                portability_summary_json=PROJECT_ROOT
                / "outputs"
                / "business_metrics"
                / "rsg_hrgv"
                / "resnet50_portability"
                / "rsg_three_seed_summary.json",
                prefix=Path(temporary) / "fig_rsg_expert_complementarity",
            )

            self.assertEqual(
                set(outputs), {"png", "svg", "pdf", "tiff", "source_description"}
            )
            for path in outputs.values():
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 100, path)

            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "Cross-granularity expert complementarity",
                "Direct role expert",
                "Mapped species expert",
                "Oracle diagnostic",
                "one-right / one-wrong",
                "Gate selection",
                "not a deployable oracle",
            ):
                self.assertIn(label, svg)

            source = outputs["source_description"].read_text(encoding="utf-8")
            self.assertIn("fixed_test", source)
            self.assertIn("photographer_holdout", source)
            self.assertIn("resnet50_portability", source)

            with Image.open(outputs["png"]) as image:
                self.assertGreaterEqual(image.width, 1800)
                self.assertGreaterEqual(image.height, 1100)


if __name__ == "__main__":
    unittest.main()
