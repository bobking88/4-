from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGGateReliabilityFigureTests(unittest.TestCase):
    def test_exports_two_theorem_diagnostics_with_claim_boundary(self) -> None:
        from generate_rsg_gate_reliability_figure import generate_rsg_gate_reliability_figure

        analysis_dir = PROJECT_ROOT / "outputs" / "business_metrics" / "rsg_hrgv" / "gate_reliability"
        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_rsg_gate_reliability_figure(
                analysis_dir / "b1_local_bound_strata.csv",
                analysis_dir / "b2_margin_strata.csv",
                analysis_dir / "gate_reliability_summary.json",
                Path(temporary) / "fig_rsg_gate_reliability",
            )

            self.assertEqual(
                set(outputs), {"png", "svg", "pdf", "tiff", "source_description"}
            )
            for path in outputs.values():
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 100, path)
            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "Theorem B.1",
                "Theorem B.2",
                "local B.1 bound",
                "soft-target deviation",
                "mechanism diagnosis",
            ):
                self.assertIn(label, svg)
            with Image.open(outputs["png"]) as image:
                self.assertGreaterEqual(image.width, 1800)
                self.assertGreaterEqual(image.height, 900)


if __name__ == "__main__":
    unittest.main()
