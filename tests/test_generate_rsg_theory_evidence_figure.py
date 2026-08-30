from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGTheoryEvidenceFigureTests(unittest.TestCase):
    def test_figure_exports_theorems_and_confirmed_routing_effects(self) -> None:
        from generate_rsg_theory_evidence_figure import generate_rsg_theory_evidence_figure

        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_rsg_theory_evidence_figure(
                PROJECT_ROOT / "outputs" / "business_metrics" / "rsg_hrgv" / "formal" / "paired_rsg_complete_vs_hrgv_reference.json",
                PROJECT_ROOT / "outputs" / "business_metrics" / "rsg_hrgv" / "source_holdout" / "paired_rsg_complete_vs_hrgv_reference.json",
                Path(temporary) / "fig_rsg_theory_evidence",
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
                "Theorem B.3",
                "routing regret",
                "Fixed test",
                "Photographer holdout",
                "classification superiority is not claimed",
            ):
                self.assertIn(label, svg)

            with Image.open(outputs["png"]) as image:
                self.assertGreaterEqual(image.width, 1800)
                self.assertGreaterEqual(image.height, 1000)

            source = outputs["source_description"].read_text(encoding="utf-8")
            self.assertIn("-1.77 pp", source)
            self.assertIn("-3.53 pp", source)


if __name__ == "__main__":
    unittest.main()
