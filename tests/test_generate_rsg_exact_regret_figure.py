from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RSGExactRegretFigureTests(unittest.TestCase):
    def test_exports_exact_convex_fusion_decomposition_with_traceability(self) -> None:
        from generate_rsg_exact_regret_figure import generate_rsg_exact_regret_figure

        summary_path = (
            PROJECT_ROOT
            / "outputs"
            / "business_metrics"
            / "rsg_hrgv"
            / "gate_reliability"
            / "gate_reliability_summary.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_rsg_exact_regret_figure(
                summary_path, Path(temporary) / "fig_rsg_exact_regret_decomposition"
            )

            self.assertEqual(
                set(outputs), {"png", "svg", "pdf", "tiff", "source_description"}
            )
            for path in outputs.values():
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 100, path)

            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "Exact convex-fusion decomposition",
                "r = -log(1 - u)",
                "u = delta d / M",
                "mechanism verification",
            ):
                self.assertIn(label, svg)

            source = json.loads(outputs["source_description"].read_text(encoding="utf-8"))
            self.assertEqual(source["sample_count"], 10242)
            self.assertEqual(source["exact_decomposition_violation_count"], 0)
            self.assertLessEqual(source["exact_decomposition_max_abs_residual"], 2e-6)
            self.assertIn("not a new classification-performance comparison", source["claim_boundary"])

            with Image.open(outputs["png"]) as image:
                self.assertGreaterEqual(image.width, 2200)
                self.assertGreaterEqual(image.height, 1200)


if __name__ == "__main__":
    unittest.main()
