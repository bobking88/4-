from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class OOSConfirmationFigureTests(unittest.TestCase):
    def test_layout_contract_keeps_notes_and_decision_footer_separated(self) -> None:
        from generate_oos_confirmation_figure import build_layout_contract

        layout = build_layout_contract()
        self.assertGreaterEqual(layout["panel_note_y"], 0.90)
        self.assertGreaterEqual(
            layout["decision_title_y"] - layout["decision_body_y"],
            0.14,
        )

    def test_builds_direction_aligned_effects(self) -> None:
        from generate_oos_confirmation_figure import build_figure_data

        analysis = json.loads(
            (
                PROJECT_ROOT
                / "outputs"
                / "training"
                / "oos_rsg_confirmation_v1"
                / "analysis"
                / "analysis.json"
            ).read_text(encoding="utf-8")
        )
        figure_data = build_figure_data(analysis)

        nll_vs_equal = next(
            item for item in figure_data["nll_benefits"]
            if item["comparison"] == "Equal fusion"
        )
        self.assertAlmostEqual(nll_vs_equal["benefit"], 0.014040491656287882)
        self.assertGreater(nll_vs_equal["ci_low"], 0)

        tradeoffs = {
            item["metric"]: item for item in figure_data["tradeoff_vs_equal"]
        }
        self.assertLess(tradeoffs["Target recall"]["benefit_pp"], -2.4)
        self.assertGreater(tradeoffs["Ti intrusion"]["benefit_pp"], 1.1)
        self.assertFalse(figure_data["promotion"]["promote_to_main_method"])

    def test_exports_reviewable_bundle(self) -> None:
        from generate_oos_confirmation_figure import generate_figure

        analysis_path = (
            PROJECT_ROOT
            / "outputs"
            / "training"
            / "oos_rsg_confirmation_v1"
            / "analysis"
            / "analysis.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            outputs = generate_figure(
                analysis_path,
                Path(temporary) / "fig_oos_rsg_confirmation",
            )

            self.assertEqual(
                set(outputs),
                {"png", "svg", "pdf", "tiff", "source_data"},
            )
            for path in outputs.values():
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 100, path)

            svg = outputs["svg"].read_text(encoding="utf-8")
            self.assertFalse(
                any(line.endswith((" ", "\t")) for line in svg.splitlines())
            )
            for label in (
                "Independent OOS routing confirmation",
                "Target recall",
                "Not promoted",
                "3/3 experts",
            ):
                self.assertIn(label, svg)

            with Image.open(outputs["png"]) as image:
                self.assertGreaterEqual(image.width, 1800)
                self.assertGreaterEqual(image.height, 800)


if __name__ == "__main__":
    unittest.main()
