"""Architecture exports are formula diagrams, never mineral results."""
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
try:
    import generate_ic_role_figure as drawing
except ModuleNotFoundError:
    drawing = None


class ICArchitectureTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(drawing, "Formula architecture renderer missing")

    def test_diagram_matches_formula_inputs_and_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            result = drawing.render_architecture(Path(temp)/"figure")
            contract = result["contract"]
            self.assertEqual(contract["dimensions"], dict(H=1280, h=64, E=23, bottleneck=4, readout=[87, 16, 4]))
            self.assertEqual(contract["current_parameters"], 3024)
            self.assertEqual(contract["reference_parameters"], 3024)
            self.assertEqual(contract["delta_divisor"], 16)
            self.assertEqual(contract["reference_input"], "original h and E, never adapted h")
            self.assertTrue(contract["auxiliary_training_only"])
            self.assertFalse(contract["backbone_trainable"])
            self.assertIn("not real effectiveness", contract["claim"])

    def test_export_has_all_nodes_and_no_clipping(self):
        with tempfile.TemporaryDirectory() as temp:
            result = drawing.render_architecture(Path(temp)/"figure")
            for key in ("svg", "pdf", "png", "caption"):
                path = Path(result[key])
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 200)
            document = ET.parse(result["svg"])
            names = {element.get("id") for element in document.iter()}
            self.assertTrue(set(result["node_ids"]) <= names)
            self.assertEqual(result["text_clipping_count"], 0)
            self.assertEqual(result["dpi"], 300)
            self.assertIn("same frozen reference", Path(result["caption"]).read_text())

    def test_budget_uses_maxima_and_requires_full_verification_coverage(self):
        try:
            import check_ic_role_implementation as acceptance
        except ModuleNotFoundError:
            acceptance = None
        self.assertIsNotNone(acceptance, "One-shot timing acceptance module missing")
        timings = dict(prepare=[1., 2., 3.], initial_kernel=[4., 5.],
                       update=[.01, .02], checkpoint=[.1, .2], artifact=[.3, .4],
                       verify_export=[6.], verification_forward_count=108*9)
        result = acceptance.budget_estimate(timings)
        self.assertAlmostEqual(result["estimated_seconds"], 1.5*(3*3+108*(5+400*.02+9*.2+.4)+6))
        self.assertTrue(result["timing_evidence_complete"])
        timings["verification_forward_count"] -= 1
        self.assertFalse(acceptance.budget_estimate(timings)["ready"])
        timings["checkpoint"] = []
        self.assertIsNone(acceptance.budget_estimate(timings)["estimated_seconds"])


if __name__ == "__main__": unittest.main()
