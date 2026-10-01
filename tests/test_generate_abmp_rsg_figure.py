import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_abmp_rsg_figure import generate_figure, run_theory_audit


class FigureTests(unittest.TestCase):
    def test_theory_audit_checks_independent_margin_and_dominance(self):
        result = run_theory_audit(samples=300)
        self.assertEqual(result["dominance_counterexamples"], 0)
        self.assertGreater(result["independent_margin_activation_count"], 0)
        for key, value in result["random_audit"].items():
            if key.endswith("violations"):
                self.assertEqual(value, 0, key)

    def test_exports_editable_diagram_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = generate_figure(Path(folder))
            self.assertEqual(source["evidence_dimension"], 18)
            self.assertEqual(source["policy_parameter_count"], 2418)
            for kind in ("png", "pdf", "svg"):
                self.assertGreater((Path(folder) / f"fig_abmp_rsg_architecture.{kind}").stat().st_size, 1000)
            self.assertIn("margin", " ".join(source["components"]))


if __name__ == "__main__":
    unittest.main()
