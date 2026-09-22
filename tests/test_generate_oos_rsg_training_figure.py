import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_oos_rsg_training_figure import generate_figure


class OOSRSGFigureTests(unittest.TestCase):
    def test_exports_figure_bundle_with_claim_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_figure(Path(directory) / "oos_rsg")
            for extension in (".png", ".svg", ".pdf", ".json"):
                self.assertTrue(outputs[extension].is_file())
                self.assertGreater(outputs[extension].stat().st_size, 0)
            source = json.loads(outputs[".json"].read_text(encoding="utf-8"))
            self.assertIn("not independent confirmation", source["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
