from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class PHRRoutingFigureTests(unittest.TestCase):
    def test_architecture_bundle_is_exported_with_a_nonperformance_claim_boundary(self) -> None:
        from generate_phr_routing_figure import generate_phr_routing_architecture

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "phr_architecture"
            outputs = generate_phr_routing_architecture(prefix)
            for extension in (".png", ".pdf", ".svg", ".tiff", ".json"):
                self.assertTrue(outputs[extension].exists(), extension)
                self.assertGreater(outputs[extension].stat().st_size, 100, extension)
            source = json.loads(outputs[".json"].read_text(encoding="utf-8"))
            self.assertEqual(source["evidence_type"], "network_contract_schematic")
            self.assertIn("no empirical performance claim", source["claim_boundary"])
            self.assertTrue(any(r"m_{f,I}" in formula for formula in source["formulae"]))
            self.assertTrue(any(r"A^\top(AA^\top)^{-1}" in formula for formula in source["formulae"]))


if __name__ == "__main__":
    unittest.main()
