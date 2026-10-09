"""Architecture diagrams must reflect the implemented network and evidence state."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from lc_role_adapter import RoleResidualModel


class FigureTests(unittest.TestCase):
    def setUp(self):
        try:
            self.figures = importlib.import_module("generate_lc_role_adaptation_figure")
        except ModuleNotFoundError:
            self.fail("Missing implemented architecture figure module")
        self.model = RoleResidualModel("R1",seed=20261002,alpha_max=.5,beta_max=.5)

    def test_counts_shapes_and_shared_auxiliary_head_come_from_model(self):
        manifest = self.figures.architecture_manifest(self.model)
        self.assertEqual(manifest["adapter_parameter_count"],[516]*3)
        self.assertEqual(manifest["shared_head_parameter_count"],1476)
        self.assertEqual(manifest["trainable_parameter_count"],3024)
        self.assertEqual(manifest["state"],"IMPLEMENTED_EFFECT_UNVERIFIED")
        self.assertEqual(manifest["example_shapes"]["delta"],[2,3,64])
        self.assertEqual(manifest["example_shapes"]["logq"],[2,4])
        self.assertEqual(manifest["modules"]["head.0"]["weight_shape"],[16,87])
        nodes = {n["id"]:n for n in manifest["nodes"]}
        self.assertTrue({"H1280","PCA64","E23","anchor","budget","delta1","delta2","delta3","head","bounded","auxiliary"} <= set(nodes))
        heads = [n for n in nodes.values() if n.get("module_path") == "head"]
        self.assertEqual(len(heads),1)
        self.assertIn(dict(source="auxiliary",target="head",style="dashed",kind="training"),manifest["edges"])
        self.assertTrue(all(e["style"] == "dashed" for e in manifest["edges"] if e["kind"] == "training"))

    def test_rejects_incompatible_model_or_unsupported_effect_claim(self):
        with self.assertRaises(ValueError):
            self.figures.architecture_manifest(RoleResidualModel("H0",seed=1,alpha_max=.5,beta_max=.5))
        manifest = self.figures.architecture_manifest(self.model)
        with tempfile.TemporaryDirectory() as folder:
            changed = copy.deepcopy(manifest)
            changed["state"] = "INDUSTRIALLY_VALIDATED"
            with self.assertRaises(ValueError):
                self.figures.render_architecture(changed,Path(folder))
            changed = copy.deepcopy(manifest)
            changed["trainable_parameter_count"] = 4000
            with self.assertRaises(ValueError):
                self.figures.render_architecture(changed,Path(folder))

    def test_export_bundle_is_editable_sized_and_has_traceable_source(self):
        manifest = self.figures.architecture_manifest(self.model)
        with tempfile.TemporaryDirectory() as folder:
            result = self.figures.render_architecture(manifest,Path(folder))
            for name in ("svg","pdf","png","source"):
                self.assertGreater(Path(result[name]).stat().st_size,1000)
            self.assertIn("<text",Path(result["svg"]).read_text(encoding="utf-8"))
            with Image.open(result["png"]) as image:
                self.assertEqual(image.size,(2160,2040))
            source = json.loads(Path(result["source"]).read_text(encoding="utf-8"))
            self.assertEqual(source["state"],"IMPLEMENTED_EFFECT_UNVERIFIED")
            self.assertEqual(len(source["code_sha256"]),2)
            self.assertEqual(result["qa"]["text_overflow"],[])


if __name__ == "__main__":
    unittest.main()
