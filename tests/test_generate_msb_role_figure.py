"""Candidate schematic exports from formula metadata, without mineral images."""
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from msb_role_model import MSBRoleModel, unit_spec
try:
    import generate_msb_role_figure as figure
except ModuleNotFoundError:
    figure=None


class FigureTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(figure,"MSB formula figure missing")

    def test_figure_matches_formulas_and_frozen_paths(self):
        spec=figure.figure_spec(); model=MSBRoleModel("MAIN",seed=20261002)
        self.assertEqual(spec["trainable_parameters"],unit_spec("MAIN")["parameters"])
        self.assertEqual(spec["fixed_reference_parameters"],sum(b.numel() for b in model.buffers()))
        self.assertEqual(spec["U_shape"],[4,64]); self.assertEqual(spec["V_shape"],[4,23])
        self.assertEqual(spec["roles"],["Ti","G","M"])
        self.assertEqual(spec["gamma"],2.); self.assertEqual(spec["tau"],.25)
        self.assertFalse(spec["trainable_shared_readout"])
        self.assertEqual(spec["evidence_dimensions"],dict(E=18,log_base=4,contradiction=1,total=23))
        for tag in ("Hadamard","fixed_initial_reference","margin_budget","common_softmax","training_loss"):
            self.assertIn(tag,spec["mechanisms"])

    def test_new_figure_never_overwrites_old_assets_or_docx(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for forbidden in (root/"outputs"/"report_assets_v9",root/"结题"/"report.docx"):
                with self.assertRaises(ValueError):
                    figure.build_architecture_figure(forbidden,status_label=figure.STATUS_LABEL)
                self.assertFalse(forbidden.exists())
            target=root/"outputs"/"msb_role_synthetic_v1"/"assets"
            with self.assertRaises(ValueError): figure.build_architecture_figure(target,status_label="已验证")
            self.assertFalse(target.exists())

    def test_connector_crossing_formula_box_is_detected(self):
        boxes=[(2,1,2,2)]
        self.assertEqual(figure.connector_intersections([[(0,2),(6,2)]],boxes),[(0,0)])
        self.assertEqual(figure.connector_intersections([[(0,0),(6,0)]],boxes),[])
        self.assertEqual(figure.connector_intersections([[(0,2),(2,2)]],boxes),[])

    def test_exports_valid_svg_png_pdf_and_candidate_label(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"outputs"/"msb_role_synthetic_v1"/"assets"
            result=figure.build_architecture_figure(output,status_label=figure.STATUS_LABEL)
            self.assertEqual(result["status"],figure.STATUS_LABEL)
            self.assertFalse(result["real_effect_verified"])
            svg=(output/"msb_rre_v1_architecture.svg").read_text(encoding="utf-8")
            tree=ET.fromstring(svg)
            self.assertGreater(len(tree.findall(".//{http://www.w3.org/2000/svg}text")),30)
            for text in ("MSB-RRE","合成验收","真实效果未验证","Hadamard","softmax"):
                self.assertIn(text,svg)
            self.assertTrue((output/"msb_rre_v1_architecture.pdf").read_bytes().startswith(b"%PDF"))
            with Image.open(output/"msb_rre_v1_architecture.png") as im:
                self.assertGreater(im.width,2000); self.assertGreater(im.height,1500)
                self.assertGreater(float(np.mean(np.asarray(im.convert("RGB"))<240)),.03)
            self.assertEqual(result["text_overflow"],[])
            with self.assertRaises(FileExistsError):
                figure.build_architecture_figure(output,status_label=figure.STATUS_LABEL)


if __name__=="__main__": unittest.main()
