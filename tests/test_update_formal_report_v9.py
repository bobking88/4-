from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FormalReportV9EvidenceTests(unittest.TestCase):
    @staticmethod
    def _write_complete_evidence(analysis_dir: Path) -> None:
        configurations = (
            "rsg_complete",
            "cgdc_complete",
            "cgdc_shared_features",
            "cgdc_unconditional",
            "cgdc_no_decomposition_loss",
        )
        metrics = (
            "accuracy",
            "macro_f1",
            "target_recall",
            "ti_to_target_intrusion_rate",
            "metallic_to_target_intrusion_rate",
            "brier_score",
            "expected_calibration_error",
        )
        summary = {
            configuration: {
                metric: {"mean": 0.70, "sample_std": 0.01, "values": [0.69, 0.70, 0.71]}
                for metric in metrics
            }
            for configuration in configurations
        }
        (analysis_dir / "cgdc_three_seed_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        for configuration in configurations[1:]:
            (analysis_dir / f"paired_{configuration}_vs_rsg_complete.json").write_text(
                json.dumps(
                    {
                        "classification": {"macro_f1": {"difference": 0.01, "ci_low": -0.01, "ci_high": 0.02}},
                        "routing_regret": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        "calibration": {
                            "brier_score": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                            "expected_calibration_error": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        },
                    }
                ),
                encoding="utf-8",
            )

    @staticmethod
    def _write_complete_rpg_evidence(analysis_dir: Path) -> None:
        configurations = (
            "rsg_complete",
            "rpg_complete",
            "rpg_without_within",
            "rpg_without_between",
            "rpg_total_entropy_only",
        )
        metrics = (
            "accuracy",
            "macro_f1",
            "target_recall",
            "brier_score",
            "expected_calibration_error",
        )
        summary = {
            configuration: {
                metric: {"mean": 0.70, "sample_std": 0.01, "values": [0.69, 0.70, 0.71]}
                for metric in metrics
            }
            for configuration in configurations
        }
        (analysis_dir / "rpg_three_seed_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        for configuration in configurations[1:]:
            (analysis_dir / f"paired_{configuration}_vs_rsg_complete.json").write_text(
                json.dumps(
                    {
                        "classification": {"macro_f1": {"difference": 0.01, "ci_low": -0.01, "ci_high": 0.02}},
                        "routing_regret": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        "calibration": {
                            "brier_score": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                            "expected_calibration_error": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        },
                    }
                ),
                encoding="utf-8",
            )

    @staticmethod
    def _write_complete_mrpg_evidence(analysis_dir: Path) -> None:
        configurations = (
            "rsg_complete",
            "rpg_complete",
            "mrpg_complete",
            "mrpg_unconstrained_between",
            "mrpg_without_between",
        )
        metrics = (
            "accuracy",
            "macro_f1",
            "target_recall",
            "brier_score",
            "expected_calibration_error",
        )
        summary = {
            configuration: {
                metric: {"mean": 0.70, "sample_std": 0.01, "values": [0.69, 0.70, 0.71]}
                for metric in metrics
            }
            for configuration in configurations
        }
        (analysis_dir / "mrpg_three_seed_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        for configuration in configurations[2:]:
            (analysis_dir / f"paired_{configuration}_vs_rsg_complete.json").write_text(
                json.dumps(
                    {
                        "classification": {
                            "macro_f1": {"difference": 0.01, "ci_low": -0.01, "ci_high": 0.02},
                            "target_recall": {"difference": 0.02, "ci_low": -0.01, "ci_high": 0.04},
                        },
                        "routing_regret": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        "calibration": {
                            "brier_score": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                            "expected_calibration_error": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        },
                    }
                ),
                encoding="utf-8",
            )
        for configuration in configurations[3:]:
            (analysis_dir / f"paired_{configuration}_vs_mrpg_complete.json").write_text(
                json.dumps(
                    {
                        "classification": {
                            "macro_f1": {"difference": 0.01, "ci_low": -0.01, "ci_high": 0.02},
                            "target_recall": {"difference": 0.02, "ci_low": -0.01, "ci_high": 0.04},
                        },
                        "routing_regret": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        "calibration": {
                            "brier_score": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                            "expected_calibration_error": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.00},
                        },
                    }
                ),
                encoding="utf-8",
            )

    def test_requires_all_formal_configurations_and_three_seeds(self) -> None:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from update_formal_report_v9 import load_formal_cgdc_evidence

        with tempfile.TemporaryDirectory() as temp_dir:
            analysis_dir = Path(temp_dir)
            configurations = (
                "rsg_complete",
                "cgdc_complete",
                "cgdc_shared_features",
                "cgdc_unconditional",
                "cgdc_no_decomposition_loss",
            )
            summary = {
                configuration: {"macro_f1": {"values": [0.70, 0.71]}}
                for configuration in configurations
            }
            (analysis_dir / "cgdc_three_seed_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "three registered seeds"):
                load_formal_cgdc_evidence(analysis_dir)

    def test_appends_cgdc_appendix_once_with_architecture_and_boundaries(self) -> None:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from update_formal_report_v9 import update_report

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source.docx"
            output = temp_root / "output.docx"
            analysis_dir = temp_root / "analysis"
            analysis_dir.mkdir()
            self._write_complete_evidence(analysis_dir)
            document = Document()
            document.add_paragraph("existing report")
            document.save(source)

            update_report(source, output, analysis_dir)
            update_report(output, output, analysis_dir)

            rendered = Document(output)
            text = "\n".join(paragraph.text for paragraph in rendered.paragraphs)
            self.assertEqual(text.count("附录 D CGDC-RSG-HRGV 网络理论与实验"), 1)
            self.assertIn("命题 P1", text)
            self.assertIn("命题 P4", text)
            self.assertIn("≤0.5", text)
            self.assertIn("Brier", text)
            self.assertIn("不等同于工业分选", text)

    def test_embeds_rendered_cgdc_formula_bundle(self) -> None:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from update_formal_report_v9 import FORMULA_DIR, update_report

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source.docx"
            output = temp_root / "output.docx"
            analysis_dir = temp_root / "analysis"
            analysis_dir.mkdir()
            self._write_complete_evidence(analysis_dir)
            document = Document()
            document.save(source)

            update_report(source, output, analysis_dir)

            expected = {
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in FORMULA_DIR.glob("cgdc_*.png")
            }
            self.assertGreaterEqual(len(expected), 3)
            with zipfile.ZipFile(output) as archive:
                embedded = {
                    hashlib.sha256(archive.read(name)).hexdigest()
                    for name in archive.namelist()
                    if name.startswith("word/media/")
                }
            self.assertLessEqual(expected, embedded)

    def test_appends_rpg_appendix_after_cgdc_without_duplication(self) -> None:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from update_formal_report_v9 import update_report

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source.docx"
            output = temp_root / "output.docx"
            cgdc_dir = temp_root / "cgdc"
            rpg_dir = temp_root / "rpg"
            cgdc_dir.mkdir()
            rpg_dir.mkdir()
            self._write_complete_evidence(cgdc_dir)
            self._write_complete_rpg_evidence(rpg_dir)
            Document().save(source)

            update_report(source, output, cgdc_dir, rpg_dir)
            update_report(output, output, cgdc_dir, rpg_dir)

            rendered = Document(output)
            text = "\n".join(paragraph.text for paragraph in rendered.paragraphs)
            self.assertEqual(text.count("附录 D CGDC-RSG-HRGV 网络理论与实验"), 1)
            self.assertEqual(text.count("附录 E RPG-HRGV 角色分区不确定性门控理论与实验"), 1)
            self.assertIn("H(S)=H(R)+H(S|R)", text)
            self.assertIn("角色间不确定性", text)
            self.assertIn("角色内不确定性", text)
            self.assertIn("不等同于工业分选", text)

    def test_appends_mrpg_appendix_with_normalization_and_monotonicity(self) -> None:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from update_formal_report_v9 import update_report

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source.docx"
            output = temp_root / "output.docx"
            cgdc_dir = temp_root / "cgdc"
            mrpg_dir = temp_root / "mrpg"
            cgdc_dir.mkdir()
            mrpg_dir.mkdir()
            self._write_complete_evidence(cgdc_dir)
            self._write_complete_mrpg_evidence(mrpg_dir)
            Document().save(source)

            update_report(source, output, cgdc_dir, mrpg_analysis_dir=mrpg_dir)
            update_report(output, output, cgdc_dir, mrpg_analysis_dir=mrpg_dir)

            rendered = Document(output)
            text = "\n".join(paragraph.text for paragraph in rendered.paragraphs)
            self.assertEqual(text.count("附录 F M-RPG-HRGV 容量归一化单调门控理论与实验"), 1)
            self.assertIn("容量归一化", text)
            self.assertIn("单调直接专家分配", text)
            self.assertIn("命题 M-R3", text)
            self.assertIn("相对完整 M-RPG", text)
            self.assertIn("不等同于工业分选", text)


if __name__ == "__main__":
    unittest.main()
