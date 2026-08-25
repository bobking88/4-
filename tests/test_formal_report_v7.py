from __future__ import annotations

import hashlib
import unittest
import zipfile
from pathlib import Path

from docx import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "结题" / "基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx"
ASSETS = tuple((PROJECT_ROOT / "outputs" / "report_assets_v7").glob("rsg_proof_*.png"))


class FormalReportV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        document = Document(REPORT)
        cls.text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    def test_formal_proof_appendix_is_present(self) -> None:
        for expected in (
            "附录 B RSG-HRGV 门控后悔理论证明",
            "定理 B.1（路由后悔上界）",
            "B.2 软门控目标的指数逼近",
            "B.3 后悔监督分支的局部梯度隔离",
            "局部梯度隔离不是“停止训练主干”",
            "摄影者留出独立确认中为−3.53个百分点",
        ):
            self.assertIn(expected, self.text)

    def test_proof_equations_are_embedded(self) -> None:
        self.assertEqual(len(ASSETS), 3)
        expected = {hashlib.sha256(path.read_bytes()).hexdigest() for path in ASSETS}
        with zipfile.ZipFile(REPORT) as archive:
            embedded = {
                hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if name.startswith("word/media/")
            }
        self.assertLessEqual(expected, embedded)


if __name__ == "__main__":
    unittest.main()
