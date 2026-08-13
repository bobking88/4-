from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class UpdateFormalReportV2Tests(unittest.TestCase):
    def test_certificate_rows_use_only_certified_seeds(self) -> None:
        from update_formal_report_v2 import certificate_rows

        payload = {
            "results": {
                "1": {"certificates": {
                    "0.1": {"status": "no_certified_threshold"},
                    "0.15": {"status": "certified", "selected": {"threshold": 0.8},
                             "test_evaluation": {"coverage": 0.5, "selective_risk": 0.1}},
                    "0.2": {"status": "certified", "selected": {"threshold": 0.6},
                            "test_evaluation": {"coverage": 0.7, "selective_risk": 0.15}},
                }},
                "2": {"certificates": {
                    "0.1": {"status": "no_certified_threshold"},
                    "0.15": {"status": "certified", "selected": {"threshold": 0.85},
                             "test_evaluation": {"coverage": 0.4, "selective_risk": 0.08}},
                    "0.2": {"status": "certified", "selected": {"threshold": 0.65},
                            "test_evaluation": {"coverage": 0.6, "selective_risk": 0.13}},
                }},
                "3": {"certificates": {
                    "0.1": {"status": "certified", "selected": {"threshold": 0.95},
                            "test_evaluation": {"coverage": 0.2, "selective_risk": 0.02}},
                    "0.15": {"status": "certified", "selected": {"threshold": 0.8},
                             "test_evaluation": {"coverage": 0.5, "selective_risk": 0.09}},
                    "0.2": {"status": "certified", "selected": {"threshold": 0.65},
                            "test_evaluation": {"coverage": 0.65, "selective_risk": 0.14}},
                }},
            }
        }

        rows = certificate_rows(payload)

        self.assertEqual(rows[0][1], "1/3")
        self.assertEqual(rows[0][3], "20.00%")
        self.assertEqual(rows[1][1], "3/3")
        self.assertEqual(rows[1][4], "9.00%")


if __name__ == "__main__":
    unittest.main()
