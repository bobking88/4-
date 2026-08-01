from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class HierarchicalAblationSummaryTests(unittest.TestCase):
    def test_component_rows_use_full_model_as_reference(self) -> None:
        from summarize_hierarchical_ablations import build_component_rows

        full = {
            "macro_f1": {"mean": 0.734, "sample_std": 0.020},
            "target_f1": {"mean": 0.720, "sample_std": 0.010},
            "target_miss_rate": {"mean": 0.224, "sample_std": 0.010},
            "ti_bearing_intrusion_rate": {"mean": 0.109, "sample_std": 0.010},
            "metallic_intrusion_rate": {"mean": 0.144, "sample_std": 0.010},
        }
        no_contrast = {
            "macro_f1": {"mean": 0.736, "sample_std": 0.020},
            "target_f1": {"mean": 0.716, "sample_std": 0.010},
            "target_miss_rate": {"mean": 0.224, "sample_std": 0.010},
            "ti_bearing_intrusion_rate": {"mean": 0.115, "sample_std": 0.010},
            "metallic_intrusion_rate": {"mean": 0.150, "sample_std": 0.010},
        }
        no_consistency = {
            "macro_f1": {"mean": 0.734, "sample_std": 0.020},
            "target_f1": {"mean": 0.709, "sample_std": 0.010},
            "target_miss_rate": {"mean": 0.228, "sample_std": 0.010},
            "ti_bearing_intrusion_rate": {"mean": 0.120, "sample_std": 0.010},
            "metallic_intrusion_rate": {"mean": 0.148, "sample_std": 0.010},
        }

        rows = build_component_rows(full, no_contrast, no_consistency)

        self.assertEqual([row["setting"] for row in rows], ["完整分层模型", "去除困难负样本约束", "去除层级一致性约束"])
        self.assertAlmostEqual(rows[1]["delta_macro_f1_vs_full"], 0.002)
        self.assertAlmostEqual(rows[1]["delta_ti_bearing_intrusion_rate_vs_full"], 0.006)
        self.assertAlmostEqual(rows[2]["delta_target_f1_vs_full"], -0.011)

    def test_component_rows_reject_missing_metrics(self) -> None:
        from summarize_hierarchical_ablations import build_component_rows

        incomplete = {"macro_f1": {"mean": 0.7, "sample_std": 0.1}}
        with self.assertRaisesRegex(ValueError, "target_f1"):
            build_component_rows(incomplete, incomplete, incomplete)


if __name__ == "__main__":
    unittest.main()
