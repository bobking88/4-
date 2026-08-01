from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class FigureSummaryTests(unittest.TestCase):
    def test_focal_seed_paths_include_all_three_formal_seeds(self) -> None:
        from generate_paper_figures import FOCAL_SEEDS, focal_seed_paths

        self.assertEqual(FOCAL_SEEDS, (20260727, 20260728, 20260729))
        self.assertEqual(
            focal_seed_paths(),
            [
                PROJECT_ROOT / "outputs" / "training" / "formal_efficientnet_b0_focal_seed20260727" / "test_metrics.json",
                PROJECT_ROOT / "outputs" / "training" / "formal_efficientnet_b0_focal_seed20260728" / "test_metrics.json",
                PROJECT_ROOT / "outputs" / "training" / "formal_efficientnet_b0_focal_seed20260729" / "test_metrics.json",
            ],
        )

    def test_role_aware_model_spec_has_three_formal_seeds(self) -> None:
        from generate_paper_figures import MODEL_SPECS

        self.assertEqual(
            MODEL_SPECS["Role-aware EfficientNet-B0"],
            [
                PROJECT_ROOT / "outputs" / "training" / f"formal_role_aware_efficientnet_b0_seed{seed}" / "test_metrics.json"
                for seed in (20260727, 20260728, 20260729)
            ],
        )

    def test_hierarchical_model_spec_has_three_formal_seeds(self) -> None:
        from generate_paper_figures import MODEL_SPECS

        self.assertEqual(
            MODEL_SPECS["Hierarchical EfficientNet-B0"],
            [
                PROJECT_ROOT / "outputs" / "training" / f"formal_hierarchical_efficientnet_b0_seed{seed}" / "test_metrics.json"
                for seed in (20260727, 20260728, 20260729)
            ],
        )

    def test_summarize_values_returns_sample_standard_deviation(self) -> None:
        from generate_paper_figures import summarize_values

        mean, sample_std = summarize_values([0.70, 0.80, 0.90])

        self.assertAlmostEqual(mean, 0.80)
        self.assertAlmostEqual(sample_std, 0.10)

    def test_summarize_values_rejects_fewer_than_two_runs(self) -> None:
        from generate_paper_figures import summarize_values

        with self.assertRaisesRegex(ValueError, "two"):
            summarize_values([0.80])

    def test_summarize_focal_ablation_uses_all_formal_seeds(self) -> None:
        from generate_paper_figures import summarize_focal_ablation

        baseline = [
            {"macro_f1": 0.70, "class_recall": {"target_mineral": 0.60}},
            {"macro_f1": 0.80, "class_recall": {"target_mineral": 0.70}},
            {"macro_f1": 0.90, "class_recall": {"target_mineral": 0.80}},
        ]
        focal = [
            {"macro_f1": 0.71, "class_recall": {"target_mineral": 0.61}},
            {"macro_f1": 0.81, "class_recall": {"target_mineral": 0.71}},
            {"macro_f1": 0.91, "class_recall": {"target_mineral": 0.81}},
        ]

        summaries = summarize_focal_ablation(baseline, focal, ["target_mineral"])

        self.assertEqual(summaries["macro_f1"]["runs"], 3)
        self.assertAlmostEqual(summaries["macro_f1"]["cross_entropy_mean"], 0.80)
        self.assertAlmostEqual(summaries["macro_f1"]["focal_loss_mean"], 0.81)
        self.assertAlmostEqual(summaries["target_mineral"]["focal_loss_mean"], 0.71)

    def test_build_target_proxy_source_rows_preserves_mean_and_variation(self) -> None:
        from generate_paper_figures import build_target_proxy_source_rows

        summary = {
            "target_precision": {"mean": 0.70, "sample_std": 0.02},
            "target_recall": {"mean": 0.71, "sample_std": 0.01},
            "target_f1": {"mean": 0.70, "sample_std": 0.01},
            "target_miss_rate": {"mean": 0.29, "sample_std": 0.01},
            "ti_bearing_intrusion_rate": {"mean": 0.10, "sample_std": 0.01},
            "metallic_intrusion_rate": {"mean": 0.11, "sample_std": 0.01},
            "gangue_intrusion_rate": {"mean": 0.03, "sample_std": 0.01},
        }

        rows = build_target_proxy_source_rows(summary)

        self.assertEqual(rows[0]["metric"], "Target precision")
        self.assertEqual(rows[0]["mean_percent"], "70.000000")
        self.assertEqual(rows[1]["sample_std_percent"], "1.000000")

    def test_build_target_proxy_comparison_rows_keeps_each_strategy(self) -> None:
        from generate_paper_figures import build_target_proxy_comparison_rows

        strategy_summaries = {
            "Cross entropy": {"target_f1": {"mean": 0.70, "sample_std": 0.01}},
            "Role-aware": {"target_f1": {"mean": 0.72, "sample_std": 0.02}},
        }

        rows = build_target_proxy_comparison_rows(strategy_summaries, [("target_f1", "Target F1")])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["strategy"], "Role-aware")
        self.assertEqual(rows[1]["mean_percent"], "72.000000")

    def test_hierarchical_architecture_figure_exports_a_png_and_source_description(self) -> None:
        from generate_paper_figures import plot_hierarchical_architecture

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            plot_hierarchical_architecture(output_dir)

            self.assertTrue((output_dir / "fig8_hierarchical_architecture.png").is_file())
            self.assertTrue((output_dir / "fig8_hierarchical_architecture.md").is_file())

    def test_theory_aware_architecture_figure_is_created(self) -> None:
        from generate_paper_figures import plot_theory_aware_hierarchical_architecture

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "architecture_cn.png"
            plot_theory_aware_hierarchical_architecture(output)

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
