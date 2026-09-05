from __future__ import annotations

import sys
import unittest
import hashlib
import json
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class PHRAnalysisTests(unittest.TestCase):
    def fixture(self, correct):
        labels = ["target_mineral", "ti_bearing_negative", "gangue_negative", "metallic_hard_negative"]
        return [{"image_id": str(i), "split_group_id": str(i), "true_label": label,
                 "predicted_label": label if correct else "target_mineral"}
                for i, label in enumerate(labels)]

    def test_macro_f1_is_f1_and_bootstrap_uses_all_seed_point_estimates(self):
        from analyze_phr_hrgv_experiments import _risk_metrics, _paired_group_bootstrap
        wrong, right = self.fixture(False), self.fixture(True)
        self.assertAlmostEqual(_risk_metrics(wrong)["macro_f1"], 0.1)
        stats = _paired_group_bootstrap([wrong, right, right], [right, wrong, right], 50, 7)
        macro = next(row for row in stats if row["metric"] == "macro_f1")
        self.assertAlmostEqual(macro["difference"], 0)

    def test_bootstrap_rejects_duplicate_ids_and_changed_groups(self):
        from analyze_phr_hrgv_experiments import _paired_group_bootstrap
        rows = self.fixture(True)
        with self.assertRaises(ValueError):
            _paired_group_bootstrap([rows + [rows[0]]], [rows], 10, 7)
        changed = [dict(row) for row in rows]
        changed[0]["split_group_id"] = "different"
        with self.assertRaises(ValueError):
            _paired_group_bootstrap([rows], [changed], 10, 7)

    def test_pairwise_summary_keeps_ti_and_metallic_edges_separate(self) -> None:
        from analyze_phr_hrgv_experiments import summarize_pairwise_rows

        rows = [
            {"phr_ti_eligible": "1", "phr_ti_margin_regret": "0.10", "phr_ti_gate_selection_correct": "1", "phr_ti_expert_sign_agreement": "1", "phr_ti_sign_preserved": "1",
             "phr_metallic_eligible": "1", "phr_metallic_margin_regret": "0.30", "phr_metallic_gate_selection_correct": "0", "phr_metallic_expert_sign_agreement": "1", "phr_metallic_sign_preserved": "1"},
            {"phr_ti_eligible": "0", "phr_ti_margin_regret": "0.00", "phr_ti_gate_selection_correct": "0", "phr_ti_expert_sign_agreement": "0", "phr_ti_sign_preserved": "0",
             "phr_metallic_eligible": "1", "phr_metallic_margin_regret": "0.10", "phr_metallic_gate_selection_correct": "1", "phr_metallic_expert_sign_agreement": "0", "phr_metallic_sign_preserved": "0"},
        ]
        summary = summarize_pairwise_rows(rows)

        self.assertEqual(summary["phr_ti_eligible_count"], 1)
        self.assertAlmostEqual(summary["phr_ti_mean_margin_regret"], 0.10)
        self.assertEqual(summary["phr_metallic_eligible_count"], 2)
        self.assertAlmostEqual(summary["phr_metallic_mean_margin_regret"], 0.20)
        self.assertAlmostEqual(summary["phr_metallic_gate_selection_accuracy"], 0.5)

    def test_formal_registration_must_match_promoted_screen_sources(self) -> None:
        from analyze_phr_hrgv_experiments import validate_formal_evidence
        from generate_phr_screen_decision import build_screen_decision

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screen_root = root / "screen"
            manifest = screen_root / "manifest.csv"
            screen_root.mkdir()
            manifest.write_text("image_id,split\na,val\n", encoding="utf-8")
            metrics = {
                "macro_f1": .60, "accuracy": .70, "target_recall": .80,
                "ti_to_target_intrusion_rate": .10, "metallic_to_target_intrusion_rate": .10,
                "phr_ti_mean_margin_regret": .50, "phr_metallic_mean_margin_regret": .50,
            }
            for configuration in (
                "rsg_reference", "phr_complete", "phr_fixed_half", "phr_hard_target",
                "phr_unweighted", "phr_coupled_features", "phr_ti_only", "phr_metallic_only",
            ):
                run = screen_root / configuration / "seed20260728"
                run.mkdir(parents=True)
                candidate_metrics = dict(metrics)
                if configuration == "phr_complete":
                    candidate_metrics.update({"macro_f1": .61, "phr_ti_mean_margin_regret": .40, "phr_metallic_mean_margin_regret": .40})
                (run / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
                (run / "environment.json").write_text(json.dumps({"validation_only": True, "smoke_run": False}), encoding="utf-8")
                (run / "val_metrics.json").write_text(json.dumps(candidate_metrics), encoding="utf-8")
                (run / "best_validation_metrics.json").write_text(json.dumps({**candidate_metrics, "selection_split": "val", "smoke_run": False}), encoding="utf-8")
                (run / "val_predictions.csv").write_text("image_id,true_label,predicted_label,split_group_id\na,target_mineral,target_mineral,g\n", encoding="utf-8")
            (screen_root / "registered_configurations.json").write_text(json.dumps({
                "protocol_version": "phr-v2", "mode": "screen",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "source_sha256": {"scripts/train_hrgv_mineral_classifier.py": "frozen"},
            }), encoding="utf-8")
            decision_path = root / "screen_decision.json"
            decision_path.write_text(
                json.dumps(build_screen_decision(screen_root, manifest)), encoding="utf-8"
            )

            formal_root = root / "formal"
            formal_root.mkdir()
            (formal_root / "registered_configurations.json").write_text(json.dumps({
                "protocol_version": "phr-v2",
                "mode": "formal",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "source_sha256": {"scripts/train_hrgv_mineral_classifier.py": "frozen"},
            }), encoding="utf-8")
            for config in ("rsg_reference", "phr_complete"):
                for seed in ("20260727", "20260728", "20260729"):
                    run = formal_root / config / f"seed{seed}"
                    run.mkdir(parents=True)
                    (run / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
                    (run / "environment.json").write_text(json.dumps({
                        "validation_only": False, "smoke_run": False,
                    }), encoding="utf-8")

            audit = validate_formal_evidence(formal_root, manifest, decision_path)
            self.assertEqual(audit["screen_decision_sha256"], hashlib.sha256(decision_path.read_bytes()).hexdigest())

            registration_path = formal_root / "registered_configurations.json"
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            registration["source_sha256"] = {"scripts/train_hrgv_mineral_classifier.py": "changed"}
            registration_path.write_text(json.dumps(registration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source"):
                validate_formal_evidence(formal_root, manifest, decision_path)


if __name__ == "__main__":
    unittest.main()
