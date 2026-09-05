from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class PHRScreenDecisionTests(unittest.TestCase):
    configs = (
        "rsg_reference", "phr_complete", "phr_fixed_half", "phr_hard_target",
        "phr_unweighted", "phr_coupled_features", "phr_ti_only", "phr_metallic_only",
    )

    def make_screen(self, root: Path, improved: bool) -> Path:
        manifest = root / "manifest.csv"
        manifest.write_text("image_id,split\na,val\n", encoding="utf-8")
        base = {
            "macro_f1": .60, "accuracy": .70, "target_recall": .80,
            "ti_to_target_intrusion_rate": .10, "metallic_to_target_intrusion_rate": .10,
            "phr_ti_mean_margin_regret": .50, "phr_metallic_mean_margin_regret": .50,
        }
        for config in self.configs:
            metrics = dict(base)
            if config == "phr_complete" and improved:
                metrics.update({"macro_f1": .61, "phr_ti_mean_margin_regret": .40, "phr_metallic_mean_margin_regret": .40})
            run = root / config / "seed20260728"
            run.mkdir(parents=True)
            (run / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            (run / "environment.json").write_text(json.dumps({"validation_only": True, "smoke_run": False}), encoding="utf-8")
            (run / "val_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (run / "best_validation_metrics.json").write_text(json.dumps({**metrics, "selection_split": "val", "smoke_run": False}), encoding="utf-8")
            (run / "val_predictions.csv").write_text("image_id,true_label,predicted_label,split_group_id\na,target_mineral,target_mineral,g\n", encoding="utf-8")
        registration = {
            "protocol_version": "phr-v2",
            "mode": "screen",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_sha256": {"scripts/train_hrgv_mineral_classifier.py": "frozen"},
        }
        (root / "registered_configurations.json").write_text(json.dumps(registration), encoding="utf-8")
        return manifest

    def test_promotes_only_when_registered_validation_criterion_is_met(self) -> None:
        from generate_phr_screen_decision import build_screen_decision, verify_screen_decision

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_screen(root, improved=True)
            decision = build_screen_decision(root, manifest)
            self.assertTrue(decision["promote_to_formal"])
            self.assertEqual(decision["selected_configuration"], "phr_complete")
            self.assertIn("C1_macro_f1", decision["criterion_ids"])
            self.assertTrue(verify_screen_decision(decision, manifest))

    def test_refuses_promotion_when_no_criterion_is_met(self) -> None:
        from generate_phr_screen_decision import build_screen_decision

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_screen(root, improved=False)
            decision = build_screen_decision(root, manifest)
            self.assertFalse(decision["promote_to_formal"])
            self.assertIsNone(decision["selected_configuration"])
            self.assertEqual(decision["criterion_ids"], [])

    def test_formal_runner_rejects_decision_after_validation_evidence_changes(self) -> None:
        from generate_phr_screen_decision import build_screen_decision
        from run_phr_hrgv_experiments import _require_promotion_decision

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_screen(root, improved=True)
            decision_path = root / "screen_decision.json"
            decision_path.write_text(json.dumps(build_screen_decision(root, manifest)), encoding="utf-8")
            metrics_path = root / "phr_complete" / "seed20260728" / "val_metrics.json"
            changed = json.loads(metrics_path.read_text(encoding="utf-8"))
            changed["macro_f1"] = .99
            metrics_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence"):
                _require_promotion_decision(decision_path, manifest)

    def test_formal_runner_rejects_source_changed_since_screening(self) -> None:
        from generate_phr_screen_decision import build_screen_decision
        from run_phr_hrgv_experiments import _require_promotion_decision

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.make_screen(root, improved=True)
            decision_path = root / "screen_decision.json"
            decision_path.write_text(json.dumps(build_screen_decision(root, manifest)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source"):
                _require_promotion_decision(decision_path, manifest, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
