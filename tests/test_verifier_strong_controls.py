import json
import hashlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import verifier_strong_controls as controls
    import run_verifier_strong_controls as runner
except ModuleNotFoundError:
    controls = runner = None


class StrongControlTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(controls, "strong-control implementation missing")
        self.assertIsNotNone(runner, "strong-control runner missing")
        torch.set_num_threads(1)
        self.p = torch.tensor([[.6, .2, .1, .1]] * 20, dtype=torch.float64)
        self.y = torch.tensor([0] * 10 + [1] * 5 + [2] * 3 + [3] * 2)

    def data(self):
        return {"p": self.p, "labels": self.y,
                "contradiction": torch.ones(20, 1, dtype=torch.float64),
                "evidence": torch.zeros(20, 18, dtype=torch.float64)}

    def test_temperature_identity_and_positive_scaling_preserve_ranking(self):
        q = controls.temperature_probabilities(self.p, torch.ones(20, 1, dtype=torch.float64))
        torch.testing.assert_close(q, self.p, atol=1e-15, rtol=1e-15)
        q = controls.temperature_probabilities(self.p, torch.full((20, 1), .1, dtype=torch.float64))
        self.assertEqual(q.argmax(1).tolist(), [0] * 20)
        with self.assertRaises(ValueError):
            controls.temperature_probabilities(self.p, torch.zeros(20, 1, dtype=torch.float64))

    def test_global_temperature_finds_known_convex_interior_and_boundaries(self):
        p = torch.tensor([[.8, .2]] * 4, dtype=torch.float64)
        fit = controls.fit_global_temperature(p, torch.tensor([0, 0, 0, 1]), [1e-6, 100.])
        self.assertAlmostEqual(fit["beta"], math.log(3) / math.log(4), places=12)
        self.assertLess(abs(fit["derivative_at_optimum"]), 1e-12)
        lo = controls.fit_global_temperature(p, torch.ones(4, dtype=torch.long), [1e-6, 100.])
        hi = controls.fit_global_temperature(p, torch.zeros(4, dtype=torch.long), [1e-6, 100.])
        self.assertEqual(lo["beta"], 1e-6)
        self.assertEqual(hi["beta"], 100.)
        self.assertEqual(lo["boundary"], "lower")

    def test_dirichlet_fit_changes_boundaries_and_checks_stationarity(self):
        p = torch.tensor([[.8, .1, .05, .05]] * 20, dtype=torch.float64)
        fit = controls.fit_dirichlet(p, self.y, .001)
        self.assertTrue(fit["converged"])
        self.assertLess(fit["gradient_max_abs"], 1e-6)
        q = controls.dirichlet_probabilities(p, fit["weight"], fit["bias"])
        torch.testing.assert_close(q.sum(1), torch.ones(20, dtype=torch.float64))
        self.assertLess(float(-q[range(20), self.y].log().mean()), .001 + float(-p[range(20), self.y].log().mean()))
        self.assertLess(float((q[0] - torch.tensor([.5, .25, .15, .1], dtype=torch.float64)).abs().max()), .01)
        swapped = torch.eye(4, dtype=torch.float64)[[1, 0, 2, 3]]
        self.assertEqual(int(controls.dirichlet_probabilities(p[:1], swapped, torch.zeros(4, dtype=torch.float64)).argmax()), 1)

    def test_heads_share_initialization_and_temperature_has_same_parameter_budget(self):
        budget = {"epochs": 2, "batch_size": 10, "learning_rate": .001, "weight_decay": .0001, "dropout": .1}
        runs = [controls.fit_head(self.data(), self.data(), arm, 19, budget) for arm in controls.LEARNED_ARMS]
        self.assertEqual(len({run["initial_sha256"] for run in runs}), 1)
        self.assertEqual({run["parameter_count"] for run in runs}, {2401})
        for run in runs:
            self.assertEqual(run["best_epoch"], min(run["history"], key=lambda row: row["stop_metrics"]["nll"])["epoch"])
            self.assertEqual(len(run["history"][0]["training_batches"]), 2)
            self.assertTrue(all(math.isfinite(b["parameter_gradient_norm"]) for b in run["history"][0]["training_batches"]))

    def test_clip_and_raw_loss_score_gradients_distinguish_saturated_plateau(self):
        data = {"p": torch.tensor([[.4, .3, .2, .1]], dtype=torch.float64),
                "contradiction": torch.ones(1, 1, dtype=torch.float64), "labels": torch.tensor([0])}
        score = torch.tensor([[math.log(4)]], dtype=torch.float64, requires_grad=True)
        loss, diagnostic = controls.objective(data, score, "P0")
        loss.backward()
        self.assertEqual(float(score.grad), 0.)
        self.assertEqual(diagnostic["saturated_count"], 1)
        score.grad = None
        loss, _ = controls.objective(data, score, "P1")
        loss.backward()
        self.assertAlmostEqual(float(score.grad), -.2, places=14)

    def test_training_reproducible_with_nonzero_bce_and_temperature_gradients(self):
        budget = {"epochs": 3, "batch_size": 10, "learning_rate": .001, "weight_decay": .0001, "dropout": .1}
        for arm in ("M0", "P1"):
            a = controls.fit_head(self.data(), self.data(), arm, 7, budget)
            b = controls.fit_head(self.data(), self.data(), arm, 7, budget)
            self.assertEqual(a["state_sha256"], b["state_sha256"])
            self.assertEqual(a["history"], b["history"])
            self.assertNotEqual(a["state_sha256"], a["initial_sha256"])

    def test_probability_validation_rejects_nonfinite_or_invalid_labels(self):
        with self.assertRaises(ValueError):
            controls.fit_dirichlet(self.p, torch.full((20,), 4), .001)
        with self.assertRaises(ValueError):
            controls.fit_dirichlet(self.p, self.y, 0.)
        with self.assertRaises(ValueError):
            controls.fit_global_temperature(self.p * float("nan"), self.y, [1e-6, 100.])

    def test_output_overwrite_refused_before_source_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                runner.run_experiment(root / "missing", root / "missing.json", root)

    def test_forbidden_fold_or_extra_arm_rejected_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for protocol in ({"protocol": "abmp_verifier_strong_controls_v1", "fold": 1},
                             {"protocol": "abmp_verifier_strong_controls_v1", "fold": 0, "arms": ["joint_loss"]}):
                path = root / "protocol.json"
                path.write_text(json.dumps(protocol), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "scope"):
                    runner.run_experiment(root, path, root / "new")

    def test_complete_delivery_replays_training_and_rejects_tampered_predictions(self):
        from test_verifier_trust_development import source_fixture, write_csv
        import run_verifier_trust_development as original_runner
        try:
            from verify_verifier_strong_controls import verify_delivery
        except ModuleNotFoundError:
            verify_delivery = None
        self.assertIsNotNone(verify_delivery, "strong-control independent replay missing")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

            def write_json(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

            _, fit_hash, _ = source_fixture(root)
            _, stop_hash, _ = source_fixture(root, "projector_stop")
            inner = root / "outputs/training/tc_oos_rsg_manifests_v1/fold_0"
            source = root / "outputs/training/tc_oos_rsg_v1/fold_0"
            training = []
            for name in ("expert_fit", "expert_stop", "gate_fit"):
                rows = [{"image_id": name, "split_group_id": name, "mindat_photo_id": name}]
                write_csv(inner / f"{name}.csv", list(rows[0]), rows)
                if name != "gate_fit":
                    training.extend(rows)
            write_csv(source / "manifests/expert_training.csv", list(training[0]), training)
            for name in ("expert/run_config.json", "expert/environment.json", "tc_projection/run_config.json"):
                write_json(source / name, {"backbone": "efficientnet_b0", "verifier_mode": "residual"})
            for name in ("expert/best_model.pt", "tc_projection/gate/best_model.pt"):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"immutable synthetic source checkpoint")
            write_json(source / "expert/selection_lock.json", {
                "manifest_sha256": {"training_manifest": digest(source / "manifests/expert_training.csv")},
                "configuration_sha256": digest(source / "expert/run_config.json"),
                "checkpoint_sha256": digest(source / "expert/best_model.pt")})
            write_json(source / "tc_projection/selection_lock.json", {
                "protocol_version": "tc_oos_rsg_v1", "status": "locked_before_outer_tensor_loading",
                "manifest_sha256": {name: digest(inner / f"{name}.csv") for name in original_runner.validate_sources.__globals__["INNER_NAMES"]},
                "model_file_sha256": {name: digest(source / file) for name, file in {
                    "expert": "expert/best_model.pt", "expert_lock": "expert/selection_lock.json",
                    "gate": "tc_projection/gate/best_model.pt", "configuration": "tc_projection/run_config.json"}.items()}})
            audit = root / "outputs/theory/abmp_candidate_capacity_v1/audit_summary.json"
            write_json(audit, {"provenance": original_runner.validate_sources(inner, source)})
            report = root / "结题" / original_runner.REPORT_NAME
            report.parent.mkdir()
            report.write_bytes(b"immutable formal report fixture")
            budget = {"epochs": 2, "batch_size": 4, "learning_rate": .001, "weight_decay": .0001, "dropout": .1}
            original_protocol = {"protocol": "abmp_verifier_trust_development_v1", "source_audit_sha256": digest(audit),
                                 "formal_report_sha256": digest(report), "input_sha256": dict(fit_hash, **stop_hash),
                                 "threads": 1, "seeds": [17], "budget": budget,
                                 "thresholds": {"nll_improvement": .001, "recall_tolerance": .01, "intrusion_tolerance": .01, "dynamic_tolerance": 1e-6}}
            write_json(root / "protocol.json", original_protocol)
            reference = root / "outputs/training/abmp_verifier_trust_v1/development_fold_0"
            original_runner.run_experiment(root, root / "protocol.json", reference)
            protocol = {"protocol": "abmp_verifier_strong_controls_v1", "fold": 0, "arms": ["T0", "D0", "M0", "P0", "P1"],
                        "subsets": list(runner.SUBSETS), "counts": {"fit": 4, "stop": 4}, "seeds": [17], "budget": budget,
                        "reference_output": str(reference.relative_to(root)).replace("\\", "/"),
                        "reference_summary_sha256": digest(reference / "development_summary.json"),
                        "input_sha256": dict(fit_hash, **stop_hash), "formal_report_sha256": digest(report),
                        "global_temperature": {"beta_bounds": [1e-6, 100.]}, "dirichlet": {"regularization": .001}}
            write_json(root / "strong.json", protocol)
            output = root / "new"
            summary = runner.run_experiment(root, root / "strong.json", output)
            self.assertEqual(len(summary["runs"]), 3)
            self.assertTrue(summary["source_and_report_unchanged"])
            verified = verify_delivery(root, root / "strong.json", output)
            self.assertEqual(verified["status"], "VERIFIED")
            self.assertEqual(verified["prediction_table_count"], 10)
            self.assertEqual(verified["deterministic_training_replays"], 3)
            path = output / "P0_seed17/stop_predictions.csv"
            path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact"):
                verify_delivery(root, root / "strong.json", output)


if __name__ == "__main__":
    unittest.main()
