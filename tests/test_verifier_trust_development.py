import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import run_verifier_trust_development as runner
except ModuleNotFoundError:
    runner = None


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_fixture(root, name="gate_stop_projector_fit"):
    folder = root / "outputs/theory/abmp_candidate_capacity_v1" / name
    manifest = root / "outputs/training/tc_oos_rsg_manifests_v1/fold_0" / f"{name}.csv"
    roles = ["target_mineral", "ti_bearing_negative", "gangue_negative", "metallic_hard_negative"]
    rows = [{"image_id": f"{name}-ID{i}", "split_group_id": f"{name}-G{i}", "mineral_label": f"mineral{i}",
             "four_class_id": str(i), "four_class_label": roles[i], "outer_fold": "0",
             "tc_subset": name} for i in range(4)]
    write_csv(manifest, list(rows[0]), rows)
    evidence = [{"image_id": r["image_id"], "split_group_id": r["split_group_id"],
                 "true_class_id": r["four_class_id"], "candidate_gate": ".4", "original_gate": ".5",
                 "ti_probability": ".25", "metal_probability": ".5", "target_scale": str(math.exp(-.5))} for r in rows]
    write_csv(folder / "routing_evidence.csv", list(evidence[0]), evidence)
    pre = [.6, .2, .1, .1]
    normalizer = .6*math.exp(-.5) + .4
    verified = [.6*math.exp(-.5)/normalizer, .2/normalizer, .1/normalizer, .1/normalizer]
    components = []
    for branch in [f"{base}_{state}" for base in ("direct", "mapped", "fixed", "oos", "original") for state in ("pre", "verified")]:
        for r in rows:
            probs = verified if branch.endswith("verified") else pre
            components.append({"image_id": r["image_id"], "split_group_id": r["split_group_id"],
                               "mineral_label": r["mineral_label"], "true_class_id": r["four_class_id"],
                               "branch": branch, "predicted_class_id": "0",
                               **{f"prob_{i}": str(p) for i, p in enumerate(probs)}})
    write_csv(folder / "component_predictions.csv", list(components[0]), components)
    paths = {"manifest.csv": manifest, "routing_evidence.csv": folder / "routing_evidence.csv",
             "component_predictions.csv": folder / "component_predictions.csv"}
    hashes = {f"{name}/{file}": hashlib.sha256(path.read_bytes()).hexdigest() for file, path in paths.items()}
    return name, hashes, paths


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "development runner implementation is missing")
        torch.set_num_threads(1)

    def test_loader_aligns_probabilities_labels_and_inference_only_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, hashes, _ = source_fixture(root)
            data = runner.load_subset(root, name, hashes)
            torch.testing.assert_close(data["p"], torch.tensor([[.6, .2, .1, .1]]*4, dtype=torch.float64))
            torch.testing.assert_close(data["contradiction"], torch.full((4, 1), .5, dtype=torch.float64))
            self.assertEqual(data["labels"].tolist(), [0, 1, 2, 3])
            self.assertEqual(tuple(data["evidence"].shape), (4, 18))
            torch.testing.assert_close(data["evidence"][:1].repeat(4, 1), data["evidence"])

    def test_loader_refuses_changed_hash_and_other_fold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, hashes, paths = source_fixture(root)
            paths["routing_evidence.csv"].write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                runner.load_subset(root, name, hashes)
            with self.assertRaisesRegex(ValueError, "subset"):
                runner.load_subset(root, "outer_eval", hashes)

    def test_loader_refuses_misaligned_labels_even_if_file_hash_registered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, hashes, paths = source_fixture(root)
            path = paths["routing_evidence.csv"]
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["true_class_id"] = "1"
            write_csv(path, list(rows[0]), rows)
            hashes[f"{name}/routing_evidence.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.load_subset(root, name, hashes)

    def test_loader_replays_frozen_branch_geometry_not_just_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, hashes, paths = source_fixture(root)
            path = paths["component_predictions.csv"]
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            row = next(row for row in rows if row["branch"] == "oos_pre")
            row.update(prob_0=".5", prob_1=".3")
            write_csv(path, list(rows[0]), rows)
            hashes[f"{name}/component_predictions.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "replay"):
                runner.load_subset(root, name, hashes)

    def test_training_changes_weights_and_selects_earliest_minimum_stop_nll(self):
        torch.manual_seed(22)
        evidence = torch.randn(32, 18, dtype=torch.float64)
        labels = torch.tensor([0, 1, 2, 3]*8)
        evidence[:, 0] = (labels == 0).double()*4 - 2
        data = {"evidence": evidence, "p": torch.tensor([[.6, .2, .1, .1]]*32, dtype=torch.float64),
                "contradiction": torch.ones(32, 1, dtype=torch.float64), "labels": labels}
        budget = {"epochs": 12, "batch_size": 16, "learning_rate": .01, "weight_decay": .0001, "dropout": 0.}
        first = runner.fit_head(data, data, "A4", 23, budget)
        second = runner.fit_head(data, data, "A3", 23, budget)
        self.assertNotEqual(first["initial_sha256"], first["state_sha256"])
        self.assertEqual(first["initial_sha256"], second["initial_sha256"])
        self.assertEqual(first["parameter_count"], second["parameter_count"])
        self.assertLess(first["selected_fit_objective"], first["initial_fit_objective"])
        want = min(first["history"], key=lambda row: row["stop_metrics"]["nll"])["epoch"]
        self.assertEqual(first["best_epoch"], want)
        prediction = runner.predict_head(first["head"], data, "A4")
        selected = first["history"][want-1]["stop_metrics"]["nll"]
        self.assertAlmostEqual(runner.calculate_probability_metrics(prediction["probabilities"], labels)["nll"], selected, places=14)
        sliced = {k: v[:1] for k, v in data.items()}
        torch.testing.assert_close(prediction["probabilities"][:1], runner.predict_head(first["head"], sliced, "A4")["probabilities"])

    def test_training_reproducible_and_invalid_mode_or_budget_refused(self):
        data = {"evidence": torch.zeros(4, 18, dtype=torch.float64),
                "p": torch.tensor([[.6, .2, .1, .1]]*4, dtype=torch.float64),
                "contradiction": torch.ones(4, 1, dtype=torch.float64), "labels": torch.arange(4)}
        budget = {"epochs": 2, "batch_size": 4, "learning_rate": .001, "weight_decay": .0001, "dropout": .1}
        a = runner.fit_head(data, data, "A3", 8, budget)
        b = runner.fit_head(data, data, "A3", 8, budget)
        self.assertEqual(a["state_sha256"], b["state_sha256"])
        with self.assertRaises(ValueError):
            runner.fit_head(data, data, "unknown", 8, budget)
        with self.assertRaises(ValueError):
            runner.fit_head(data, data, "A3", 8, dict(budget, epochs=0))

    def test_gate_uses_strongest_controls_and_rejects_constant_or_increased_intrusion(self):
        controls = {"A0": dict(nll=.8, macro_f1=.65, target_recall=.6, ti_intrusion_to_target=.1, metallic_intrusion_to_target=.1),
                    "A1": dict(nll=.9, macro_f1=.66, target_recall=.6, ti_intrusion_to_target=.08, metallic_intrusion_to_target=.1)}
        controls["A2"] = dict(controls["A0"])
        metrics = dict(nll=.79, macro_f1=.67, target_recall=.6, ti_intrusion_to_target=.08, metallic_intrusion_to_target=.1)
        diag = {"gamma_std_active": .2, "minimum_fixed_output_distance": .01, "fit_mean_replay_output_distance": .01}
        thresholds = {"nll_improvement": .001, "recall_tolerance": .01, "intrusion_tolerance": .01, "dynamic_tolerance": 1e-6}
        self.assertTrue(runner.development_gate(metrics, controls, diag, thresholds)["passed"])
        for changed, diagnostics in [(dict(metrics, nll=.81), diag), (dict(metrics, ti_intrusion_to_target=.1), diag),
                                     (metrics, dict(diag, gamma_std_active=0))]:
            self.assertFalse(runner.development_gate(changed, controls, diagnostics, thresholds)["passed"])
        with self.assertRaises(ValueError):
            runner.development_gate(metrics, {"A1": controls["A1"]}, diag, thresholds)

    def test_existing_output_refused_before_any_source_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaises(FileExistsError):
                runner.run_experiment(folder / "missing", folder / "missing.json", folder)

    def test_empty_or_duplicate_seeds_refused_before_source_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seeds in ([], [3, 3]):
                path = root / "protocol.json"
                path.write_text(json.dumps({"protocol": "abmp_verifier_trust_development_v1", "seeds": seeds}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "seeds"):
                    runner.run_experiment(root, path, root / "output")

    def test_complete_experiment_preserves_sources_and_delivers_every_arm_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, fit_hashes, _ = source_fixture(root)
            _, stop_hashes, _ = source_fixture(root, "projector_stop")
            inner = root / "outputs/training/tc_oos_rsg_manifests_v1/fold_0"
            source = root / "outputs/training/tc_oos_rsg_v1/fold_0"
            write_json = lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text(json.dumps(value), encoding="utf-8"))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            training = []
            for name in ("expert_fit", "expert_stop", "gate_fit"):
                rows = [{"image_id": name, "split_group_id": name, "mindat_photo_id": name}]
                write_csv(inner / f"{name}.csv", list(rows[0]), rows)
                if name != "gate_fit":
                    training.extend(rows)
            write_csv(source / "manifests/expert_training.csv", list(training[0]), training)
            config = {"backbone": "efficientnet_b0", "verifier_mode": "residual"}
            write_json(source / "expert/run_config.json", config)
            write_json(source / "expert/environment.json", config)
            write_json(source / "tc_projection/run_config.json", config)
            for path in (source / "expert/best_model.pt", source / "tc_projection/gate/best_model.pt"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frozen checkpoint hash fixture; no weight loading")
            write_json(source / "expert/selection_lock.json", {
                "manifest_sha256": {"training_manifest": digest(source / "manifests/expert_training.csv")},
                "configuration_sha256": digest(source / "expert/run_config.json"),
                "checkpoint_sha256": digest(source / "expert/best_model.pt")})
            write_json(source / "tc_projection/selection_lock.json", {
                "protocol_version": "tc_oos_rsg_v1", "status": "locked_before_outer_tensor_loading",
                "manifest_sha256": {name: digest(inner / f"{name}.csv") for name in runner.validate_sources.__globals__["INNER_NAMES"]},
                "model_file_sha256": {name: digest(path) for name, path in {
                    "expert": source / "expert/best_model.pt", "expert_lock": source / "expert/selection_lock.json",
                    "gate": source / "tc_projection/gate/best_model.pt", "configuration": source / "tc_projection/run_config.json"}.items()}})
            audit_path = root / "outputs/theory/abmp_candidate_capacity_v1/audit_summary.json"
            write_json(audit_path, {"provenance": runner.validate_sources(inner, source)})
            report = root / "结题" / runner.REPORT_NAME
            report.parent.mkdir()
            report.write_bytes(b"immutable formal report hash fixture")
            protocol = {"protocol": "abmp_verifier_trust_development_v1", "source_audit_sha256": digest(audit_path),
                        "formal_report_sha256": digest(report), "input_sha256": dict(fit_hashes, **stop_hashes),
                        "threads": 1, "seeds": [17], "budget": {"epochs": 2, "batch_size": 4, "learning_rate": .001, "weight_decay": .0001, "dropout": .1},
                        "thresholds": {"nll_improvement": .001, "recall_tolerance": .01, "intrusion_tolerance": .01, "dynamic_tolerance": 1e-6}}
            protocol_path = root / "protocol.json"
            write_json(protocol_path, protocol)
            output = root / "experiment"
            summary = runner.run_experiment(root, protocol_path, output)
            self.assertEqual(set(summary["controls"]), {"A0", "A1", "A2"})
            self.assertEqual([run["arm"] for run in summary["runs"]], ["A3", "A4"])
            self.assertTrue(summary["source_and_report_unchanged"])
            self.assertFalse(summary["outer_manifest_read"])
            self.assertEqual(json.loads((output / "run_status.json").read_text())["status"], "COMPLETED")
            for run in summary["runs"]:
                folder = output / f"{run['arm']}_seed17"
                self.assertTrue((folder / "best_head.pt").is_file())
                self.assertEqual(len(json.loads((folder / "history.json").read_text())), 2)
                self.assertEqual(run["diagnostics"]["checkpoint_replay_max_abs_residual"], 0.)
            self.assertEqual(digest(report), protocol["formal_report_sha256"])
            try:
                import verify_verifier_trust_delivery as verifier
            except ModuleNotFoundError:
                verifier = None
            self.assertIsNotNone(verifier, "complete delivery verification is missing")
            verified = verifier.verify_delivery(root, output)
            self.assertEqual(verified["status"], "VERIFIED")
            self.assertEqual(verified["prediction_table_count"], 12)
            self.assertEqual(verified["max_abs_replay_residual"], 0.)
            summary_path = output / "development_summary.json"
            changed = json.loads(summary_path.read_text(encoding="utf-8"))
            changed["aggregate"]["A4"]["all_seeds_passed"] = not changed["aggregate"]["A4"]["all_seeds_passed"]
            summary_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aggregate"):
                verifier.verify_delivery(root, output)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            path = output / "A4_seed17/stop_predictions.csv"
            path.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                verifier.verify_delivery(root, output)


if __name__ == "__main__":
    unittest.main()
