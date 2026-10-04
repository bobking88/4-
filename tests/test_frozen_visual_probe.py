import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import frozen_visual_probe as probe
    import run_frozen_visual_probe as runner
except ModuleNotFoundError:
    probe = runner = None


class FrozenVisualProbeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(probe, "frozen visual probe implementation missing")
        self.assertIsNotNone(runner, "bounded probe runner missing")
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_fit_only_pca_standardization_and_constant_columns(self):
        e = torch.randn(25, 18, dtype=torch.float64)
        h = torch.randn(25, 12, dtype=torch.float64)
        e[:, 0] = 3.
        state = probe.fit_preprocessing(e, h, components=6)
        train = probe.apply_preprocessing(e, h, state)
        self.assertEqual(train["E"].shape, (25, 18))
        self.assertEqual(train["H"].shape, (25, 6))
        self.assertLess(float(train["E"].mean(0).abs().max()), 1e-14)
        self.assertTrue(torch.equal(train["E"][:, 0], torch.zeros(25, dtype=torch.float64)))
        snapshot = copy.deepcopy(state)
        other = probe.apply_preprocessing(e + 100, h + 100, state)
        self.assertGreater(float(other["E"].abs().max()), 10.)
        for key, value in snapshot.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        basis = state["basis"]
        torch.testing.assert_close(basis.T @ basis, torch.eye(6, dtype=torch.float64), atol=1e-12, rtol=1e-12)

    def test_input_ablations_share_dimensions_without_label_access(self):
        blocks = {"E": torch.randn(25, 18, dtype=torch.float64), "H": torch.randn(25, 6, dtype=torch.float64)}
        e = probe.make_input(blocks, "E", 17)
        h = probe.make_input(blocks, "H", 17)
        both = probe.make_input(blocks, "EH", 17)
        shuffled = probe.make_input(blocks, "EH_permuted", 17)
        self.assertEqual({x.shape for x in (e, h, both, shuffled)}, {(25, 24)})
        self.assertTrue(torch.equal(e[:, 18:], torch.zeros(25, 6, dtype=torch.float64)))
        self.assertTrue(torch.equal(h[:, :18], torch.zeros(25, 18, dtype=torch.float64)))
        torch.testing.assert_close(both[:, :18], blocks["E"])
        torch.testing.assert_close(shuffled[:, :18], blocks["E"])
        self.assertFalse(torch.equal(shuffled[:, 18:], blocks["H"]))
        self.assertEqual(sorted(shuffled[:, 18].tolist()), sorted(blocks["H"][:, 0].tolist()))
        torch.testing.assert_close(shuffled, probe.make_input(blocks, "EH_permuted", 17), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            probe.make_input(blocks, "new_network", 17)

    def test_probability_metrics_distinguish_promotions_losses_and_intrusions(self):
        p = torch.tensor([[.2, .6, .1, .1], [.6, .2, .1, .1], [.1, .7, .1, .1], [.1, .1, .1, .7]], dtype=torch.float64)
        q = p[[1, 0, 1, 1]]
        counts = probe.transition_counts(q, p, torch.tensor([0, 0, 1, 3]))
        self.assertEqual(counts["new_correct_targets"], 1)
        self.assertEqual(counts["lost_correct_targets"], 1)
        self.assertEqual(counts["net_correct_targets"], 0)
        self.assertEqual(counts["new_false_target_by_class"], {"1": 1, "2": 0, "3": 1})

    def test_small_heads_replay_training_and_select_earliest_minimum(self):
        x = torch.randn(28, 24, dtype=torch.float64)
        y = torch.tensor([0, 1, 2, 3] * 7)
        budget = {"epochs": 3, "batch_size": 10, "learning_rate": .001, "weight_decay": .0001, "dropout": .1, "hidden_dim": 64}
        for architecture in ("linear", "mlp"):
            a = probe.fit_classifier(x, y, x, y, architecture, 19, budget)
            b = probe.fit_classifier(x, y, x, y, architecture, 19, budget)
            self.assertEqual(a["state_sha256"], b["state_sha256"])
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(a["best_epoch"], min(a["history"], key=lambda r: r["stop_nll"])["epoch"])
            self.assertNotEqual(a["initial_sha256"], a["state_sha256"])
            self.assertEqual(a["parameter_count"], 100 if architecture == "linear" else 1860)

    def test_feature_model_frozen_and_batchnorm_state_cannot_update(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 5))
        model = probe.freeze_extractor(model)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        before = copy.deepcopy(model.state_dict())
        with torch.inference_mode():
            model(torch.ones(8, 3))
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_feature_checkpoint_subset_is_strict(self):
        module = torch.nn.Linear(3, 4)
        checkpoint = {"model_state_dict": {"features.weight": module.weight.detach(), "features.bias": module.bias.detach(), "role_head.weight": torch.ones(4, 4)}}
        state = probe.extract_feature_state(checkpoint)
        self.assertEqual(set(state), {"weight", "bias"})
        module.load_state_dict(state, strict=True)
        with self.assertRaises(ValueError):
            probe.extract_feature_state({"model_state_dict": {"role_head.weight": torch.ones(4, 4)}})

    def test_path_scope_rejects_traversal_absolute_and_alias_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.jpg"
            image.write_bytes(b"fixture")
            self.assertEqual(probe.scoped_image_path(root, "image.jpg"), image.resolve())
            for bad in ("../escape.jpg", str(image.resolve()), "C:/elsewhere/a.jpg"):
                with self.assertRaises(ValueError):
                    probe.scoped_image_path(root, bad)

    def test_invalid_values_and_dimensions_fail_before_training(self):
        e = torch.zeros(10, 18, dtype=torch.float64)
        h = torch.zeros(9, 12, dtype=torch.float64)
        with self.assertRaises(ValueError):
            probe.fit_preprocessing(e, h, 3)
        with self.assertRaises(ValueError):
            probe.fit_preprocessing(e, torch.full((10, 12), float("nan"), dtype=torch.float64), 3)

    def test_registered_protocol_rejects_scope_budget_or_architecture_changes(self):
        path = Path(__file__).resolve().parents[1] / "docs/experiment_protocols/abmp_frozen_visual_probe_v1.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        runner.validate_protocol(protocol)
        for field, value in (("fold", 1), ("arms", ["residual"]), ("counts", {"fit": 340, "stop": 341}), ("architectures", ["transformer"])):
            bad = dict(protocol, **{field: value})
            with self.assertRaisesRegex(ValueError, "scope"):
                runner.validate_protocol(bad)
        bad = copy.deepcopy(protocol)
        bad["budget"]["epochs"] = 60
        with self.assertRaisesRegex(ValueError, "budget"):
            runner.validate_protocol(bad)
        bad = dict(protocol, strong_control_summary_sha256=protocol["strong_control_summary_sha256"] + "extra")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            runner.validate_protocol(bad)

    def test_output_overwrite_fails_before_any_source_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                runner.run_experiment(Path(tmp), Path(tmp) / "missing.json", Path(tmp))


if __name__ == "__main__":
    unittest.main()
