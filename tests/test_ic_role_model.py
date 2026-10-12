"""Synthetic IC-RFA tests: centering, references, matched controls and loss."""
import copy
import io
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import ic_role_model as model_code
except ModuleNotFoundError:
    model_code = None


UNITS = ("Z-O-U", "Z-O-R", "Z-S-U", "Z-S-R", "C-O-U", "C-O-R",
         "C-S-U", "C-S-R", "HC-S", "FC-S", "C-S-U-noaux", "C-S-R-noaux")


def make_fixture(n=120, seed=7):
    g = torch.Generator().manual_seed(seed)
    return dict(h=torch.randn(n, 64, generator=g, dtype=torch.float64),
                e=torch.randn(n, 23, generator=g, dtype=torch.float64),
                log_anchor=torch.log_softmax(torch.randn(n, 4, generator=g, dtype=torch.float64), 1),
                labels=torch.arange(n, dtype=torch.long) % 4)


class ICModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(model_code, "IC centered model is not implemented")
        torch.set_num_threads(1)
        self.batch = make_fixture(12)

    def build(self, unit="C-S-R"):
        return model_code.ICRoleModel(unit, seed=20261002)

    def forward(self, model, **kwargs):
        return model(self.batch["h"], self.batch["e"], self.batch["log_anchor"], **kwargs)

    def test_all_units_identity_parameter_and_storage_counts(self):
        for unit in UNITS:
            with self.subTest(unit=unit):
                model = self.build(unit)
                identity = model_code.reference_identity(model)
                trainable = 1476 if unit == "HC-S" else 3132 if unit == "FC-S" else 3024
                fixed = 0 if unit.startswith("Z") else trainable
                snapshot = trainable if unit.startswith("Z-S") else 0
                self.assertEqual(identity["trainable_count"], trainable)
                self.assertEqual(identity["fixed_function_count"], fixed)
                self.assertEqual(identity["regularization_snapshot_count"], snapshot)
                self.assertTrue(identity["references_unaliased"])
                result = self.forward(model, auxiliary=True)
                torch.testing.assert_close(result["logq"], self.batch["log_anchor"], atol=1e-12, rtol=0)
                for view in range(3):
                    torch.testing.assert_close(result["aux_logq"][:, view], self.batch["log_anchor"], atol=1e-12, rtol=0)

    def test_reference_is_frozen_unaliased_and_reload_stable(self):
        model = self.build()
        initial = model_code.reference_identity(model)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        loss = model_code.objective(model, self.forward(model, auxiliary=True), self.batch["labels"])
        loss["total"].backward()
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optimizer.step()
        after = model_code.reference_identity(model)
        self.assertEqual(initial["reference_current_sha256"], after["reference_current_sha256"])
        self.assertTrue(after["references_match_initial"])
        packet = io.BytesIO()
        torch.save(model.state_dict(), packet)
        packet.seek(0)
        restored = self.build()
        restored.load_state_dict(torch.load(packet, weights_only=True))
        torch.testing.assert_close(self.forward(model)["logq"], self.forward(restored)["logq"], atol=0, rtol=0)

    def test_same_seed_controls_have_same_initial_current_parameters(self):
        for prefix in ("Z", "C"):
            base = dict(self.build(prefix+"-O-U").named_parameters())
            for unit in (prefix+"-O-R", prefix+"-S-U", prefix+"-S-R"):
                other = dict(self.build(unit).named_parameters())
                for name, parameter in base.items():
                    torch.testing.assert_close(parameter, other[name], atol=0, rtol=0)
        c, z = self.build("C-S-R"), self.build("Z-S-R")
        torch.testing.assert_close(c.head[0].weight, z.head[0].weight, atol=0, rtol=0)
        for a, b in zip(c.adapters, z.adapters):
            torch.testing.assert_close(a.down.weight, b.down.weight, atol=0, rtol=0)
            self.assertEqual(int(torch.count_nonzero(b.up.weight)), 0)
            self.assertGreater(int(torch.count_nonzero(a.up.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(z.head[2].weight)), 0)
        self.assertGreater(int(torch.count_nonzero(c.head[2].weight)), 0)

    def test_pair_loss_paths_and_regularization_centers(self):
        for unit in UNITS:
            model = self.build(unit)
            result = self.forward(model, auxiliary=True)
            loss = model_code.objective(model, result, self.batch["labels"])
            expected_pair = []
            for j in (1, 2, 3):
                mask = (self.batch["labels"] == 0) | (self.batch["labels"] == j)
                logits = (result["logq"] if unit in ("HC-S", "FC-S") else result["aux_logq"][:, j-1])[mask]
                log_odds = logits[:, 0]-logits[:, j]
                targets = (self.batch["labels"][mask] == 0).double()
                expected_pair.append(torch.nn.functional.binary_cross_entropy_with_logits(log_odds, targets))
            if unit.endswith("noaux"):
                self.assertEqual(float(loss["pair"].detach()), 0.)
            else:
                torch.testing.assert_close(loss["pair"], torch.stack(expected_pair).mean(), rtol=1e-12, atol=1e-12)
            if "-O-" in unit:
                self.assertGreater(float(loss["regularization"].detach()), 0.)
            else:
                self.assertEqual(float(loss["regularization"].detach()), 0.)
            torch.testing.assert_close(loss["total"], loss["data"]+loss["regularization"])

    def test_reference_uses_original_h_and_delta_difference_is_outer(self):
        model = self.build()
        with torch.no_grad():
            model.adapters[0].up.weight.add_(.5)
        result = self.forward(model, auxiliary=True)
        expected = (model.adapters[0](self.batch["h"])-model.adapter_refs[0](self.batch["h"]))/16
        torch.testing.assert_close(result["delta"][:, 0], expected, atol=0, rtol=0)
        self.assertGreater(float(result["scores"].detach().norm()), 1e-8)
        original = torch.cat([self.batch["h"], self.batch["e"]], 1)
        adapted = torch.cat([result["adapted_h"], self.batch["e"]], 1)
        torch.testing.assert_close(result["scores"], model.head(adapted)-model.head_ref(original), atol=0, rtol=0)

    def test_uniform_route_and_same_weight_off(self):
        model = self.build("C-S-U")
        with torch.no_grad():
            model.adapters[0].up.weight.add_(.3)
            model.head[2].bias[0] += .2
        result = self.forward(model)
        off = self.forward(model, adaptation=False)
        torch.testing.assert_close(result["g"], result["b"].expand(-1, 3)/3, atol=0, rtol=0)
        torch.testing.assert_close(off["adapted_h"], self.batch["h"], atol=0, rtol=0)
        self.assertGreater(float((off["logq"]-self.batch["log_anchor"]).detach().norm()), 1e-8)
        self.assertTrue(bool((result["delta"].norm(dim=2) <= 1).all()))

    def test_batch_size_does_not_change_prediction(self):
        model = self.build()
        with torch.no_grad():
            model.head[2].weight.add_(.1)
        full = self.forward(model)["logq"]
        rows = [model(self.batch["h"][i:i+1], self.batch["e"][i:i+1],
                      self.batch["log_anchor"][i:i+1])["logq"] for i in range(12)]
        torch.testing.assert_close(full, torch.cat(rows), rtol=1e-12, atol=1e-12)

    def test_numeric_underflow_and_invalid_inputs_fail(self):
        model = self.build()
        for key in ("h", "e", "log_anchor"):
            broken = copy.deepcopy(self.batch)
            broken[key][0, 0] = float("nan")
            with self.assertRaises(ValueError):
                model(broken["h"], broken["e"], broken["log_anchor"])
        anchor = torch.log_softmax(torch.tensor([[-900., 0., -2., -3.]], dtype=torch.float64), 1)
        with self.assertRaises(ValueError):
            model(self.batch["h"][:1], self.batch["e"][:1], anchor)
        with self.assertRaises(ValueError):
            self.build("C-X-R")

    def test_simultaneous_role_reordering_preserves_prediction_and_loss(self):
        model = self.build()
        with torch.no_grad():
            model.adapters[0].up.weight.add_(.3)
            model.head[2].weight[0].add_(.1)
        order = torch.tensor([0, 3, 1, 2])
        reordered = copy.deepcopy(model)
        reordered.adapters = torch.nn.ModuleList([reordered.adapters[j] for j in (2, 0, 1)])
        reordered.adapter_refs = torch.nn.ModuleList([reordered.adapter_refs[j] for j in (2, 0, 1)])
        with torch.no_grad():
            for head in (reordered.head, reordered.head_ref):
                head[2].weight.copy_(head[2].weight[order].clone())
                head[2].bias.copy_(head[2].bias[order].clone())
        old = self.forward(model, auxiliary=True)
        new = reordered(self.batch["h"], self.batch["e"], self.batch["log_anchor"][:, order], auxiliary=True)
        torch.testing.assert_close(new["logq"], old["logq"][:, order], atol=1e-12, rtol=1e-12)
        old_loss = model_code.objective(model, old, self.batch["labels"])
        new_loss = model_code.objective(reordered, new, order.argsort()[self.batch["labels"]])
        torch.testing.assert_close(new_loss["total"], old_loss["total"], atol=1e-12, rtol=1e-12)
        wrong = copy.deepcopy(model)
        wrong.adapters = torch.nn.ModuleList([wrong.adapters[j] for j in (2, 0, 1)])
        self.assertGreater(float((self.forward(wrong)["logq"]-old["logq"]).detach().norm()), 1e-8)

    def test_alias_and_trainable_reference_are_rejected(self):
        model = self.build()
        model.adapter_refs[0] = model.adapters[0]
        self.assertFalse(model_code.reference_identity(model)["references_unaliased"])
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.forward(model)
        model = self.build()
        model.head_ref[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.forward(model)

    def test_initialization_ranges_and_rng_are_preserved(self):
        before = torch.random.get_rng_state().clone()
        model = self.build()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(bool((model.head[0].weight.abs() <= 1/(87**.5)).all()))
        self.assertTrue(bool((model.head[0].bias.abs() <= 1/(87**.5)).all()))
        self.assertTrue(bool((model.head[2].weight.abs() <= .25).all()))
        self.assertTrue(bool((model.head[2].bias == 0).all()))
        for branch in model.adapters:
            self.assertTrue(bool((branch.down.weight.abs() <= .125).all()))
            self.assertTrue(bool((branch.down.bias.abs() <= .125).all()))
            self.assertTrue(bool((branch.up.weight.abs() <= .5).all()))


if __name__ == "__main__":
    unittest.main()
