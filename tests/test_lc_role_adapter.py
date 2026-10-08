import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import lc_role_adapter as adapter
except ModuleNotFoundError:
    adapter = None


class RoleAdapterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(adapter, "LC-RFA adapter module missing")
        torch.set_num_threads(1)
        gen = torch.Generator().manual_seed(9)
        self.h = torch.randn(40, 64, generator=gen, dtype=torch.float64)
        self.e = torch.randn(40, 23, generator=gen, dtype=torch.float64)
        self.log_anchor = torch.log_softmax(torch.randn(40, 4, generator=gen, dtype=torch.float64), 1)
        self.labels = torch.arange(40) % 4

    def build(self, arm="R1"):
        return adapter.RoleResidualModel(arm, seed=20261002, alpha_max=.5, beta_max=.5)

    def forward(self, model, auxiliary=True):
        return model(self.h, self.e, self.log_anchor, auxiliary=auxiliary)

    def activate(self, model):
        gen = torch.Generator().manual_seed(12)
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "up.weight" in name or name.startswith("head.2"):
                    value.copy_(.2 * torch.randn(value.shape, generator=gen, dtype=value.dtype))
        return model

    def test_parameter_counts_identity_and_comparable_initialization(self):
        counts = {"H0": 1476, "F0": 3132, "S0": 3024, "S1": 3024,
                  "R0": 3024, "R1": 3024, "U1": 3024}
        reference = self.build().state_dict()
        for arm, count in counts.items():
            model = self.build(arm)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            result = self.forward(model)
            torch.testing.assert_close(result["logq"], self.log_anchor, rtol=1e-10, atol=1e-10)
            self.assertFalse(any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.Dropout)) for m in model.modules()))
            if arm != "F0":
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, reference[key], rtol=0, atol=0)
            for branch in model.adapters:
                self.assertIsNone(branch.up.bias)
                self.assertEqual(sum(p.numel() for p in branch.parameters()), 516)

    def test_role_and_uniform_gates_and_shared_auxiliary_views(self):
        for arm in ("S1", "R1"):
            model = self.activate(self.build(arm))
            result = self.forward(model)
            p = self.log_anchor.exp()
            b = (4 * p[:, :1] * p[:, 1:]).sum(1, keepdim=True)
            expected = b.expand(-1, 3) / 3 if arm == "S1" else 4 * p[:, :1] * p[:, 1:]
            torch.testing.assert_close(result["g"], expected)
            self.assertTrue((result["delta"].norm(dim=2) <= 1 + 1e-12).all())
            torch.testing.assert_close(result["adapted_h"], self.h + (result["g"][:, :, None] * result["delta"]).sum(1))
            for j in range(3):
                scores = model.head(torch.cat([self.h + result["delta"][:, j], self.e], 1))
                q = adapter.bounded_log_probabilities(self.log_anchor, scores, b, alpha_max=.5, beta_max=.5)
                torch.testing.assert_close(result["aux_logq"][:, j], q["logq"])
            self.assertEqual(len([name for name, _ in model.named_modules() if name == "head"]), 1)

    def test_pair_loss_averages_each_mask_before_equal_pair_weight(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 3])
        logs = torch.log_softmax(torch.arange(28, dtype=torch.float64).reshape(7, 4) / 7, 1)
        views = torch.stack([logs, logs.roll(1, 1), logs.roll(2, 1)], 1)
        expected = []
        for j in range(1, 4):
            mask = (labels == 0) | (labels == j)
            logpair = views[mask, j-1][:, [0, j]]
            conditional = logpair - torch.logsumexp(logpair, 1, keepdim=True)
            expected.append(-conditional[range(int(mask.sum())), (labels[mask] != 0).long()].mean())
        torch.testing.assert_close(adapter.pair_loss(views, labels), torch.stack(expected).mean())
        duplicate = logs[:, None, :].expand(-1, 3, -1)
        torch.testing.assert_close(adapter.pair_loss(logs, labels), adapter.pair_loss(duplicate, labels))
        with self.assertRaises(ValueError):
            adapter.pair_loss(logs[:3], labels[:3])

    def test_objective_pair_arm_scope_and_all_parameter_l2(self):
        for arm in ("H0", "F0", "S0", "S1", "R0", "R1", "U1"):
            model = self.activate(self.build(arm))
            result = self.forward(model)
            loss = adapter.objective(model, result, self.labels, .01)
            expected_l2 = .005 * sum(p.square().sum() for p in model.parameters())
            torch.testing.assert_close(loss["l2"], expected_l2)
            torch.testing.assert_close(loss["total"], loss["nll"] + .1 * loss["pair"] + expected_l2)
            if arm in ("H0", "S0", "R0"):
                self.assertEqual(float(loss["pair"]), 0)
            else:
                source = result["logq"] if arm == "F0" else result["aux_logq"]
                torch.testing.assert_close(loss["pair"], adapter.pair_loss(source, self.labels))
        with self.assertRaises(ValueError):
            adapter.objective(model, result, self.labels, float("nan"))

    def test_data_gradient_not_confused_with_regularization_shrink(self):
        model = self.build()
        opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
        rows = []
        for step in range(5):
            loss = adapter.objective(model, self.forward(model), self.labels, .01)
            parameters = tuple(model.parameters())
            data_grad = torch.autograd.grad(loss["nll"] + .1*loss["pair"], parameters, retain_graph=True, allow_unused=True)
            total_grad = torch.autograd.grad(loss["total"], parameters, allow_unused=True)
            row = {name: (0 if d is None else float(d.norm()), 0 if t is None else float(t.norm()))
                   for (name, _), d, t in zip(model.named_parameters(), data_grad, total_grad)}
            rows.append(row)
            opt.zero_grad()
            for parameter, gradient in zip(parameters, total_grad):
                parameter.grad = gradient
            opt.step()
        self.assertGreater(rows[0]["head.2.weight"][0], 0)
        self.assertEqual(rows[0]["adapters.0.down.weight"][0], 0)
        self.assertGreater(rows[0]["adapters.0.down.weight"][1], 0)
        self.assertTrue(any(row["adapters.0.up.weight"][0] > 0 for row in rows[1:]))
        self.assertTrue(any(row["adapters.0.down.weight"][0] > 0 for row in rows[2:]))

    def test_batch_independence_and_output_bounds(self):
        model = self.activate(self.build())
        result = self.forward(model)
        single = model(self.h[:1], self.e[:1], self.log_anchor[:1])
        torch.testing.assert_close(single["logq"], result["logq"][:1], atol=1e-12, rtol=1e-12)
        self.assertTrue((result["u"].abs() <= result["alpha"] + 1e-12).all())
        self.assertTrue(((result["v"].max(1).values-result["v"].min(1).values) <= result["beta"][:, 0] + 1e-12).all())
        torch.testing.assert_close(result["logq"].exp().sum(1), torch.ones(40, dtype=torch.float64))

    def test_near_one_anchor_is_stable_and_underflow_is_explicit(self):
        anchor = torch.log_softmax(torch.tensor([[0., -50., -55., -60.]], dtype=torch.float64), 1)
        result = adapter.bounded_log_probabilities(anchor, torch.ones_like(anchor), torch.zeros(1, 1, dtype=torch.float64), alpha_max=.5, beta_max=.5)
        torch.testing.assert_close(result["logq"], anchor, atol=1e-12, rtol=1e-12)
        with self.assertRaises(adapter.NumericRangeFailure) as caught:
            adapter.bounded_log_probabilities(self.log_anchor[:1], torch.tensor([[1000., 0, 0, 0]], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64), alpha_max=.5, beta_max=.5, bounded=False)
        self.assertIn("NUMERIC_RANGE_FAILURE", str(caught.exception))
        self.assertTrue(torch.isfinite(caught.exception.logq).all())

    def test_invalid_anchor_shape_scores_budget_and_arm_rejected(self):
        for anchor in (self.log_anchor + 1, self.log_anchor*float("nan"), self.log_anchor[:1, :3], self.log_anchor*float("inf")):
            with self.assertRaises(ValueError):
                adapter.bounded_log_probabilities(anchor, torch.zeros_like(self.log_anchor), torch.ones(40, 1, dtype=torch.float64), alpha_max=.5, beta_max=.5)
        for kwargs in (dict(alpha_max=-1, beta_max=.5), dict(alpha_max=.5, beta_max=float("nan"))):
            with self.assertRaises(ValueError):
                adapter.RoleResidualModel("R1", seed=7, **kwargs)
        with self.assertRaises(ValueError):
            self.build("unknown")
        with self.assertRaises(ValueError):
            self.build()(self.h.float(), self.e, self.log_anchor)


if __name__ == "__main__":
    unittest.main()
