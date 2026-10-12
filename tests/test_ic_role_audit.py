"""Synthetic proofs and counterexamples for IC-RFA implementation claims."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ic_role_model import ICRoleModel, UNITS, _digest, objective, reference_identity
from lc_role_adapter import bounded_log_probabilities
from test_ic_role_model import make_fixture

try:
    import ic_role_audit as audit
except ModuleNotFoundError:
    audit = None


def batch(n=4, *, constructed=False):
    out = make_fixture(n, seed=20261006)
    out["records"] = [dict(image_id=f"synthetic-{i}", split_group_id=f"group-{i}") for i in range(n)]
    if constructed:
        out["h"].fill_(1)
        out["e"].zero_()
        out["log_anchor"].fill_(-torch.log(torch.tensor(4., dtype=torch.float64)))
    return out


def constructed_model(cls=ICRoleModel):
    model = cls("C-S-R", seed=20261002)
    with torch.no_grad():
        for branch in model.adapters:
            branch.down.weight.fill_(.03)
            branch.down.bias.zero_()
            branch.up.weight.fill_(.04)
        model.head[0].weight.zero_()
        model.head[0].weight[:, :64].fill_(.02)
        model.head[0].bias.zero_()
        model.head[2].weight.fill_(-.10)
        model.head[2].weight[0].fill_(.10)
        model.head[2].bias.zero_()
    model.adapter_refs = copy.deepcopy(model.adapters)
    model.head_ref = copy.deepcopy(model.head)
    for _, p in model.named_function_reference():
        p.requires_grad_(False)
    model.initial_current_sha256 = _digest(model.named_current())
    model.initial_reference_sha256 = _digest(model.named_function_reference())
    return model


class WrongReferenceModel(ICRoleModel):
    """Intentionally wrong test-only wiring, with an optional fake gradient."""
    detach_wrong_reference = False

    def forward(self, h, e, log_anchor, *, auxiliary=False, adaptation=True):
        result = super().forward(h, e, log_anchor, auxiliary=auxiliary, adaptation=adaptation)

        def wrong_readout(visual):
            joined = torch.cat([visual, e], 1)
            reference = self.head_ref(joined)
            if self.detach_wrong_reference:
                reference = reference.detach()
            scores = self.head(joined)-reference
            return dict(bounded_log_probabilities(log_anchor, scores, result["b"], alpha_max=.5, beta_max=.5), scores=scores)

        result.update(wrong_readout(result["adapted_h"]))
        if auxiliary:
            result["aux_logq"] = torch.stack([wrong_readout(h+result["delta"][:, j])["logq"] for j in range(3)], 1)
        return result


class ICAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(audit, "IC theory audit module is not implemented")
        torch.set_num_threads(1)

    def test_identity_has_nonzero_constructed_branch_jacobian(self):
        model, data = constructed_model(), batch(4, constructed=True)
        result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
        torch.testing.assert_close(result["logq"], data["log_anchor"], atol=1e-12, rtol=0)
        jac = audit.scalar_branch_jacobian(model, data, 1, "aux")
        self.assertEqual(jac["jacobian"].shape, (4, 516))
        for name in ("down.weight", "up.weight"):
            parameter = dict(model.adapters[0].named_parameters())[name]
            direction = torch.zeros_like(parameter)
            direction[0, 0] = 1
            check = audit.audit_directional_derivative(model, data, 1, name, direction)
            self.assertTrue(check["passed"])
            self.assertGreater(float(check["analytic"].norm()), 1e-8)
            self.assertEqual(check["step"], 1e-6)

    def test_wrong_reference_and_detach_negative_fixture(self):
        data = batch(4, constructed=True)
        wrong = constructed_model(WrongReferenceModel)
        for p in wrong.head.parameters():
            p.requires_grad_(False)
        direction = torch.zeros_like(wrong.adapters[0].up.weight)
        direction[0, 0] = 1
        correct_zero = audit.audit_directional_derivative(wrong, data, 1, "up.weight", direction)
        self.assertTrue(correct_zero["passed"])
        self.assertEqual(float(correct_zero["analytic"].norm()), 0.)
        fake = constructed_model(WrongReferenceModel)
        fake.detach_wrong_reference = True
        for p in fake.head.parameters():
            p.requires_grad_(False)
        bad = audit.audit_directional_derivative(fake, data, 1, "up.weight", direction)
        self.assertFalse(bad["passed"])
        self.assertGreater(bad["max_abs_error"], 1e-7)
        self.assertEqual(float(bad["finite_difference"].norm()), 0.)

    def test_zero_u_and_zero_w2_have_distinct_blocking_effects(self):
        model, data = constructed_model(), batch(4, constructed=True)
        with torch.no_grad():
            model.adapters[0].up.weight.zero_()
            model.adapter_refs[0].up.weight.zero_()
        model.initial_reference_sha256 = _digest(model.named_function_reference())
        jac = audit.scalar_branch_jacobian(model, data, 1, "aux")["jacobian"]
        self.assertEqual(float(jac[:, :260].norm()), 0.)
        self.assertGreater(float(jac[:, 260:].norm()), 1e-8)
        model = constructed_model()
        with torch.no_grad():
            model.head[2].weight.zero_()
            model.head_ref[2].weight.zero_()
        model.initial_reference_sha256 = _digest(model.named_function_reference())
        jac = audit.scalar_branch_jacobian(model, data, 1, "aux")["jacobian"]
        self.assertEqual(float(jac.norm()), 0.)

    def test_single_row_bound_violation_is_not_averaged_away(self):
        model, data = ICRoleModel("C-S-R", seed=20261002), batch(20)
        with torch.no_grad():
            model.adapters[0].up.weight.add_(.5)
        result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
        self.assertEqual(audit.audit_bounds(model, data, result)["violation_count"], 0)
        broken = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in result.items()}
        broken["delta"][3, 0].fill_(2)
        check = audit.audit_bounds(model, data, broken)
        self.assertEqual(check["violation_count"], 1)
        self.assertEqual(check["violating_image_ids"], ["synthetic-3"])

    def test_gram_congruence_and_duplicate_input_counterexample(self):
        model, data = constructed_model(), batch(4, constructed=True)
        check = audit.audit_initial_kernel(model, data, 1)
        self.assertTrue(check["passed"])
        self.assertLessEqual(check["max_jacobian_residual"], 1e-10)
        self.assertLessEqual(check["max_gram_residual"], 1e-10)
        self.assertEqual(check["aux_numeric_rank"], 1)
        self.assertFalse(check["aux_numeric_positive_definite"])
        self.assertIsNone(check["conditional_lambda_lower_bound"])
        torch.testing.assert_close(check["G"].diag(), torch.full((4,), .25, dtype=torch.float64))

    def test_same_weight_off_is_not_retrained_control(self):
        model, data = ICRoleModel("C-S-R", seed=20261002), batch(20)
        with torch.no_grad():
            model.head[2].bias[0] += .2
            model.adapters[0].up.weight.add_(.3)
        result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
        check = audit.audit_bounds(model, data, result)
        self.assertEqual(check["violation_count"], 0)
        self.assertGreater(check["off_anchor_max_log_difference"], 1e-8)
        self.assertTrue(all(r["eta_aux"] >= r["eta_main"] for r in check["rows"]))
        for row in check["rows"]:
            self.assertAlmostEqual(row["eta_main"], row["L_F"]*row["b"]**2)
            self.assertAlmostEqual(row["eta_aux"], row["L_F"]*row["b"])

    def test_kernel_requires_initial_state_and_head_only_is_not_applicable(self):
        model, data = ICRoleModel("C-S-R", seed=20261002), batch(4)
        with torch.no_grad():
            model.head[2].bias[0] += .01
        with self.assertRaisesRegex(ValueError, "(?i)initial"):
            audit.audit_initial_kernel(model, data, 1)
        head = ICRoleModel("HC-S", seed=20261002)
        self.assertEqual(audit.audit_initial_kernel(head, data, 1)["status"], "NOT_APPLICABLE")

    def test_finite_difference_restores_parameters_after_error(self):
        model, data = constructed_model(), batch(4, constructed=True)
        before = copy.deepcopy(model.state_dict())
        original = model.forward
        count = 0

        def interrupted(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError("Synthetic directional interruption")
            return original(*args, **kwargs)

        with patch.object(model, "forward", side_effect=interrupted):
            with self.assertRaises(ValueError):
                audit.audit_directional_derivative(model, data, 1, "up.weight", torch.ones_like(model.adapters[0].up.weight))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(reference_identity(model)["references_match_initial"])

    def test_5000_synthetic_bounds_and_zero_budget_function(self):
        model, data = ICRoleModel("C-S-R", seed=20261002), batch(5000)
        data["log_anchor"][0] = torch.log_softmax(torch.tensor([0., -50., -55., -60.], dtype=torch.float64), 0)
        data["log_anchor"][1] = torch.log_softmax(torch.tensor([0., 0., -1., -2.], dtype=torch.float64), 0)
        data["h"][2] = data["h"][3]
        data["e"][2] = data["e"][3]
        data["log_anchor"][2] = data["log_anchor"][3]
        with torch.no_grad():
            model.adapters[0].up.weight.add_(.3)
            model.head[2].weight.add_(.2)
        result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
        check = audit.audit_bounds(model, data, result)
        self.assertEqual(check["row_count"], 5000)
        self.assertEqual(check["violation_count"], 0)
        self.assertEqual(check["auxiliary_row_count"], 15000)
        q = bounded_log_probabilities(data["log_anchor"][:4], torch.ones(4, 4, dtype=torch.float64),
            torch.ones(4, 1, dtype=torch.float64), alpha_max=0., beta_max=0.)
        torch.testing.assert_close(q["logq"], data["log_anchor"][:4], atol=1e-12, rtol=0)

    def test_aux_global_bound_and_definition_cannot_hide_in_large_eta(self):
        model, data = ICRoleModel("C-S-R", seed=20261002), batch(20)
        data["log_anchor"].fill_(-torch.log(torch.tensor(4., dtype=torch.float64)))
        with torch.no_grad():
            model.head[2].weight.mul_(100)
        result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
        self.assertEqual(audit.audit_bounds(model, data, result)["violation_count"], 0)
        broken = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in result.items()}
        bad = torch.log_softmax(torch.tensor([9., 0., 0., 0.], dtype=torch.float64), 0)
        broken["aux_logq"][7] = bad
        check = audit.audit_bounds(model, data, broken)
        self.assertEqual(check["violation_count"], 1)
        self.assertEqual(check["violating_image_ids"], ["synthetic-7"])
        self.assertIn("aux_nll_drift", check["rows"][7]["violations"])
        self.assertIn("aux_definition", check["rows"][7]["violations"])

    def test_wrong_reference_value_definition_is_detected(self):
        model, data = constructed_model(WrongReferenceModel), batch(4, constructed=True)
        with torch.no_grad():
            model.adapters[0].up.weight.add_(.2)
        result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
        check = audit.audit_bounds(model, data, result)
        self.assertEqual(check["violation_count"], 4)
        self.assertIn("main_score_definition", check["rows"][0]["violations"])

    def test_all_units_have_main_and_auxiliary_certificates(self):
        data = batch(16)
        for unit in UNITS:
            model = ICRoleModel(unit, seed=20261002)
            with torch.no_grad():
                for _, parameter in model.named_current():
                    parameter.add_(.01)
            result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
            self.assertEqual(audit.audit_bounds(model, data, result)["violation_count"], 0, unit)

    def test_nonzero_jacobian_does_not_guarantee_loss_gradient(self):
        model, data = constructed_model(), batch(4, constructed=True)
        jac = audit.scalar_branch_jacobian(model, data, 1, "aux")["jacobian"]
        self.assertGreater(float(jac.norm()), 1e-8)
        result = model(data["h"], data["e"], data["log_anchor"], auxiliary=True)
        loss = objective(model, result, data["labels"])
        parameters = tuple(p for _, p in model.named_current())
        gradients = torch.autograd.grad(loss["data"], parameters)
        self.assertLessEqual(float(torch.cat([g.flatten() for g in gradients]).norm()), 1e-12)


if __name__ == "__main__":
    unittest.main()
