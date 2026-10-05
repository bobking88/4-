import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import anchor_linear_controls as controls
    import run_anchor_linear_controls as runner
except ModuleNotFoundError:
    controls = runner = None


class AnchorLinearControlTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(controls, "anchor control implementation missing")
        self.assertIsNotNone(runner, "anchor control runner missing")
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.x = torch.randn(24, 3, dtype=torch.float64)
        self.anchor = torch.randn(24, 4, dtype=torch.float64)
        self.y = torch.tensor([0, 1, 2, 3] * 6)

    def test_zero_residual_preserves_same_strong_anchor(self):
        theta = torch.zeros(4, 4, dtype=torch.float64)
        q = controls.probabilities(self.x, self.anchor, theta)
        torch.testing.assert_close(q, self.anchor.softmax(1), rtol=0, atol=0)

    def test_analytic_objective_gradient_hessian_match_autograd(self):
        theta = torch.randn(4, 4, dtype=torch.float64)*.03
        objective = controls.RidgeObjective(self.x, self.anchor, self.y, .001)
        value, gradient = objective.value_gradient(theta.reshape(-1).numpy())
        variable = theta.clone().requires_grad_(True)
        augmented = torch.cat([self.x, torch.ones(24,1,dtype=self.x.dtype)], 1)
        loss = F.cross_entropy(self.anchor+augmented@variable,self.y)+.001/2*variable.square().sum()
        auto_gradient, = torch.autograd.grad(loss, variable, create_graph=True)
        auto_hessian = torch.stack([torch.autograd.grad(v,variable,retain_graph=True)[0].reshape(-1) for v in auto_gradient.reshape(-1)])
        self.assertAlmostEqual(value, float(loss.detach()), places=13)
        np.testing.assert_allclose(gradient, auto_gradient.detach().reshape(-1).numpy(), atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(objective.hessian(theta.reshape(-1).numpy()), auto_hessian.numpy(), atol=1e-13, rtol=1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(objective.hessian(theta.reshape(-1).numpy())).min(), .001-1e-13)

    def test_convex_fit_certifies_stationarity_and_identity_fit_bound(self):
        fitted = controls.fit(self.x, self.anchor, self.y, .001, "zero", runner.SOLVER)
        self.assertTrue(fitted["converged"])
        self.assertLessEqual(fitted["gradient_l2"], 1e-7)
        self.assertLessEqual(fitted["gap_upper_bound"], 1e-8)
        self.assertLessEqual(fitted["fit_nll"], float(F.cross_entropy(self.anchor,self.y))+1e-8)

    def test_restarts_are_unique_solution_checks_and_replay_exact(self):
        a = controls.fit(self.x, self.anchor, self.y, .001, "zero", runner.SOLVER)
        b = controls.fit(self.x, self.anchor, self.y, .001, "zero", runner.SOLVER)
        self.assertTrue(torch.equal(a["theta"], b["theta"]))
        c = controls.fit(self.x, self.anchor, self.y, .001, "perturbed_20261003", runner.SOLVER)
        difference = float((a["theta"]-c["theta"]).norm())
        self.assertLessEqual(difference,(a["gradient_l2"]+c["gradient_l2"])/.001+1e-10)
        torch.testing.assert_close(controls.probabilities(self.x,self.anchor,a["theta"]),controls.probabilities(self.x,self.anchor,c["theta"]),atol=1e-6,rtol=0)

    def test_feature_statistics_fit_only_and_capacity_not_padded(self):
        evidence = torch.randn(24,18,dtype=torch.float64)
        p = self.anchor.softmax(1)
        L = torch.zeros(24,1,dtype=torch.float64)
        h = torch.randn(24,64,dtype=torch.float64)
        state = controls.fit_representation(evidence,p,L)
        before = copy.deepcopy(state)
        blocks = controls.representations(evidence,p,L,h,state)
        other = controls.representations(evidence+100,p,L,h,state)
        self.assertEqual({k:v.shape[1] for k,v in blocks.items()},{"P":4,"E":23,"H":64})
        self.assertGreater(float(other["E"][:,:18].abs().max()), 10)
        for key in state:
            self.assertTrue(torch.equal(state[key], before[key]))
        self.assertEqual(controls.make_input(blocks,"EH",20271002).shape,(24,87))
        shuffled = controls.make_input(blocks,"EH_permuted",20271002)
        self.assertTrue(torch.equal(shuffled[:,:23],blocks["E"]))
        self.assertFalse(torch.equal(shuffled[:,23:],h))
        self.assertTrue(torch.equal(shuffled,controls.make_input(blocks,"EH_permuted",20271002)))

    def test_invalid_inputs_rejected_before_solver(self):
        for bad in (self.x.float(),self.x*float("nan")):
            with self.assertRaises(ValueError):
                controls.fit(bad,self.anchor,self.y,.001,"zero",runner.SOLVER)
        with self.assertRaises(ValueError):
            controls.fit(self.x,self.anchor,torch.full_like(self.y,4),.001,"zero",runner.SOLVER)
        with self.assertRaises(ValueError):
            controls.fit(self.x,self.anchor,self.y,0,"zero",runner.SOLVER)
        with self.assertRaises(ValueError):
            controls.fit(self.x,self.anchor,self.y,.001,"unknown",runner.SOLVER)

    def test_registered_protocol_rejects_scope_and_solver_tuning(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root/"docs/experiment_protocols/abmp_anchor_linear_controls_v1.json").read_text(encoding="utf-8"))
        runner.validate_protocol(protocol)
        for key,value in (("fold",1),("arms",["role_network"]),("regularization",.0001),("primary_initialization","perturbed_20261003")):
            with self.assertRaisesRegex(ValueError,"scope|registered"):
                runner.validate_protocol(dict(protocol,**{key:value}))
        bad = copy.deepcopy(protocol)
        bad["solver"]["lbfgs_maxiter"] = 20
        with self.assertRaisesRegex(ValueError,"registered"):
            runner.validate_protocol(bad)

    def test_overwrite_refused_before_source_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                runner.run_experiment(root,root/"missing.json",root)

    def test_screen_requires_strong_probability_and_role_controls(self):
        metric = {"nll":.80}
        transition = {"new_correct_targets":2,"net_correct_targets":1,"false_target_count_by_class":{"1":14,"2":0,"3":7}}
        base = {"false_target_count_by_class":{"1":13,"2":0,"3":6}}
        result = runner.screen(metric,{"T0":{"nll":.82},"P":{"nll":.82},"E":{"nll":.82},"EH_permuted":{"nll":.82}},transition,base)
        self.assertTrue(result["role_usable_signal"])
        transition["false_target_count_by_class"]["3"] = 8
        self.assertFalse(runner.screen(metric,{"T0":{"nll":.82},"P":{"nll":.82},"E":{"nll":.82},"EH_permuted":{"nll":.82}},transition,base)["role_usable_signal"])

    def test_summary_replay_rejects_primary_restart_and_screen_tampering(self):
        self.assertTrue(hasattr(runner, "check_summary_fields"), "aggregate verification missing")
        transition = {"new_correct_targets":2,"net_correct_targets":1,"false_target_count_by_class":{"1":14,"3":7}}
        baseline = {"metrics":{"stop":{"nll":.82}},"transitions":{"stop":{"false_target_count_by_class":{"1":13,"3":6}}}}
        primary = {arm:{"metrics":{"stop":{"nll":.80 if arm == "EH" else .82}},"transitions":{"stop":transition}} for arm in controls.ARMS}
        restarts = {arm:[{"passed":True},{"passed":True}] for arm in controls.ARMS}
        screening = runner.screen(primary["EH"]["metrics"]["stop"], {"T0":baseline["metrics"]["stop"],**{arm:primary[arm]["metrics"]["stop"] for arm in ("P","E","EH_permuted")}}, transition, baseline["transitions"]["stop"])
        summary = {"baseline":baseline,"primary":primary,"restart_checks":restarts,"screening":screening,"numerical_stability_passed":True}
        runner.check_summary_fields(summary,baseline,primary,restarts)
        for key,value in (("baseline",{}),("primary",{}),("restart_checks",{}),("screening",{}),("numerical_stability_passed",False)):
            with self.assertRaisesRegex(ValueError,"summary"):
                runner.check_summary_fields(dict(summary,**{key:value}),baseline,primary,restarts)

    def test_prediction_replay_rejects_changed_probability(self):
        subset = {"records":[{"image_id":str(i),"relative_path":f"{i}.jpg","split_group_id":str(i),"mineral_label":"test"} for i in range(24)],"labels":self.y}
        q = self.anchor.softmax(1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"predictions.csv"
            runner.write_predictions(path,subset,{"probabilities":q})
            self.assertEqual(runner.check_prediction(path,subset,q),24)
            changed = q.clone()
            changed[0,0] += 1e-8
            with self.assertRaisesRegex(ValueError,"mismatch"):
                runner.check_prediction(path,subset,changed)


if __name__ == "__main__":
    unittest.main()
