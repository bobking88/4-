"""Formula checks and explicit counterexamples on synthetic inputs only."""
import math
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from msb_synthetic_helpers import make_prepared_fixture
from msb_role_model import MSBRoleModel
from msb_role_controls import make_convex_control
try:
    import msb_role_audit as aa
except ModuleNotFoundError:
    aa=None


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.batch=make_prepared_fixture(24)["batch"]
        cls.report=aa.audit_synthetic_theory(MSBRoleModel("MAIN",seed=20261002),cls.batch) if aa else None

    def setUp(self):
        self.assertIsNotNone(aa,"MSB theoretical audits missing")
        self.model=MSBRoleModel("MAIN",seed=20261002)

    def forward(self,model=None):
        b=self.batch
        return (model or self.model)(b["h"],b["e"],b["log_anchor"])

    def test_all_class_interior_witnesses(self):
        self.assertEqual(self.report["interior_witness_count"],500)
        self.assertGreaterEqual(self.report["min_witness_winner_margin"],.125-1e-10)
        self.assertGreaterEqual(self.report["min_witness_negative_margin"],.25-1e-10)
        self.assertAlmostEqual(self.report["witness_relative_energy"],.25,places=12)
        self.assertAlmostEqual(self.report["witness_sp_squared_distance_max"],3*math.atanh(.5)**2,places=12)
        self.assertFalse(self.report["joint_fit_guaranteed"])

    def test_pair_boundary_bias_derivative(self):
        self.assertGreaterEqual(self.report["min_boundary_bias_derivative"],.1875-1e-10)
        self.assertGreaterEqual(self.report["min_boundary_pair_loss_derivative"],.09375-1e-10)
        self.assertLessEqual(self.report["max_finite_difference_error"],1e-5)
        self.assertTrue(self.report["pair_boundary_can_be_nonwinning"])

    def test_wrong_pair_initial_gradient_bound(self):
        self.assertGreaterEqual(self.report["min_wrong_pair_initial_bias_gradient"],.125-1e-10)
        self.assertFalse(self.report["global_convergence_proved"])

    def test_root_odds_recover_correction_and_branch_cross_zero(self):
        with torch.no_grad(): self.model.branches[0].zeta.add_(.2)
        r=self.forward()
        check=aa.audit_forward_contract(self.batch,r)
        self.assertLessEqual(check["max_root_odds_error"],1e-10)
        margins=r["logq"][:,:1]-r["logq"][:,1:]
        biases=[b.zeta for b in self.model.branches]
        for j in range(3):
            gradients=torch.autograd.grad(margins[:,j].sum(),biases,retain_graph=True)
            for k,g in enumerate(gradients):
                if j!=k: self.assertLessEqual(float(g.abs()),1e-10)

    def test_common_normalizer_changes_other_probabilities(self):
        before=self.forward()["logq"].detach()
        with torch.no_grad(): self.model.branches[0].zeta.add_(.2)
        after=self.forward()["logq"].detach()
        self.assertTrue(bool(((after-before).abs().max(0).values>1e-8).all()))

    def test_single_input_oracle_not_joint_fit(self):
        self.assertEqual(self.report["conflicting_input_correct_count"],1)
        self.assertFalse(self.report["parameter_identifiability_proved"])

    def test_correct_high_confidence_can_be_harmed(self):
        self.assertEqual(self.report["correct_anchor_label"],0)
        self.assertNotEqual(self.report["harmful_witness_label"],0)
        self.assertFalse(self.report["all_confident_correct_protected"])

    def test_mixed_hessian_not_claim_mlp_superiority(self):
        self.assertGreater(self.report["mixed_hessian_norm"],1e-8)
        self.assertLessEqual(self.report["mixed_hessian_analytic_error"],1e-10)
        self.assertFalse(self.report["non_equivalence_to_general_mlp_proved"])

    def test_data_regularizer_gradient_decomposition(self):
        with torch.no_grad(): self.model.branches[0].zeta.add_(.2)
        r=self.forward()
        grads=aa.gradient_components(self.model,r,self.batch["labels"])
        self.assertEqual(grads["vectors"]["total"].numel(),1083)
        torch.testing.assert_close(grads["vectors"]["data"],grads["vectors"]["nll"]+grads["vectors"]["pair_weighted"],atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(grads["vectors"]["total"],grads["vectors"]["data"]+grads["vectors"]["regularization"],atol=1e-10,rtol=1e-10)

    def test_underflow_and_tanh_saturation_explicit(self):
        with torch.no_grad(): self.model.branches[0].zeta.add_(20.)
        r=self.forward()
        self.assertGreater(r["saturation_count"],0)
        self.assertTrue(aa.audit_forward_contract(self.batch,r)["has_saturation"])
        p=torch.log_softmax(torch.tensor([[0.,-1000.,-1.,-1.]],dtype=torch.float64),1)
        with self.assertRaises(ValueError): self.model(torch.zeros(1,64,dtype=torch.float64),torch.zeros(1,23,dtype=torch.float64),p)
        p=torch.log_softmax(torch.tensor([[0.,-500.,-500.,-500.]],dtype=torch.float64),1)
        with self.assertRaises(ValueError): self.model(torch.zeros(1,64,dtype=torch.float64),torch.zeros(1,23,dtype=torch.float64),p)

    def test_bias_submodel_and_tampered_output_rejected(self):
        control=make_convex_control("BIAS3-CVX",self.batch)
        c=torch.tensor([.2,-.1,.3],dtype=torch.float64)
        control["coordinates"]=c
        with torch.no_grad():
            for j,b in enumerate(self.model.branches): b.zeta.copy_(c[j].atanh())
        result=aa.audit_control_equivalence(self.model,control,self.batch)
        self.assertLessEqual(result["max_logq_error"],1e-12)
        bad=dict(self.forward()); bad["d"]=bad["d"]+1
        with self.assertRaises(ValueError): aa.audit_forward_contract(self.batch,bad)

    def test_radius_has_declared_log_odds_units(self):
        p=torch.log_softmax(torch.tensor([[0.,-2.,3.,-4.]],dtype=torch.float64),1)
        b=dict(h=torch.zeros(1,64,dtype=torch.float64),e=torch.zeros(1,23,dtype=torch.float64),log_anchor=p)
        r=self.model(b["h"],b["e"],p)
        torch.testing.assert_close(r["R_ref"],torch.tensor([[4.25,6.25,8.25]],dtype=torch.float64),atol=1e-12,rtol=1e-12)
        aa.audit_forward_contract(b,r)

    def test_real_identity_rejected_by_synthetic_probe(self):
        bad=dict(self.batch)
        bad["records"]=[dict(row,image_id="not-synthetic") for row in bad["records"]]
        with self.assertRaises(ValueError): aa.audit_synthetic_theory(self.model,bad)
