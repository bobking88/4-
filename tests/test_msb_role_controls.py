"""Synthetic losses and fully solved convex controls."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from msb_synthetic_helpers import make_prepared_fixture
from msb_role_model import MSBRoleModel
try:
    import msb_role_controls as cc
except ModuleNotFoundError:
    cc=None


class ControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(cc,"MSB objective and convex controls missing")
        self.batch=make_prepared_fixture(24)["batch"]

    def result(self,model):
        b=self.batch
        return model(b["h"],b["e"],b["log_anchor"])

    def test_pair_uses_main_odds_and_original_denominators(self):
        model=MSBRoleModel("MAIN",seed=20261002)
        h=torch.zeros(6,64,dtype=torch.float64); e=torch.zeros(6,23,dtype=torch.float64)
        p=torch.log_softmax(torch.tensor([0.,-1.,-2.,-3.],dtype=torch.float64),0).expand(6,4)
        y=torch.tensor([0,0,1,1,2,3])
        losses=cc.objective(model,model(h,e,p),y)
        soft=lambda x:math.log1p(math.exp(x))
        expected=((2*soft(-1)+2*soft(1))/4+(2*soft(-2)+soft(2))/3+(2*soft(-3)+soft(3))/3)/3
        self.assertAlmostEqual(float(losses["pair_raw"].detach()),expected,places=12)
        self.assertAlmostEqual(float(losses["pair_weighted"].detach()),.1*expected,places=12)
        torch.testing.assert_close(losses["total"],losses["data"]+losses["regularization"])
        noaux=MSBRoleModel("NOAUX",seed=20261002)
        self.assertEqual(float(cc.objective(noaux,noaux(h,e,p),y)["pair_weighted"].detach()),0)

    def test_fix1_regularizer_uses_main_reference_radius(self):
        model=MSBRoleModel("FIX1",seed=20261002)
        with torch.no_grad(): model.branches[0].zeta.add_(.5)
        result=self.result(model)
        losses=cc.objective(model,result,self.batch["labels"])
        expected=.01*(result["d"]/result["R_ref"]).square().mean()
        torch.testing.assert_close(losses["relative_penalty"],expected)
        self.assertAlmostEqual(float(losses["sp"].detach()),.001*.25/2,places=12)
        self.assertNotAlmostEqual(float(expected.detach()),.01*math.tanh(.5)**2/3,places=7)

    def test_bias_objective_is_main_submodel(self):
        control=cc.make_convex_control("BIAS3-CVX",self.batch)
        c=torch.tensor([.2,-.1,.3],dtype=torch.float64)
        model=MSBRoleModel("MAIN",seed=20261002)
        with torch.no_grad():
            for j,branch in enumerate(model.branches): branch.zeta.copy_(c[j].atanh())
        expected=cc.objective(model,self.result(model),self.batch["labels"])["total"]
        actual=cc.convex_objective(control,c,self.batch)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)

    def test_bias_hessian_lower_bound_and_endpoint_rejection(self):
        control=cc.make_convex_control("BIAS3-CVX",self.batch)
        c=torch.tensor([.2,-.1,.3],dtype=torch.float64)
        H=torch.autograd.functional.hessian(lambda x:cc.convex_objective(control,x,self.batch),c)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(H).min()),.001-1e-10)
        for value in (1.,-1.,float("nan")):
            bad=c.clone(); bad[0]=value
            with self.assertRaises(ValueError): cc.convex_objective(control,bad,self.batch)

    def test_lin_classwise_comp_positive_ridge(self):
        for unit,dim in (("LIN-UNB",264),("CLASSWISE4",3),("REPAIR4-COMP",7)):
            with self.subTest(unit=unit):
                control=cc.make_convex_control(unit,self.batch)
                x=torch.zeros(dim,dtype=torch.float64)
                H=torch.autograd.functional.hessian(lambda z:cc.convex_objective(control,z,self.batch),x,vectorize=True)
                self.assertGreaterEqual(float(torch.linalg.eigvalsh(H).min()),.001-1e-10)
                self.assertEqual(control["coordinates"].numel(),dim)

    def test_repair_features_sign_ties_counts_and_degeneracy(self):
        p=torch.full((1,4),-math.log(4),dtype=torch.float64)
        counts=torch.tensor([2.,3.,4.,5.],dtype=torch.float64)
        phi=cc.competition_features(p,counts)
        expected=torch.tensor([0.,1.,math.log(2/3),0.],dtype=torch.float64)
        torch.testing.assert_close(phi[0,0,1],expected,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(phi[:,:,:,0],phi[:,:,:,3],atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.equal(phi[0,1,0],-phi[0,0,1]))

    def test_unsolved_control_fails_certificate(self):
        control=cc.make_convex_control("CLASSWISE4",self.batch)
        with patch.object(cc,"MAX_ITERATIONS",0):
            with self.assertRaisesRegex(RuntimeError,"certificate"):
                cc.solve_convex(control,self.batch,deadline_check=lambda:None)

    def test_actual_convex_solution_certificate_and_readout(self):
        for unit in ("LIN-UNB","BIAS3-CVX","CLASSWISE4","REPAIR4-COMP"):
            control=cc.make_convex_control(unit,self.batch)
            initial=float(cc.convex_objective(control,control["coordinates"],self.batch))
            calls=[]
            solved=cc.solve_convex(control,self.batch,deadline_check=lambda:calls.append(1))
            self.assertTrue(solved["converged"])
            self.assertLessEqual(solved["gap_upper"],1e-8)
            self.assertLessEqual(solved["objective"],initial+1e-12)
            self.assertGreater(len(calls),0)
            x=solved["coordinates"].clone().requires_grad_()
            g=torch.autograd.grad(cc.convex_objective(control,x,self.batch),x)[0]
            self.assertLessEqual(float(g.square().sum()/(2*.001)),1e-8)
            loaded=solved["model_state"]
            q=cc.control_forward(loaded,self.batch["h"],self.batch["e"],self.batch["log_anchor"])["logq"]
            torch.testing.assert_close(q.exp().sum(1),torch.ones(24,dtype=torch.float64))

    def test_known_quadratic_and_deadline_interrupt(self):
        control=cc.make_convex_control("CLASSWISE4",self.batch)
        target=torch.tensor([.2,-.3,.4],dtype=torch.float64)
        with patch.object(cc,"convex_objective",side_effect=lambda ctl,x,b:.001*(x-target).square().sum()/2):
            solved=cc.solve_convex(control,self.batch,deadline_check=lambda:None)
        torch.testing.assert_close(solved["coordinates"],target,atol=1e-12,rtol=1e-12)
        def interrupted(): raise RuntimeError("TIMED_OUT")
        with self.assertRaisesRegex(RuntimeError,"TIMED_OUT"):
            cc.solve_convex(control,self.batch,deadline_check=interrupted)

    def test_certificate_checked_after_final_allowed_update(self):
        control=cc.make_convex_control("CLASSWISE4",self.batch)
        target=torch.tensor([.2,-.3,.4],dtype=torch.float64)
        with patch.object(cc,"MAX_ITERATIONS",1), patch.object(cc,"convex_objective",side_effect=lambda ctl,x,b:.001*(x-target).square().sum()/2):
            solved=cc.solve_convex(control,self.batch,deadline_check=lambda:None)
        self.assertTrue(solved["converged"])
        self.assertEqual(solved["iterations"],1)
        torch.testing.assert_close(solved["coordinates"],target,atol=1e-12,rtol=1e-12)
