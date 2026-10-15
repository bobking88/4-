"""Synthetic identity, reference and ordinary-head equivalence tests."""
import copy
import io
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from msb_synthetic_helpers import make_prepared_fixture
try:
    import msb_role_model as mm
except ModuleNotFoundError:
    mm = None

UNITS = ("MAIN", "SHARED-BIL", "CONCAT-MLP", "ADD", "FIX1", "UNB-SCALE", "NOAUX", "LIN-UNB")


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(mm, "MSB centered model missing")
        self.batch = make_prepared_fixture(24)["batch"]

    def forward(self, model, **kw):
        b = self.batch
        return model(b["h"], b["e"], b["log_anchor"], **kw)

    def test_initial_identity_all_neural_units(self):
        for unit in UNITS:
            result = self.forward(mm.MSBRoleModel(unit, seed=20261002))
            torch.testing.assert_close(result["logq"], self.batch["log_anchor"], atol=1e-12, rtol=1e-12)
            self.assertTrue(torch.equal(result["d"],torch.zeros(24,3,dtype=torch.float64)))

    def test_counts_and_seed_streams_exact(self):
        for unit, count in (("MAIN",1083),("SHARED-BIL",1107),("CONCAT-MLP",1095),("LIN-UNB",264)):
            model = mm.MSBRoleModel(unit,seed=20261002)
            self.assertEqual(sum(p.numel() for _,p in model.named_current()),count)
            self.assertEqual(sum(b.numel() for _,b in model.named_buffers()),count)
        model = mm.MSBRoleModel("MAIN",seed=20261002)
        generator = torch.Generator().manual_seed(20261202)
        expected = (2*torch.rand(4,64,generator=generator,dtype=torch.float64)-1)/8
        self.assertTrue(torch.equal(model.branches[0].U,expected))

    def test_reference_no_alias_after_update_and_reload(self):
        model = mm.MSBRoleModel("MAIN",seed=20261002)
        before = mm.reference_digest(model)
        for name,p in model.named_current():
            self.assertNotEqual(p.data_ptr(),model.initial_parameter(name).data_ptr())
        loss = -self.forward(model)["logq"][:,0].mean()
        loss.backward()
        with torch.no_grad():
            for _,p in model.named_current(): p.add_(p.grad,alpha=-.01)
        self.assertEqual(mm.reference_digest(model),before)
        memory = io.BytesIO(); torch.save(model.state_dict(),memory); memory.seek(0)
        other = mm.MSBRoleModel("MAIN",seed=20261003)
        other.load_state_dict(torch.load(memory,weights_only=True))
        self.assertEqual(mm.reference_digest(other),before)
        torch.testing.assert_close(self.forward(model)["logq"],self.forward(other)["logq"])

    def test_matched_ablation_initial_bytes(self):
        main = mm.MSBRoleModel("MAIN",seed=20261002)
        for unit in ("ADD","FIX1","UNB-SCALE","NOAUX"):
            other = mm.MSBRoleModel(unit,seed=20261002)
            for (name,p),(oname,q) in zip(main.named_current(),other.named_current()):
                self.assertEqual(name,oname); self.assertTrue(torch.equal(p,q))
            self.assertEqual(mm.reference_digest(main),mm.reference_digest(other))

    def test_shared_block_embedding_contains_main(self):
        main=mm.MSBRoleModel("MAIN",seed=20261002)
        shared=mm.MSBRoleModel("SHARED-BIL",seed=20261002)
        self.assertTrue(shared.head.A.requires_grad)
        with torch.no_grad():
            main.branches[0].U.add_(.04); main.branches[1].a.add_(.1)
            main.branches[2].zeta.add_(.2)
            shared.head.A.zero_()
            for j,branch in enumerate(main.branches):
                sl=slice(4*j,4*j+4)
                for name in ("U","u","V","v"):
                    getattr(shared.head,name)[sl].copy_(getattr(branch,name))
                shared.head.A[j,sl].copy_(branch.a)
                shared.head.zeta[j].copy_(branch.zeta)
        torch.testing.assert_close(self.forward(main)["logq"],self.forward(shared)["logq"],atol=1e-12,rtol=1e-12)

    def test_shared_bias_is_not_scaled(self):
        model=mm.MSBRoleModel("SHARED-BIL",seed=20261002)
        with torch.no_grad(): model.head.zeta.copy_(torch.tensor([.2,.4,.6],dtype=torch.float64))
        score=model.head.score(torch.zeros(24,64,dtype=torch.float64),torch.zeros(24,23,dtype=torch.float64))
        expected=torch.tensor([.2,.4,.6],dtype=torch.float64).expand(24,3)
        torch.testing.assert_close(score,expected,atol=1e-12,rtol=1e-12)

    def test_flat_decoder_rewrite_equivalent(self):
        model=mm.MSBRoleModel("MAIN",seed=20261002)
        with torch.no_grad(): model.branches[0].zeta.add_(.35)
        result=self.forward(model)
        logits=self.batch["log_anchor"]+torch.cat([torch.zeros(24,1,dtype=torch.float64),-result["d"]],1)
        flat=torch.log_softmax(logits,1)
        torch.testing.assert_close(result["logq"],flat,atol=1e-12,rtol=1e-12)
        params=[p for _,p in model.named_current()]
        g1=torch.autograd.grad(-result["logq"][:,0].sum(),params,retain_graph=True)
        g2=torch.autograd.grad(-flat[:,0].sum(),params)
        for a,b in zip(g1,g2): torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_all_roles_off_returns_anchor(self):
        model=mm.MSBRoleModel("MAIN",seed=20261002)
        with torch.no_grad():
            for branch in model.branches: branch.zeta.add_(.3)
        result=self.forward(model,role_mask=(False,False,False))
        torch.testing.assert_close(result["logq"],self.batch["log_anchor"],atol=1e-12,rtol=1e-12)
        with self.assertRaises(ValueError): self.forward(mm.MSBRoleModel("SHARED-BIL",seed=20261002),role_mask=(False,True,True))

    def test_invalid_inputs_and_unknown_units_rejected(self):
        with self.assertRaises(ValueError): mm.MSBRoleModel("unknown",seed=20261002)
        model=mm.MSBRoleModel("MAIN",seed=20261002)
        with self.assertRaises(ValueError): model(self.batch["h"].float(),self.batch["e"],self.batch["log_anchor"])
        bad=self.batch["log_anchor"].clone(); bad[0,0]=float("nan")
        with self.assertRaises(ValueError): model(self.batch["h"],self.batch["e"],bad)
