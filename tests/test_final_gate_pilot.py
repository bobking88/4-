import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from run_frozen_expert_gate_pilot import final_gate_nll


class FinalGateLossTests(unittest.TestCase):
    def test_neutral_verifier_equals_mixture_nll(self):
        a = torch.tensor([[.7, .1, .1, .1]], dtype=torch.float64)
        b = torch.tensor([[.1, .3, .3, .3]], dtype=torch.float64)
        g = torch.tensor([[.4]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0])
        loss = final_gate_nll(g, a, b, y, torch.ones_like(g), torch.ones_like(g))
        self.assertTrue(torch.allclose(loss, -(g*.7+(1-g)*.1).log().mean()))

    def test_active_verifier_has_final_output_gradient(self):
        a = torch.tensor([[.7, .1, .1, .1]], dtype=torch.float64)
        b = torch.tensor([[.1, .3, .3, .3]], dtype=torch.float64)
        z = torch.tensor([[.2]], dtype=torch.float64, requires_grad=True)
        g = z.sigmoid()
        qloss = final_gate_nll(g, a, b, torch.tensor([0]), torch.zeros_like(g), torch.zeros_like(g))
        c = torch.tensor([[torch.exp(torch.tensor(-2.)).item(), 1, 1, 1]], dtype=torch.float64)
        za, zb = (c*a).sum(), (c*b).sum()
        expected = g*za/(g*za+(1-g)*zb)-g*.7/(g*.7+(1-g)*.1)
        self.assertTrue(torch.allclose(torch.autograd.grad(qloss, z)[0], expected, atol=1e-8))


if __name__ == '__main__':
    unittest.main()
