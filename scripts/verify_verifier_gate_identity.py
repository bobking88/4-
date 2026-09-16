"""Audit final-output gate derivatives against the existing residual verifier."""
import json
from pathlib import Path

import torch

from hrgv_network import apply_residual_target_verifiers


def main():
    torch.manual_seed(20260916)
    n = 4096
    a = torch.softmax(torch.randn(n, 4, dtype=torch.float64), 1)
    b = torch.softmax(torch.randn(n, 4, dtype=torch.float64), 1)
    ti = torch.rand(n, 1, dtype=torch.float64)
    metal = torch.rand(n, 1, dtype=torch.float64)
    z = torch.randn(n, 1, dtype=torch.float64, requires_grad=True)
    g = torch.sigmoid(z)
    c = torch.ones_like(a)
    c[:, :1] = torch.exp(-torch.relu(1-2*ti)-torch.relu(1-2*metal))
    za, zb = (c*a).sum(1, keepdim=True), (c*b).sum(1, keepdim=True)
    ga = g*za/(g*za+(1-g)*zb)
    mixture = g*a+(1-g)*b
    final = apply_residual_target_verifiers(mixture, ti, metal)
    reparameterized = ga*(c*a/za)+(1-ga)*(c*b/zb)
    labels = torch.arange(n) % 4
    ay = a.gather(1, labels[:, None])
    my = mixture.gather(1, labels[:, None])
    rho = g*ay/my
    final_loss = -final.gather(1, labels[:, None]).log().sum()
    pre_loss = -my.log().sum()
    final_grad = torch.autograd.grad(final_loss, z, retain_graph=True)[0]
    pre_grad = torch.autograd.grad(pre_loss, z)[0]
    residuals = {
        'posterior_identity': float((final-reparameterized).abs().max().detach()),
        'logit_shift_identity': float((ga-torch.sigmoid(z+za.log()-zb.log())).abs().max().detach()),
        'final_gradient_identity': float((final_grad-(ga-rho)).abs().max().detach()),
        'pre_gradient_identity': float((pre_grad-(g-rho)).abs().max().detach()),
        'gradient_difference_identity': float((final_grad-pre_grad-(ga-g)).abs().max().detach()),
    }
    tolerance = 1e-12
    assert all(value < tolerance for value in residuals.values()), residuals
    neutral = apply_residual_target_verifiers(mixture.detach(), torch.ones_like(ti), torch.ones_like(metal))
    neutral_error = float((neutral-mixture.detach()).abs().max())
    assert neutral_error < tolerance
    result = {
        'scope': 'synthetic algebra and autograd check; not classification improvement',
        'seed': 20260916, 'samples': n, 'dtype': 'float64',
        'assumptions': 'fixed experts and verifier outputs with respect to gate logit; no extra calibration; inactive probability clamps',
        'max_absolute_residuals': residuals,
        'neutral_verifier_identity_error': neutral_error,
        'max_pre_post_gradient_difference': float((final_grad-pre_grad).abs().max()),
        'strict_gradient_sign_reversals': int(((final_grad*pre_grad)<-1e-12).sum()),
        'tolerance': tolerance, 'all_checks_passed': True,
    }
    output = Path(__file__).resolve().parents[1]/'outputs/theory/verifier_gate_identity.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
