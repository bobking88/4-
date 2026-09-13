"""Exact finite-support counterexample to NLL consistency of RSG soft supervision."""
import hashlib
import json
from pathlib import Path

import torch

from hrgv_network import regret_gate_targets, weighted_soft_gate_loss


def verify():
    dtype = torch.float64
    truth = torch.tensor([.4, .4, .1, .1], dtype=dtype)
    direct = torch.tensor([.72, .08, .1, .1], dtype=dtype)
    mapped = torch.tensor([.48, .32, .1, .1], dtype=dtype)
    targets = regret_gate_targets(direct.repeat(4, 1), mapped.repeat(4, 1), torch.arange(4), .2, .5, torch)
    effective_weights = truth[:, None] * targets['gate_gap_weight']
    optimum = (effective_weights * targets['soft_oracle_gate']).sum() / effective_weights.sum()
    gate = optimum.clone().requires_grad_(True)
    bce = weighted_soft_gate_loss(gate.expand(4, 1), targets['soft_oracle_gate'], effective_weights, torch)
    bce_gradient = torch.autograd.grad(bce, gate)[0]

    def nll(g):
        probabilities = g[..., None]*direct+(1-g[..., None])*mapped
        return -(truth*probabilities.log()).sum(-1)

    nll_gradient = torch.autograd.grad(nll(gate), gate)[0]
    grid = torch.linspace(0, 1, 10001, dtype=dtype)
    risks = nll(grid)
    checks = {
        'bce_stationarity': abs(float(bce_gradient)) < 1e-12,
        'bce_optimum_interior': 0 < float(optimum) < 1,
        'nll_derivative_positive_at_bce_optimum': float(nll_gradient) > 0,
        'nll_grid_minimum_at_zero': int(risks.argmin()) == 0,
        'strict_positive_excess_nll': float(nll(optimum)-nll(torch.tensor(0., dtype=dtype))) > .001,
    }
    return {
        'evidence_type': 'finite_support_mathematical_counterexample_not_dataset_experiment',
        'truth': truth.tolist(), 'direct': direct.tolist(), 'mapped': mapped.tolist(),
        'temperature': .2, 'gap_temperature': .5,
        'weighted_bce_optimal_gate': float(optimum), 'fusion_nll_optimal_gate': 0.,
        'nll_at_bce_optimum_nats': float(nll(optimum)),
        'nll_at_fusion_optimum_nats': float(nll(torch.tensor(0., dtype=dtype))),
        'excess_nll_nats': float(nll(optimum)-nll(torch.tensor(0., dtype=dtype))),
        'bce_gradient': float(bce_gradient), 'nll_gradient': float(nll_gradient),
        'checks': checks, 'all_checks_passed': all(checks.values()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == '__main__':
    result = verify()
    path = Path(__file__).resolve().parents[1] / 'outputs/theory/rsg_population_counterexample.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['all_checks_passed']:
        raise SystemExit(1)
