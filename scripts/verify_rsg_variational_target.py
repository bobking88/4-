"""Check the RSG target's variational interpretation using synthetic probabilities."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hrgv_network import regret_gate_targets


def verify():
    generator = torch.Generator().manual_seed(20260912)
    direct = torch.softmax(torch.rand(3000, 4, generator=generator, dtype=torch.float64) * 2, 1)
    mapped = torch.softmax(torch.rand(3000, 4, generator=generator, dtype=torch.float64) * 2, 1)
    mapped[0] = direct[0]
    labels = torch.arange(3000) % 4
    index = labels[:, None]
    ld, lm = -direct.gather(1, index).log(), -mapped.gather(1, index).log()
    gate = .01 + .98 * torch.rand(3000, 1, generator=generator, dtype=torch.float64)
    records = []
    for temperature in (.2, .5, 1., 2.):
        result = regret_gate_targets(direct, mapped, labels, temperature, .5, torch)
        target = result['soft_oracle_gate']

        def negative_entropy(t):
            return torch.special.xlogy(t, t) + torch.special.xlogy(1-t, 1-t)

        def objective(t):
            return t * ld + (1-t) * lm + temperature * negative_entropy(t)

        kl_gate_target = negative_entropy(gate) - gate * target.log() - (1-gate) * (1-target).log()
        kl_target_gate = negative_entropy(target) - target * gate.log() - (1-target) * (1-gate).log()
        stationarity = ld - lm + temperature * (target.log() - (1-target).log())
        excess = objective(gate) - objective(target)
        bce_excess = F.binary_cross_entropy(gate, target, reduction='none') + negative_entropy(target)
        # Independent closed-form optimal value and endpoint checks.
        optimum = -temperature * torch.logsumexp(torch.cat((-ld, -lm), 1) / temperature, 1, keepdim=True)
        errors = {
            'stationarity': float(stationarity.abs().max()),
            'objective_kl_identity': float((excess - temperature * kl_gate_target).abs().max()),
            'bce_reverse_kl_identity': float((bce_excess - kl_target_gate).abs().max()),
            'optimal_value': float((objective(target) - optimum).abs().max()),
        }
        checks = {
            'identities': all(v < 1e-10 for v in errors.values()),
            'sampled_gate_not_better': bool((excess >= -1e-10).all()),
            'endpoints_not_better': bool((objective(target) <= torch.minimum(ld, lm) + 1e-10).all()),
            'strict_convexity': bool((temperature / (target * (1-target)) > 0).all()),
            'tie_has_zero_supervision_weight': bool(target[0] == .5 and result['gate_gap_weight'][0] == 0),
            'kl_directions_not_equal': bool((kl_gate_target - kl_target_gate).abs().max() > .01),
        }
        records.append({'temperature': temperature, 'max_absolute_errors': errors, 'checks': checks})
    root = Path(__file__).resolve().parents[1]
    return {
        'evidence_type': 'synthetic_numerical_verification_not_performance',
        'seed': 20260912, 'samples_per_temperature': 3000, 'dtype': 'float64',
        'scope': 'bounded positive probabilities; includes an exact expert tie; no dataset access',
        'records': records,
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), root / 'scripts/hrgv_network.py')},
        'all_checks_passed': all(all(r['checks'].values()) for r in records),
    }


if __name__ == '__main__':
    result = verify()
    destination = Path(__file__).resolve().parents[1] / 'outputs/theory/rsg_variational_target.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['all_checks_passed']:
        raise SystemExit(1)
