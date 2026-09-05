"""Reproducible numerical checks, not empirical evidence of model improvement."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hrgv_network import apply_pairwise_log_odds_correction


def verify_properties(seed: int = 20260905, samples: int = 10000) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    direct = torch.randn(samples, 2, generator=generator, dtype=dtype) * 5
    mapped = torch.randn(samples, 2, generator=generator, dtype=dtype) * 5
    # Include exact ties and both signs independently of the random draw.
    direct = torch.cat((direct, torch.tensor([[0., 0.], [2., -2.]], dtype=dtype)))
    mapped = torch.cat((mapped, torch.tensor([[0., 0.], [2., -2.]], dtype=dtype)))
    gate = torch.sigmoid(torch.randn(direct.shape, generator=generator, dtype=dtype) * 3)
    gap = direct - mapped
    oracle = (gap >= 0).to(dtype)
    fused = gate * direct + (1 - gate) * mapped
    regret = torch.maximum(direct, mapped) - fused
    identity = (gate - oracle).abs() * gap.abs()
    excess = F.softplus(-fused) - F.softplus(-torch.maximum(direct, mapped))
    temperature, weight_temperature = .2, .5
    target = torch.sigmoid(gap / temperature)
    kl = (torch.special.xlogy(target, target) + torch.special.xlogy(1-target, 1-target)
          - target * gate.log() - (1-target) * (1-gate).log()).clamp_min(0)
    soft_error_bound = torch.exp(-gap.abs() / temperature)
    gate_bound = gap.abs() * ((kl / 2).sqrt() + soft_error_bound)
    weight = torch.tanh(gap.abs() / weight_temperature)
    safe_weight = torch.where(weight > 0, weight, torch.ones_like(weight))
    weighted_kl = (weight * kl).mean()
    weighted_bound = ((gap.square() / safe_weight).mean() * weighted_kl / 2).sqrt() + temperature / math.e

    probabilities = torch.softmax(torch.randn(samples, 4, generator=generator, dtype=dtype), 1)
    base = probabilities[:, [0]].log() - probabilities[:, [1, 3]].log()
    delta = torch.randn(samples, 2, generator=generator, dtype=dtype) * 3
    correction = apply_pairwise_log_odds_correction(probabilities, base + delta, 0, 1, 3, torch)
    shift = correction["logit_adjustments"]
    incidence = torch.tensor([[1., -1., 0.], [1., 0., -1.]], dtype=dtype)
    independent = delta @ torch.linalg.pinv(incidence).T
    norm_identity = 2 / 3 * (delta[:, 0].square() - delta[:, 0]*delta[:, 1] + delta[:, 1].square())
    loss_change = (-F.log_softmax(probabilities.log() + shift, 1) + probabilities.log()).abs()
    loss_bound = math.sqrt(2) * delta.norm(dim=1)
    same_sign = direct * mapped >= 0

    checks = {
        "regret_identity": bool(torch.allclose(regret, identity, atol=1e-10, rtol=0)),
        "binary_logistic_excess": bool(((excess >= -1e-10) & (excess <= regret + 1e-10)).all()),
        "soft_oracle_bound": bool(((target - oracle).abs() <= soft_error_bound + 1e-10).all()),
        "gate_kl_regret_bound": bool((regret <= gate_bound + 1e-10).all()),
        "weighted_mean_regret_bound": bool(regret.mean() <= weighted_bound + 1e-10),
        "same_sign_preservation": bool(((fused*direct >= 0) & (fused*mapped >= 0))[same_sign].all()),
        "minimum_norm_pseudoinverse": bool(torch.allclose(shift[:, [0, 1, 3]], independent, atol=1e-10, rtol=0)),
        "margin_constraints": bool(torch.allclose(shift[:, [0]] - shift[:, [1, 3]], delta, atol=1e-10, rtol=0)),
        "norm_identity": bool(torch.allclose(shift.square().sum(1), norm_identity, atol=1e-10, rtol=0)),
        "four_class_nll_bound": bool((loss_change <= loss_bound[:, None] + 1e-10).all()),
    }
    uniform = torch.full((1, 4), .25, dtype=dtype)
    example = apply_pairwise_log_odds_correction(uniform, torch.tensor([[3., 3.]], dtype=dtype), 0, 1, 3, torch)
    result = {
        "evidence_type": "synthetic_numerical_verification",
        "seed": seed, "random_samples": samples, "added_tie_cases": 2,
        "dtype": "float64", "temperature": temperature, "weight_temperature": weight_temperature,
        "checks": checks, "all_checks_passed": all(checks.values()),
        "max_identity_error": float((regret - identity).abs().max()),
        "max_projection_error": float((shift[:, [0, 1, 3]] - independent).abs().max()),
        "mean_regret": float(regret.mean()), "weighted_kl_mean_bound": float(weighted_bound),
        "counterexample": {
            "true_class": "gangue_negative", "base_probabilities": uniform.tolist()[0],
            "requested_target_pair_margins": [3., 3.],
            "corrected_probabilities": example["corrected_role_probabilities"].tolist()[0],
            "gangue_logit_change": float(example["logit_adjustments"][0, 2]),
            "gangue_nll_change": float(-example["corrected_role_probabilities"][0, 2].log() + uniform[0, 2].log()),
        },
        "claim_boundary": "Numerical checks supplement analytic proofs; no accuracy, generalization, or novelty claim.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    result = verify_properties(args.seed, args.samples)
    result["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name("hrgv_network.py"))
    }
    result["torch_version"] = torch.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
