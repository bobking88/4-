"""Numerically verify structural geometry, not mineral classification efficacy."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from audit_abmp_candidate_capacity import build_component_branches, target_segment_interval
from run_tc_oos_rsg_experiments import _write_json
from tc_oos_rsg import _validate_probability_matrix


def minimum_target_preserving_scale(probabilities, margin=.01):
    """Minimum positive target scale for a normalised margin, capped at 2 if infeasible."""
    _validate_probability_matrix(probabilities, "probabilities")
    if not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError("margin must lie in (0, 1).")
    q = probabilities.double()
    q = q / q.sum(1, keepdim=True)
    target = q[:, :1]
    numerator = q[:, 1:].max(1, keepdim=True).values + margin*(1-target)
    denominator = (1-margin)*target
    ratio = numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    scale = torch.where(denominator > 0, ratio, torch.full_like(ratio, 2.0)).clamp(max=2.0)
    return {"minimum_scale": scale, "feasible": scale <= 1.0}


def verify_geometry_properties(count=25000, seed=20261002):
    if count < 1:
        raise ValueError("count must be positive.")
    generator = torch.Generator().manual_seed(seed)
    direct = torch.softmax(torch.randn(count, 4, generator=generator, dtype=torch.float64), 1)
    mapped = torch.softmax(torch.randn(count, 4, generator=generator, dtype=torch.float64), 1)
    random_column = lambda: torch.rand(count, 1, generator=generator, dtype=torch.float64)
    cache = {"direct": direct, "mapped": mapped, "original_gate": random_column(), "ti": random_column(), "metal": random_column()}
    branches = build_component_branches(cache, random_column(), {})
    effective = branches["effective_candidate_gate"]
    replay = effective*branches["direct_verified"]+(1-effective)*branches["mapped_verified"]
    full = target_segment_interval(branches["direct_verified"], branches["mapped_verified"])
    restricted = target_segment_interval(branches["oos_verified"], branches["fixed_verified"])
    target_promotions = sum(int(((branches[f"{name}_verified"].argmax(1) == 0) & (branches[f"{name}_pre"].argmax(1) != 0)).sum()) for name in ("direct", "mapped", "fixed", "oos", "original"))
    minimum = minimum_target_preserving_scale(direct)
    mask = minimum["feasible"].flatten()
    scale = torch.maximum(random_column()[mask], minimum["minimum_scale"][mask])
    scaled = direct[mask].clone()
    scaled[:, :1] *= scale
    scaled /= scaled.sum(1, keepdim=True)
    actual_margin = scaled[:, :1]-scaled[:, 1:]
    return {"purpose": "structural_formula_check_not_classifier_performance", "dtype": "float64",
            "sample_count": count, "seed": seed, "tolerance": 1e-12,
            "commutation_max_abs_residual": float((replay-branches["oos_verified"]).abs().max()),
            "nested_segment_violations": int((restricted["feasible"] & ~full["feasible"]).sum()),
            "target_set_promotions": target_promotions, "scale_margin_feasible_count": int(mask.sum()),
            "scale_margin_violations": int((actual_margin < .01-1e-12).any(1).sum()),
            "minimum_observed_protected_margin": float(actual_margin.min()) if bool(mask.any()) else None,
            "interpretation": "Random checks supplement algebraic proofs; they do not prove population recall or novelty."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/theory/abmp_candidate_capacity_v1/geometry_properties.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    result = verify_geometry_properties(args.count, args.seed)
    _write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
