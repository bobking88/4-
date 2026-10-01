"""Diagnose routing capacity without opening any additional evaluation fold."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from abmp_rsg import apply_abmp_projection
from run_abmp_rsg_development import ANCHORS
from run_tc_oos_rsg_experiments import calculate_probability_metrics, _file_sha256


ROOT = Path(__file__).resolve().parents[1]


def diagnose_geometry(q0, qphi, labels, raw_route, epsilon, config):
    q0_ties = (q0 == q0.max(1, keepdim=True).values).sum(1) > 1
    candidate_ties = (qphi == qphi.max(1, keepdim=True).values).sum(1) > 1
    target0 = q0.argmax(1) == 0
    target_phi = qphi.argmax(1) == 0
    disagreement = q0.argmax(1) != qphi.argmax(1)
    true_target = labels == 0
    true_target_count = int(true_target.sum())
    vulnerable = target0 & ~target_phi
    initial_margin = q0[:, 0] - q0[:, 1:].max(1).values
    geometry = []
    for cap in (.02, .04, .08):
        for tau_p, tau_m, delta in ANCHORS:
            options = dict(tau_p=tau_p, tau_m=tau_m, delta=delta)
            maximum_route = torch.ones_like(raw_route)
            full = apply_abmp_projection(q0, qphi, maximum_route, cap, **options)
            post = apply_abmp_projection(q0, qphi, maximum_route, cap, mode="posterior", **options)
            independent = full["projected_route"] < post["projected_route"] - 1e-7
            anchors = full["anchor_mask"].flatten()
            geometry.append({
                "epsilon_max": cap, **options,
                "anchor_count": int(anchors.sum()),
                "vulnerable_anchor_count": int((anchors & vulnerable).sum()),
                "independent_margin_capacity_count": int(independent.sum()),
                "independent_margin_true_target_count": int((independent.flatten() & (labels == 0)).sum()),
                "independent_margin_false_positive_count": int((independent.flatten() & (labels != 0)).sum()),
            })
    options = {key: config[key] for key in ("tau_p", "tau_m", "delta")}
    adaptive = apply_abmp_projection(q0, qphi, raw_route, epsilon, **options)
    fixed = apply_abmp_projection(q0, qphi, raw_route, float(epsilon.mean()), **options)
    difference = (adaptive["final_probabilities"] - fixed["final_probabilities"]).abs()
    adaptive_metrics = calculate_probability_metrics(adaptive["final_probabilities"], labels)
    fixed_metrics = calculate_probability_metrics(fixed["final_probabilities"], labels)
    return {
        "purpose": "posthoc_development_geometry_not_confirmation",
        "rows": len(q0),
        "q0_argmax_tie_count": int(q0_ties.sum()),
        "candidate_argmax_tie_count": int(candidate_ties.sum()),
        "unique_endpoint_argmax_assumption_verified": not bool((q0_ties | candidate_ties).any()),
        "q0_candidate_argmax_agreement": float((q0.argmax(1) == qphi.argmax(1)).double().mean()),
        "expert_argmax_disagreement_count": int(disagreement.sum()),
        "true_target_count": true_target_count,
        "true_target_disagreement_count": int((true_target & disagreement).sum()),
        "true_target_recall_gain_capacity_count": int((true_target & ~target0 & disagreement).sum()),
        "target_recall_absolute_change_bound": int((true_target & disagreement).sum()) / true_target_count if true_target_count else None,
        "target_recall_gain_bound": int((true_target & ~target0 & disagreement).sum()) / true_target_count if true_target_count else None,
        "accuracy_absolute_change_bound": float(disagreement.double().mean()),
        "q0_target_count": int(target0.sum()),
        "q0_target_to_candidate_nontarget_count": int(vulnerable.sum()),
        "vulnerable_true_target_count": int((vulnerable & (labels == 0)).sum()),
        "vulnerable_false_positive_count": int((vulnerable & (labels != 0)).sum()),
        "vulnerable_margin_min": float(initial_margin[vulnerable].min()) if bool(vulnerable.any()) else None,
        "vulnerable_margin_max": float(initial_margin[vulnerable].max()) if bool(vulnerable.any()) else None,
        "capacity_by_config": geometry,
        "fixed_mean_budget": float(epsilon.mean()),
        "fixed_budget_comparison_scope": "Same route weights; development mean budget. Not separately trained or selected.",
        "fixed_mean_budget_max_probability_difference": float(difference.max()),
        "fixed_mean_budget_mean_probability_difference": float(difference.mean()),
        "fixed_mean_budget_prediction_disagreement_count": int((adaptive["final_probabilities"].argmax(1) != fixed["final_probabilities"].argmax(1)).sum()),
        "adaptive_metrics": adaptive_metrics,
        "fixed_mean_budget_metrics": fixed_metrics,
        "adaptive_minus_fixed_nll": adaptive_metrics["nll"] - fixed_metrics["nll"],
    }


def _read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_directory(source: Path):
    paths = {name: source / "predictions" / f"{name}.csv" for name in ("A0", "A1", "A6_routing")}
    rows = {name: _read_rows(path) for name, path in paths.items()}
    identities = [(r["image_id"], r["split_group_id"]) for r in rows["A0"]]
    for name in ("A1", "A6_routing"):
        if [(r["image_id"], r["split_group_id"]) for r in rows[name]] != identities:
            raise ValueError(f"Prediction alignment mismatch: {name}")
    if [r["true_class_id"] for r in rows["A0"]] != [r["true_class_id"] for r in rows["A1"]]:
        raise ValueError("Class labels differ between experts.")
    if [r["true_class_id"] for r in rows["A0"]] != [r["class_id"] for r in rows["A6_routing"]]:
        raise ValueError("Routing diagnostics have different labels.")
    posterior = lambda name: torch.tensor([[float(r[f"prob_{k}"]) for k in range(4)] for r in rows[name]], dtype=torch.float64)
    route = torch.tensor([[float(r["raw_route"])] for r in rows["A6_routing"]], dtype=torch.float64)
    eps = torch.tensor([[float(r["epsilon"])] for r in rows["A6_routing"]], dtype=torch.float64)
    labels = torch.tensor([int(r["true_class_id"]) for r in rows["A0"]])
    summary_path = source / "development_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = diagnose_geometry(posterior("A0"), posterior("A1"), labels, route, eps, summary["diagnostic_config"])
    result["source_protocol"] = summary["protocol"]
    result["source_status"] = summary["status"]
    result["candidates_with_independent_margin_activation"] = sum(c["audit"]["independent_margin_activation_count"] > 0 for c in summary["candidates"])
    result["source_sha256"] = {**{name: _file_sha256(path) for name, path in paths.items()}, "summary": _file_sha256(summary_path)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/training/abmp_rsg_v2/development_fold_0_r2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/theory/abmp_rsg_development_geometry.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze_directory(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
