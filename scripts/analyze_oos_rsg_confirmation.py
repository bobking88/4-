"""Analyze the preregistered three-expert OOS-RSG confirmation study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


EXPERT_SEEDS = (20260924, 20260925, 20260926)
CLASS_COUNT = 4
METRICS = (
    "final_nll_nats",
    "accuracy",
    "macro_f1",
    "target_recall",
    "ti_intrusion",
    "metal_intrusion",
)
METHOD_PATHS = {
    "equal": Path("equal_final_predictions.csv"),
    "original_joint_gate": Path("original_joint_gate_final_predictions.csv"),
    "seen": Path("seen/final_predictions.csv"),
    "unseen": Path("unseen/final_predictions.csv"),
}


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _probabilities(row: dict[str, str]) -> tuple[float, ...]:
    values = tuple(float(row[f"prob_{index}"]) for index in range(CLASS_COUNT))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Probabilities must be finite and non-negative.")
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError("Probability rows must sum to one.")
    return values


def calculate_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Metric rows must not be empty.")
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    nll = 0.0
    for row in rows:
        true_id = int(row["true_class_id"])
        predicted_id = int(row["predicted_class_id"])
        if true_id not in range(CLASS_COUNT) or predicted_id not in range(CLASS_COUNT):
            raise ValueError("Class IDs must lie in [0, 3].")
        probabilities = _probabilities(row)
        confusion[true_id, predicted_id] += 1
        nll -= math.log(max(probabilities[true_id], np.finfo(float).eps))
    return _metrics_from_sufficient_statistics(confusion, nll, len(rows))


def _metrics_from_sufficient_statistics(
    confusion: np.ndarray, nll_sum: float, count: int
) -> dict[str, float]:
    if count <= 0 or confusion.shape != (CLASS_COUNT, CLASS_COUNT):
        raise ValueError("Invalid sufficient statistics.")
    true_counts = confusion.sum(axis=1).astype(float)
    predicted_counts = confusion.sum(axis=0).astype(float)
    true_positive = np.diag(confusion).astype(float)
    f1_denominator = true_counts + predicted_counts
    class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0,
    )
    if any(true_counts[index] <= 0 for index in (0, 1, 3)):
        raise ValueError("Target and both hard-negative classes must be represented.")
    return {
        "final_nll_nats": float(nll_sum / count),
        "accuracy": float(true_positive.sum() / count),
        "macro_f1": float(class_f1.mean()),
        "target_recall": float(confusion[0, 0] / true_counts[0]),
        "ti_intrusion": float(confusion[1, 0] / true_counts[1]),
        "metal_intrusion": float(confusion[3, 0] / true_counts[3]),
    }


def align_prediction_rows(
    baseline_rows: list[dict[str, str]], comparison_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    baseline = {row["image_id"]: row for row in baseline_rows}
    comparison = {row["image_id"]: row for row in comparison_rows}
    if len(baseline) != len(baseline_rows) or len(comparison) != len(comparison_rows):
        raise ValueError("Prediction files contain duplicate image IDs.")
    if set(baseline) != set(comparison):
        raise ValueError("Prediction image ID sets do not match.")
    aligned = []
    for image_id in sorted(baseline):
        first = baseline[image_id]
        second = comparison[image_id]
        identity_fields = (
            "split_group_id",
            "mineral_label",
            "true_class_id",
        )
        if any(first[field] != second[field] for field in identity_fields):
            raise ValueError(f"Prediction identity mismatch for {image_id}.")
        aligned.append(
            {
                "image_id": image_id,
                "split_group_id": first["split_group_id"],
                "true_class_id": int(first["true_class_id"]),
                "baseline_predicted_id": int(first["predicted_class_id"]),
                "comparison_predicted_id": int(second["predicted_class_id"]),
                "baseline_probabilities": _probabilities(first),
                "comparison_probabilities": _probabilities(second),
            }
        )
    return aligned


def _cluster_statistics(rows: list[dict[str, object]]):
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        group_id = str(row["split_group_id"])
        if not group_id:
            raise ValueError("Every prediction must have a group ID.")
        grouped[group_id].append(row)
    strata: dict[int, list[tuple[np.ndarray, float, np.ndarray, float, int]]] = defaultdict(list)
    for group_id, members in grouped.items():
        true_ids = {int(row["true_class_id"]) for row in members}
        if len(true_ids) != 1:
            raise ValueError(f"Group {group_id} crosses true-class strata.")
        baseline_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        comparison_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        baseline_nll = 0.0
        comparison_nll = 0.0
        for row in members:
            true_id = int(row["true_class_id"])
            baseline_confusion[true_id, int(row["baseline_predicted_id"])] += 1
            comparison_confusion[true_id, int(row["comparison_predicted_id"])] += 1
            baseline_nll -= math.log(
                max(row["baseline_probabilities"][true_id], np.finfo(float).eps)
            )
            comparison_nll -= math.log(
                max(row["comparison_probabilities"][true_id], np.finfo(float).eps)
            )
        strata[next(iter(true_ids))].append(
            (
                baseline_confusion,
                baseline_nll,
                comparison_confusion,
                comparison_nll,
                len(members),
            )
        )
    return strata


def _sample_metrics(strata, rng: np.random.Generator):
    baseline_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    comparison_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    baseline_nll = comparison_nll = 0.0
    count = 0
    for true_id in sorted(strata):
        clusters = strata[true_id]
        draws = rng.multinomial(
            len(clusters), np.full(len(clusters), 1.0 / len(clusters))
        )
        for multiplicity, cluster in zip(draws, clusters, strict=True):
            if multiplicity == 0:
                continue
            b_conf, b_nll, c_conf, c_nll, cluster_count = cluster
            baseline_confusion += multiplicity * b_conf
            comparison_confusion += multiplicity * c_conf
            baseline_nll += multiplicity * b_nll
            comparison_nll += multiplicity * c_nll
            count += multiplicity * cluster_count
    return (
        _metrics_from_sufficient_statistics(baseline_confusion, baseline_nll, count),
        _metrics_from_sufficient_statistics(comparison_confusion, comparison_nll, count),
    )


def paired_two_stage_bootstrap(
    pairs_by_seed: dict[int, list[dict[str, object]]],
    replicates: int = 10000,
    rng_seed: int = 20260923,
) -> dict[str, object]:
    if set(pairs_by_seed) != set(EXPERT_SEEDS):
        raise ValueError("Exactly the three preregistered expert seeds are required.")
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive.")
    per_seed: dict[str, object] = {}
    for seed, rows in pairs_by_seed.items():
        baseline_rows = []
        comparison_rows = []
        for row in rows:
            common = {
                "image_id": row["image_id"],
                "split_group_id": row["split_group_id"],
                "mineral_label": "",
                "true_class_id": str(row["true_class_id"]),
            }
            baseline_rows.append(
                {
                    **common,
                    "predicted_class_id": str(row["baseline_predicted_id"]),
                    **{
                        f"prob_{index}": str(value)
                        for index, value in enumerate(row["baseline_probabilities"])
                    },
                }
            )
            comparison_rows.append(
                {
                    **common,
                    "predicted_class_id": str(row["comparison_predicted_id"]),
                    **{
                        f"prob_{index}": str(value)
                        for index, value in enumerate(row["comparison_probabilities"])
                    },
                }
            )
        baseline_metrics = calculate_metrics(baseline_rows)
        comparison_metrics = calculate_metrics(comparison_rows)
        per_seed[str(seed)] = {
            metric: {
                "baseline": baseline_metrics[metric],
                "comparison": comparison_metrics[metric],
                "difference": comparison_metrics[metric] - baseline_metrics[metric],
            }
            for metric in METRICS
        }

    tables = {seed: _cluster_statistics(rows) for seed, rows in pairs_by_seed.items()}
    rng = np.random.default_rng(rng_seed)
    bootstrap_values = {metric: np.empty(replicates, dtype=float) for metric in METRICS}
    seed_values = np.asarray(EXPERT_SEEDS)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        differences = {metric: [] for metric in METRICS}
        for seed in sampled_seeds:
            baseline, comparison = _sample_metrics(tables[int(seed)], rng)
            for metric in METRICS:
                differences[metric].append(comparison[metric] - baseline[metric])
        for metric in METRICS:
            bootstrap_values[metric][replicate] = mean(differences[metric])

    summary = {}
    for metric in METRICS:
        differences = [per_seed[str(seed)][metric]["difference"] for seed in EXPERT_SEEDS]
        values = bootstrap_values[metric]
        summary[metric] = {
            "difference": mean(differences),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "probability_less_than_zero": float(np.mean(values < 0.0)),
            "per_seed_differences": differences,
        }
    return {"per_seed": per_seed, "summary": summary}


def assess_promotion(unseen_minus_equal: dict[str, object]) -> dict[str, object]:
    criteria = {
        "nll_ci_below_zero": unseen_minus_equal["final_nll_nats"]["ci_high"] < 0.0,
        "nll_favorable_in_at_least_two_seeds": unseen_minus_equal[
            "nll_favorable_seed_count"
        ]
        >= 2,
        "macro_f1_guardrail": unseen_minus_equal["macro_f1"]["difference"] >= -0.005,
        "target_recall_guardrail": unseen_minus_equal["target_recall"]["difference"]
        >= -0.01,
        "ti_intrusion_guardrail": unseen_minus_equal["ti_intrusion"]["difference"] <= 0.01,
        "metal_intrusion_guardrail": unseen_minus_equal["metal_intrusion"]["difference"]
        <= 0.01,
    }
    return {
        "criteria": criteria,
        "promote_to_main_method": all(criteria.values()),
        "rule": "all preregistered NLL and harm-guardrail criteria must pass",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV.")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze locked OOS-RSG confirmation.")
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260923)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite an existing analysis directory.")
    args.output_dir.mkdir(parents=True)

    all_rows = {
        seed: {
            method: read_prediction_rows(
                args.gate_root / f"seed{seed}" / relative_path
            )
            for method, relative_path in METHOD_PATHS.items()
        }
        for seed in EXPERT_SEEDS
    }
    comparisons = {}
    comparison_specs = {
        "unseen_minus_seen": ("seen", "unseen"),
        "unseen_minus_equal": ("equal", "unseen"),
        "unseen_minus_original_joint_gate": ("original_joint_gate", "unseen"),
    }
    for name, (baseline, comparison) in comparison_specs.items():
        pairs = {
            seed: align_prediction_rows(
                all_rows[seed][baseline], all_rows[seed][comparison]
            )
            for seed in EXPERT_SEEDS
        }
        comparisons[name] = paired_two_stage_bootstrap(
            pairs, args.bootstrap_replicates, args.rng_seed
        )

    unseen_equal = comparisons["unseen_minus_equal"]["summary"]
    unseen_equal["nll_favorable_seed_count"] = sum(
        value < 0.0
        for value in unseen_equal["final_nll_nats"]["per_seed_differences"]
    )
    decision = assess_promotion(unseen_equal)
    analysis = {
        "scope": "locked final-eval analysis after expert and gate selection",
        "expert_seeds": list(EXPERT_SEEDS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "rng_seed": args.rng_seed,
        "comparisons": comparisons,
        "promotion_decision": decision,
        "evidence_boundary": (
            "Public specimen images only; no grade, recovery, industrial belt, or "
            "external-site claim."
        ),
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    effect_rows = []
    for comparison_name, result in comparisons.items():
        for metric, values in result["summary"].items():
            if metric == "nll_favorable_seed_count":
                continue
            effect_rows.append(
                {"comparison": comparison_name, "metric": metric, **values}
            )
    _write_csv(args.output_dir / "paired_effects.csv", effect_rows)

    per_seed_rows = []
    for seed in EXPERT_SEEDS:
        for method, rows in all_rows[seed].items():
            per_seed_rows.append(
                {"expert_seed": seed, "method": method, **calculate_metrics(rows)}
            )
    _write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
