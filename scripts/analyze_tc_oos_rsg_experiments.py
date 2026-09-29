"""Analyze grouped outer-fold predictions for TC-OOS-RSG."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CLASS_COUNT = 4
TARGET_CLASS_ID = 0
TI_CLASS_ID = 1
METALLIC_CLASS_ID = 3
BOOTSTRAP_METRICS = (
    "nll",
    "accuracy",
    "macro_f1",
    "target_recall",
    "ti_intrusion_to_target",
    "metallic_intrusion_to_target",
)


def _probabilities(row: Mapping[str, object]) -> tuple[float, ...]:
    values = tuple(float(row[f"prob_{index}"]) for index in range(CLASS_COUNT))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Probabilities must be finite and non-negative.")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError("Probability rows must sum to one.")
    return values


def _metrics_from_statistics(
    confusion: np.ndarray, nll_sum: float, row_count: int
) -> dict[str, float]:
    if confusion.shape != (CLASS_COUNT, CLASS_COUNT) or row_count <= 0:
        raise ValueError("Invalid metric sufficient statistics.")
    true_counts = confusion.sum(axis=1).astype(float)
    predicted_counts = confusion.sum(axis=0).astype(float)
    true_positive = np.diag(confusion).astype(float)
    required = (TARGET_CLASS_ID, TI_CLASS_ID, METALLIC_CLASS_ID)
    if any(true_counts[class_id] <= 0 for class_id in required):
        raise ValueError("Target and both hard-negative roles must be represented.")
    f1_denominator = true_counts + predicted_counts
    class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0,
    )
    return {
        "nll": float(nll_sum / row_count),
        "accuracy": float(true_positive.sum() / row_count),
        "macro_f1": float(class_f1.mean()),
        "target_recall": float(
            confusion[TARGET_CLASS_ID, TARGET_CLASS_ID]
            / true_counts[TARGET_CLASS_ID]
        ),
        "ti_intrusion_to_target": float(
            confusion[TI_CLASS_ID, TARGET_CLASS_ID] / true_counts[TI_CLASS_ID]
        ),
        "metallic_intrusion_to_target": float(
            confusion[METALLIC_CLASS_ID, TARGET_CLASS_ID]
            / true_counts[METALLIC_CLASS_ID]
        ),
    }


def calculate_role_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Metric rows must not be empty.")
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    nll_sum = 0.0
    for row in rows:
        true_id = int(row["true_class_id"])
        predicted_id = int(row["predicted_class_id"])
        if true_id not in range(CLASS_COUNT) or predicted_id not in range(CLASS_COUNT):
            raise ValueError("Class IDs must lie in [0, 3].")
        probabilities = _probabilities(row)
        if predicted_id != int(np.argmax(probabilities)):
            raise ValueError("Predicted class must equal the posterior argmax.")
        confusion[true_id, predicted_id] += 1
        nll_sum -= math.log(max(probabilities[true_id], np.finfo(float).eps))
    return _metrics_from_statistics(confusion, nll_sum, len(rows))


def _align_prediction_rows(
    baseline_rows: Sequence[Mapping[str, object]],
    method_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = {str(row["image_id"]): row for row in baseline_rows}
    method = {str(row["image_id"]): row for row in method_rows}
    if len(baseline) != len(baseline_rows) or len(method) != len(method_rows):
        raise ValueError("Prediction files contain duplicate image IDs.")
    if set(baseline) != set(method):
        raise ValueError("Prediction image ID mismatch.")

    aligned = []
    for image_id in sorted(baseline):
        first = baseline[image_id]
        second = method[image_id]
        for field in ("split_group_id", "true_class_id", "outer_fold"):
            if str(first[field]) != str(second[field]):
                raise ValueError(f"Prediction {field} mismatch for {image_id}.")
        aligned.append(
            {
                "image_id": image_id,
                "split_group_id": str(first["split_group_id"]),
                "outer_fold": int(first["outer_fold"]),
                "true_class_id": int(first["true_class_id"]),
                "baseline_predicted_id": int(first["predicted_class_id"]),
                "method_predicted_id": int(second["predicted_class_id"]),
                "baseline_probabilities": _probabilities(first),
                "method_probabilities": _probabilities(second),
            }
        )
    return aligned


def _statistics_from_aligned(
    rows: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, float, np.ndarray, float, int]:
    baseline_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    method_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    baseline_nll = 0.0
    method_nll = 0.0
    for row in rows:
        true_id = int(row["true_class_id"])
        baseline_id = int(row["baseline_predicted_id"])
        method_id = int(row["method_predicted_id"])
        baseline_confusion[true_id, baseline_id] += 1
        method_confusion[true_id, method_id] += 1
        baseline_nll -= math.log(
            max(float(row["baseline_probabilities"][true_id]), np.finfo(float).eps)
        )
        method_nll -= math.log(
            max(float(row["method_probabilities"][true_id]), np.finfo(float).eps)
        )
    return (
        baseline_confusion,
        baseline_nll,
        method_confusion,
        method_nll,
        len(rows),
    )


def _cluster_statistics(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int], list[tuple[np.ndarray, float, np.ndarray, float, int]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        group_id = str(row["split_group_id"])
        if not group_id:
            raise ValueError("Every prediction must have a split_group_id.")
        grouped[group_id].append(row)

    strata: dict[
        tuple[int, int], list[tuple[np.ndarray, float, np.ndarray, float, int]]
    ] = defaultdict(list)
    for group_id, members in grouped.items():
        folds = {int(row["outer_fold"]) for row in members}
        classes = {int(row["true_class_id"]) for row in members}
        if len(folds) != 1 or len(classes) != 1:
            raise ValueError(f"Group {group_id} crosses a fold or class stratum.")
        strata[(next(iter(folds)), next(iter(classes)))].append(
            _statistics_from_aligned(members)
        )
    return strata


def _sample_statistics(
    strata: Mapping[
        tuple[int, int],
        Sequence[tuple[np.ndarray, float, np.ndarray, float, int]],
    ],
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    baseline_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    method_confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    baseline_nll = 0.0
    method_nll = 0.0
    row_count = 0
    for stratum in sorted(strata):
        clusters = strata[stratum]
        multiplicities = rng.multinomial(
            len(clusters), np.full(len(clusters), 1.0 / len(clusters))
        )
        for multiplicity, cluster in zip(multiplicities, clusters, strict=True):
            if multiplicity == 0:
                continue
            b_conf, b_nll, m_conf, m_nll, count = cluster
            baseline_confusion += multiplicity * b_conf
            method_confusion += multiplicity * m_conf
            baseline_nll += multiplicity * b_nll
            method_nll += multiplicity * m_nll
            row_count += multiplicity * count
    return (
        _metrics_from_statistics(baseline_confusion, baseline_nll, row_count),
        _metrics_from_statistics(method_confusion, method_nll, row_count),
    )


def cluster_paired_bootstrap(
    baseline_rows: Sequence[Mapping[str, object]],
    method_rows: Sequence[Mapping[str, object]],
    iterations: int = 10000,
    seed: int = 20260927,
) -> dict[str, object]:
    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive.")
    aligned = _align_prediction_rows(baseline_rows, method_rows)
    strata = _cluster_statistics(aligned)
    baseline_metrics = calculate_role_metrics(baseline_rows)
    method_metrics = calculate_role_metrics(method_rows)
    differences = {
        metric: method_metrics[metric] - baseline_metrics[metric]
        for metric in BOOTSTRAP_METRICS
    }

    fold_differences: dict[str, dict[str, float]] = {}
    for fold in sorted({int(row["outer_fold"]) for row in aligned}):
        fold_rows = [row for row in aligned if int(row["outer_fold"]) == fold]
        b_conf, b_nll, m_conf, m_nll, count = _statistics_from_aligned(fold_rows)
        baseline_fold = _metrics_from_statistics(b_conf, b_nll, count)
        method_fold = _metrics_from_statistics(m_conf, m_nll, count)
        fold_differences[str(fold)] = {
            metric: method_fold[metric] - baseline_fold[metric]
            for metric in BOOTSTRAP_METRICS
        }

    rng = np.random.default_rng(seed)
    replicate_values = {
        metric: np.empty(iterations, dtype=float) for metric in BOOTSTRAP_METRICS
    }
    for iteration in range(iterations):
        baseline_sample, method_sample = _sample_statistics(strata, rng)
        for metric in BOOTSTRAP_METRICS:
            replicate_values[metric][iteration] = (
                method_sample[metric] - baseline_sample[metric]
            )

    confidence_intervals = {}
    for metric, values in replicate_values.items():
        confidence_intervals[metric] = {
            "difference": differences[metric],
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
        if metric == "target_recall":
            confidence_intervals[metric]["one_sided_95_lower"] = float(
                np.quantile(values, 0.05)
            )

    return {
        "baseline_metrics": baseline_metrics,
        "method_metrics": method_metrics,
        "confidence_intervals": confidence_intervals,
        "fold_differences": fold_differences,
        "audit": {
            "sampling_unit": "split_group_id",
            "paired": True,
            "stratified_by": ["outer_fold", "true_class_id"],
            "group_count": sum(len(clusters) for clusters in strata.values()),
            "paired_image_count": len(aligned),
            "iterations": iterations,
            "seed": seed,
        },
    }


def verify_theory_invariants(
    rows: Sequence[Mapping[str, object]], tolerance: float = 1e-6
) -> dict[str, object]:
    if not rows:
        raise ValueError("Invariant rows must not be empty.")
    if tolerance < 0.0:
        raise ValueError("Tolerance must be non-negative.")
    counts = {
        "simplex_violation_count": 0,
        "target_safety_violation_count": 0,
        "decomposition_violation_count": 0,
        "convex_nll_bound_violation_count": 0,
        "nonfinite_or_negative_count": 0,
    }
    max_target_harm = -math.inf
    max_decomposition_residual = 0.0
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        q0 = tuple(float(value) for value in row["q0_probabilities"])
        qphi = tuple(float(value) for value in row["qphi_probabilities"])
        qtc = tuple(float(value) for value in row["qtc_probabilities"])
        if any(len(values) != CLASS_COUNT for values in (q0, qphi, qtc)):
            raise ValueError("Every posterior must contain four probabilities.")
        route = float(row["route"])
        epsilon_target = float(row["epsilon_target"])
        all_values = (*q0, *qphi, *qtc, route, epsilon_target)
        finite_nonnegative = all(math.isfinite(value) and value >= 0.0 for value in all_values)
        if not finite_nonnegative or route > 1.0 + tolerance:
            counts["nonfinite_or_negative_count"] += 1
            continue
        if any(abs(sum(values) - 1.0) > tolerance for values in (q0, qphi, qtc)):
            counts["simplex_violation_count"] += 1

        expected = tuple(
            (1.0 - route) * first + route * second
            for first, second in zip(q0, qphi)
        )
        residual = max(abs(observed - target) for observed, target in zip(qtc, expected))
        max_decomposition_residual = max(max_decomposition_residual, residual)
        if residual > tolerance:
            counts["decomposition_violation_count"] += 1

        target_harm = q0[TARGET_CLASS_ID] - qtc[TARGET_CLASS_ID]
        max_target_harm = max(max_target_harm, target_harm)
        if target_harm > epsilon_target + tolerance:
            counts["target_safety_violation_count"] += 1

        true_id = int(row["true_class_id"])
        floor = np.finfo(float).eps
        final_nll = -math.log(max(qtc[true_id], floor))
        convex_bound = (1.0 - route) * -math.log(max(q0[true_id], floor)) + route * -math.log(
            max(qphi[true_id], floor)
        )
        if final_nll > convex_bound + tolerance:
            counts["convex_nll_bound_violation_count"] += 1

        group_folds[str(row["split_group_id"])].add(int(row["outer_fold"]))

    cross_fold_count = sum(len(folds) > 1 for folds in group_folds.values())
    result = {
        "row_count": len(rows),
        **counts,
        "cross_fold_group_overlap_count": cross_fold_count,
        "max_target_harm": float(max_target_harm),
        "max_decomposition_residual": float(max_decomposition_residual),
        "tolerance": tolerance,
    }
    result["all_invariants_pass"] = not any(
        result[name]
        for name in (
            "simplex_violation_count",
            "target_safety_violation_count",
            "decomposition_violation_count",
            "convex_nll_bound_violation_count",
            "nonfinite_or_negative_count",
            "cross_fold_group_overlap_count",
        )
    )
    return result


def assess_tc_promotion(
    summary: Mapping[str, object],
    recall_margin: float = 0.01,
    intrusion_margin: float = 0.01,
) -> dict[str, object]:
    intervals = summary["confidence_intervals"]
    invariants = summary["invariants"]
    fold_differences = summary["fold_differences"]
    invariant_fields = (
        "target_safety_violation_count",
        "simplex_violation_count",
        "decomposition_violation_count",
        "convex_nll_bound_violation_count",
        "nonfinite_or_negative_count",
        "cross_fold_group_overlap_count",
    )
    ti_difference = float(intervals["ti_intrusion_to_target"]["difference"])
    metallic_difference = float(
        intervals["metallic_intrusion_to_target"]["difference"]
    )
    favorable_fold_count = sum(
        float(values["nll"]) < 0.0 for values in fold_differences.values()
    )
    criteria = {
        "nll_ci_below_zero": float(intervals["nll"]["ci_high"]) < 0.0,
        "target_recall_noninferior": float(
            intervals["target_recall"]["one_sided_95_lower"]
        )
        >= -recall_margin,
        "zero_theory_violations": all(int(invariants[field]) == 0 for field in invariant_fields),
        "hard_negative_intrusion_guardrail": (
            min(ti_difference, metallic_difference) < 0.0
            and max(ti_difference, metallic_difference) <= intrusion_margin
        ),
        "two_fold_nll_consistency": favorable_fold_count >= 2,
    }
    passed = all(criteria.values())
    return {
        "criteria": criteria,
        "promote_to_main_method": passed,
        "decision": "promote_tc_oos_rsg" if passed else "retain_q0_fallback",
        "failed_criteria": [name for name, value in criteria.items() if not value],
        "recall_margin": recall_margin,
        "intrusion_margin": intrusion_margin,
        "favorable_nll_fold_count": favorable_fold_count,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze locked TC-OOS-RSG outer-fold predictions."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260927)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite an existing analysis directory.")
    baseline_rows = []
    method_rows = []
    candidate_rows = []
    epsilon_by_fold = {}
    for fold in range(3):
        root = args.experiment_root / f"fold_{fold}" / "tc_projection"
        baseline_rows.extend(_read_csv(root / "outer_predictions" / "M1.csv"))
        candidate_rows.extend(_read_csv(root / "outer_predictions" / "M2.csv"))
        method_rows.extend(_read_csv(root / "outer_predictions" / "M5.csv"))
        lock = json.loads((root / "selection_lock.json").read_text(encoding="utf-8"))
        epsilon_by_fold[fold] = float(lock["selected_epsilon_target"])

    bootstrap = cluster_paired_bootstrap(
        baseline_rows, method_rows, iterations=args.iterations, seed=args.seed
    )
    q0_method = _align_prediction_rows(baseline_rows, method_rows)
    q0_candidate = _align_prediction_rows(baseline_rows, candidate_rows)
    candidate_by_id = {row["image_id"]: row for row in q0_candidate}
    method_by_id = {str(row["image_id"]): row for row in method_rows}
    invariant_rows = []
    for aligned in q0_method:
        image_id = str(aligned["image_id"])
        candidate = candidate_by_id[image_id]
        raw_method = method_by_id[image_id]
        invariant_rows.append(
            {
                "image_id": image_id,
                "split_group_id": aligned["split_group_id"],
                "outer_fold": aligned["outer_fold"],
                "true_class_id": aligned["true_class_id"],
                "q0_probabilities": aligned["baseline_probabilities"],
                "qphi_probabilities": candidate["method_probabilities"],
                "qtc_probabilities": aligned["method_probabilities"],
                "route": float(raw_method["route"]),
                "epsilon_target": epsilon_by_fold[int(aligned["outer_fold"])],
            }
        )
    invariants = verify_theory_invariants(invariant_rows)
    summary = {**bootstrap, "invariants": invariants}
    decision = assess_tc_promotion(summary)
    payload = {
        "analysis": summary,
        "promotion_decision": decision,
        "claim_boundary": (
            "Deterministic posterior safety and paired public-specimen evidence only; "
            "no hard-recall, grade, recovery, or industrial-sorting guarantee."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "theory_invariants.json").write_text(
        json.dumps(invariants, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
