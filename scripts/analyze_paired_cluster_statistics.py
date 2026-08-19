from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from train_mineral_classifier import CLASS_LABELS


METRIC_SPECS = {
    "accuracy": {"label": "Accuracy", "favorable_direction": 1},
    "macro_f1": {"label": "Macro F1", "favorable_direction": 1},
    "target_recall": {"label": "Target recall", "favorable_direction": 1},
    "target_miss_rate": {"label": "Target miss rate", "favorable_direction": -1},
    "ti_to_target_intrusion": {"label": "Ti-bearing intrusion", "favorable_direction": -1},
    "metallic_to_target_intrusion": {"label": "Metallic intrusion", "favorable_direction": -1},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def align_prediction_rows(
    baseline_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    baseline = {row["image_id"]: row for row in baseline_rows}
    comparison = {row["image_id"]: row for row in comparison_rows}
    if len(baseline) != len(baseline_rows) or len(comparison) != len(comparison_rows):
        raise ValueError("Prediction files contain duplicate image_id values.")
    if set(baseline) != set(comparison):
        missing_baseline = sorted(set(comparison) - set(baseline))[:5]
        missing_comparison = sorted(set(baseline) - set(comparison))[:5]
        raise ValueError(
            "Prediction image_id sets do not match: "
            f"missing baseline={missing_baseline}, missing comparison={missing_comparison}."
        )

    pairs = []
    for image_id in sorted(baseline):
        first = baseline[image_id]
        second = comparison[image_id]
        if first["true_label"] != second["true_label"]:
            raise ValueError(f"Prediction true label mismatch for {image_id}.")
        if first["split_group_id"] != second["split_group_id"]:
            raise ValueError(f"Prediction split group mismatch for {image_id}.")
        pairs.append(
            {
                "image_id": image_id,
                "split_group_id": first["split_group_id"],
                "mineral_label": first.get("mineral_label", ""),
                "true_label": first["true_label"],
                "baseline_prediction": first["predicted_label"],
                "comparison_prediction": second["predicted_label"],
            }
        )
    return pairs


def resample_clusters(rows: list[dict[str, str]], rng: random.Random) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("split_group_id", "")
        if not group_id:
            raise ValueError("Every row must have a non-empty split_group_id.")
        grouped[group_id].append(row)
    strata: dict[str, list[str]] = defaultdict(list)
    can_stratify = all(row.get("true_label") for row in rows)
    if can_stratify:
        for group_id, values in grouped.items():
            labels = {row["true_label"] for row in values}
            if len(labels) != 1:
                raise ValueError(f"Cluster {group_id} crosses true-label strata.")
            strata[next(iter(labels))].append(group_id)
    else:
        strata["all"] = sorted(grouped)

    sampled = []
    for label in sorted(strata):
        group_ids = sorted(strata[label])
        for group_id in rng.choices(group_ids, k=len(group_ids)):
            sampled.extend(grouped[group_id])
    return sampled


def _conditional_rate(rows: list[dict[str, str]], true_label: str, predicted_label: str) -> float:
    relevant = [row for row in rows if row["true_label"] == true_label]
    if not relevant:
        return math.nan
    return sum(row["prediction"] == predicted_label for row in relevant) / len(relevant)


def calculate_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Metric rows must not be empty.")
    true = [row["true_label"] for row in rows]
    predicted = [row["prediction"] for row in rows]
    target_recall = _conditional_rate(rows, "target_mineral", "target_mineral")
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "target_recall": target_recall,
        "target_miss_rate": 1.0 - target_recall,
        "ti_to_target_intrusion": _conditional_rate(rows, "ti_bearing_negative", "target_mineral"),
        "metallic_to_target_intrusion": _conditional_rate(rows, "metallic_hard_negative", "target_mineral"),
    }


def confusion_to_metrics(confusion: np.ndarray) -> dict[str, float]:
    expected_shape = (len(CLASS_LABELS), len(CLASS_LABELS))
    if confusion.shape != expected_shape:
        raise ValueError(f"Confusion matrix must have shape {expected_shape}.")
    total = float(confusion.sum())
    if total <= 0:
        raise ValueError("Confusion matrix must contain at least one observation.")

    true_counts = confusion.sum(axis=1).astype(float)
    predicted_counts = confusion.sum(axis=0).astype(float)
    true_positive = np.diag(confusion).astype(float)
    denominator = 2.0 * true_positive + (predicted_counts - true_positive) + (true_counts - true_positive)
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    label_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    target = label_index["target_mineral"]
    ti_bearing = label_index["ti_bearing_negative"]
    metallic = label_index["metallic_hard_negative"]
    target_recall = float(confusion[target, target] / true_counts[target])
    return {
        "accuracy": float(true_positive.sum() / total),
        "macro_f1": float(class_f1.mean()),
        "target_recall": target_recall,
        "target_miss_rate": 1.0 - target_recall,
        "ti_to_target_intrusion": float(confusion[ti_bearing, target] / true_counts[ti_bearing]),
        "metallic_to_target_intrusion": float(confusion[metallic, target] / true_counts[metallic]),
    }


def _paired_metrics(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float]]:
    baseline_rows = [{**row, "prediction": row["baseline_prediction"]} for row in rows]
    comparison_rows = [{**row, "prediction": row["comparison_prediction"]} for row in rows]
    return calculate_metrics(baseline_rows), calculate_metrics(comparison_rows)


def _cluster_confusion_tables(rows: list[dict[str, str]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    label_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("split_group_id", "")
        if not group_id:
            raise ValueError("Every row must have a non-empty split_group_id.")
        grouped[group_id].append(row)

    strata: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    shape = (len(CLASS_LABELS), len(CLASS_LABELS))
    for group_id, values in grouped.items():
        labels = {row["true_label"] for row in values}
        if len(labels) != 1:
            raise ValueError(f"Cluster {group_id} crosses true-label strata.")
        true_label = next(iter(labels))
        baseline = np.zeros(shape, dtype=np.int64)
        comparison = np.zeros(shape, dtype=np.int64)
        for row in values:
            true_index = label_index[row["true_label"]]
            baseline[true_index, label_index[row["baseline_prediction"]]] += 1
            comparison[true_index, label_index[row["comparison_prediction"]]] += 1
        strata[true_label].append((baseline, comparison))

    return {
        label: (
            np.stack([pair[0] for pair in pairs]),
            np.stack([pair[1] for pair in pairs]),
        )
        for label, pairs in strata.items()
    }


def _sample_cluster_confusions(
    tables: dict[str, tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(CLASS_LABELS), len(CLASS_LABELS))
    baseline = np.zeros(shape, dtype=np.int64)
    comparison = np.zeros(shape, dtype=np.int64)
    for label in sorted(tables):
        baseline_clusters, comparison_clusters = tables[label]
        cluster_count = baseline_clusters.shape[0]
        draw_counts = rng.multinomial(cluster_count, np.full(cluster_count, 1.0 / cluster_count))
        baseline += np.tensordot(draw_counts, baseline_clusters, axes=(0, 0)).astype(np.int64)
        comparison += np.tensordot(draw_counts, comparison_clusters, axes=(0, 0)).astype(np.int64)
    return baseline, comparison


def exact_mcnemar(rows: list[dict[str, str]]) -> dict[str, float | int]:
    baseline_only = 0
    comparison_only = 0
    for row in rows:
        baseline_correct = row["baseline_prediction"] == row["true_label"]
        comparison_correct = row["comparison_prediction"] == row["true_label"]
        baseline_only += int(baseline_correct and not comparison_correct)
        comparison_only += int(comparison_correct and not baseline_correct)
    discordant = baseline_only + comparison_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(baseline_only, comparison_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "baseline_only_correct": baseline_only,
        "comparison_only_correct": comparison_only,
        "discordant_total": discordant,
        "p_value_two_sided_exact": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_two_stage_bootstrap(
    runs_by_seed: dict[str, list[dict[str, str]]],
    replicates: int,
    rng_seed: int,
) -> dict[str, object]:
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    if not runs_by_seed:
        raise ValueError("At least one paired seed is required.")
    seed_ids = sorted(runs_by_seed)
    rng = np.random.default_rng(rng_seed)

    per_seed = {}
    for seed, rows in runs_by_seed.items():
        baseline, comparison = _paired_metrics(rows)
        per_seed[seed] = {
            metric: {
                "baseline": baseline[metric],
                "comparison": comparison[metric],
                "difference": comparison[metric] - baseline[metric],
            }
            for metric in METRIC_SPECS
        }

    point = {}
    for metric, spec in METRIC_SPECS.items():
        baseline_mean = mean(per_seed[seed][metric]["baseline"] for seed in seed_ids)
        comparison_mean = mean(per_seed[seed][metric]["comparison"] for seed in seed_ids)
        difference = comparison_mean - baseline_mean
        point[metric] = {
            "label": spec["label"],
            "favorable_direction": spec["favorable_direction"],
            "baseline_mean": baseline_mean,
            "comparison_mean": comparison_mean,
            "difference": difference,
            "oriented_improvement": difference * spec["favorable_direction"],
        }

    cluster_tables = {seed: _cluster_confusion_tables(rows) for seed, rows in runs_by_seed.items()}
    bootstrap_rows = []
    for bootstrap_index in range(replicates):
        sampled_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        differences: dict[str, list[float]] = {metric: [] for metric in METRIC_SPECS}
        for seed in sampled_seeds:
            baseline_confusion, comparison_confusion = _sample_cluster_confusions(cluster_tables[str(seed)], rng)
            baseline = confusion_to_metrics(baseline_confusion)
            comparison = confusion_to_metrics(comparison_confusion)
            for metric in METRIC_SPECS:
                differences[metric].append(comparison[metric] - baseline[metric])
        bootstrap_rows.append(
            {
                "bootstrap_index": bootstrap_index,
                **{metric: mean(values) for metric, values in differences.items()},
            }
        )

    summary = {}
    for metric, spec in METRIC_SPECS.items():
        values = np.asarray([row[metric] for row in bootstrap_rows], dtype=float)
        direction = spec["favorable_direction"]
        summary[metric] = {
            **point[metric],
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "probability_favorable": float(np.mean(values * direction > 0.0)),
            "bootstrap_replicates": replicates,
        }
    return {"per_seed": per_seed, "summary": summary, "replicates": bootstrap_rows}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot_effects(summary: dict[str, dict[str, float]], figure_prefix: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )
    metrics = list(METRIC_SPECS)
    labels = [METRIC_SPECS[metric]["label"] for metric in metrics]
    oriented = []
    lower = []
    upper = []
    colors = []
    for metric in metrics:
        result = summary[metric]
        direction = int(result["favorable_direction"])
        effect = 100.0 * result["difference"] * direction
        ci_a = 100.0 * result["ci_low"] * direction
        ci_b = 100.0 * result["ci_high"] * direction
        ci_low, ci_high = sorted((ci_a, ci_b))
        oriented.append(effect)
        lower.append(effect - ci_low)
        upper.append(ci_high - effect)
        colors.append("#2A9D8F" if ci_low > 0 else "#4C78A8")

    y = np.arange(len(metrics))[::-1]
    figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    axis.axvline(0.0, color="#606060", linewidth=0.8, linestyle="--", zorder=1)
    axis.errorbar(
        oriented,
        y,
        xerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="#394B59",
        elinewidth=1.2,
        capsize=3,
        zorder=2,
    )
    axis.scatter(oriented, y, c=colors, s=34, edgecolor="white", linewidth=0.6, zorder=3)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Oriented improvement (percentage points; >0 favors hierarchical model)")
    axis.grid(axis="x", color="#D9E1E8", linewidth=0.6, alpha=0.8)
    axis.set_title("Paired two-stage cluster bootstrap on the fixed test set", loc="left", fontweight="bold", fontsize=9)
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(figure_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(figure_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        figure_prefix.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired cluster-bootstrap and exact McNemar inference.")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--baseline-prefix", default="formal_role_aware_efficientnet_b0_seed")
    parser.add_argument("--comparison-prefix", default="formal_hierarchical_efficientnet_b0_seed")
    parser.add_argument("--seeds", nargs="+", default=("20260727", "20260728", "20260729"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260819)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = {}
    run_provenance = {}
    for seed in args.seeds:
        baseline_path = args.training_root / f"{args.baseline_prefix}{seed}" / "test_predictions.csv"
        comparison_path = args.training_root / f"{args.comparison_prefix}{seed}" / "test_predictions.csv"
        runs[str(seed)] = align_prediction_rows(_read_csv(baseline_path), _read_csv(comparison_path))
        run_provenance[str(seed)] = {
            "baseline_run": baseline_path.parent.name,
            "comparison_run": comparison_path.parent.name,
            "row_count": len(runs[str(seed)]),
            "cluster_count": len({row["split_group_id"] for row in runs[str(seed)]}),
        }

    bootstrap = paired_two_stage_bootstrap(runs, args.bootstrap_replicates, args.rng_seed)
    mcnemar_rows = []
    for seed in args.seeds:
        mcnemar_rows.append({"seed": seed, **exact_mcnemar(runs[str(seed)])})
    adjusted = holm_adjust([float(row["p_value_two_sided_exact"]) for row in mcnemar_rows])
    for row, value in zip(mcnemar_rows, adjusted):
        row["p_value_holm_three_seeds"] = value

    seed_metric_rows = []
    for seed, metrics in bootstrap["per_seed"].items():
        for metric, values in metrics.items():
            seed_metric_rows.append({"seed": seed, "metric": metric, **values})
    summary_rows = []
    for metric, values in bootstrap["summary"].items():
        summary_rows.append({"metric": metric, **values})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paired_seed_metrics.csv", seed_metric_rows)
    _write_csv(args.output_dir / "bootstrap_distribution.csv", bootstrap["replicates"])
    _write_csv(args.output_dir / "paired_cluster_bootstrap_summary.csv", summary_rows)
    _write_csv(args.output_dir / "mcnemar_exact.csv", mcnemar_rows)
    payload = {
        "analysis": {
            "comparison": "hierarchical_minus_baseline",
            "bootstrap": "two-stage resampling of seeds and true-role-stratified split_group_id clusters",
            "confidence_interval": "percentile 95%",
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
            "mcnemar": "two-sided exact binomial test per seed with Holm correction across three seeds",
            "effect_interpretation": "raw difference is comparison minus baseline; oriented improvement reverses lower-is-better metrics",
        },
        "run_provenance": run_provenance,
        "summary": bootstrap["summary"],
        "mcnemar": mcnemar_rows,
        "limitations": [
            "Only three training seeds are available, so seed-level uncertainty remains imprecise.",
            "The bootstrap quantifies uncertainty over observed seeds and image groups, not unseen industrial domains.",
            "McNemar tests accuracy discordance only and does not replace interval estimates for Macro F1 or class-specific risks.",
        ],
    }
    with (args.output_dir / "paired_cluster_bootstrap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _plot_effects(bootstrap["summary"], args.figure_prefix)
    print(json.dumps({"summary": bootstrap["summary"], "mcnemar": mcnemar_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
