from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

from sklearn.metrics import roc_auc_score

from analyze_paired_cluster_statistics import (
    align_prediction_rows,
    exact_mcnemar,
    holm_adjust,
    paired_two_stage_bootstrap,
)


FORMAL_SEEDS = ("20260727", "20260728", "20260729")
METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "target_recall",
    "target_miss_rate",
    "ti_to_target_intrusion_rate",
    "metallic_to_target_intrusion_rate",
    "species_accuracy",
    "mean_gate",
    "mean_expert_js_divergence",
)
FAVORABLE_DIRECTIONS = {
    "accuracy": 1,
    "macro_f1": 1,
    "target_recall": 1,
    "target_miss_rate": -1,
    "ti_to_target_intrusion_rate": -1,
    "metallic_to_target_intrusion_rate": -1,
    "species_accuracy": 1,
    "mean_gate": 0,
    "mean_expert_js_divergence": 0,
}
DEFAULT_CONFIGURATIONS = (
    "residual_complete",
    "equal_fusion",
    "no_contrast",
    "decoupled_residual",
    "gate_only",
    "complete",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metric_rows(
    rows: Sequence[dict[str, float]],
    metric_names: Iterable[str] | None = None,
) -> dict[str, dict[str, object]]:
    if len(rows) < 2:
        raise ValueError("At least two metric rows are required for sample variation.")
    names = tuple(metric_names or rows[0].keys())
    summary: dict[str, dict[str, object]] = {}
    for name in names:
        values = [float(row[name]) for row in rows]
        summary[name] = {
            "mean": mean(values),
            "sample_std": stdev(values),
            "values": values,
        }
    return summary


def summarize_gate_by_role(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["true_label"]].append(float(row["gate"]))
    if not grouped:
        raise ValueError("Prediction rows must not be empty.")
    return {
        label: {
            "count": len(values),
            "mean": mean(values),
            "sample_std": stdev(values) if len(values) > 1 else 0.0,
        }
        for label, values in sorted(grouped.items())
    }


def calculate_verifier_auc(
    rows: Sequence[dict[str, str]],
    negative_label: str,
    probability_field: str,
) -> dict[str, float | int]:
    eligible = [
        row
        for row in rows
        if row["true_label"] in {"target_mineral", negative_label}
    ]
    labels = [int(row["true_label"] == "target_mineral") for row in eligible]
    if len(set(labels)) != 2:
        raise ValueError("Verifier ROC-AUC requires target and selected negative examples.")
    scores = [float(row[probability_field]) for row in eligible]
    return {
        "eligible_count": len(eligible),
        "target_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def compare_summaries(
    baseline: dict[str, dict[str, object]],
    comparison: dict[str, dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    common = [name for name in baseline if name in comparison]
    result = {}
    for name in common:
        baseline_mean = float(baseline[name]["mean"])
        comparison_mean = float(comparison[name]["mean"])
        difference = comparison_mean - baseline_mean
        direction = FAVORABLE_DIRECTIONS.get(name, 0)
        result[name] = {
            "baseline_mean": baseline_mean,
            "comparison_mean": comparison_mean,
            "difference": difference,
            "favorable_direction": direction,
            "oriented_improvement": difference * direction,
        }
    return result


def load_configuration(
    training_root: Path,
    configuration: str,
    seeds: Sequence[str],
) -> dict[str, object]:
    metrics = []
    predictions: dict[str, list[dict[str, str]]] = {}
    environments = {}
    for seed in seeds:
        run_dir = training_root / f"formal_hrgv_{configuration}_seed{seed}"
        metrics_path = run_dir / "test_metrics.json"
        predictions_path = run_dir / "test_predictions.csv"
        environment_path = run_dir / "environment.json"
        missing = [
            str(path)
            for path in (metrics_path, predictions_path, environment_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Incomplete formal run: {missing}")
        metric = read_json(metrics_path)
        metrics.append({name: float(metric[name]) for name in METRIC_NAMES})
        predictions[seed] = read_csv(predictions_path)
        environments[seed] = read_json(environment_path)
    return {
        "metrics": metrics,
        "summary": summarize_metric_rows(metrics, METRIC_NAMES),
        "predictions": predictions,
        "environments": environments,
    }


def build_paired_inference(
    reference_predictions: dict[str, list[dict[str, str]]],
    comparison_predictions: dict[str, list[dict[str, str]]],
    replicates: int,
    rng_seed: int,
) -> dict[str, object]:
    paired = {
        seed: align_prediction_rows(reference_predictions[seed], comparison_predictions[seed])
        for seed in sorted(reference_predictions)
    }
    bootstrap = paired_two_stage_bootstrap(paired, replicates, rng_seed)
    mcnemar = [{"seed": seed, **exact_mcnemar(paired[seed])} for seed in sorted(paired)]
    adjusted = holm_adjust(
        [float(row["p_value_two_sided_exact"]) for row in mcnemar]
    )
    for row, p_value in zip(mcnemar, adjusted):
        row["p_value_holm_three_seeds"] = p_value
    return {
        "bootstrap_summary": bootstrap["summary"],
        "mcnemar": mcnemar,
        "seed_count": len(paired),
        "bootstrap_replicates": replicates,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze formal HRGV-Net experiments.")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", default="residual_complete")
    parser.add_argument(
        "--configuration",
        action="append",
        choices=DEFAULT_CONFIGURATIONS,
        help="Analyze selected configurations; repeat for multiple values.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260821)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configurations = tuple(args.configuration or DEFAULT_CONFIGURATIONS)
    if args.reference not in configurations:
        raise ValueError("The reference configuration must be included in the analysis.")
    if args.bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    loaded = {
        name: load_configuration(args.training_root, name, FORMAL_SEEDS)
        for name in configurations
    }

    summary_rows = []
    for configuration, artifact in loaded.items():
        for metric, values in artifact["summary"].items():
            summary_rows.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "mean": values["mean"],
                    "sample_std": values["sample_std"],
                    **{
                        f"seed_{seed}": value
                        for seed, value in zip(FORMAL_SEEDS, values["values"])
                    },
                }
            )
    write_csv(args.output_dir / "hrgv_three_seed_summary.csv", summary_rows)
    write_json(
        args.output_dir / "hrgv_three_seed_summary.json",
        {name: artifact["summary"] for name, artifact in loaded.items()},
    )

    reference = loaded[args.reference]
    delta_rows = []
    gate_rows = []
    verifier_rows = []
    for configuration, artifact in loaded.items():
        if configuration != args.reference:
            for metric, values in compare_summaries(
                reference["summary"], artifact["summary"]
            ).items():
                delta_rows.append(
                    {"reference": args.reference, "comparison": configuration, "metric": metric, **values}
                )
            paired = build_paired_inference(
                reference["predictions"],
                artifact["predictions"],
                args.bootstrap_replicates,
                args.rng_seed,
            )
            write_json(args.output_dir / f"paired_{configuration}_vs_{args.reference}.json", paired)

        for seed in FORMAL_SEEDS:
            predictions = artifact["predictions"][seed]
            for role, values in summarize_gate_by_role(predictions).items():
                gate_rows.append(
                    {"configuration": configuration, "seed": seed, "true_role": role, **values}
                )
            verifier_specs = (
                ("ti", "ti_bearing_negative", "ti_target_probability"),
                ("metallic", "metallic_hard_negative", "metallic_target_probability"),
            )
            for verifier, negative_label, field in verifier_specs:
                verifier_rows.append(
                    {
                        "configuration": configuration,
                        "seed": seed,
                        "verifier": verifier,
                        **calculate_verifier_auc(predictions, negative_label, field),
                    }
                )
    write_csv(args.output_dir / "hrgv_ablation_deltas.csv", delta_rows)
    write_csv(args.output_dir / "hrgv_gate_by_role.csv", gate_rows)
    write_csv(args.output_dir / "hrgv_verifier_auc.csv", verifier_rows)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "training_root": str(args.training_root.resolve()),
            "reference": args.reference,
            "configurations": list(configurations),
            "seeds": list(FORMAL_SEEDS),
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
        },
    )


if __name__ == "__main__":
    main()
