from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import numpy as np

from analyze_paired_cluster_statistics import (
    align_prediction_rows,
    paired_two_stage_bootstrap,
    resample_clusters,
)


FORMAL_SEEDS = ("20260727", "20260728", "20260729")
PILOT_SEEDS = ("20260728",)
DEFAULT_CONFIGURATIONS = (
    "hrgv_reference",
    "rsg_complete",
    "rsg_hard_target",
    "rsg_unweighted",
    "rsg_coupled_gate",
)
CLASSIFICATION_METRICS = (
    "accuracy",
    "macro_f1",
    "target_recall",
    "target_miss_rate",
    "ti_to_target_intrusion_rate",
    "metallic_to_target_intrusion_rate",
)
ROUTING_METRICS = (
    "direct_accuracy",
    "mapped_accuracy",
    "fused_accuracy",
    "oracle_accuracy",
    "expert_prediction_disagreement_rate",
    "one_right_one_wrong_rate",
    "one_right_gate_selection_accuracy",
    "mean_routing_regret_nll",
    "complementarity_recovery",
)
FAVORABLE_DIRECTIONS = {
    "accuracy": 1,
    "macro_f1": 1,
    "target_recall": 1,
    "target_miss_rate": -1,
    "ti_to_target_intrusion_rate": -1,
    "metallic_to_target_intrusion_rate": -1,
    "direct_accuracy": 0,
    "mapped_accuracy": 0,
    "fused_accuracy": 1,
    "oracle_accuracy": 0,
    "expert_prediction_disagreement_rate": 0,
    "one_right_one_wrong_rate": 0,
    "one_right_gate_selection_accuracy": 1,
    "mean_routing_regret_nll": -1,
    "complementarity_recovery": 1,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_formal_seeds(seeds: Sequence[str]) -> None:
    if len(set(seeds)) != len(seeds):
        raise ValueError("Formal seed records contain duplicate seeds.")
    missing = sorted(set(FORMAL_SEEDS) - set(seeds))
    extra = sorted(set(seeds) - set(FORMAL_SEEDS))
    if missing:
        raise ValueError(f"Formal seed records are missing seeds: {missing}")
    if extra:
        raise ValueError(f"Formal seed records contain unexpected seeds: {extra}")


def validate_custom_formal_seeds(seeds: Sequence[str]) -> None:
    """Validate an explicitly declared, independent three-seed repeat set."""
    if len(seeds) != 3:
        raise ValueError("A formal comparison requires exactly three seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Formal seeds contain duplicates.")


def calculate_routing_metrics(rows: Sequence[dict[str, str]]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("Prediction rows must not be empty.")
    direct_correct = [row["direct_predicted_label"] == row["true_label"] for row in rows]
    mapped_correct = [row["mapped_predicted_label"] == row["true_label"] for row in rows]
    fused_correct = [row["predicted_label"] == row["true_label"] for row in rows]
    oracle_correct = [first or second for first, second in zip(direct_correct, mapped_correct)]
    disagreement = [
        row["direct_predicted_label"] != row["mapped_predicted_label"] for row in rows
    ]
    one_right = [first != second for first, second in zip(direct_correct, mapped_correct)]
    one_right_indices = [index for index, value in enumerate(one_right) if value]
    gate_selection = [row["gate_selection_correct"] in {"1", "true", "True"} for row in rows]
    count = len(rows)
    direct_accuracy = sum(direct_correct) / count
    mapped_accuracy = sum(mapped_correct) / count
    fused_accuracy = sum(fused_correct) / count
    oracle_accuracy = sum(oracle_correct) / count
    expert_best = max(direct_accuracy, mapped_accuracy)
    oracle_headroom = oracle_accuracy - expert_best
    complementarity_recovery = None
    if oracle_headroom > 0:
        complementarity_recovery = (fused_accuracy - expert_best) / oracle_headroom
    one_right_gate_accuracy = None
    if one_right_indices:
        one_right_gate_accuracy = sum(gate_selection[index] for index in one_right_indices) / len(
            one_right_indices
        )
    return {
        "count": count,
        "direct_accuracy": direct_accuracy,
        "mapped_accuracy": mapped_accuracy,
        "fused_accuracy": fused_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "expert_prediction_disagreement_rate": sum(disagreement) / count,
        "one_right_one_wrong_count": len(one_right_indices),
        "one_right_one_wrong_rate": len(one_right_indices) / count,
        "one_right_gate_selection_accuracy": one_right_gate_accuracy,
        "mean_routing_regret_nll": mean(float(row["routing_regret_nll"]) for row in rows),
        "complementarity_recovery": complementarity_recovery,
    }


def _summarize_values(values: Sequence[float]) -> dict[str, object]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


def _align_routing_rows(
    reference: Sequence[dict[str, str]], comparison: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    reference_by_id = {row["image_id"]: row for row in reference}
    comparison_by_id = {row["image_id"]: row for row in comparison}
    if set(reference_by_id) != set(comparison_by_id):
        raise ValueError("Prediction image_id sets do not match for routing analysis.")
    aligned = []
    for image_id in sorted(reference_by_id):
        first = reference_by_id[image_id]
        second = comparison_by_id[image_id]
        if first["true_label"] != second["true_label"]:
            raise ValueError(f"Prediction true label mismatch for {image_id}.")
        aligned.append(
            {
                "image_id": image_id,
                "split_group_id": first["split_group_id"],
                "true_label": first["true_label"],
                "reference_regret": first["routing_regret_nll"],
                "comparison_regret": second["routing_regret_nll"],
            }
        )
    return aligned


def paired_routing_regret_bootstrap(
    reference: dict[str, list[dict[str, str]]],
    comparison: dict[str, list[dict[str, str]]],
    replicates: int,
    rng_seed: int,
) -> dict[str, object]:
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    aligned = {
        seed: _align_routing_rows(reference[seed], comparison[seed])
        for seed in sorted(reference)
    }
    seed_ids = sorted(aligned)
    point_by_seed = {
        seed: mean(
            float(row["comparison_regret"]) - float(row["reference_regret"])
            for row in rows
        )
        for seed, rows in aligned.items()
    }
    rng = random.Random(rng_seed)
    samples = []
    for _ in range(replicates):
        selected_seeds = rng.choices(seed_ids, k=len(seed_ids))
        seed_differences = []
        for seed in selected_seeds:
            sampled = resample_clusters(aligned[seed], rng)
            seed_differences.append(
                mean(
                    float(row["comparison_regret"]) - float(row["reference_regret"])
                    for row in sampled
                )
            )
        samples.append(mean(seed_differences))
    return {
        "difference": mean(point_by_seed.values()),
        "oriented_improvement": -mean(point_by_seed.values()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_favorable": sum(value < 0 for value in samples) / len(samples),
        "per_seed_difference": point_by_seed,
        "bootstrap_replicates": replicates,
    }


def load_configuration(
    training_root: Path,
    stage: str,
    configuration: str,
    seeds: Sequence[str],
) -> dict[str, object]:
    metric_rows = []
    routing_rows = []
    predictions = {}
    for seed in seeds:
        run_dir = training_root / f"{stage}_{configuration}_seed{seed}"
        metrics_path = run_dir / "test_metrics.json"
        predictions_path = run_dir / "test_predictions.csv"
        missing = [str(path) for path in (metrics_path, predictions_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete {stage} run: {missing}")
        metrics = read_json(metrics_path)
        predictions[seed] = read_csv(predictions_path)
        metric_rows.append({name: float(metrics[name]) for name in CLASSIFICATION_METRICS})
        routing_rows.append(calculate_routing_metrics(predictions[seed]))
    summary = {
        name: _summarize_values([float(row[name]) for row in metric_rows])
        for name in CLASSIFICATION_METRICS
    }
    for name in ROUTING_METRICS:
        values = [row[name] for row in routing_rows]
        if any(value is None for value in values):
            summary[name] = {"mean": None, "sample_std": None, "values": values}
        else:
            summary[name] = _summarize_values([float(value) for value in values])
    return {
        "metrics": metric_rows,
        "routing": routing_rows,
        "summary": summary,
        "predictions": predictions,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RSG-HRGV experiments.")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--reference", default="hrgv_reference")
    parser.add_argument(
        "--config", action="append", choices=DEFAULT_CONFIGURATIONS
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260822)
    parser.add_argument(
        "--seeds",
        nargs="+",
        help=(
            "Explicit seed set for a formal comparison. Use only for a separately "
            "documented protocol; the default formal set remains frozen."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configurations = tuple(args.config or DEFAULT_CONFIGURATIONS)
    if args.reference not in configurations:
        raise ValueError("The reference configuration must be included.")
    seeds = tuple(args.seeds) if args.seeds else (
        PILOT_SEEDS if args.stage == "pilot" else FORMAL_SEEDS
    )
    if args.stage == "formal":
        if args.seeds:
            validate_custom_formal_seeds(seeds)
        else:
            validate_formal_seeds(seeds)
    loaded = {
        configuration: load_configuration(
            args.training_root, args.stage, configuration, seeds
        )
        for configuration in configurations
    }

    summary_rows = []
    routing_rows = []
    for configuration, artifact in loaded.items():
        for metric, summary in artifact["summary"].items():
            summary_rows.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "mean": summary["mean"],
                    "sample_std": summary["sample_std"],
                    **{
                        f"seed_{seed}": value
                        for seed, value in zip(seeds, summary["values"])
                    },
                }
            )
        for seed, values in zip(seeds, artifact["routing"]):
            routing_rows.append({"configuration": configuration, "seed": seed, **values})
    write_csv(args.output_dir / "rsg_three_seed_summary.csv", summary_rows)
    write_json(
        args.output_dir / "rsg_three_seed_summary.json",
        {configuration: artifact["summary"] for configuration, artifact in loaded.items()},
    )
    write_csv(args.output_dir / "rsg_routing_metrics.csv", routing_rows)

    reference = loaded[args.reference]
    delta_rows = []
    for configuration, artifact in loaded.items():
        if configuration == args.reference:
            continue
        for metric, comparison_summary in artifact["summary"].items():
            reference_summary = reference["summary"][metric]
            if reference_summary["mean"] is None or comparison_summary["mean"] is None:
                continue
            difference = float(comparison_summary["mean"]) - float(reference_summary["mean"])
            direction = FAVORABLE_DIRECTIONS[metric]
            delta_rows.append(
                {
                    "reference": args.reference,
                    "comparison": configuration,
                    "metric": metric,
                    "reference_mean": reference_summary["mean"],
                    "comparison_mean": comparison_summary["mean"],
                    "difference": difference,
                    "favorable_direction": direction,
                    "oriented_improvement": difference * direction,
                }
            )
        aligned = {
            seed: align_prediction_rows(
                reference["predictions"][seed], artifact["predictions"][seed]
            )
            for seed in seeds
        }
        classification_bootstrap = paired_two_stage_bootstrap(
            aligned, args.bootstrap_replicates, args.rng_seed
        )
        routing_bootstrap = paired_routing_regret_bootstrap(
            reference["predictions"],
            artifact["predictions"],
            args.bootstrap_replicates,
            args.rng_seed,
        )
        write_json(
            args.output_dir / f"paired_{configuration}_vs_{args.reference}.json",
            {
                "classification": classification_bootstrap["summary"],
                "routing_regret": routing_bootstrap,
            },
        )
    write_csv(args.output_dir / "rsg_ablation_deltas.csv", delta_rows)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "training_root": str(args.training_root.resolve()),
            "stage": args.stage,
            "reference": args.reference,
            "configurations": list(configurations),
            "seeds": list(seeds),
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
        },
    )


if __name__ == "__main__":
    main()
