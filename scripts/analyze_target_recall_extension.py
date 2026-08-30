from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

from analyze_cgdc_rsg_experiments import (
    CALIBRATION_METRICS,
    paired_calibration_bootstrap,
    summarize_calibration,
)
from analyze_paired_cluster_statistics import align_prediction_rows, paired_two_stage_bootstrap
from analyze_rsg_hrgv_experiment import (
    CLASSIFICATION_METRICS,
    FAVORABLE_DIRECTIONS,
    ROUTING_METRICS,
    calculate_routing_metrics,
    paired_routing_regret_bootstrap,
    read_csv,
    read_json,
    write_csv,
    write_json,
)


ORIGINAL_SEEDS = ("20260727", "20260728", "20260729")
EXTENSION_SEEDS = ("20260730", "20260731")
FIVE_SEEDS = ORIGINAL_SEEDS + EXTENSION_SEEDS
REGISTERED_CONFIGURATIONS = ("rsg_complete", "mrpg_complete")


def registered_run_directories(
    rsg_original_training_root: Path,
    mrpg_original_training_root: Path,
    extension_training_root: Path,
) -> dict[str, dict[str, Path]]:
    """Return the fixed five-seed run locations for the registered extension."""
    original_roots = {
        "rsg_complete": rsg_original_training_root,
        "mrpg_complete": mrpg_original_training_root,
    }
    return {
        configuration: {
            **{
                seed: original_roots[configuration] / f"formal_{configuration}_seed{seed}"
                for seed in ORIGINAL_SEEDS
            },
            **{
                seed: extension_training_root / f"extension_{configuration}_seed{seed}"
                for seed in EXTENSION_SEEDS
            },
        }
        for configuration in REGISTERED_CONFIGURATIONS
    }


def _summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        raise ValueError("Cannot summarize an empty metric sequence.")
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


def load_five_seed_configuration(
    run_directories: dict[str, Path], seeds: Sequence[str] = FIVE_SEEDS
) -> dict[str, object]:
    if tuple(seeds) != FIVE_SEEDS:
        raise ValueError("Five-seed extension analysis requires the registered seed sequence.")
    if set(run_directories) != set(seeds):
        raise ValueError("Run directories must contain exactly the registered five seeds.")

    metric_rows: list[dict[str, float]] = []
    routing_rows: list[dict[str, float | int | None]] = []
    predictions: dict[str, list[dict[str, str]]] = {}
    calibration_rows: list[dict[str, float]] = []
    for seed in seeds:
        run_dir = run_directories[seed]
        metrics_path = run_dir / "test_metrics.json"
        predictions_path = run_dir / "test_predictions.csv"
        missing = [str(path) for path in (metrics_path, predictions_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete registered extension run: {missing}")
        metrics = read_json(metrics_path)
        predictions[seed] = read_csv(predictions_path)
        metric_rows.append({name: float(metrics[name]) for name in CLASSIFICATION_METRICS})
        routing_rows.append(calculate_routing_metrics(predictions[seed]))
        calibration_rows.append(summarize_calibration(predictions[seed]))

    summary = {
        metric: _summary([row[metric] for row in metric_rows])
        for metric in CLASSIFICATION_METRICS
    }
    for metric in ROUTING_METRICS:
        values = [row[metric] for row in routing_rows]
        if any(value is None for value in values):
            summary[metric] = {"mean": None, "sample_std": None, "values": values}
        else:
            summary[metric] = _summary([float(value) for value in values])
    summary.update(
        {
            metric: _summary([row[metric] for row in calibration_rows])
            for metric in CALIBRATION_METRICS
        }
    )
    return {
        "metrics": metric_rows,
        "routing": routing_rows,
        "calibration": calibration_rows,
        "summary": summary,
        "predictions": predictions,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze the registered five-seed RSG/M-RPG target-recall extension."
    )
    parser.add_argument(
        "--rsg-original-training-root",
        type=Path,
        default=project_root / "outputs" / "training" / "cgdc_formal",
    )
    parser.add_argument(
        "--mrpg-original-training-root",
        type=Path,
        default=project_root / "outputs" / "training" / "mrpg_formal",
    )
    parser.add_argument(
        "--extension-training-root",
        type=Path,
        default=project_root / "outputs" / "training" / "target_recall_extension",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "business_metrics" / "target_recall_extension" / "five_seed",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--rng-seed", type=int, default=20260830)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")

    all_run_directories = registered_run_directories(
        args.rsg_original_training_root,
        args.mrpg_original_training_root,
        args.extension_training_root,
    )
    loaded = {
        configuration: load_five_seed_configuration(run_directories)
        for configuration, run_directories in all_run_directories.items()
    }
    metrics = (*CLASSIFICATION_METRICS, *ROUTING_METRICS, *CALIBRATION_METRICS)
    summary_rows: list[dict[str, object]] = []
    for configuration, artifact in loaded.items():
        for metric in metrics:
            summary = artifact["summary"][metric]
            summary_rows.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "mean": summary["mean"],
                    "sample_std": summary["sample_std"],
                    **{
                        f"seed_{seed}": value
                        for seed, value in zip(FIVE_SEEDS, summary["values"])
                    },
                }
            )
    write_csv(args.output_dir / "five_seed_summary.csv", summary_rows)
    write_json(
        args.output_dir / "five_seed_summary.json",
        {configuration: artifact["summary"] for configuration, artifact in loaded.items()},
    )

    reference_name = "rsg_complete"
    comparison_name = "mrpg_complete"
    reference = loaded[reference_name]
    comparison = loaded[comparison_name]
    delta_rows: list[dict[str, object]] = []
    for metric in metrics:
        baseline = reference["summary"][metric]
        candidate = comparison["summary"][metric]
        if baseline["mean"] is None or candidate["mean"] is None:
            continue
        difference = float(candidate["mean"]) - float(baseline["mean"])
        direction = FAVORABLE_DIRECTIONS.get(metric, -1)
        delta_rows.append(
            {
                "reference": reference_name,
                "comparison": comparison_name,
                "metric": metric,
                "reference_mean": baseline["mean"],
                "comparison_mean": candidate["mean"],
                "difference": difference,
                "favorable_direction": direction,
                "oriented_improvement": difference * direction,
            }
        )
    write_csv(args.output_dir / "five_seed_deltas.csv", delta_rows)

    aligned = {
        seed: align_prediction_rows(reference["predictions"][seed], comparison["predictions"][seed])
        for seed in FIVE_SEEDS
    }
    write_json(
        args.output_dir / "paired_mrpg_complete_vs_rsg_complete.json",
        {
            "classification": paired_two_stage_bootstrap(
                aligned, args.bootstrap_replicates, args.rng_seed
            )["summary"],
            "routing_regret": paired_routing_regret_bootstrap(
                reference["predictions"], comparison["predictions"],
                args.bootstrap_replicates, args.rng_seed,
            ),
            "calibration": paired_calibration_bootstrap(
                reference["predictions"], comparison["predictions"],
                args.bootstrap_replicates, args.rng_seed,
            ),
        },
    )
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "protocol": "registered_two_seed_extension_combined_with_original_three_seed_formal_runs",
            "configurations": list(REGISTERED_CONFIGURATIONS),
            "reference": reference_name,
            "seeds": list(FIVE_SEEDS),
            "rsg_original_training_root": str(args.rsg_original_training_root.resolve()),
            "mrpg_original_training_root": str(args.mrpg_original_training_root.resolve()),
            "extension_training_root": str(args.extension_training_root.resolve()),
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
            "classification_metrics": list(CLASSIFICATION_METRICS),
            "routing_metrics": list(ROUTING_METRICS),
            "calibration_metrics": list(CALIBRATION_METRICS),
        },
    )


if __name__ == "__main__":
    main()
