from __future__ import annotations

import argparse
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
from analyze_rsg_hrgv_experiment import (
    CLASSIFICATION_METRICS,
    FAVORABLE_DIRECTIONS,
    FORMAL_SEEDS,
    ROUTING_METRICS,
    load_configuration,
    paired_routing_regret_bootstrap,
    validate_formal_seeds,
    write_csv,
    write_json,
)
from train_mineral_classifier import CLASS_LABELS


CALIBRATION_METRICS = ("brier_score", "expected_calibration_error")


def _summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


def summarize_calibration(rows: Sequence[dict[str, str]], bins: int = 10) -> dict[str, float]:
    """Compute ECE and Brier score from the exported four-role posterior columns."""
    if not rows:
        raise ValueError("Prediction rows must not be empty.")
    if bins <= 0:
        raise ValueError("The number of calibration bins must be positive.")
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    brier_sum = 0.0
    bins_data: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for row in rows:
        try:
            target = label_to_index[row["true_label"]]
            posterior = [float(row[f"role_probability_{label}"]) for label in CLASS_LABELS]
        except KeyError as exc:
            raise ValueError(f"Missing calibration field: {exc.args[0]}") from exc
        if min(posterior) < 0 or abs(sum(posterior) - 1.0) > 1e-4:
            raise ValueError("Exported role posteriors must be non-negative and sum to one.")
        prediction = max(range(len(CLASS_LABELS)), key=posterior.__getitem__)
        confidence = posterior[prediction]
        brier_sum += sum(
            (probability - float(index == target)) ** 2
            for index, probability in enumerate(posterior)
        )
        bins_data[min(int(confidence * bins), bins - 1)].append(
            (confidence, prediction == target)
        )
    ece = 0.0
    count = len(rows)
    for values in bins_data:
        if values:
            mean_confidence = mean(value[0] for value in values)
            mean_accuracy = mean(value[1] for value in values)
            ece += len(values) / count * abs(mean_accuracy - mean_confidence)
    return {"brier_score": brier_sum / count, "expected_calibration_error": ece}


def _align_calibration_rows(
    reference: Sequence[dict[str, str]], comparison: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    reference_by_id = {row["image_id"]: row for row in reference}
    comparison_by_id = {row["image_id"]: row for row in comparison}
    if set(reference_by_id) != set(comparison_by_id):
        raise ValueError("Prediction image_id sets do not match for calibration analysis.")
    aligned: list[dict[str, str]] = []
    for image_id in sorted(reference_by_id):
        baseline = reference_by_id[image_id]
        candidate = comparison_by_id[image_id]
        if baseline["true_label"] != candidate["true_label"]:
            raise ValueError(f"Prediction true label mismatch for {image_id}.")
        if baseline["split_group_id"] != candidate["split_group_id"]:
            raise ValueError(f"Prediction split group mismatch for {image_id}.")
        aligned.append(
            {
                "image_id": image_id,
                "split_group_id": baseline["split_group_id"],
                "true_label": baseline["true_label"],
                **{
                    f"reference_{label}": baseline[f"role_probability_{label}"]
                    for label in CLASS_LABELS
                },
                **{
                    f"comparison_{label}": candidate[f"role_probability_{label}"]
                    for label in CLASS_LABELS
                },
            }
        )
    return aligned


def _calibration_rows(rows: Sequence[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    return [
        {
            "true_label": row["true_label"],
            **{
                f"role_probability_{label}": row[f"{prefix}_{label}"]
                for label in CLASS_LABELS
            },
        }
        for row in rows
    ]


def paired_calibration_bootstrap(
    reference: dict[str, list[dict[str, str]]],
    comparison: dict[str, list[dict[str, str]]],
    replicates: int,
    rng_seed: int,
) -> dict[str, dict[str, object]]:
    """Cluster-and-seed resampled intervals for calibration metric differences."""
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    if set(reference) != set(comparison) or not reference:
        raise ValueError("Reference and comparison must share non-empty seed sets.")
    aligned = {
        seed: _align_calibration_rows(reference[seed], comparison[seed])
        for seed in sorted(reference)
    }
    point_by_seed = {
        seed: {
            metric: summarize_calibration(_calibration_rows(rows, "comparison"))[metric]
            - summarize_calibration(_calibration_rows(rows, "reference"))[metric]
            for metric in CALIBRATION_METRICS
        }
        for seed, rows in aligned.items()
    }
    rng = random.Random(rng_seed)
    samples = {metric: [] for metric in CALIBRATION_METRICS}
    seed_ids = sorted(aligned)
    for _ in range(replicates):
        sampled_by_seed = []
        for seed in rng.choices(seed_ids, k=len(seed_ids)):
            sampled = resample_clusters(aligned[seed], rng)
            reference_metrics = summarize_calibration(_calibration_rows(sampled, "reference"))
            comparison_metrics = summarize_calibration(_calibration_rows(sampled, "comparison"))
            sampled_by_seed.append(
                {
                    metric: comparison_metrics[metric] - reference_metrics[metric]
                    for metric in CALIBRATION_METRICS
                }
            )
        for metric in CALIBRATION_METRICS:
            samples[metric].append(mean(item[metric] for item in sampled_by_seed))
    return {
        metric: {
            "difference": mean(values[metric] for values in point_by_seed.values()),
            "oriented_improvement": -mean(values[metric] for values in point_by_seed.values()),
            "ci_low": float(np.quantile(samples[metric], 0.025)),
            "ci_high": float(np.quantile(samples[metric], 0.975)),
            "probability_favorable": sum(value < 0 for value in samples[metric]) / len(samples[metric]),
            "per_seed_difference": {seed: values[metric] for seed, values in point_by_seed.items()},
            "bootstrap_replicates": replicates,
        }
        for metric in CALIBRATION_METRICS
    }


def parse_config_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --config-root value must be CONFIGURATION=PATH.")
        configuration, raw_path = (item.strip() for item in value.split("=", 1))
        if not configuration or not raw_path or configuration in roots:
            raise ValueError("Configuration roots must be non-empty and unique.")
        roots[configuration] = Path(raw_path)
    if not roots:
        raise ValueError("At least one --config-root value is required.")
    return roots


def load_cgdc_configuration(
    training_root: Path, configuration: str, seeds: Sequence[str]
) -> dict[str, object]:
    artifact = load_configuration(training_root, "formal", configuration, seeds)
    calibration_by_seed = [summarize_calibration(artifact["predictions"][seed]) for seed in seeds]
    artifact["summary"].update(
        {
            metric: _summary([float(values[metric]) for values in calibration_by_seed])
            for metric in CALIBRATION_METRICS
        }
    )
    artifact["calibration"] = calibration_by_seed
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the registered CGDC-RSG-HRGV matrix.")
    parser.add_argument("--config-root", action="append", required=True, metavar="CONFIGURATION=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", default="rsg_complete")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260826)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    roots = parse_config_roots(args.config_root)
    if args.reference not in roots:
        raise ValueError("The reference configuration must have a --config-root entry.")
    seeds = FORMAL_SEEDS
    validate_formal_seeds(seeds)
    loaded = {
        configuration: load_cgdc_configuration(root, configuration, seeds)
        for configuration, root in roots.items()
    }
    metric_names = (*CLASSIFICATION_METRICS, *ROUTING_METRICS, *CALIBRATION_METRICS)
    summary_rows: list[dict[str, object]] = []
    for configuration, artifact in loaded.items():
        for metric in metric_names:
            summary = artifact["summary"][metric]
            summary_rows.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "mean": summary["mean"],
                    "sample_std": summary["sample_std"],
                    **{f"seed_{seed}": value for seed, value in zip(seeds, summary["values"])},
                }
            )
    write_csv(args.output_dir / "cgdc_three_seed_summary.csv", summary_rows)
    write_json(
        args.output_dir / "cgdc_three_seed_summary.json",
        {configuration: artifact["summary"] for configuration, artifact in loaded.items()},
    )

    reference = loaded[args.reference]
    delta_rows: list[dict[str, object]] = []
    for configuration, artifact in loaded.items():
        if configuration == args.reference:
            continue
        for metric in metric_names:
            baseline = reference["summary"][metric]
            comparison = artifact["summary"][metric]
            if baseline["mean"] is None or comparison["mean"] is None:
                continue
            direction = FAVORABLE_DIRECTIONS.get(metric, -1)
            difference = float(comparison["mean"]) - float(baseline["mean"])
            delta_rows.append(
                {
                    "reference": args.reference,
                    "comparison": configuration,
                    "metric": metric,
                    "reference_mean": baseline["mean"],
                    "comparison_mean": comparison["mean"],
                    "difference": difference,
                    "favorable_direction": direction,
                    "oriented_improvement": difference * direction,
                }
            )
        aligned = {
            seed: align_prediction_rows(reference["predictions"][seed], artifact["predictions"][seed])
            for seed in seeds
        }
        write_json(
            args.output_dir / f"paired_{configuration}_vs_{args.reference}.json",
            {
                "classification": paired_two_stage_bootstrap(
                    aligned, args.bootstrap_replicates, args.rng_seed
                )["summary"],
                "routing_regret": paired_routing_regret_bootstrap(
                    reference["predictions"], artifact["predictions"],
                    args.bootstrap_replicates, args.rng_seed,
                ),
                "calibration": paired_calibration_bootstrap(
                    reference["predictions"], artifact["predictions"],
                    args.bootstrap_replicates, args.rng_seed,
                ),
            },
        )
    write_csv(args.output_dir / "cgdc_ablation_deltas.csv", delta_rows)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "training_roots": {name: str(path.resolve()) for name, path in roots.items()},
            "reference": args.reference,
            "seeds": list(seeds),
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
            "calibration_metrics": list(CALIBRATION_METRICS),
        },
    )


if __name__ == "__main__":
    main()
