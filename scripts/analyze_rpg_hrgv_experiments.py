from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from analyze_cgdc_rsg_experiments import (
    CALIBRATION_METRICS,
    load_cgdc_configuration,
    paired_calibration_bootstrap,
    parse_config_roots,
)
from analyze_paired_cluster_statistics import align_prediction_rows, paired_two_stage_bootstrap
from analyze_rsg_hrgv_experiment import (
    CLASSIFICATION_METRICS,
    FAVORABLE_DIRECTIONS,
    FORMAL_SEEDS,
    ROUTING_METRICS,
    paired_routing_regret_bootstrap,
    validate_formal_seeds,
    write_csv,
    write_json,
)


REQUIRED_CONFIGURATIONS = (
    "rsg_complete",
    "rpg_complete",
    "rpg_without_within",
    "rpg_without_between",
    "rpg_total_entropy_only",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the registered RPG-HRGV matrix.")
    parser.add_argument("--config-root", action="append", required=True, metavar="CONFIGURATION=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260827)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    roots = parse_config_roots(args.config_root)
    if set(roots) != set(REQUIRED_CONFIGURATIONS):
        missing = sorted(set(REQUIRED_CONFIGURATIONS) - set(roots))
        extra = sorted(set(roots) - set(REQUIRED_CONFIGURATIONS))
        raise ValueError(f"RPG analysis requires the registered configurations; missing={missing}, extra={extra}.")
    seeds = FORMAL_SEEDS
    validate_formal_seeds(seeds)
    loaded = {
        configuration: load_cgdc_configuration(roots[configuration], configuration, seeds)
        for configuration in REQUIRED_CONFIGURATIONS
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
    write_csv(args.output_dir / "rpg_three_seed_summary.csv", summary_rows)
    write_json(
        args.output_dir / "rpg_three_seed_summary.json",
        {configuration: artifact["summary"] for configuration, artifact in loaded.items()},
    )

    reference = loaded["rsg_complete"]
    delta_rows: list[dict[str, object]] = []
    for configuration in REQUIRED_CONFIGURATIONS[1:]:
        artifact = loaded[configuration]
        for metric in metric_names:
            baseline = reference["summary"][metric]
            comparison = artifact["summary"][metric]
            if baseline["mean"] is None or comparison["mean"] is None:
                continue
            difference = float(comparison["mean"]) - float(baseline["mean"])
            direction = FAVORABLE_DIRECTIONS.get(metric, -1)
            delta_rows.append(
                {
                    "reference": "rsg_complete",
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
            args.output_dir / f"paired_{configuration}_vs_rsg_complete.json",
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
    write_csv(args.output_dir / "rpg_ablation_deltas.csv", delta_rows)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "training_roots": {name: str(path.resolve()) for name, path in roots.items()},
            "reference": "rsg_complete",
            "seeds": list(seeds),
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
            "required_configurations": list(REQUIRED_CONFIGURATIONS),
            "calibration_metrics": list(CALIBRATION_METRICS),
        },
    )


if __name__ == "__main__":
    main()
