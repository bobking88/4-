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
    "mrpg_complete",
    "mrpg_unconstrained_between",
    "mrpg_without_between",
)
MRPG_CONFIGURATIONS = REQUIRED_CONFIGURATIONS[2:]
REFERENCE_CONFIGURATIONS = REQUIRED_CONFIGURATIONS[:2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the registered M-RPG-HRGV matrix.")
    parser.add_argument("--config-root", action="append", required=True, metavar="CONFIGURATION=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260828)
    return parser.parse_args(argv)


def _write_pairwise_analysis(
    output_dir: Path,
    reference_name: str,
    comparison_name: str,
    reference: dict[str, object],
    comparison: dict[str, object],
    seeds: Sequence[str],
    replicates: int,
    rng_seed: int,
) -> None:
    aligned = {
        seed: align_prediction_rows(reference["predictions"][seed], comparison["predictions"][seed])
        for seed in seeds
    }
    write_json(
        output_dir / f"paired_{comparison_name}_vs_{reference_name}.json",
        {
            "classification": paired_two_stage_bootstrap(aligned, replicates, rng_seed)["summary"],
            "routing_regret": paired_routing_regret_bootstrap(
                reference["predictions"], comparison["predictions"], replicates, rng_seed
            ),
            "calibration": paired_calibration_bootstrap(
                reference["predictions"], comparison["predictions"], replicates, rng_seed
            ),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    roots = parse_config_roots(args.config_root)
    if set(roots) != set(REQUIRED_CONFIGURATIONS):
        missing = sorted(set(REQUIRED_CONFIGURATIONS) - set(roots))
        extra = sorted(set(roots) - set(REQUIRED_CONFIGURATIONS))
        raise ValueError(f"M-RPG analysis requires the registered configurations; missing={missing}, extra={extra}.")
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
    write_csv(args.output_dir / "mrpg_three_seed_summary.csv", summary_rows)
    write_json(
        args.output_dir / "mrpg_three_seed_summary.json",
        {configuration: artifact["summary"] for configuration, artifact in loaded.items()},
    )

    delta_rows: list[dict[str, object]] = []
    for reference_name in REFERENCE_CONFIGURATIONS:
        reference = loaded[reference_name]
        for comparison_name in MRPG_CONFIGURATIONS:
            comparison = loaded[comparison_name]
            for metric in metric_names:
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
            _write_pairwise_analysis(
                args.output_dir,
                reference_name,
                comparison_name,
                reference,
                comparison,
                seeds,
                args.bootstrap_replicates,
                args.rng_seed,
            )
    write_csv(args.output_dir / "mrpg_ablation_deltas.csv", delta_rows)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "training_roots": {name: str(path.resolve()) for name, path in roots.items()},
            "references": list(REFERENCE_CONFIGURATIONS),
            "seeds": list(seeds),
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
            "required_configurations": list(REQUIRED_CONFIGURATIONS),
            "calibration_metrics": list(CALIBRATION_METRICS),
        },
    )


if __name__ == "__main__":
    main()
