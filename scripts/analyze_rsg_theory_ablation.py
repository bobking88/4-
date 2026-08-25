from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from analyze_paired_cluster_statistics import align_prediction_rows, paired_two_stage_bootstrap
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


def parse_config_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --config-root value must be CONFIGURATION=PATH.")
        configuration, raw_path = value.split("=", 1)
        configuration = configuration.strip()
        raw_path = raw_path.strip()
        if not configuration or not raw_path:
            raise ValueError("Each --config-root value must be CONFIGURATION=PATH.")
        if configuration in roots:
            raise ValueError(f"Found duplicate configuration root: {configuration}")
        roots[configuration] = Path(raw_path)
    if not roots:
        raise ValueError("At least one --config-root value is required.")
    return roots


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the three-seed RSG-HRGV theory ablation across training roots."
    )
    parser.add_argument(
        "--config-root",
        action="append",
        required=True,
        metavar="CONFIGURATION=PATH",
        help="Training root for one configuration; repeat for every configuration.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", default="rsg_complete")
    parser.add_argument("--stage", choices=("formal",), default="formal")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260825)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    roots = parse_config_roots(args.config_root)
    if args.reference not in roots:
        raise ValueError("The reference configuration must have a --config-root entry.")
    seeds = FORMAL_SEEDS
    validate_formal_seeds(seeds)
    loaded = {
        configuration: load_configuration(root, args.stage, configuration, seeds)
        for configuration, root in roots.items()
    }

    summary_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
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
    write_csv(args.output_dir / "rsg_theory_ablation_summary.csv", summary_rows)
    write_json(
        args.output_dir / "rsg_theory_ablation_summary.json",
        {configuration: artifact["summary"] for configuration, artifact in loaded.items()},
    )
    write_csv(args.output_dir / "rsg_theory_ablation_routing_metrics.csv", routing_rows)

    reference = loaded[args.reference]
    delta_rows: list[dict[str, object]] = []
    for configuration, artifact in loaded.items():
        if configuration == args.reference:
            continue
        for metric, comparison_summary in artifact["summary"].items():
            reference_summary = reference["summary"][metric]
            if reference_summary["mean"] is None or comparison_summary["mean"] is None:
                continue
            difference = float(comparison_summary["mean"]) - float(reference_summary["mean"])
            delta_rows.append(
                {
                    "reference": args.reference,
                    "comparison": configuration,
                    "metric": metric,
                    "reference_mean": reference_summary["mean"],
                    "comparison_mean": comparison_summary["mean"],
                    "difference": difference,
                    "favorable_direction": FAVORABLE_DIRECTIONS[metric],
                    "oriented_improvement": difference * FAVORABLE_DIRECTIONS[metric],
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
            reference["predictions"], artifact["predictions"], args.bootstrap_replicates, args.rng_seed
        )
        write_json(
            args.output_dir / f"paired_{configuration}_vs_{args.reference}.json",
            {
                "classification": classification_bootstrap["summary"],
                "routing_regret": routing_bootstrap,
            },
        )
    write_csv(args.output_dir / "rsg_theory_ablation_deltas.csv", delta_rows)
    write_json(
        args.output_dir / "analysis_manifest.json",
        {
            "training_roots": {name: str(path.resolve()) for name, path in roots.items()},
            "stage": args.stage,
            "reference": args.reference,
            "configurations": list(roots),
            "seeds": list(seeds),
            "bootstrap_replicates": args.bootstrap_replicates,
            "rng_seed": args.rng_seed,
            "classification_metrics": list(CLASSIFICATION_METRICS),
            "routing_metrics": list(ROUTING_METRICS),
        },
    )


if __name__ == "__main__":
    main()
