from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence


FORMAL_SEEDS = ("20260727", "20260728", "20260729")
CLASSIFICATION_METRICS = ("accuracy", "macro_f1", "target_recall", "ti_to_target_intrusion_rate", "metallic_to_target_intrusion_rate")


def _bool(value: str) -> bool:
    return value in {"1", "true", "True"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize_pairwise_rows(rows: Sequence[dict[str, str]]) -> dict[str, float | int | None]:
    if not rows:
        raise ValueError("Prediction rows must not be empty.")
    result: dict[str, float | int | None] = {}
    for edge in ("ti", "metallic"):
        prefix = f"phr_{edge}"
        eligible = [row for row in rows if _bool(row[f"{prefix}_eligible"])]
        sign_agree = [row for row in eligible if _bool(row[f"{prefix}_expert_sign_agreement"])]
        result[f"{prefix}_eligible_count"] = len(eligible)
        result[f"{prefix}_gate_selection_accuracy"] = (
            mean(_bool(row[f"{prefix}_gate_selection_correct"]) for row in eligible) if eligible else None
        )
        result[f"{prefix}_mean_margin_regret"] = (
            mean(float(row[f"{prefix}_margin_regret"]) for row in eligible) if eligible else None
        )
        result[f"{prefix}_sign_agreement_count"] = len(sign_agree)
        result[f"{prefix}_sign_preservation_rate"] = (
            mean(_bool(row[f"{prefix}_sign_preserved"]) for row in sign_agree) if sign_agree else None
        )
    return result


def _risk_metrics(rows: Sequence[dict[str, str]]) -> dict[str, float | None]:
    count = len(rows)
    correct = mean(row["true_label"] == row["predicted_label"] for row in rows)
    labels = sorted({row["true_label"] for row in rows})
    recalls = [
        mean(row["predicted_label"] == label for row in rows if row["true_label"] == label)
        for label in labels
    ]
    def rate(source: str) -> float | None:
        subset = [row for row in rows if row["true_label"] == source]
        return mean(row["predicted_label"] == "target_mineral" for row in subset) if subset else None
    target = [row for row in rows if row["true_label"] == "target_mineral"]
    return {
        "accuracy": correct, "macro_f1_proxy": mean(recalls),
        "target_recall": mean(row["predicted_label"] == "target_mineral" for row in target) if target else None,
        "ti_to_target_intrusion_rate": rate("ti_bearing_negative"),
        "metallic_to_target_intrusion_rate": rate("metallic_hard_negative"),
        "count": count,
    }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))]


def _paired_group_bootstrap(references: Sequence[Sequence[dict[str, str]]], comparisons: Sequence[Sequence[dict[str, str]]], replicates: int, seed: int) -> list[dict[str, object]]:
    if len(references) != len(comparisons) or not references:
        raise ValueError("Aligned formal reference and comparison seed sets are required.")
    grouped_seeds = []
    for reference, comparison in zip(references, comparisons):
        by_id = {row["image_id"]: row for row in comparison}
        if set(by_id) != {row["image_id"] for row in reference}:
            raise ValueError("Prediction image IDs do not match.")
        groups: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
        for row in reference:
            other = by_id[row["image_id"]]
            if row["true_label"] != other["true_label"]:
                raise ValueError("Prediction true labels do not match.")
            groups.setdefault(row["split_group_id"], []).append((row, other))
        grouped_seeds.append(groups)
    rng, samples = random.Random(seed), []
    for _ in range(replicates):
        selected_seeds = rng.choices(grouped_seeds, k=len(grouped_seeds))
        pairs = []
        for groups in selected_seeds:
            group_ids = list(groups)
            pairs.extend(pair for group in rng.choices(group_ids, k=len(group_ids)) for pair in groups[group])
        ref, comp = zip(*pairs)
        ref_metrics, comp_metrics = _risk_metrics(ref), _risk_metrics(comp)
        samples.append({key: (comp_metrics[key] - ref_metrics[key]) if ref_metrics[key] is not None and comp_metrics[key] is not None else None for key in ("accuracy", "target_recall", "ti_to_target_intrusion_rate", "metallic_to_target_intrusion_rate")})
    rows = []
    favorable = {"accuracy": 1, "target_recall": 1, "ti_to_target_intrusion_rate": -1, "metallic_to_target_intrusion_rate": -1}
    for metric, direction in favorable.items():
        values = [sample[metric] for sample in samples if sample[metric] is not None]
        rows.append({"metric": metric, "difference": mean(values), "ci_low": _quantile(values, .025), "ci_high": _quantile(values, .975), "probability_favorable": mean(value * direction > 0 for value in values), "bootstrap_replicates": replicates})
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze registered PHR-HRGV experiments.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260905)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    artifacts: dict[str, dict[str, list[object]]] = {name: {"metrics": [], "predictions": []} for name in ("rsg_reference", "phr_complete")}
    for config in artifacts:
        for seed in FORMAL_SEEDS:
            run = args.experiment_root / config / f"seed{seed}"
            metrics_path, predictions_path = run / "test_metrics.json", run / "test_predictions.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                raise FileNotFoundError(f"Incomplete formal run: {run}")
            artifacts[config]["metrics"].append(json.loads(metrics_path.read_text(encoding="utf-8")))
            artifacts[config]["predictions"].append(_read_csv(predictions_path))
    summary_rows, pair_rows = [], []
    for config, artifact in artifacts.items():
        for metric in CLASSIFICATION_METRICS:
            values = [float(item[metric]) for item in artifact["metrics"] if not math.isnan(float(item[metric]))]
            summary_rows.append({"configuration": config, "metric": metric, "mean": mean(values), "sample_std": stdev(values), **{f"seed_{seed}": value for seed, value in zip(FORMAL_SEEDS, values)}})
        for seed, rows in zip(FORMAL_SEEDS, artifact["predictions"]):
            pair_rows.append({"configuration": config, "seed": seed, **summarize_pairwise_rows(rows)})
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "pairwise_routing_summary.csv", pair_rows)
    bootstrap_rows = _paired_group_bootstrap(artifacts["rsg_reference"]["predictions"], artifacts["phr_complete"]["predictions"], args.bootstrap_replicates, args.rng_seed)
    _write_csv(args.output_dir / "paired_cluster_bootstrap.csv", bootstrap_rows)
    summary = {(row["configuration"], row["metric"]): row["mean"] for row in summary_rows}
    favorable = summary[("phr_complete", "macro_f1")] >= summary[("rsg_reference", "macro_f1")]
    _write_json(args.output_dir / "analysis.json", {"formal_evidence_supports_claim": favorable, "criterion": "PHR mean Macro F1 is not lower than RSG reference over the registered seeds.", "claim_boundary": None if favorable else "Registered formal comparison did not support an overall Macro F1 improvement; report PHR as a negative or boundary result.", "formal_seeds": list(FORMAL_SEEDS), "bootstrap_replicates": args.bootstrap_replicates})


if __name__ == "__main__":
    main()
