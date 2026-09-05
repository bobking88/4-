from __future__ import annotations

import argparse
import hashlib
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

from analyze_paired_cluster_statistics import (
    align_prediction_rows, calculate_metrics, paired_two_stage_bootstrap,
)
from generate_phr_screen_decision import verify_screen_decision


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_formal_evidence(
    experiment_root: Path, manifest: Path, screen_decision_path: Path,
) -> dict[str, object]:
    """Validate that formal PHR comparison artifacts descend from a promoted screen."""
    experiment_root, manifest, screen_decision_path = (
        experiment_root.resolve(), manifest.resolve(), screen_decision_path.resolve(),
    )
    if not screen_decision_path.exists():
        raise FileNotFoundError(f"Missing promoted screen decision: {screen_decision_path}")
    decision = json.loads(screen_decision_path.read_text(encoding="utf-8"))
    verify_screen_decision(decision, manifest)
    if not decision.get("promote_to_formal") or decision.get("selected_configuration") != "phr_complete":
        raise ValueError("Formal analysis requires a promoted phr_complete screen decision.")

    registration_path = experiment_root / "registered_configurations.json"
    if not registration_path.exists():
        raise FileNotFoundError(f"Missing formal registration: {registration_path}")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("protocol_version") != "phr-v2" or registration.get("mode") != "formal":
        raise ValueError("Formal registration must use phr-v2 in formal mode.")
    if registration.get("manifest_sha256") != _sha256(manifest):
        raise ValueError("Formal registration manifest hash mismatch.")
    screened_sources = decision["evidence"].get("registration_source_sha256")
    if registration.get("source_sha256") != screened_sources:
        raise ValueError("Formal registration source hashes differ from the promoted validation screen.")

    for config in ("rsg_reference", "phr_complete"):
        for seed in FORMAL_SEEDS:
            run = experiment_root / config / f"seed{seed}"
            status_path, environment_path = run / "run_status.json", run / "environment.json"
            if not status_path.exists() or not environment_path.exists():
                raise FileNotFoundError(f"Incomplete formal provenance: {run}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise ValueError(f"Formal run is not complete: {run}")
            if environment.get("validation_only") is not False or environment.get("smoke_run") is not False:
                raise ValueError(f"Formal evidence must be non-smoke test evaluation: {run}")
    return {
        "registration_path": str(registration_path),
        "registration_sha256": _sha256(registration_path),
        "screen_decision_path": str(screen_decision_path),
        "screen_decision_sha256": _sha256(screen_decision_path),
        "screen_criterion_ids": decision["criterion_ids"],
        "source_sha256": registration["source_sha256"],
    }


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
    raw = calculate_metrics([{**row, "prediction": row["predicted_label"]} for row in rows])
    return {_metric_name(key): value for key, value in raw.items()}


def _metric_name(key: str) -> str:
    return key + "_rate" if key.endswith("_intrusion") else key


def _paired_group_bootstrap(references: Sequence[Sequence[dict[str, str]]], comparisons: Sequence[Sequence[dict[str, str]]], replicates: int, seed: int) -> list[dict[str, object]]:
    if len(references) != len(comparisons) or not references:
        raise ValueError("Aligned formal reference and comparison seed sets are required.")
    aligned = {str(index): align_prediction_rows(list(first), list(second))
               for index, (first, second) in enumerate(zip(references, comparisons))}
    identity = {(row["image_id"], row["true_label"], row["split_group_id"]) for row in aligned["0"]}
    for rows in aligned.values():
        if {(row["image_id"], row["true_label"], row["split_group_id"]) for row in rows} != identity:
            raise ValueError("Seeds must evaluate the same frozen images and groups.")
    result = paired_two_stage_bootstrap(aligned, replicates, seed)
    return [{"metric": _metric_name(metric), **summary} for metric, summary in result["summary"].items()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze registered PHR-HRGV experiments.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--screen-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260905)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    provenance = validate_formal_evidence(
        args.experiment_root, args.manifest, args.screen_decision,
    )
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
            values = [float(item[metric]) for item in artifact["metrics"]]
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Missing/non-finite formal seed metric: {config}/{metric}")
            summary_rows.append({"configuration": config, "metric": metric, "mean": mean(values), "sample_std": stdev(values), **{f"seed_{seed}": value for seed, value in zip(FORMAL_SEEDS, values)}})
        for seed, rows in zip(FORMAL_SEEDS, artifact["predictions"]):
            pair_rows.append({"configuration": config, "seed": seed, **summarize_pairwise_rows(rows)})
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "pairwise_routing_summary.csv", pair_rows)
    bootstrap_rows = _paired_group_bootstrap(artifacts["rsg_reference"]["predictions"], artifacts["phr_complete"]["predictions"], args.bootstrap_replicates, args.rng_seed)
    _write_csv(args.output_dir / "paired_cluster_bootstrap.csv", bootstrap_rows)
    summary = {(row["configuration"], row["metric"]): row["mean"] for row in summary_rows}
    macro = next(row for row in bootstrap_rows if row["metric"] == "macro_f1")
    _write_json(args.output_dir / "analysis.json", {
        "formal_evidence_supports_claim": False,
        "fixed_split_macro_f1_delta": macro["difference"],
        "fixed_split_macro_f1_ci_excludes_zero": macro["ci_low"] > 0,
        "claim_boundary": "Fixed-split results alone do not establish the registered broad claim. Mechanism, source-holdout and backbone portability evidence must be audited separately. Bootstrap intervals describe sampling and seed uncertainty under this protocol, not probability that the theory is true.",
        "formal_seeds": list(FORMAL_SEEDS), "bootstrap_replicates": args.bootstrap_replicates,
        "provenance": provenance,
    })


if __name__ == "__main__":
    main()
