"""Make an immutable, validation-only promotion decision for the PHR experiment plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from run_phr_hrgv_experiments import SCREEN_CONFIGURATION_FLAGS, SCREEN_SEEDS


REQUIRED_METRICS = (
    "macro_f1", "accuracy", "target_recall", "ti_to_target_intrusion_rate",
    "metallic_to_target_intrusion_rate", "phr_ti_mean_margin_regret",
    "phr_metallic_mean_margin_regret",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _finite_metrics(metrics: dict[str, Any], path: Path) -> dict[str, float]:
    result = {}
    for name in REQUIRED_METRICS:
        if name not in metrics or not math.isfinite(float(metrics[name])):
            raise ValueError(f"Missing/non-finite validation metric {name}: {path}")
        result[name] = float(metrics[name])
    return result


def _collect_evidence(screen_root: Path, manifest: Path) -> dict[str, Any]:
    registration_path = screen_root / "registered_configurations.json"
    registration = _read_json(registration_path)
    if registration.get("protocol_version") != "phr-v2" or registration.get("mode") != "screen":
        raise ValueError("Screen registration must be phr-v2 and mode=screen.")
    manifest_hash = _sha256(manifest)
    if registration.get("manifest_sha256") != manifest_hash:
        raise ValueError("Screen registration manifest hash mismatch.")
    evidence: dict[str, Any] = {
        "registration_path": str(registration_path.resolve()),
        "registration_sha256": _sha256(registration_path),
        "registration_source_sha256": registration.get("source_sha256"),
        "runs": {},
    }
    for configuration in SCREEN_CONFIGURATION_FLAGS:
        for seed in SCREEN_SEEDS:
            run = screen_root / configuration / f"seed{seed}"
            status_path = run / "run_status.json"
            environment_path = run / "environment.json"
            metrics_path = run / "val_metrics.json"
            best_metrics_path = run / "best_validation_metrics.json"
            predictions_path = run / "val_predictions.csv"
            for path in (status_path, environment_path, metrics_path, best_metrics_path, predictions_path):
                if not path.exists():
                    raise FileNotFoundError(f"Incomplete validation screen artifact: {path}")
            status = _read_json(status_path)
            environment = _read_json(environment_path)
            best_metrics = _read_json(best_metrics_path)
            metrics = _finite_metrics(_read_json(metrics_path), metrics_path)
            if status.get("status") != "complete":
                raise ValueError(f"Screen run is not complete: {run}")
            if environment.get("validation_only") is not True or environment.get("smoke_run") is not False:
                raise ValueError(f"Screen evidence must be non-smoke validation-only: {run}")
            if best_metrics.get("selection_split") != "val" or best_metrics.get("smoke_run") is not False:
                raise ValueError(f"Best validation evidence is malformed: {run}")
            evidence["runs"][f"{configuration}/seed{seed}"] = {
                "metrics": metrics,
                "artifacts": {
                    "run_status": _sha256(status_path),
                    "environment": _sha256(environment_path),
                    "val_metrics": _sha256(metrics_path),
                    "best_validation_metrics": _sha256(best_metrics_path),
                    "val_predictions": _sha256(predictions_path),
                },
            }
    return evidence


def _criteria(reference: dict[str, float], comparison: dict[str, float]) -> list[str]:
    f1_gain = comparison["macro_f1"] - reference["macro_f1"]
    ti_not_worse = comparison["ti_to_target_intrusion_rate"] <= reference["ti_to_target_intrusion_rate"]
    metallic_not_worse = comparison["metallic_to_target_intrusion_rate"] <= reference["metallic_to_target_intrusion_rate"]
    c1 = f1_gain >= .005 and (ti_not_worse or metallic_not_worse)
    c2 = (
        reference["phr_ti_mean_margin_regret"] > 0
        and reference["phr_metallic_mean_margin_regret"] > 0
        and comparison["phr_ti_mean_margin_regret"] <= .90 * reference["phr_ti_mean_margin_regret"]
        and comparison["phr_metallic_mean_margin_regret"] <= .90 * reference["phr_metallic_mean_margin_regret"]
        and comparison["target_recall"] >= reference["target_recall"] - .003
    )
    c3 = (
        abs(comparison["accuracy"] - reference["accuracy"]) <= .003
        and comparison["target_recall"] > reference["target_recall"]
        and (
            comparison["ti_to_target_intrusion_rate"] < reference["ti_to_target_intrusion_rate"]
            or comparison["metallic_to_target_intrusion_rate"] < reference["metallic_to_target_intrusion_rate"]
        )
    )
    return [name for name, passed in (("C1_macro_f1", c1), ("C2_pair_regret", c2), ("C3_risk_tradeoff", c3)) if passed]


def build_screen_decision(screen_root: Path, manifest: Path) -> dict[str, Any]:
    screen_root, manifest = screen_root.resolve(), manifest.resolve()
    evidence = _collect_evidence(screen_root, manifest)
    reference = evidence["runs"][f"rsg_reference/seed{SCREEN_SEEDS[0]}"]["metrics"]
    comparison = evidence["runs"][f"phr_complete/seed{SCREEN_SEEDS[0]}"]["metrics"]
    criterion_ids = _criteria(reference, comparison)
    return {
        "protocol_version": "phr-v2",
        "selection_split": "val",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "screen_root": str(screen_root),
        "screen_seed": SCREEN_SEEDS[0],
        "comparison": {"reference": "rsg_reference", "candidate": "phr_complete"},
        "criterion_ids": criterion_ids,
        "promote_to_formal": bool(criterion_ids),
        "selected_configuration": "phr_complete" if criterion_ids else None,
        "evidence": evidence,
        "evidence_sha256": _canonical_sha256(evidence),
        "claim_boundary": "This is a validation-only promotion decision, not evidence of final test performance or a general theoretical advantage.",
    }


def verify_screen_decision(decision: dict[str, Any], manifest: Path) -> bool:
    if decision.get("protocol_version") != "phr-v2" or decision.get("selection_split") != "val":
        raise ValueError("Decision must be a phr-v2 validation-only decision.")
    if decision.get("manifest_sha256") != _sha256(manifest):
        raise ValueError("Decision manifest hash mismatch.")
    evidence = _collect_evidence(Path(decision["screen_root"]), manifest)
    if decision.get("evidence") != evidence or decision.get("evidence_sha256") != _canonical_sha256(evidence):
        raise ValueError("Decision validation evidence no longer matches registered artifacts.")
    expected = _criteria(
        evidence["runs"][f"rsg_reference/seed{SCREEN_SEEDS[0]}"]["metrics"],
        evidence["runs"][f"phr_complete/seed{SCREEN_SEEDS[0]}"]["metrics"],
    )
    if decision.get("criterion_ids") != expected:
        raise ValueError("Decision criteria do not match validation evidence.")
    promoted = bool(expected)
    if bool(decision.get("promote_to_formal")) != promoted:
        raise ValueError("Decision promotion flag does not match the registered criteria.")
    if decision.get("selected_configuration") != ("phr_complete" if promoted else None):
        raise ValueError("Decision selected configuration does not match the registered criteria.")
    return True


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    decision = build_screen_decision(args.screen_root, args.manifest)
    decision["created_at"] = datetime.now().isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
