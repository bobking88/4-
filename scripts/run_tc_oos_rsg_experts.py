"""Train and lock fold-matched TC-OOS-RSG experts and M0 baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_VERSION = "tc_oos_rsg_v1"
BASE_SEED = 20260927


@dataclass(frozen=True)
class TrainingCommand:
    name: str
    output_dir: Path
    arguments: tuple[str, ...]
    configuration: dict[str, object]


def _value(record: object, field: str) -> str:
    if isinstance(record, Mapping):
        value = record.get(field, "")
    else:
        value = getattr(record, field, "")
    return str(value).strip()


def _unique_values(records: Sequence[object], field: str, subset: str) -> set[str]:
    values = [_value(record, field) for record in records]
    if not values or any(not value for value in values):
        raise ValueError(f"{subset} must contain nonempty {field} values.")
    if len(values) != len(set(values)):
        raise ValueError(f"{subset} contains duplicate {field} values.")
    return set(values)


def _present_values(records: Sequence[object], field: str, subset: str) -> set[str]:
    values = [_value(record, field) for record in records]
    if not values or any(not value for value in values):
        raise ValueError(f"{subset} must contain nonempty {field} values.")
    return set(values)


def _species_role_mapping(records: Sequence[object], subset: str) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for record in records:
        species = _value(record, "mineral_label")
        pair = (
            _value(record, "four_class_label"),
            _value(record, "four_class_id"),
        )
        if not species or not all(pair):
            raise ValueError(f"{subset} has an incomplete species-role mapping.")
        previous = mapping.setdefault(species, pair)
        if previous != pair:
            raise ValueError(
                f"{subset} has an inconsistent species-role mapping for {species}."
            )
    return mapping


def audit_expert_manifests(
    fit_records: Sequence[object],
    stop_records: Sequence[object],
    outer_eval_records: Sequence[object],
) -> dict[str, object]:
    """Validate selection isolation without opening any referenced image."""
    named = {
        "expert_fit": list(fit_records),
        "expert_stop": list(stop_records),
        "outer_eval": list(outer_eval_records),
    }
    image_ids = {
        name: _unique_values(records, "image_id", name)
        for name, records in named.items()
    }
    groups = {
        name: _present_values(records, "split_group_id", name)
        for name, records in named.items()
    }
    names = tuple(named)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            image_overlap = image_ids[left] & image_ids[right]
            if image_overlap:
                raise ValueError(
                    f"image overlap between {left} and {right}: "
                    f"{sorted(image_overlap)[:3]}"
                )
            group_overlap = groups[left] & groups[right]
            if group_overlap:
                raise ValueError(
                    f"group overlap between {left} and {right}: "
                    f"{sorted(group_overlap)[:3]}"
                )

    fit_mapping = _species_role_mapping(named["expert_fit"], "expert_fit")
    for subset in ("expert_stop", "outer_eval"):
        candidate_mapping = _species_role_mapping(named[subset], subset)
        for species, pair in candidate_mapping.items():
            if species not in fit_mapping or fit_mapping[species] != pair:
                raise ValueError(
                    f"species-role mapping mismatch for {species} in {subset}."
                )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "row_counts": {name: len(records) for name, records in named.items()},
        "group_counts": {name: len(groups[name]) for name in names},
        "species_count": len(fit_mapping),
        "species_role_mapping": {
            species: {"four_class_label": pair[0], "four_class_id": pair[1]}
            for species, pair in sorted(fit_mapping.items())
        },
        "image_overlap_count": 0,
        "group_overlap_count": 0,
        "outer_images_loaded": False,
    }


def build_training_commands(
    project_root: Path,
    expert_manifest: Path,
    baseline_manifest: Path,
    dataset_root: Path,
    output_root: Path,
    python_executable: Path,
    torch_home: Path,
    fold: int,
    device: str = "auto",
    epochs: int = 30,
    batch_size: int = 16,
    num_workers: int = 0,
    smoke_run: bool = False,
) -> dict[str, TrainingCommand]:
    seed = BASE_SEED + fold
    shared = {
        "seed": seed,
        "pretrained": not smoke_run,
        "epochs": 1 if smoke_run else epochs,
        "batch_size": min(batch_size, 4) if smoke_run else batch_size,
        "num_workers": 0 if smoke_run else num_workers,
        "selection_split": "expert_stop",
        "outer_eval_used_for_selection": False,
        "protocol_version": PROTOCOL_VERSION,
    }

    expert_output = output_root / "expert"
    expert_args = [
        str(python_executable),
        str(project_root / "scripts" / "train_hrgv_mineral_classifier.py"),
        "--manifest",
        str(expert_manifest),
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(expert_output),
        "--backbone",
        "efficientnet_b0",
        "--device",
        device,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--patience",
        "8",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--lambda-gate-regret",
        "0.0",
        "--torch-home",
        str(torch_home),
        "--validation-only",
    ]
    if smoke_run:
        expert_args.append("--smoke-run")

    baseline_output = output_root / "baseline"
    baseline_args = [
        str(python_executable),
        str(project_root / "scripts" / "train_mineral_classifier.py"),
        "--manifest",
        str(baseline_manifest),
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(baseline_output),
        "--model",
        "efficientnet_b0",
        "--loss",
        "cross_entropy",
        "--device",
        device,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--patience",
        "8",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--torch-home",
        str(torch_home),
    ]
    if smoke_run:
        baseline_args.append("--smoke-run")

    return {
        "expert": TrainingCommand(
            name="expert",
            output_dir=expert_output,
            arguments=tuple(expert_args),
            configuration={
                **shared,
                "model": "hrgv",
                "backbone": "efficientnet_b0",
                "verifier_mode": "residual",
                "lambda_gate_regret": 0.0,
                "validation_only": True,
            },
        ),
        "baseline": TrainingCommand(
            name="baseline",
            output_dir=baseline_output,
            arguments=tuple(baseline_args),
            configuration={
                **shared,
                "model": "efficientnet_b0",
                "loss": "cross_entropy",
                "validation_only": False,
                "test_split_is_selection_duplicate": True,
            },
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_csv(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_training_manifests(
    fit_path: Path,
    stop_path: Path,
    outer_path: Path,
    output_root: Path,
) -> dict[str, object]:
    fit, fit_fields = _read_csv(fit_path)
    stop, stop_fields = _read_csv(stop_path)
    outer, _ = _read_csv(outer_path)
    audit = audit_expert_manifests(fit, stop, outer)
    fields = list(dict.fromkeys([*fit_fields, *stop_fields]))
    if "split" not in fields:
        fields.append("split")

    expert_manifest = output_root / "manifests" / "expert_training.csv"
    baseline_manifest = output_root / "manifests" / "baseline_training.csv"
    expert_rows = [
        {**row, "split": "train"} for row in fit
    ] + [{**row, "split": "val"} for row in stop]
    baseline_rows = [
        {**row, "split": "train"} for row in fit
    ] + [{**row, "split": "val"} for row in stop] + [
        {**row, "split": "test"} for row in stop
    ]
    _write_csv(expert_manifest, expert_rows, fields)
    _write_csv(baseline_manifest, baseline_rows, fields)
    audit_path = output_root / "manifest_audit.json"
    _write_json(audit_path, audit)
    return {
        "expert_manifest": expert_manifest,
        "baseline_manifest": baseline_manifest,
        "audit_path": audit_path,
        "audit": audit,
    }


def write_selection_lock(
    output_dir: Path,
    source_manifests: Mapping[str, Path],
    config_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
) -> dict[str, object]:
    required = [*source_manifests.values(), config_path, checkpoint_path, metrics_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot lock missing artifacts: {missing}")
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "selected_before_outer_evaluation": True,
        "outer_images_loaded": False,
        "manifest_sha256": {
            name: _sha256(path) for name, path in sorted(source_manifests.items())
        },
        "configuration_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best_validation_metrics_sha256": _sha256(metrics_path),
    }
    _write_json(output_dir / "selection_lock.json", lock)
    return lock


def write_baseline_best_validation_metrics(output_dir: Path) -> dict[str, object]:
    history_path = output_dir / "metrics_history.csv"
    rows, _ = _read_csv(history_path)
    if not rows:
        raise ValueError("Baseline metrics_history.csv is empty.")
    best = max(
        enumerate(rows),
        key=lambda item: (float(item[1]["macro_f1"]), -item[0]),
    )[1]
    metrics: dict[str, object] = {
        key: float(value)
        for key, value in best.items()
        if key != "epoch" and value not in (None, "")
    }
    metrics.update(
        {
            "epoch": int(best["epoch"]),
            "selection_split": "val",
            "source": "metrics_history.csv",
        }
    )
    target = output_dir / "best_validation_metrics.json"
    _write_json(target, metrics)
    return metrics


def _run_command(command: TrainingCommand, project_root: Path, force: bool) -> None:
    marker = command.output_dir / "best_model.pt"
    if marker.exists() and not force:
        raise RuntimeError(f"Locked training output already exists: {command.output_dir}")
    if command.output_dir.exists() and any(command.output_dir.iterdir()) and not force:
        raise RuntimeError(f"Incomplete output exists: {command.output_dir}")
    command.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(command.output_dir / "run_config.json", command.configuration)
    _write_json(
        command.output_dir / "command.json",
        {"arguments": list(command.arguments)},
    )
    subprocess.run(command.arguments, cwd=project_root, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train one locked TC-OOS-RSG expert and M0 baseline for a fold."
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=root / "outputs" / "training" / "tc_oos_rsg_manifests_v1",
    )
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs" / "training" / "tc_oos_rsg_v1",
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--torch-home", type=Path, default=root / ".torch-cache")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.execute and args.dry_run:
        raise ValueError("Choose either --execute or --dry-run, not both.")
    fold_dir = args.protocol_dir / f"fold_{args.fold}"
    run_root = args.output_root / f"fold_{args.fold}"
    sources = {
        "expert_fit": fold_dir / "expert_fit.csv",
        "expert_stop": fold_dir / "expert_stop.csv",
        "outer_eval": fold_dir / "outer_eval.csv",
    }
    prepared = prepare_training_manifests(
        sources["expert_fit"],
        sources["expert_stop"],
        sources["outer_eval"],
        run_root,
    )
    commands = build_training_commands(
        project_root=args.project_root,
        expert_manifest=prepared["expert_manifest"],
        baseline_manifest=prepared["baseline_manifest"],
        dataset_root=args.dataset_root,
        output_root=run_root,
        python_executable=args.python_executable,
        torch_home=args.torch_home,
        fold=args.fold,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smoke_run=args.smoke_run,
    )
    summary: dict[str, object] = {
        "fold": args.fold,
        "protocol_version": PROTOCOL_VERSION,
        "audit": prepared["audit"],
        "outer_images_loaded": False,
        "commands": {name: list(command.arguments) for name, command in commands.items()},
    }
    if not args.execute:
        for command in commands.values():
            print(subprocess.list2cmdline(list(command.arguments)))
        _write_json(run_root / "dry_run_summary.json", summary)
        return summary

    for command in commands.values():
        _run_command(command, args.project_root, args.force)
        if command.name == "baseline":
            write_baseline_best_validation_metrics(command.output_dir)
        source_manifests = {
            **sources,
            "training_manifest": (
                prepared["expert_manifest"]
                if command.name == "expert"
                else prepared["baseline_manifest"]
            ),
        }
        write_selection_lock(
            output_dir=command.output_dir,
            source_manifests=source_manifests,
            config_path=command.output_dir / "run_config.json",
            checkpoint_path=command.output_dir / "best_model.pt",
            metrics_path=command.output_dir / "best_validation_metrics.json",
        )
    summary["completed"] = True
    _write_json(run_root / "run_summary.json", summary)
    return summary


if __name__ == "__main__":
    main()
