"""Run the preregistered independent experts for OOS-RSG confirmation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CONFIRMATION_EXPERT_SEEDS = (20260924, 20260925, 20260926)


@dataclass(frozen=True)
class ExpertCommand:
    seed: int
    output_dir: Path
    arguments: tuple[str, ...]


def build_expert_commands(
    project_root: Path,
    manifest: Path,
    dataset_root: Path,
    output_root: Path,
    python_executable: Path,
    torch_home: Path,
    device: str,
) -> list[ExpertCommand]:
    training_script = project_root / "scripts" / "train_hrgv_mineral_classifier.py"
    commands: list[ExpertCommand] = []
    for seed in CONFIRMATION_EXPERT_SEEDS:
        output_dir = output_root / "experts" / f"seed{seed}"
        arguments = (
            str(python_executable),
            str(training_script),
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--device",
            device,
            "--seed",
            str(seed),
            "--epochs",
            "30",
            "--patience",
            "8",
            "--batch-size",
            "16",
            "--num-workers",
            "0",
            "--lambda-gate-regret",
            "0.0",
            "--torch-home",
            str(torch_home),
            "--validation-only",
        )
        commands.append(ExpertCommand(seed, output_dir, arguments))
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train three locked independent experts for OOS-RSG confirmation."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "outputs/training/oos_rsg_confirmation_manifests_v1/expert.csv",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "outputs/training/oos_rsg_confirmation_v1",
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--torch-home", type=Path, default=root / ".torch-cache")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run and args.execute:
        raise ValueError("Choose either --dry-run or --execute, not both.")
    commands = build_expert_commands(
        project_root=args.project_root,
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        python_executable=args.python_executable,
        torch_home=args.torch_home,
        device=args.device,
    )
    for command in commands:
        if args.dry_run or not args.execute:
            print(subprocess.list2cmdline(list(command.arguments)))
            continue
        completion_marker = command.output_dir / "val_metrics.json"
        if completion_marker.exists() and not args.force:
            print(f"SKIP complete expert: {command.output_dir}", flush=True)
            continue
        if command.output_dir.exists() and not args.force:
            raise RuntimeError(
                f"Incomplete output exists; inspect before rerun: {command.output_dir}"
            )
        print(f"RUN independent expert seed={command.seed}", flush=True)
        subprocess.run(command.arguments, cwd=args.project_root, check=True)


if __name__ == "__main__":
    main()
