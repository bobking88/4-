from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FORMAL_SEEDS = (20260727, 20260728, 20260729)
PILOT_SEEDS = (20260728,)
RSG_FLAGS = (
    "--verifier-mode", "residual", "--lambda-gate-regret", "0.1",
    "--gate-regret-temperature", "0.2", "--gate-gap-temperature", "0.5",
    "--detach-gate-features",
)
CONFIGURATION_FLAGS = {
    "mrpg_complete": (*RSG_FLAGS, "--enable-mrpg"),
    "mrpg_unconstrained_between": (
        *RSG_FLAGS, "--enable-mrpg", "--mrpg-between-mode", "unconstrained",
    ),
    "mrpg_without_between": (
        *RSG_FLAGS, "--enable-mrpg", "--mrpg-between-mode", "disabled",
    ),
}


@dataclass(frozen=True)
class ExperimentCommand:
    configuration: str
    seed: int
    output_dir: Path
    arguments: tuple[str, ...]


def build_experiment_commands(
    project_root: Path,
    manifest: Path,
    dataset_root: Path,
    output_root: Path,
    python_executable: Path,
    device: str,
    torch_home: Path,
    stage: str,
) -> list[ExperimentCommand]:
    if stage not in {"pilot", "formal"}:
        raise ValueError("stage must be pilot or formal.")
    seeds = PILOT_SEEDS if stage == "pilot" else FORMAL_SEEDS
    training_script = project_root / "scripts" / "train_hrgv_mineral_classifier.py"
    commands: list[ExperimentCommand] = []
    for configuration, extra_flags in CONFIGURATION_FLAGS.items():
        for seed in seeds:
            output_dir = output_root / f"{stage}_{configuration}_seed{seed}"
            arguments = (
                str(python_executable), str(training_script),
                "--manifest", str(manifest),
                "--dataset-root", str(dataset_root),
                "--output-dir", str(output_dir),
                "--seed", str(seed),
                "--device", device,
                "--torch-home", str(torch_home),
                "--couple-verifier-features",
                *extra_flags,
            )
            commands.append(ExperimentCommand(configuration, seed, output_dir, arguments))
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the registered M-RPG-HRGV experiment matrix.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "training" / "mrpg_formal",
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".torch-cache",
    )
    parser.add_argument("--stage", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--config", action="append", choices=tuple(CONFIGURATION_FLAGS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run and args.execute:
        raise ValueError("Choose either --dry-run or --execute, not both.")
    commands = build_experiment_commands(
        args.project_root,
        args.manifest,
        args.dataset_root,
        args.output_root,
        args.python_executable,
        args.device,
        args.torch_home,
        args.stage,
    )
    selected = set(args.config or CONFIGURATION_FLAGS)
    for command in commands:
        if command.configuration not in selected:
            continue
        if args.dry_run or not args.execute:
            print(subprocess.list2cmdline(list(command.arguments)))
            continue
        metrics_path = command.output_dir / "test_metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"SKIP complete run: {command.output_dir}", flush=True)
            continue
        print(f"RUN configuration={command.configuration} seed={command.seed}", flush=True)
        subprocess.run(command.arguments, cwd=args.project_root, check=True)


if __name__ == "__main__":
    main()
