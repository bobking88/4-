from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FORMAL_SEEDS = (20260727, 20260728, 20260729)
PILOT_SEEDS = (20260728,)
RSG_COMMON_FLAGS = (
    "--verifier-mode",
    "residual",
    "--lambda-gate-regret",
    "0.1",
    "--gate-regret-temperature",
    "0.2",
    "--gate-gap-temperature",
    "0.5",
)
CONFIGURATION_FLAGS = {
    "hrgv_reference": ("--verifier-mode", "residual", "--disable-gate-regret"),
    "rsg_complete": (*RSG_COMMON_FLAGS, "--detach-gate-features"),
    "rsg_hard_target": (*RSG_COMMON_FLAGS, "--detach-gate-features", "--hard-gate-target"),
    "rsg_unweighted": (
        *RSG_COMMON_FLAGS,
        "--detach-gate-features",
        "--unweighted-gate-regret",
    ),
    "rsg_coupled_gate": (*RSG_COMMON_FLAGS, "--couple-gate-features"),
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
    backbone: str = "efficientnet_b0",
) -> list[ExperimentCommand]:
    if stage not in {"pilot", "formal"}:
        raise ValueError("stage must be pilot or formal.")
    seeds = PILOT_SEEDS if stage == "pilot" else FORMAL_SEEDS
    training_script = project_root / "scripts" / "train_hrgv_mineral_classifier.py"
    commands: list[ExperimentCommand] = []
    for configuration, extra_flags in CONFIGURATION_FLAGS.items():
        for seed in seeds:
            output_dir = output_root / backbone / f"{stage}_{configuration}_seed{seed}"
            arguments = (
                str(python_executable),
                str(training_script),
                "--manifest",
                str(manifest),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--backbone",
                backbone,
                "--seed",
                str(seed),
                "--device",
                device,
                "--torch-home",
                str(torch_home),
                "--couple-verifier-features",
                *extra_flags,
            )
            commands.append(
                ExperimentCommand(configuration, seed, output_dir, arguments)
            )
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the registered RSG-HRGV experiment matrix.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "training",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".torch-cache",
    )
    parser.add_argument("--stage", choices=("pilot", "formal"), default="pilot")
    parser.add_argument(
        "--backbone",
        choices=("efficientnet_b0", "resnet50"),
        default="efficientnet_b0",
        help="Run a registered RSG configuration on the selected visual backbone.",
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(CONFIGURATION_FLAGS),
        help="Run only selected configurations; repeat for more than one.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run and args.execute:
        raise ValueError("Choose either --dry-run or --execute, not both.")
    commands = build_experiment_commands(
        project_root=args.project_root,
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        python_executable=args.python_executable,
        device=args.device,
        torch_home=args.torch_home,
        stage=args.stage,
        backbone=args.backbone,
    )
    selected = set(args.config or CONFIGURATION_FLAGS)
    commands = [command for command in commands if command.configuration in selected]
    for command in commands:
        printable = subprocess.list2cmdline(list(command.arguments))
        if args.dry_run or not args.execute:
            print(printable)
            continue
        metrics_path = command.output_dir / "test_metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"SKIP complete run: {command.output_dir}", flush=True)
            continue
        print(f"RUN configuration={command.configuration} seed={command.seed}", flush=True)
        subprocess.run(command.arguments, cwd=args.project_root, check=True)


if __name__ == "__main__":
    main()
