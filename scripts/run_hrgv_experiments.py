from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FORMAL_SEEDS = (20260727, 20260728, 20260729)
CONFIGURATION_FLAGS = {
    "decoupled_residual": ("--verifier-mode", "residual"),
    "residual_coupled": (
        "--verifier-mode",
        "residual",
        "--couple-verifier-features",
    ),
    "gate_only": ("--disable-verifiers", "--lambda-verifier", "0"),
    "equal_fusion": (
        "--verifier-mode",
        "residual",
        "--fixed-gate",
        "0.5",
        "--couple-verifier-features",
    ),
    "no_contrast": (
        "--verifier-mode",
        "residual",
        "--lambda-contrast",
        "0",
        "--couple-verifier-features",
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
) -> list[ExperimentCommand]:
    commands: list[ExperimentCommand] = []
    training_script = project_root / "scripts" / "train_hrgv_mineral_classifier.py"
    for configuration, extra_flags in CONFIGURATION_FLAGS.items():
        for seed in FORMAL_SEEDS:
            output_dir = output_root / f"formal_hrgv_{configuration}_seed{seed}"
            arguments = (
                str(python_executable),
                str(training_script),
                "--manifest",
                str(manifest),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(seed),
                "--device",
                device,
                "--torch-home",
                str(torch_home),
                *extra_flags,
            )
            commands.append(
                ExperimentCommand(
                    configuration=configuration,
                    seed=seed,
                    output_dir=output_dir,
                    arguments=arguments,
                )
            )
    return commands


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the registered HRGV-Net experiment matrix.")
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
    parser.add_argument(
        "--configuration",
        action="append",
        choices=tuple(CONFIGURATION_FLAGS),
        help="Run only selected configurations; repeat the flag for more than one.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    commands = build_experiment_commands(
        project_root=args.project_root,
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        python_executable=args.python_executable,
        device=args.device,
        torch_home=args.torch_home,
    )
    selected = set(args.configuration or CONFIGURATION_FLAGS)
    commands = [command for command in commands if command.configuration in selected]
    for command in commands:
        printable = subprocess.list2cmdline(list(command.arguments))
        if not args.execute:
            print(printable)
            continue
        metrics_path = command.output_dir / "test_metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"SKIP complete run: {command.output_dir}", flush=True)
            continue
        print(
            f"RUN configuration={command.configuration} seed={command.seed}", flush=True
        )
        subprocess.run(command.arguments, cwd=args.project_root, check=True)


if __name__ == "__main__":
    main()
