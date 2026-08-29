from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


EXTENSION_SEEDS = (20260730, 20260731)
REGISTERED_CONFIGURATIONS = ("rsg_complete", "mrpg_complete")
RSG_FLAGS = (
    "--verifier-mode", "residual", "--lambda-gate-regret", "0.1",
    "--gate-regret-temperature", "0.2", "--gate-gap-temperature", "0.5",
    "--detach-gate-features",
)
CONFIGURATION_FLAGS = {
    "rsg_complete": RSG_FLAGS,
    "mrpg_complete": (*RSG_FLAGS, "--enable-mrpg"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered two-seed RSG/M-RPG target-recall extension."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "training" / "target_recall_extension",
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".torch-cache",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    training_script = args.project_root / "scripts" / "train_hrgv_mineral_classifier.py"
    for configuration in REGISTERED_CONFIGURATIONS:
        for seed in EXTENSION_SEEDS:
            output_dir = args.output_root / f"extension_{configuration}_seed{seed}"
            arguments = (
                str(args.python_executable), str(training_script),
                "--manifest", str(args.manifest),
                "--dataset-root", str(args.dataset_root),
                "--output-dir", str(output_dir),
                "--seed", str(seed),
                "--device", args.device,
                "--torch-home", str(args.torch_home),
                "--couple-verifier-features",
                *CONFIGURATION_FLAGS[configuration],
            )
            if not args.execute:
                print(subprocess.list2cmdline(list(arguments)))
                continue
            if (output_dir / "test_metrics.json").is_file():
                print(f"SKIP complete run: {output_dir}", flush=True)
                continue
            print(f"RUN configuration={configuration} seed={seed}", flush=True)
            subprocess.run(arguments, cwd=args.project_root, check=True)


if __name__ == "__main__":
    main()
