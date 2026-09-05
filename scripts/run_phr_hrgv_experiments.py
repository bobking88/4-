from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


SCREEN_SEEDS = (20260728,)
FORMAL_SEEDS = (20260727, 20260728, 20260729)
SCREEN_CONFIGURATION_FLAGS = {
    "rsg_reference": ("--disable-gate-regret",),
    "phr_complete": ("--enable-phr", "--lambda-phr", "0.10"),
    "phr_fixed_half": ("--enable-phr", "--lambda-phr", "0.10", "--fixed-gate", "0.5"),
    "phr_hard_target": ("--enable-phr", "--lambda-phr", "0.10", "--phr-hard-gate-target"),
    "phr_unweighted": ("--enable-phr", "--lambda-phr", "0.10", "--phr-unweighted"),
    "phr_coupled_features": ("--enable-phr", "--lambda-phr", "0.10", "--couple-phr-gate-features"),
    "phr_ti_only": ("--enable-phr", "--lambda-phr", "0.10", "--phr-edges", "ti"),
    "phr_metallic_only": ("--enable-phr", "--lambda-phr", "0.10", "--phr-edges", "metallic"),
}
FORMAL_CONFIGURATION_FLAGS = {
    name: SCREEN_CONFIGURATION_FLAGS[name] for name in ("rsg_reference", "phr_complete")
}


@dataclass(frozen=True)
class ExperimentCommand:
    configuration: str
    seed: int
    output_dir: Path
    arguments: tuple[str, ...]


def build_experiment_commands(
    project_root: Path, manifest: Path, dataset_root: Path, output_root: Path,
    python_executable: Path, device: str, torch_home: Path, mode: str,
) -> list[ExperimentCommand]:
    if mode not in {"screen", "formal"}:
        raise ValueError("mode must be screen or formal.")
    configurations = SCREEN_CONFIGURATION_FLAGS if mode == "screen" else FORMAL_CONFIGURATION_FLAGS
    seeds = SCREEN_SEEDS if mode == "screen" else FORMAL_SEEDS
    trainer = project_root / "scripts" / "train_hrgv_mineral_classifier.py"
    commands = []
    for configuration, flags in configurations.items():
        for seed in seeds:
            output_dir = output_root / configuration / f"seed{seed}"
            arguments = (
                str(python_executable), str(trainer), "--manifest", str(manifest),
                "--dataset-root", str(dataset_root), "--output-dir", str(output_dir),
                "--seed", str(seed), "--device", device, "--torch-home", str(torch_home),
                "--couple-verifier-features", "--detach-gate-features", *flags,
            )
            commands.append(ExperimentCommand(configuration, seed, output_dir, arguments))
    return commands


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_promotion_decision(output_root: Path) -> None:
    decision_path = output_root.parent / "phr_hrgv_screen" / "screen_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError("Formal PHR runs require a validation-only screen_decision.json.")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("selected_configuration") != "phr_complete" or not decision.get("promote_to_formal"):
        raise ValueError("Formal PHR runs require a promoted phr_complete validation decision.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the registered PHR-HRGV experiment matrix.")
    parser.add_argument("--mode", choices=("screen", "formal"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--torch-home", type=Path, default=Path(__file__).resolve().parents[1] / ".torch-cache")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run and args.execute:
        raise ValueError("Choose either --dry-run or --execute, not both.")
    if args.mode == "formal":
        _require_promotion_decision(args.output_root)
    commands = build_experiment_commands(
        args.project_root, args.manifest, args.dataset_root, args.output_root,
        args.python_executable, args.device, args.torch_home, args.mode,
    )
    _write_json(args.output_root / "registered_configurations.json", {
        "timestamp": datetime.now().isoformat(), "mode": args.mode,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": _sha256(args.manifest),
        "dataset_root": str(args.dataset_root.resolve()), "seeds": list(SCREEN_SEEDS if args.mode == "screen" else FORMAL_SEEDS),
        "commands": [{"configuration": item.configuration, "seed": item.seed, "output_dir": str(item.output_dir), "arguments": list(item.arguments)} for item in commands],
    })
    for command in commands:
        if args.dry_run or not args.execute:
            print(subprocess.list2cmdline(list(command.arguments)))
            continue
        if (command.output_dir / "test_metrics.json").exists() and not args.force:
            print(f"SKIP complete run: {command.output_dir}", flush=True)
            continue
        print(f"RUN configuration={command.configuration} seed={command.seed}", flush=True)
        subprocess.run(command.arguments, cwd=args.project_root, check=True)


if __name__ == "__main__":
    main()
