from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence


FLOAT32_EPSILON = 1.1920928955078125e-7
NUMERICAL_TOLERANCE = 2e-6


def _analyze_rows(path: Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Replay CSV is empty: {path}")
    b1_residuals: list[float] = []
    b2_residuals: list[float] = []
    regrets: list[float] = []
    bounds: list[float] = []
    minimum_probability = math.inf
    for row in rows:
        direct = float(row["direct_true_probability"])
        mapped = float(row["mapped_true_probability"])
        gate = float(row["gate"])
        hard_oracle = float(row["hard_oracle_gate"])
        soft_oracle = float(row["soft_oracle_gate"])
        regret = float(row["routing_regret_nll"])
        if direct < FLOAT32_EPSILON or mapped < FLOAT32_EPSILON:
            raise ValueError("Replay probabilities must retain the float32 clamp floor.")
        minimum_probability = min(minimum_probability, direct, mapped)
        b1_bound = abs(gate - hard_oracle) * abs(direct - mapped) / FLOAT32_EPSILON
        gap = math.log(mapped) - math.log(direct)
        b2_bound = math.exp(-abs(gap) / 0.2)
        b1_residuals.append(regret - b1_bound)
        b2_residuals.append(abs(soft_oracle - hard_oracle) - b2_bound)
        regrets.append(regret)
        bounds.append(b1_bound)
    return {
        "sample_count": len(rows),
        "minimum_true_probability": minimum_probability,
        "mean_routing_regret_nll": sum(regrets) / len(regrets),
        "mean_b1_upper_bound": sum(bounds) / len(bounds),
        "b1_max_residual": max(b1_residuals),
        "b1_violation_count": sum(value > NUMERICAL_TOLERANCE for value in b1_residuals),
        "b2_max_residual": max(b2_residuals),
        "b2_violation_count": sum(value > NUMERICAL_TOLERANCE for value in b2_residuals),
    }


def analyze_rsg_theory_replay(replay_root: Path, output_dir: Path) -> dict[str, object]:
    """Summarize B.1 and B.2 numerical consistency from high-precision checkpoint replay."""
    replay_root = Path(replay_root)
    csv_paths = sorted(replay_root.glob("*/*.csv"))
    if not csv_paths:
        raise ValueError("No protocol/seed replay CSV files were found.")
    runs: list[dict[str, object]] = []
    for path in csv_paths:
        run = _analyze_rows(path)
        run.update({"protocol": path.parent.name, "seed": path.stem, "csv": str(path)})
        runs.append(run)
    totals: dict[str, float] = defaultdict(float)
    total_samples = sum(int(run["sample_count"]) for run in runs)
    for run in runs:
        count = int(run["sample_count"])
        totals["mean_routing_regret_nll"] += float(run["mean_routing_regret_nll"]) * count
        totals["mean_b1_upper_bound"] += float(run["mean_b1_upper_bound"]) * count
        totals["b1_violation_count"] += int(run["b1_violation_count"])
        totals["b2_violation_count"] += int(run["b2_violation_count"])
    summary = {
        "runs": runs,
        "overall": {
            "run_count": len(runs),
            "sample_count": total_samples,
            "minimum_true_probability": min(
                float(run["minimum_true_probability"]) for run in runs
            ),
            "mean_routing_regret_nll": totals["mean_routing_regret_nll"] / total_samples,
            "mean_b1_upper_bound": totals["mean_b1_upper_bound"] / total_samples,
            "b1_max_residual": max(float(run["b1_max_residual"]) for run in runs),
            "b1_violation_count": int(totals["b1_violation_count"]),
            "b2_max_residual": max(float(run["b2_max_residual"]) for run in runs),
            "b2_violation_count": int(totals["b2_violation_count"]),
        },
        "numeric_settings": {
            "float32_epsilon": FLOAT32_EPSILON,
            "tolerance": NUMERICAL_TOLERANCE,
            "soft_target_temperature": 0.2,
        },
        "claim_boundary": (
            "Theorem B.1 and B.2 are checked for numerical consistency on high-precision "
            "checkpoint replay. This validates the implemented routing formulas, not a new "
            "classification, industrial-sorting, grade, recovery, external-generalization, or OOD claim."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "theory_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "protocol",
        "seed",
        "sample_count",
        "minimum_true_probability",
        "mean_routing_regret_nll",
        "mean_b1_upper_bound",
        "b1_max_residual",
        "b1_violation_count",
        "b2_max_residual",
        "b2_violation_count",
    )
    with (output_dir / "theory_replay_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{name: run[name] for name in fields} for run in runs])
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze high-precision RSG-HRGV routing replay diagnostics."
    )
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = analyze_rsg_theory_replay(args.replay_root, args.output_dir)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
