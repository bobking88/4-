from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


FLOAT32_EPSILON = 1.1920928955078125e-7
NUMERICAL_TOLERANCE = 2e-6
SOFT_TARGET_TEMPERATURE = 0.2


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Replay CSV is empty: {path}")
    return rows


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot summarize empty diagnostic rows.")
    return {
        "sample_count": len(rows),
        "mean_routing_regret_nll": _mean(rows, "routing_regret_nll"),
        "mean_b1_local_bound": _mean(rows, "b1_local_bound"),
        "mean_oracle_margin": _mean(rows, "oracle_margin"),
        "mean_soft_hard_deviation": _mean(rows, "soft_hard_deviation"),
        "mean_b2_bound": _mean(rows, "b2_bound"),
        "mean_exact_decomposition_abs_residual": _mean(
            rows, "exact_decomposition_abs_residual"
        ),
        "exact_decomposition_max_abs_residual": max(
            float(row["exact_decomposition_abs_residual"]) for row in rows
        ),
        "exact_decomposition_violation_count": sum(
            float(row["exact_decomposition_abs_residual"]) > NUMERICAL_TOLERANCE
            for row in rows
        ),
        "b1_local_max_residual": max(float(row["b1_local_residual"]) for row in rows),
        "b1_local_violation_count": sum(
            float(row["b1_local_residual"]) > NUMERICAL_TOLERANCE for row in rows
        ),
        "b2_max_residual": max(float(row["b2_residual"]) for row in rows),
        "b2_violation_count": sum(
            float(row["b2_residual"]) > NUMERICAL_TOLERANCE for row in rows
        ),
    }


def _rank_strata(
    rows: Sequence[dict[str, Any]], metric: str, strata_count: int
) -> list[tuple[str, list[dict[str, Any]]]]:
    if strata_count < 2:
        raise ValueError("At least two strata are required.")
    ordered = sorted(
        rows,
        key=lambda row: (float(row[metric]), str(row["seed"]), str(row["image_id"])),
    )
    groups: list[list[dict[str, Any]]] = [[] for _ in range(strata_count)]
    for index, row in enumerate(ordered):
        group_index = min(index * strata_count // len(ordered), strata_count - 1)
        groups[group_index].append(row)
    return [(f"T{index + 1}", group) for index, group in enumerate(groups) if group]


def _stratify(
    protocol: str,
    rows: Sequence[dict[str, Any]],
    *,
    diagnostic: str,
    rank_metric: str,
    strata_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stratum, group in _rank_strata(rows, rank_metric, strata_count):
        summary = _summarize_rows(group)
        output.append(
            {
                "protocol": protocol,
                "diagnostic": diagnostic,
                "stratum": stratum,
                "rank_metric": rank_metric,
                "rank_metric_min": min(float(row[rank_metric]) for row in group),
                "rank_metric_max": max(float(row[rank_metric]) for row in group),
                **summary,
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_row(
    raw: dict[str, str], protocol: str, seed: str
) -> dict[str, float | str]:
    direct = float(raw["direct_true_probability"])
    mapped = float(raw["mapped_true_probability"])
    gate = float(raw["gate"])
    fused = float(raw["fused_true_probability"])
    hard_oracle = float(raw["hard_oracle_gate"])
    soft_oracle = float(raw["soft_oracle_gate"])
    regret = float(raw["routing_regret_nll"])
    if direct < FLOAT32_EPSILON or mapped < FLOAT32_EPSILON:
        raise ValueError("Replay probabilities must preserve the float32 clamp floor.")
    local_epsilon = min(direct, mapped)
    probability_gap = abs(direct - mapped)
    gate_error = abs(gate - hard_oracle)
    best_expert_probability = max(direct, mapped)
    exact_fused_probability = best_expert_probability - gate_error * probability_gap
    exact_regret_nll = math.log(best_expert_probability / exact_fused_probability)
    exact_fusion_abs_residual = abs(fused - exact_fused_probability)
    exact_regret_abs_residual = abs(regret - exact_regret_nll)
    margin = abs(math.log(direct) - math.log(mapped))
    b1_local_bound = gate_error * probability_gap / local_epsilon
    b2_bound = math.exp(-margin / SOFT_TARGET_TEMPERATURE)
    soft_hard_deviation = abs(soft_oracle - hard_oracle)
    return {
        "protocol": protocol,
        "seed": seed,
        "image_id": raw["image_id"],
        "routing_regret_nll": regret,
        "exact_fused_probability": exact_fused_probability,
        "exact_regret_nll": exact_regret_nll,
        "exact_fusion_abs_residual": exact_fusion_abs_residual,
        "exact_regret_abs_residual": exact_regret_abs_residual,
        "exact_decomposition_abs_residual": max(
            exact_fusion_abs_residual, exact_regret_abs_residual
        ),
        "gate_error": gate_error,
        "probability_gap": probability_gap,
        "oracle_margin": margin,
        "b1_local_bound": b1_local_bound,
        "b1_local_residual": regret - b1_local_bound,
        "soft_hard_deviation": soft_hard_deviation,
        "b2_bound": b2_bound,
        "b2_residual": soft_hard_deviation - b2_bound,
    }


def analyze_rsg_gate_reliability(
    replay_root: Path, output_dir: Path, *, strata_count: int = 3
) -> dict[str, object]:
    """Diagnose B.1/B.2 behavior by protocol and ranked mechanism strata.

    This is a descriptive formula-mechanism analysis of fixed checkpoint replays.
    It is not an additional classification-performance comparison.
    """
    replay_root = Path(replay_root)
    csv_paths = sorted(replay_root.glob("*/*.csv"))
    if not csv_paths:
        raise ValueError("No high-precision replay CSV files were found.")
    by_protocol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in csv_paths:
        protocol = path.parent.name
        seed = path.stem
        for raw in _read_csv(path):
            by_protocol[protocol].append(_build_row(raw, protocol, seed))

    b1_rows: list[dict[str, Any]] = []
    b2_rows: list[dict[str, Any]] = []
    protocols: dict[str, object] = {}
    for protocol in sorted(by_protocol):
        rows = by_protocol[protocol]
        run_count = len({str(row["seed"]) for row in rows})
        protocols[protocol] = {"run_count": run_count, **_summarize_rows(rows)}
        b1_rows.extend(
            _stratify(
                protocol,
                rows,
                diagnostic="B.1 local bound",
                rank_metric="b1_local_bound",
                strata_count=strata_count,
            )
        )
        b2_rows.extend(
            _stratify(
                protocol,
                rows,
                diagnostic="B.2 margin",
                rank_metric="oracle_margin",
                strata_count=strata_count,
            )
        )

    all_rows = [row for rows in by_protocol.values() for row in rows]
    summary = {
        "protocols": protocols,
        "overall": {
            "run_count": sum(int(value["run_count"]) for value in protocols.values()),
            **_summarize_rows(all_rows),
        },
        "numeric_settings": {
            "float32_epsilon": FLOAT32_EPSILON,
            "tolerance": NUMERICAL_TOLERANCE,
            "soft_target_temperature": SOFT_TARGET_TEMPERATURE,
            "strata_count": strata_count,
        },
        "claim_boundary": (
            "This is a descriptive mechanism diagnosis of the exact convex-fusion "
            "decomposition, the pointwise B.1 local bound, and B.2 exponential "
            "soft-target bound on fixed high-precision checkpoint "
            "replays. It is not a new classification-performance comparison, industrial "
            "sorting, grade, recovery, external-generalization, or OOD claim."
        ),
    }
    output_dir = Path(output_dir)
    _write_csv(output_dir / "b1_local_bound_strata.csv", b1_rows)
    _write_csv(output_dir / "b2_margin_strata.csv", b2_rows)
    (output_dir / "gate_reliability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze RSG-HRGV gate reliability from high-precision replays."
    )
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strata-count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = analyze_rsg_gate_reliability(
        args.replay_root, args.output_dir, strata_count=args.strata_count
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
