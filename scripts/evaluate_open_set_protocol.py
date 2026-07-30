"""Evaluate an open-set rejection protocol from known and unknown prediction scores.

The current project has no independently verified unknown-mineral image set.  This
tool intentionally evaluates score tables only after such a set is collected; it
does not fabricate an OOD result from the closed-set four-class test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def _validate_scores(scores: Iterable[float], name: str) -> list[float]:
    values = [float(value) for value in scores]
    if not values:
        raise ValueError(f"{name} scores must not be empty.")
    return values


def _average_rank(values: list[float]) -> list[float]:
    """Return one-based average ranks while handling tied scores deterministically."""
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    position = 0
    while position < len(ranked):
        end = position
        while end + 1 < len(ranked) and ranked[end + 1][1] == ranked[position][1]:
            end += 1
        average = ((position + 1) + (end + 1)) / 2
        for index in range(position, end + 1):
            result[ranked[index][0]] = average
        position = end + 1
    return result


def auroc(known_scores: Iterable[float], unknown_scores: Iterable[float]) -> float:
    """Calculate AUROC where a larger score means the sample is accepted as known."""
    known = _validate_scores(known_scores, "Known")
    unknown = _validate_scores(unknown_scores, "Unknown")
    ranks = _average_rank(known + unknown)
    known_rank_sum = sum(ranks[: len(known)])
    return (known_rank_sum - len(known) * (len(known) + 1) / 2) / (len(known) * len(unknown))


def threshold_for_target_tpr(known_scores: Iterable[float], target_tpr: float = 0.95) -> float:
    """Return the largest acceptance threshold retaining at least target_tpr known samples."""
    known = _validate_scores(known_scores, "Known")
    if not 0 < target_tpr <= 1:
        raise ValueError("target_tpr must be within (0, 1].")
    retain_count = max(1, int(__import__("math").ceil(target_tpr * len(known))))
    return sorted(known, reverse=True)[retain_count - 1]


def evaluate_open_set_scores(
    known_scores: Iterable[float], unknown_scores: Iterable[float], target_tpr: float = 0.95
) -> dict[str, float | int]:
    """Summarize rejection quality using an acceptance threshold fixed on known data."""
    known = _validate_scores(known_scores, "Known")
    unknown = _validate_scores(unknown_scores, "Unknown")
    threshold = threshold_for_target_tpr(known, target_tpr)
    known_accept_rate = sum(value >= threshold for value in known) / len(known)
    unknown_accept_rate = sum(value >= threshold for value in unknown) / len(unknown)
    return {
        "known_count": len(known),
        "unknown_count": len(unknown),
        "target_tpr": target_tpr,
        "threshold": threshold,
        "known_accept_rate": known_accept_rate,
        "fpr_at_95_tpr": unknown_accept_rate,
        "auroc": auroc(known, unknown),
    }


def load_score_column(path: Path, score_column: str) -> list[float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or score_column not in rows[0]:
        raise ValueError(f"{path} must contain a non-empty '{score_column}' column.")
    return [float(row[score_column]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate open-set known-versus-unknown prediction score tables.")
    parser.add_argument("--known-csv", type=Path, required=True, help="Closed-set known-mineral prediction table.")
    parser.add_argument("--unknown-csv", type=Path, required=True, help="Independent unknown-mineral prediction table.")
    parser.add_argument("--score-column", default="confidence", help="Larger values must indicate stronger known-class acceptance.")
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_open_set_scores(
        load_score_column(args.known_csv, args.score_column),
        load_score_column(args.unknown_csv, args.score_column),
        args.target_tpr,
    )
    result.update(
        {
            "score_column": args.score_column,
            "known_csv": str(args.known_csv),
            "unknown_csv": str(args.unknown_csv),
            "interpretation": "Higher scores are accepted as known; fpr_at_95_tpr is the unknown acceptance rate at the retained known fraction.",
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
