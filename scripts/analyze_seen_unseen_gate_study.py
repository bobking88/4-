"""Summarize the paired expert-seen versus expert-unseen gate study."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "final_nll_nats",
    "final_macro_f1",
    "final_target_recall",
    "final_ti_intrusion",
    "final_metal_intrusion",
    "mean_gate",
)


def summarize_runs(summary):
    paired = defaultdict(dict)
    grouped = defaultdict(list)
    for run in summary.get("runs", {}).values():
        key = (run["objective"], int(run["seed"]))
        source = run["source"]
        if source in paired[key]:
            raise ValueError(f"duplicate paired run for {key} and {source}")
        paired[key][source] = run["selected"]
        grouped[(source, run["objective"])].append(run["selected"])

    differences = []
    direction_counts = defaultdict(lambda: {
        "pairs": 0,
        "nll_improved": 0,
        "macro_f1_improved": 0,
        "target_recall_improved": 0,
    })
    for (objective, seed), sources in sorted(paired.items()):
        if set(sources) != {"seen", "unseen"}:
            raise ValueError(f"missing paired run for {(objective, seed)}")
        row = {"objective": objective, "seed": seed}
        for metric in METRICS:
            if metric in sources["seen"] and metric in sources["unseen"]:
                row[f"unseen_minus_seen_{metric}"] = (
                    float(sources["unseen"][metric]) - float(sources["seen"][metric])
                )
        differences.append(row)
        counts = direction_counts[objective]
        counts["pairs"] += 1
        counts["nll_improved"] += row.get("unseen_minus_seen_final_nll_nats", 0.0) < 0
        counts["macro_f1_improved"] += row.get("unseen_minus_seen_final_macro_f1", 0.0) > 0
        counts["target_recall_improved"] += row.get("unseen_minus_seen_final_target_recall", 0.0) > 0

    aggregates = []
    for (source, objective), selected_rows in sorted(grouped.items()):
        row = {"source": source, "objective": objective, "count": len(selected_rows)}
        for metric in METRICS:
            values = [float(selected[metric]) for selected in selected_rows if metric in selected]
            if values:
                row[f"mean_{metric}"] = statistics.mean(values)
                row[f"sample_sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        row["selected_epochs"] = [int(selected["epoch"]) for selected in selected_rows]
        aggregates.append(row)

    paired_aggregates = []
    for objective in sorted(direction_counts):
        rows = [row for row in differences if row["objective"] == objective]
        aggregate = {"objective": objective, "count": len(rows)}
        for metric in METRICS:
            key = f"unseen_minus_seen_{metric}"
            values = [float(row[key]) for row in rows if key in row]
            if values:
                aggregate[f"mean_{key}"] = statistics.mean(values)
                aggregate[f"sample_sd_{key}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        paired_aggregates.append(aggregate)

    return {
        "references": summary.get("references", {}),
        "aggregates": aggregates,
        "paired_differences": differences,
        "paired_aggregates": paired_aggregates,
        "direction_counts": dict(direction_counts),
        "independence_warning": (
            "The three gate initializations share one frozen expert and one stop set; "
            "they are paired optimization replicates, not independent expert replications."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=ROOT / "outputs/training/seen_unseen_gate_study_v1",
    )
    args = parser.parse_args()
    summary = json.loads((args.study_dir / "summary.json").read_text(encoding="utf-8"))
    analysis = summarize_runs(summary)
    (args.study_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = analysis["paired_differences"]
    with (args.study_dir / "paired_differences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
