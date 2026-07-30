from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRIC_KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "species_accuracy")
ROLE_LABELS = ("target_mineral", "ti_bearing_negative", "gangue_negative", "metallic_hard_negative")


def load_run_metrics(run_dirs: list[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing test metrics: {metrics_path}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        missing = [key for key in METRIC_KEYS if key not in payload]
        if missing:
            raise ValueError(f"Metrics file {metrics_path} is missing: {', '.join(missing)}")
        class_recall = payload.get("class_recall", {})
        row: dict[str, float | str] = {"run_name": run_dir.name}
        row.update({key: float(payload[key]) for key in METRIC_KEYS})
        for label in ROLE_LABELS:
            if label not in class_recall:
                raise ValueError(f"Metrics file {metrics_path} is missing class recall for {label}")
            row[f"{label}_recall"] = float(class_recall[label])
        rows.append(row)
    return rows


def summarize_formal_runs(run_rows: list[dict[str, float | str]]) -> dict[str, dict[str, float]]:
    if len(run_rows) != 3:
        raise ValueError("Formal hierarchical analysis requires exactly three runs.")
    summary: dict[str, dict[str, float]] = {}
    for key in run_rows[0]:
        if key == "run_name":
            continue
        values = [float(row[key]) for row in run_rows]
        summary[key] = {"mean": mean(values), "sample_std": stdev(values)}
    return summary


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, dict[str, float]]) -> None:
    labels = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "species_accuracy": "Species Accuracy",
        "target_mineral_recall": "Target Recall",
        "ti_bearing_negative_recall": "Ti-bearing-negative Recall",
        "gangue_negative_recall": "Gangue Recall",
        "metallic_hard_negative_recall": "Metallic-hard-negative Recall",
    }
    lines = [
        "# Hierarchical Mineral Recognition Three-Seed Summary",
        "",
        "| Metric | Mean | Sample standard deviation |",
        "|---|---:|---:|",
    ]
    for key, values in summary.items():
        lines.append(f"| {labels.get(key, key)} | {values['mean']:.2%} | {values['sample_std']:.2%} |")
    lines.extend(
        [
            "",
            "The summary is a closed-set result on the fixed public mineral-specimen image split. It is not a mineral grade, recovery, or industrial sorting result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exactly three hierarchical mineral classifier runs.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_run_metrics(args.run_dir)
    summary = summarize_formal_runs(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "hierarchical_metrics_per_run.csv", rows)
    (args.output_dir / "hierarchical_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(args.output_dir / "hierarchical_metrics_three_seed.md", summary)
    print(json.dumps({"output_dir": str(args.output_dir), "run_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
