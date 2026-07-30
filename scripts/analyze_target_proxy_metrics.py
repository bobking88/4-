from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


TARGET_LABEL = "target_mineral"
HIGH_RISK_NEGATIVE_LABELS = {
    "ti_bearing_negative": "ti_bearing",
    "metallic_hard_negative": "metallic",
    "gangue_negative": "gangue",
}
REQUIRED_COLUMNS = ("true_label", "predicted_label")


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_target_proxy_metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    for row_index, row in enumerate(rows, start=1):
        for column in REQUIRED_COLUMNS:
            if column not in row:
                raise ValueError(f"Row {row_index} is missing required column: {column}")

    true_target = [row["true_label"] == TARGET_LABEL for row in rows]
    predicted_target = [row["predicted_label"] == TARGET_LABEL for row in rows]
    true_positive = sum(truth and prediction for truth, prediction in zip(true_target, predicted_target))
    false_negative = sum(truth and not prediction for truth, prediction in zip(true_target, predicted_target))
    false_positive = sum(not truth and prediction for truth, prediction in zip(true_target, predicted_target))
    support = true_positive + false_negative
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, support)
    metrics: dict[str, float | int] = {
        "sample_count": len(rows),
        "target_support": support,
        "target_true_positive": true_positive,
        "target_false_negative": false_negative,
        "target_false_positive": false_positive,
        "target_precision": precision,
        "target_recall": recall,
        "target_f1": safe_ratio(2 * precision * recall, precision + recall),
        "target_miss_rate": safe_ratio(false_negative, support),
    }
    for label, prefix in HIGH_RISK_NEGATIVE_LABELS.items():
        support_for_label = sum(row["true_label"] == label for row in rows)
        intrusion_count = sum(
            row["true_label"] == label and row["predicted_label"] == TARGET_LABEL for row in rows
        )
        metrics[f"{prefix}_support"] = support_for_label
        metrics[f"{prefix}_intrusion_count"] = intrusion_count
        metrics[f"{prefix}_intrusion_rate"] = safe_ratio(intrusion_count, support_for_label)
    return metrics


def summarize_metric_runs(metric_rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    if len(metric_rows) < 2:
        raise ValueError("At least two runs are required to calculate a sample standard deviation.")
    summary: dict[str, dict[str, float]] = {}
    for key in metric_rows[0]:
        if key == "run_name" or key.endswith("_count") or key.endswith("_support") or key in {"sample_count", "target_true_positive", "target_false_negative", "target_false_positive"}:
            continue
        values = [float(row[key]) for row in metric_rows]
        summary[key] = {"mean": mean(values), "sample_std": stdev(values)}
    return summary


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_name_from_path(path: Path) -> str:
    return path.parent.name


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows supplied for CSV output.")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, run_rows: list[dict[str, object]], summary: dict[str, dict[str, float]]) -> None:
    lines = [
        "# 目标代理类/非目标类业务指标汇总",
        "",
        "## 指标定义",
        "",
        "将 `target_mineral` 折叠为目标代理类，其余三类折叠为非目标类。"
        "“误入目标比例”表示某个非目标类别被预测为 `target_mineral` 的比例；"
        "“漏选率”表示真实目标代理类未被预测为 `target_mineral` 的比例。",
        "",
        "## 各运行结果",
        "",
        "| 运行 | 目标 Precision | 目标 Recall | 目标 F1 | 漏选率 | 含钛干扰误入目标 | 金属光泽干扰误入目标 | 脉石误入目标 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            "| {run_name} | {target_precision:.2%} | {target_recall:.2%} | {target_f1:.2%} | "
            "{target_miss_rate:.2%} | {ti_bearing_intrusion_rate:.2%} | "
            "{metallic_intrusion_rate:.2%} | {gangue_intrusion_rate:.2%} |".format(**row)
        )
    lines.extend(["", "## 三随机种子统计", "", "| 指标 | 均值 | 样本标准差 |", "|---|---:|---:|"])
    for key, stats in summary.items():
        lines.append(f"| {key} | {stats['mean']:.2%} | {stats['sample_std']:.2%} |")
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "上述指标衡量公开矿物标本图像上的类别层面代理识别风险，"
            "不等同于工业分选回收率、精矿品位或生产线实际漏选率。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize target-proxy business metrics from prediction CSV files.")
    parser.add_argument("--prediction-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rows: list[dict[str, object]] = []
    for prediction_path in args.prediction_csv:
        metrics = calculate_target_proxy_metrics(load_prediction_rows(prediction_path))
        run_rows.append({"run_name": run_name_from_path(prediction_path), **metrics})
    summary = summarize_metric_runs(run_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target_proxy_metrics_per_run.csv", run_rows)
    (args.output_dir / "target_proxy_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(args.output_dir / "target_proxy_metrics_three_seed.md", run_rows, summary)
    print(json.dumps({"output_dir": str(args.output_dir), "run_count": len(run_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
