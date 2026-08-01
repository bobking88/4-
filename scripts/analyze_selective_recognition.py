from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


TARGET_LABEL = "target_mineral"
TITANIUM_LABEL = "ti_bearing_negative"
METALLIC_LABEL = "metallic_hard_negative"
REQUIRED_COLUMNS = ("true_label", "predicted_label", "confidence")
EXPECTED_RUN_NAMES = (
    "formal_hierarchical_efficientnet_b0_seed20260727",
    "formal_hierarchical_efficientnet_b0_seed20260728",
    "formal_hierarchical_efficientnet_b0_seed20260729",
)
THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(20))


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Prediction CSV contains no rows.")
    for row_index, row in enumerate(rows, start=1):
        for column in REQUIRED_COLUMNS:
            if column not in row:
                raise ValueError(f"Row {row_index} is missing required column: {column}")
        try:
            float(row["confidence"])
        except ValueError as error:
            raise ValueError(f"Row {row_index} has a non-numeric confidence value.") from error


def calculate_selective_metrics(
    rows: list[dict[str, str]], thresholds: tuple[float, ...]
) -> list[dict[str, float | int | None]]:
    """Calculate retained-record error rates for each confidence threshold."""
    _validate_rows(rows)
    values: list[dict[str, float | int | None]] = []
    for threshold in thresholds:
        retained = [row for row in rows if float(row["confidence"]) >= threshold]
        target_rows = [row for row in retained if row["true_label"] == TARGET_LABEL]
        titanium_rows = [row for row in retained if row["true_label"] == TITANIUM_LABEL]
        metallic_rows = [row for row in retained if row["true_label"] == METALLIC_LABEL]
        error_count = sum(row["true_label"] != row["predicted_label"] for row in retained)
        values.append(
            {
                "threshold": threshold,
                "retained_count": len(retained),
                "coverage": len(retained) / len(rows),
                "risk": _optional_ratio(error_count, len(retained)),
                "target_proxy_miss_rate": _optional_ratio(
                    sum(row["predicted_label"] != TARGET_LABEL for row in target_rows),
                    len(target_rows),
                ),
                "titanium_interference_intrusion_rate": _optional_ratio(
                    sum(row["predicted_label"] == TARGET_LABEL for row in titanium_rows),
                    len(titanium_rows),
                ),
                "metallic_hard_negative_intrusion_rate": _optional_ratio(
                    sum(row["predicted_label"] == TARGET_LABEL for row in metallic_rows),
                    len(metallic_rows),
                ),
            }
        )
    return values


def _mean_and_sample_std(values: list[float | int | None]) -> tuple[float | None, float | None]:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None, None
    return mean(defined), stdev(defined) if len(defined) > 1 else None


def aggregate_threshold_metrics(
    seed_metrics: dict[str, list[dict[str, float | int | None]]]
) -> list[dict[str, object]]:
    """Keep per-seed values and calculate defined-value mean/sample standard deviation."""
    if not seed_metrics:
        raise ValueError("At least one seed is required for aggregation.")
    seed_names = list(seed_metrics)
    threshold_count = len(seed_metrics[seed_names[0]])
    if any(len(values) != threshold_count for values in seed_metrics.values()):
        raise ValueError("All seeds must contain the same number of thresholds.")

    aggregated: list[dict[str, object]] = []
    for index in range(threshold_count):
        seed_values = {seed_name: seed_metrics[seed_name][index] for seed_name in seed_names}
        thresholds = {float(value["threshold"]) for value in seed_values.values()}
        if len(thresholds) != 1:
            raise ValueError("Seed threshold sequences do not align.")
        metric_names = [name for name in next(iter(seed_values.values())) if name != "threshold"]
        mean_values: dict[str, float | None] = {}
        std_values: dict[str, float | None] = {}
        for metric_name in metric_names:
            metric_mean, metric_std = _mean_and_sample_std(
                [value[metric_name] for value in seed_values.values()]
            )
            mean_values[metric_name] = metric_mean
            std_values[metric_name] = metric_std
        aggregated.append(
            {
                "threshold": thresholds.pop(),
                "seed_values": seed_values,
                "mean": mean_values,
                "sample_std": std_values,
            }
        )
    return aggregated


def _load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_input_paths(input_glob: str) -> list[Path]:
    candidates = {Path(path).resolve() for path in glob.glob(input_glob)}
    by_name = {path.parent.name: path for path in candidates}
    unexpected_names = sorted(set(by_name) - set(EXPECTED_RUN_NAMES))
    missing_names = [name for name in EXPECTED_RUN_NAMES if name not in by_name]
    if unexpected_names or missing_names:
        raise ValueError(
            "Input must be exactly the three full-hierarchical seed prediction files; "
            f"missing={missing_names}, unexpected={unexpected_names}."
        )
    return [by_name[name] for name in EXPECTED_RUN_NAMES]


def _plot(aggregated: list[dict[str, object]], figure_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    thresholds = [float(item["threshold"]) for item in aggregated]
    coverage = [item["mean"]["coverage"] for item in aggregated]  # type: ignore[index]
    coverage_std = [item["sample_std"]["coverage"] for item in aggregated]  # type: ignore[index]
    risk = [item["mean"]["risk"] for item in aggregated]  # type: ignore[index]
    risk_std = [item["sample_std"]["risk"] for item in aggregated]  # type: ignore[index]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].errorbar(thresholds, coverage, yerr=coverage_std, marker="o", capsize=3, color="#1f77b4")
    axes[0].set_title("置信度阈值与保留覆盖率")
    axes[0].set_xlabel("置信度阈值")
    axes[0].set_ylabel("保留覆盖率")
    axes[0].set_xlim(-0.02, 0.97)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.25)

    paired = sorted(zip(coverage, risk, risk_std), key=lambda values: values[0])
    axes[1].errorbar(
        [item[0] for item in paired],
        [item[1] for item in paired],
        yerr=[item[2] for item in paired],
        marker="o",
        capsize=3,
        color="#d62728",
    )
    axes[1].set_title("保留风险与保留覆盖率")
    axes[1].set_xlabel("保留覆盖率")
    axes[1].set_ylabel("保留风险")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.25)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)


def _format_rate(value: float | None) -> str:
    return "未定义" if value is None else f"{value:.2%}"


def _write_markdown(path: Path, aggregated: list[dict[str, object]], figure_path: Path) -> None:
    figure_reference = os.path.relpath(figure_path, start=path.parent).replace("\\", "/")
    lines = [
        "# 置信度选择性识别风险分析",
        "",
        "本分析在冻结的测试划分上，对三个完整层级识别随机种子的预测置信度实施阈值筛选。低于阈值的样本仅建议留待后续检查，不改变原始测试划分，也不对其作工业处置推断。",
        "",
        "## 阈值汇总",
        "",
        "| 阈值 | 覆盖率（均值±标准差） | 保留风险（均值±标准差） | 目标代理漏选率（均值±标准差） | 含钛干扰误入目标率（均值±标准差） | 金属强负样本误入目标率（均值±标准差） |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregated:
        averages = item["mean"]
        deviations = item["sample_std"]
        rows = [
            f"{_format_rate(averages[name])} ± {_format_rate(deviations[name])}"
            for name in (
                "coverage",
                "risk",
                "target_proxy_miss_rate",
                "titanium_interference_intrusion_rate",
                "metallic_hard_negative_intrusion_rate",
            )
        ]
        lines.append(f"| {item['threshold']:.2f} | " + " | ".join(rows) + " |")
    lines.extend(
        [
            "",
            "## 图9",
            "",
            f"![图9：置信度选择性识别分析]({figure_reference})",
            "",
            "图中较低的覆盖率表示更多样本被延后，建议进行后续检查。",
            "",
            "## 使用边界",
            "",
            "保留风险和各类误入/漏选率仅描述该固定公开图像测试划分上的模型预测行为。该结果不是工业成本最优策略，也不是实际 XRF 验证、选矿回收率或生产线性能结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze confidence-based selective mineral recognition.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = _resolve_input_paths(args.input_glob)
    seed_metrics = {
        input_path.parent.name: calculate_selective_metrics(_load_prediction_rows(input_path), THRESHOLDS)
        for input_path in input_paths
    }
    aggregated = aggregate_threshold_metrics(seed_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_type": "confidence-based selective-recognition risk analysis",
        "fixed_split_consumed_unchanged": True,
        "input_files": [str(path) for path in input_paths],
        "thresholds": list(THRESHOLDS),
        "seed_level_metrics": seed_metrics,
        "threshold_summary": aggregated,
        "defer_interpretation": "Lower coverage means more samples are deferred for later inspection.",
        "limitations": [
            "This is not an industrial cost-optimal policy.",
            "This is not real XRF validation.",
        ],
    }
    (args.output_dir / "selective_recognition_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot(aggregated, args.figure)
    _write_markdown(args.output_dir / "selective_recognition_summary.md", aggregated, args.figure)
    print(json.dumps({"input_count": len(input_paths), "threshold_count": len(THRESHOLDS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
