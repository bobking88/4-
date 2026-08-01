from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "macro_f1",
    "target_f1",
    "target_miss_rate",
    "ti_bearing_intrusion_rate",
    "metallic_intrusion_rate",
)


def load_summary(metrics_dir: Path) -> dict[str, dict[str, float]]:
    hierarchical_path = metrics_dir / "hierarchical_metrics_summary.json"
    target_path = metrics_dir / "target_proxy_metrics_summary.json"
    if not hierarchical_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing summary files in {metrics_dir}")
    hierarchical = json.loads(hierarchical_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    return {**hierarchical, **target}


def value(summary: dict[str, dict[str, float]], metric: str) -> dict[str, float]:
    if metric not in summary:
        raise ValueError(f"Missing required metric: {metric}")
    result = summary[metric]
    if "mean" not in result or "sample_std" not in result:
        raise ValueError(f"Metric {metric} must contain mean and sample_std")
    return {"mean": float(result["mean"]), "sample_std": float(result["sample_std"])}


def build_component_rows(
    full: dict[str, dict[str, float]],
    no_contrast: dict[str, dict[str, float]],
    no_consistency: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    configurations = (
        ("完整分层模型", full),
        ("去除困难负样本约束", no_contrast),
        ("去除层级一致性约束", no_consistency),
    )
    full_values = {metric: value(full, metric)["mean"] for metric in METRICS}
    rows: list[dict[str, float | str]] = []
    for setting, summary in configurations:
        row: dict[str, float | str] = {"setting": setting}
        for metric in METRICS:
            stats = value(summary, metric)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sample_std"] = stats["sample_std"]
            row[f"delta_{metric}_vs_full"] = stats["mean"] - full_values[metric]
        rows.append(row)
    return rows


def format_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def write_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# 分层模型组件消融汇总",
        "",
        "三组设置均采用相同数据版本、固定训练/验证/测试划分和三个随机种子。",
        "变化量为该设置均值减去完整分层模型均值；正值不一定代表更好，"
        "例如漏选率和误入目标比例应越低越好。",
        "",
        "| 设置 | Macro F1 (%) | 目标代理 F1 (%) | 目标漏选率 (%) | 含钛干扰误入目标 (%) | 金属光泽干扰误入目标 (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {macro_f1} | {target_f1} | {miss} | {ti} | {metallic} |".format(
                setting=row["setting"],
                macro_f1=format_percent(float(row["macro_f1_mean"]), float(row["macro_f1_sample_std"])),
                target_f1=format_percent(float(row["target_f1_mean"]), float(row["target_f1_sample_std"])),
                miss=format_percent(float(row["target_miss_rate_mean"]), float(row["target_miss_rate_sample_std"])),
                ti=format_percent(float(row["ti_bearing_intrusion_rate_mean"]), float(row["ti_bearing_intrusion_rate_sample_std"])),
                metallic=format_percent(float(row["metallic_intrusion_rate_mean"]), float(row["metallic_intrusion_rate_sample_std"])),
            )
        )
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "样本量为三个随机种子，均值差与标准差相近时只能作为趋势，"
            "不能表述为统计显著或稳定优于。所有风险指标均为公开矿物标本图像上的类别代理指标，"
            "不等同于工业回收率、精矿品位或生产线误分率。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize hierarchical model component ablations.")
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--no-contrast-dir", type=Path, required=True)
    parser.add_argument("--no-consistency-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_component_rows(
        load_summary(args.full_dir),
        load_summary(args.no_contrast_dir),
        load_summary(args.no_consistency_dir),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hierarchical_component_ablation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(args.output_dir / "hierarchical_component_ablation.md", rows)
    print(json.dumps({"output_dir": str(args.output_dir), "configuration_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
