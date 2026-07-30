from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "outputs" / "training"
ERROR_ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "error_analysis" / "efficientnet_b0_seed20260728"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures_v1"
FOCAL_SEEDS = (20260727, 20260728, 20260729)

MODEL_SPECS = {
    "ResNet50": [TRAINING_DIR / f"formal_resnet50_seed{seed}" / "test_metrics.json" for seed in (20260727, 20260728, 20260729)],
    "EfficientNet-B0": [TRAINING_DIR / f"formal_efficientnet_b0_seed{seed}" / "test_metrics.json" for seed in (20260727, 20260728, 20260729)],
    "Role-aware EfficientNet-B0": [
        TRAINING_DIR / f"formal_role_aware_efficientnet_b0_seed{seed}" / "test_metrics.json"
        for seed in (20260727, 20260728, 20260729)
    ],
}
CLASS_DISPLAY = {
    "target_mineral": "Target mineral",
    "ti_bearing_negative": "Ti-bearing\nnegative",
    "gangue_negative": "Gangue",
    "metallic_hard_negative": "Metallic\nhard negative",
}
SHORT_CLASS_DISPLAY = {
    "target_mineral": "Target",
    "ti_bearing_negative": "Ti-bearing",
    "gangue_negative": "Gangue",
    "metallic_hard_negative": "Metallic",
}
COLORS = {
    "ResNet50": "#6f8fb3",
    "EfficientNet-B0": "#d58b5a",
    "Role-aware EfficientNet-B0": "#709c7a",
    "Focal loss": "#b55d60",
}
MODEL_DISPLAY = {
    "ResNet50": "ResNet50",
    "EfficientNet-B0": "EfficientNet-B0",
    "Role-aware EfficientNet-B0": "Role-aware\nEfficientNet-B0",
}
TARGET_PROXY_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "business_metrics" / "efficientnet_b0_cross_entropy" / "target_proxy_metrics_summary.json"
)
TARGET_PROXY_METRICS = (
    ("target_precision", "Target precision"),
    ("target_recall", "Target recall"),
    ("target_f1", "Target F1"),
    ("target_miss_rate", "Target miss rate"),
    ("ti_bearing_intrusion_rate", "Ti-bearing -> target"),
    ("metallic_intrusion_rate", "Metallic -> target"),
    ("gangue_intrusion_rate", "Gangue -> target"),
)
TARGET_PROXY_COMPARISON_METRICS = (
    ("target_f1", "Target F1", True),
    ("target_miss_rate", "Target miss rate", False),
    ("ti_bearing_intrusion_rate", "Ti-bearing -> target", False),
    ("metallic_intrusion_rate", "Metallic -> target", False),
)
TARGET_PROXY_STRATEGY_PATHS = {
    "Cross entropy": TARGET_PROXY_SUMMARY_PATH,
    "Focal loss": PROJECT_ROOT / "outputs" / "business_metrics" / "efficientnet_b0_focal" / "target_proxy_metrics_summary.json",
    "Role-aware": PROJECT_ROOT / "outputs" / "business_metrics" / "role_aware_efficientnet_b0" / "target_proxy_metrics_summary.json",
}
TARGET_PROXY_STRATEGY_COLORS = {
    "Cross entropy": "#d58b5a",
    "Focal loss": "#b55d60",
    "Role-aware": "#709c7a",
}


def focal_seed_paths() -> list[Path]:
    return [
        TRAINING_DIR / f"formal_efficientnet_b0_focal_seed{seed}" / "test_metrics.json"
        for seed in FOCAL_SEEDS
    ]


def summarize_values(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("At least two runs are required to calculate a sample standard deviation.")
    return mean(values), stdev(values)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, name: str) -> None:
    for extension, kwargs in {
        "png": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(OUTPUT_DIR / f"{name}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runs() -> dict[str, list[dict]]:
    return {model: [load_json(path) for path in paths] for model, paths in MODEL_SPECS.items()}


def summarize_focal_ablation(
    baseline_runs: list[dict], focal_runs: list[dict], class_labels: list[str]
) -> dict[str, dict[str, float | int]]:
    """Summarize three-seed loss-function results for overall and class-wise metrics."""
    if len(baseline_runs) != len(focal_runs):
        raise ValueError("Cross-entropy and Focal Loss runs must have matching seed counts.")
    metric_values = {
        "macro_f1": (
            [run["macro_f1"] for run in baseline_runs],
            [run["macro_f1"] for run in focal_runs],
        )
    }
    metric_values.update(
        {
            label: (
                [run["class_recall"][label] for run in baseline_runs],
                [run["class_recall"][label] for run in focal_runs],
            )
            for label in class_labels
        }
    )
    summaries = {}
    for metric, (baseline_values, focal_values) in metric_values.items():
        baseline_mean, baseline_std = summarize_values(baseline_values)
        focal_mean, focal_std = summarize_values(focal_values)
        summaries[metric] = {
            "cross_entropy_mean": baseline_mean,
            "cross_entropy_std": baseline_std,
            "focal_loss_mean": focal_mean,
            "focal_loss_std": focal_std,
            "delta_focal_minus_baseline": focal_mean - baseline_mean,
            "runs": len(baseline_values),
        }
    return summaries


def build_target_proxy_source_rows(summary: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    """Convert target-proxy metrics into chart-ready, auditable source rows."""
    return [
        {
            "metric": label,
            "metric_key": key,
            "mean_percent": f"{summary[key]['mean'] * 100:.6f}",
            "sample_std_percent": f"{summary[key]['sample_std'] * 100:.6f}",
            "runs": "3",
        }
        for key, label in TARGET_PROXY_METRICS
    ]


def build_target_proxy_comparison_rows(
    strategy_summaries: dict[str, dict[str, dict[str, float]]],
    metrics: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """Create auditable strategy comparison rows for selected target-proxy metrics."""
    rows = []
    for key, label in metrics:
        for strategy, summary in strategy_summaries.items():
            rows.append(
                {
                    "metric": label,
                    "metric_key": key,
                    "strategy": strategy,
                    "mean_percent": f"{summary[key]['mean'] * 100:.6f}",
                    "sample_std_percent": f"{summary[key]['sample_std'] * 100:.6f}",
                    "runs": "3",
                }
            )
    return rows


def plot_model_comparison(runs: dict[str, list[dict]], source_dir: Path) -> None:
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), constrained_layout=True)
    for axis, (metric_key, metric_label) in zip(axes, metrics):
        model_names = list(runs)
        summaries = [summarize_values([run[metric_key] for run in runs[name]]) for name in model_names]
        bars = axis.bar(
            [MODEL_DISPLAY[name] for name in model_names],
            [item[0] * 100 for item in summaries],
            yerr=[item[1] * 100 for item in summaries],
            capsize=3,
            color=[COLORS[name] for name in model_names],
            edgecolor="#3b3b3b",
            linewidth=0.6,
        )
        axis.set_ylim(65, 80)
        axis.set_ylabel("Test-set score (%)")
        axis.set_title(metric_label, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#dedede", linewidth=0.5)
        axis.set_axisbelow(True)
        for bar, (value, _) in zip(bars, summaries):
            axis.text(bar.get_x() + bar.get_width() / 2, value * 100 + 0.55, f"{value * 100:.2f}", ha="center", va="bottom", fontsize=7)
        for name, (value, deviation) in zip(model_names, summaries):
            source_rows.append({"model": name, "metric": metric_label, "mean_percent": f"{value * 100:.6f}", "sample_std_percent": f"{deviation * 100:.6f}", "runs": 3})
    save_figure(fig, "fig1_model_comparison")
    write_csv(source_dir / "fig1_model_comparison.csv", source_rows, ["model", "metric", "mean_percent", "sample_std_percent", "runs"])


def plot_class_recall(runs: dict[str, list[dict]], source_dir: Path) -> None:
    labels = list(CLASS_DISPLAY)
    x = np.arange(len(labels))
    width = 0.78 / len(runs)
    fig, axis = plt.subplots(figsize=(6.8, 3.1), constrained_layout=True)
    source_rows = []
    for index, (model_name, model_runs) in enumerate(runs.items()):
        summaries = [summarize_values([run["class_recall"][label] for run in model_runs]) for label in labels]
        axis.bar(
            x + (index - (len(runs) - 1) / 2) * width,
            [value * 100 for value, _ in summaries],
            width,
            yerr=[deviation * 100 for _, deviation in summaries],
            capsize=2,
            color=COLORS[model_name],
            label=model_name,
            edgecolor="#3b3b3b",
            linewidth=0.5,
        )
        for label, (value, deviation) in zip(labels, summaries):
            source_rows.append({"model": model_name, "class_label": label, "mean_recall_percent": f"{value * 100:.6f}", "sample_std_percent": f"{deviation * 100:.6f}", "runs": 3})
    axis.set_xticks(x, [CLASS_DISPLAY[label] for label in labels])
    axis.set_ylim(40, 90)
    axis.set_ylabel("Recall (%)")
    axis.set_title("Class-wise test recall", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(ncol=2, loc="lower left")
    save_figure(fig, "fig2_class_recall")
    write_csv(source_dir / "fig2_class_recall.csv", source_rows, ["model", "class_label", "mean_recall_percent", "sample_std_percent", "runs"])


def plot_confusion_matrix(source_dir: Path) -> None:
    path = TRAINING_DIR / "formal_efficientnet_b0_seed20260728" / "confusion_matrix.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    labels = rows[0][1:]
    matrix = np.array([[int(value) for value in row[1:]] for row in rows[1:]])
    fig, axis = plt.subplots(figsize=(4.6, 3.9), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, shrink=0.85, label="Sample count")
    axis.set_xticks(np.arange(len(labels)), [SHORT_CLASS_DISPLAY[label] for label in labels], rotation=30, ha="right")
    axis.set_yticks(np.arange(len(labels)), [SHORT_CLASS_DISPLAY[label] for label in labels])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("EfficientNet-B0 confusion matrix (seed 20260728)", loc="left", fontweight="bold")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "#1a1a1a", fontsize=8)
    source_rows = []
    for true_label, values in zip(labels, matrix):
        for predicted_label, count in zip(labels, values):
            source_rows.append({"true_label": true_label, "predicted_label": predicted_label, "count": int(count)})
    save_figure(fig, "fig3_confusion_matrix")
    write_csv(source_dir / "fig3_confusion_matrix.csv", source_rows, ["true_label", "predicted_label", "count"])


def plot_error_pairs(source_dir: Path) -> None:
    with (ERROR_ANALYSIS_DIR / "error_pair_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))[:6]
    labels = [" -> ".join(SHORT_CLASS_DISPLAY[item] for item in row["error_pair"].split("__as__")) for row in rows][::-1]
    counts = [int(row["count"]) for row in rows][::-1]
    fig, axis = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    bars = axis.barh(labels, counts, color="#b55d60", edgecolor="#3b3b3b", linewidth=0.5)
    axis.set_xlabel("Misclassified test images")
    axis.set_title("Six most frequent error pairs", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#dedede", linewidth=0.5)
    axis.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        axis.text(count + 0.6, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=7)
    save_figure(fig, "fig4_top_error_pairs")
    write_csv(source_dir / "fig4_top_error_pairs.csv", rows, ["error_pair", "count", "share_of_test"])


def plot_focal_ablation(source_dir: Path) -> None:
    baseline_paths = MODEL_SPECS["EfficientNet-B0"]
    baseline_runs = [load_json(path) for path in baseline_paths]
    focal_runs = [load_json(path) for path in focal_seed_paths()]
    labels = list(CLASS_DISPLAY)
    summaries = summarize_focal_ablation(baseline_runs, focal_runs, labels)
    metrics = [("macro_f1", "Macro F1"), *[(label, SHORT_CLASS_DISPLAY[label]) for label in labels]]
    x = np.arange(len(metrics))
    width = 0.35
    fig, axis = plt.subplots(figsize=(6.8, 3.05), constrained_layout=True)
    baseline_means = [summaries[key]["cross_entropy_mean"] for key, _ in metrics]
    baseline_stds = [summaries[key]["cross_entropy_std"] for key, _ in metrics]
    focal_means = [summaries[key]["focal_loss_mean"] for key, _ in metrics]
    focal_stds = [summaries[key]["focal_loss_std"] for key, _ in metrics]
    axis.bar(x - width / 2, np.array(baseline_means) * 100, width, yerr=np.array(baseline_stds) * 100, capsize=2, label="Cross entropy", color=COLORS["EfficientNet-B0"], edgecolor="#3b3b3b", linewidth=0.5)
    axis.bar(x + width / 2, np.array(focal_means) * 100, width, yerr=np.array(focal_stds) * 100, capsize=2, label="Focal loss", color=COLORS["Focal loss"], edgecolor="#3b3b3b", linewidth=0.5)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(55, 85)
    axis.set_ylabel("Test-set score (%)")
    axis.set_title("Loss-function ablation (three seeds, mean +/- SD)", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(ncol=2, loc="lower left")
    source_rows = []
    for metric, _ in metrics:
        summary = summaries[metric]
        source_rows.append({
            "metric": metric,
            "cross_entropy_mean_percent": f"{summary['cross_entropy_mean'] * 100:.6f}",
            "cross_entropy_std_percent": f"{summary['cross_entropy_std'] * 100:.6f}",
            "focal_loss_mean_percent": f"{summary['focal_loss_mean'] * 100:.6f}",
            "focal_loss_std_percent": f"{summary['focal_loss_std'] * 100:.6f}",
            "delta_focal_minus_baseline_percent": f"{summary['delta_focal_minus_baseline'] * 100:.6f}",
            "runs": summary["runs"],
        })
    save_figure(fig, "fig5_focal_loss_ablation")
    write_csv(source_dir / "fig5_focal_loss_ablation.csv", source_rows, ["metric", "cross_entropy_mean_percent", "cross_entropy_std_percent", "focal_loss_mean_percent", "focal_loss_std_percent", "delta_focal_minus_baseline_percent", "runs"])


def plot_target_proxy_metrics(source_dir: Path) -> None:
    summary = load_json(TARGET_PROXY_SUMMARY_PATH)
    source_rows = build_target_proxy_source_rows(summary)
    labels = [row["metric"] for row in source_rows][::-1]
    means = [float(row["mean_percent"]) for row in source_rows][::-1]
    deviations = [float(row["sample_std_percent"]) for row in source_rows][::-1]
    colors = ["#d58b5a" if "Target" in label and "->" not in label else "#b55d60" for label in labels]
    fig, axis = plt.subplots(figsize=(6.8, 3.45), constrained_layout=True)
    bars = axis.barh(labels, means, xerr=deviations, capsize=2, color=colors, edgecolor="#3b3b3b", linewidth=0.5)
    axis.set_xlim(0, 80)
    axis.set_xlabel("Rate (%)")
    axis.set_title("Target-proxy decision metrics (three seeds, mean +/- SD)", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#dedede", linewidth=0.5)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, means):
        axis.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=7)
    save_figure(fig, "fig6_target_proxy_metrics")
    write_csv(source_dir / "fig6_target_proxy_metrics.csv", source_rows, ["metric", "metric_key", "mean_percent", "sample_std_percent", "runs"])


def plot_target_proxy_strategy_comparison(source_dir: Path) -> None:
    strategy_summaries = {name: load_json(path) for name, path in TARGET_PROXY_STRATEGY_PATHS.items()}
    compact_metrics = [(key, label) for key, label, _ in TARGET_PROXY_COMPARISON_METRICS]
    source_rows = build_target_proxy_comparison_rows(strategy_summaries, compact_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), constrained_layout=True)
    strategies = list(strategy_summaries)
    for axis, (key, label, higher_is_better) in zip(axes.flat, TARGET_PROXY_COMPARISON_METRICS):
        means = [strategy_summaries[strategy][key]["mean"] * 100 for strategy in strategies]
        deviations = [strategy_summaries[strategy][key]["sample_std"] * 100 for strategy in strategies]
        bars = axis.bar(
            strategies,
            means,
            yerr=deviations,
            capsize=2,
            color=[TARGET_PROXY_STRATEGY_COLORS[strategy] for strategy in strategies],
            edgecolor="#3b3b3b",
            linewidth=0.5,
        )
        axis.set_ylim(0, 80 if key == "target_f1" else 40)
        axis.set_ylabel("Rate (%)")
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(f"{label} ({direction})", loc="left", fontweight="bold")
        axis.grid(axis="y", color="#dedede", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelrotation=18)
        for bar, value in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)
    save_figure(fig, "fig7_target_proxy_strategy_comparison")
    write_csv(source_dir / "fig7_target_proxy_strategy_comparison.csv", source_rows, ["metric", "metric_key", "strategy", "mean_percent", "sample_std_percent", "runs"])


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = OUTPUT_DIR / "source_data"
    source_dir.mkdir(exist_ok=True)
    runs = load_runs()
    plot_model_comparison(runs, source_dir)
    plot_class_recall(runs, source_dir)
    plot_confusion_matrix(source_dir)
    plot_error_pairs(source_dir)
    plot_focal_ablation(source_dir)
    plot_target_proxy_metrics(source_dir)
    plot_target_proxy_strategy_comparison(source_dir)
    print(f"Exported figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
