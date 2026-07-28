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

MODEL_SPECS = {
    "ResNet50": [TRAINING_DIR / f"formal_resnet50_seed{seed}" / "test_metrics.json" for seed in (20260727, 20260728, 20260729)],
    "EfficientNet-B0": [TRAINING_DIR / f"formal_efficientnet_b0_seed{seed}" / "test_metrics.json" for seed in (20260727, 20260728, 20260729)],
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
COLORS = {"ResNet50": "#6f8fb3", "EfficientNet-B0": "#d58b5a", "Focal loss": "#b55d60"}


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


def plot_model_comparison(runs: dict[str, list[dict]], source_dir: Path) -> None:
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), constrained_layout=True)
    for axis, (metric_key, metric_label) in zip(axes, metrics):
        model_names = list(runs)
        summaries = [summarize_values([run[metric_key] for run in runs[name]]) for name in model_names]
        bars = axis.bar(
            model_names,
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
    width = 0.35
    fig, axis = plt.subplots(figsize=(6.8, 3.1), constrained_layout=True)
    source_rows = []
    for index, (model_name, model_runs) in enumerate(runs.items()):
        summaries = [summarize_values([run["class_recall"][label] for run in model_runs]) for label in labels]
        axis.bar(
            x + (index - 0.5) * width,
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
    baseline = load_json(TRAINING_DIR / "formal_efficientnet_b0_seed20260728" / "test_metrics.json")
    focal = load_json(TRAINING_DIR / "formal_efficientnet_b0_focal_seed20260728" / "test_metrics.json")
    metrics = [("macro_f1", "Macro F1"), *[(f"class_recall.{label}", SHORT_CLASS_DISPLAY[label]) for label in CLASS_DISPLAY]]
    baseline_values = [baseline["macro_f1"]] + [baseline["class_recall"][label] for label in CLASS_DISPLAY]
    focal_values = [focal["macro_f1"]] + [focal["class_recall"][label] for label in CLASS_DISPLAY]
    x = np.arange(len(metrics))
    width = 0.35
    fig, axis = plt.subplots(figsize=(6.8, 3.05), constrained_layout=True)
    axis.bar(x - width / 2, np.array(baseline_values) * 100, width, label="Cross entropy", color=COLORS["EfficientNet-B0"], edgecolor="#3b3b3b", linewidth=0.5)
    axis.bar(x + width / 2, np.array(focal_values) * 100, width, label="Focal loss", color=COLORS["Focal loss"], edgecolor="#3b3b3b", linewidth=0.5)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(55, 85)
    axis.set_ylabel("Test-set score (%)")
    axis.set_title("Loss-function ablation (same seed and split)", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#dedede", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(ncol=2, loc="lower left")
    source_rows = []
    for metric, baseline_value, focal_value in zip([key for key, _ in metrics], baseline_values, focal_values):
        source_rows.append({"metric": metric, "cross_entropy_percent": f"{baseline_value * 100:.6f}", "focal_loss_percent": f"{focal_value * 100:.6f}", "delta_focal_minus_baseline_percent": f"{(focal_value - baseline_value) * 100:.6f}"})
    save_figure(fig, "fig5_focal_loss_ablation")
    write_csv(source_dir / "fig5_focal_loss_ablation.csv", source_rows, ["metric", "cross_entropy_percent", "focal_loss_percent", "delta_focal_minus_baseline_percent"])


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
    print(f"Exported figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
