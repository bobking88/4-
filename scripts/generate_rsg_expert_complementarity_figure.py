from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ink": "#20303C",
    "muted": "#647681",
    "grid": "#D8E0E4",
    "direct": "#527A9E",
    "mapped": "#2F7D72",
    "fused": "#C28A3A",
    "oracle": "#B35C4C",
    "accent": "#3C876F",
    "note": "#F6E7B8",
}


def _load_complete_summary(path: Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload["rsg_complete"]
    keys = (
        "direct_accuracy",
        "mapped_accuracy",
        "fused_accuracy",
        "oracle_accuracy",
        "expert_prediction_disagreement_rate",
        "one_right_one_wrong_rate",
        "one_right_gate_selection_accuracy",
    )
    return {key: float(values[key]["mean"]) for key in keys}


def _write_source_description(path: Path, protocols: list[tuple[str, str, dict[str, float]]]) -> None:
    payload = {
        "purpose": (
            "Diagnostic only: quantify two-expert complementarity and gate selection using "
            "frozen three-seed summaries. Oracle accuracy uses the true label and is not deployable."
        ),
        "protocols": {
            key: {metric: value for metric, value in summary.items()}
            for key, _label, summary in protocols
        },
        "metric_definitions": {
            "direct_accuracy": "Accuracy of the direct four-role expert.",
            "mapped_accuracy": "Accuracy after mapping the 17-species posterior into four roles.",
            "fused_accuracy": "Accuracy of the learned convex fusion after the complete RSG model.",
            "oracle_accuracy": "Per-image correctness if the better of two experts were selected using the true label; diagnostic upper bound only.",
            "expert_prediction_disagreement_rate": "Rate at which the direct and mapped role argmax predictions differ.",
            "one_right_one_wrong_rate": "Rate at which exactly one expert predicts the true role.",
            "one_right_gate_selection_accuracy": "Gate correctness conditional on the one-right / one-wrong subset.",
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_rsg_expert_complementarity_figure(
    fixed_summary_json: Path,
    holdout_summary_json: Path,
    portability_summary_json: Path,
    prefix: Path,
) -> dict[str, Path]:
    """Render frozen three-seed diagnostics for the two RSG evidence experts."""
    protocols = [
        ("fixed_test", "Fixed test", _load_complete_summary(fixed_summary_json)),
        ("photographer_holdout", "Photographer holdout", _load_complete_summary(holdout_summary_json)),
        ("resnet50_portability", "ResNet50 portability", _load_complete_summary(portability_summary_json)),
    ]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )
    figure = plt.figure(figsize=(12.6, 7.4))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.88), width_ratios=(1.20, 1.0), hspace=0.34, wspace=0.28)
    accuracy_axis = figure.add_subplot(grid[0, 0])
    routing_axis = figure.add_subplot(grid[0, 1])
    note_axis = figure.add_subplot(grid[1, :])
    figure.patch.set_facecolor("white")

    x = np.arange(len(protocols))
    bar_width = 0.18
    accuracy_specs = [
        ("direct_accuracy", "Direct role expert", COLORS["direct"]),
        ("mapped_accuracy", "Mapped species expert", COLORS["mapped"]),
        ("fused_accuracy", "Learned fusion", COLORS["fused"]),
        ("oracle_accuracy", "Oracle diagnostic", COLORS["oracle"]),
    ]
    for index, (metric, label, color) in enumerate(accuracy_specs):
        values = [summary[metric] * 100 for _key, _label, summary in protocols]
        bars = accuracy_axis.bar(x + (index - 1.5) * bar_width, values, bar_width, label=label, color=color)
        for bar, value in zip(bars, values):
            accuracy_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.33, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5, color=COLORS["ink"])
    accuracy_axis.set_title("Cross-granularity expert complementarity", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    accuracy_axis.set_ylabel("Three-seed mean accuracy (%)", fontsize=7.7)
    accuracy_axis.set_xticks(x, [label for _key, label, _summary in protocols], fontsize=7.3)
    accuracy_axis.set_ylim(64, 82)
    accuracy_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    accuracy_axis.spines[["top", "right"]].set_visible(False)
    accuracy_axis.legend(loc="lower left", fontsize=6.7, ncol=2, frameon=False)
    accuracy_axis.text(0.985, 0.055, "Oracle diagnostic is not a deployable oracle.", transform=accuracy_axis.transAxes, ha="right", va="bottom", fontsize=6.5, color=COLORS["muted"])

    routing_specs = [
        ("expert_prediction_disagreement_rate", "Expert disagreement"),
        ("one_right_one_wrong_rate", "one-right / one-wrong"),
        ("one_right_gate_selection_accuracy", "Gate selection"),
    ]
    values = np.array([[summary[metric] * 100 for metric, _label in routing_specs] for _key, _protocol_label, summary in protocols])
    for index, (_metric, label) in enumerate(routing_specs):
        line = routing_axis.plot(x, values[:, index], marker="o", linewidth=2.0, markersize=6, label=label)[0]
        for x_value, value in zip(x, values[:, index]):
            routing_axis.text(x_value, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5, color=line.get_color())
    routing_axis.set_title("Where the gate has a decision to make", fontsize=10.0, fontweight="bold", color=COLORS["ink"])
    routing_axis.set_ylabel("Rate / conditional accuracy (%)", fontsize=7.7)
    routing_axis.set_xticks(x, [label for _key, label, _summary in protocols], fontsize=7.2)
    routing_axis.set_ylim(0, 70)
    routing_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    routing_axis.spines[["top", "right"]].set_visible(False)
    routing_axis.legend(loc="upper left", fontsize=6.6, frameon=False)

    note_axis.set_axis_off()
    note_axis.text(0.02, 0.93, "Interpretation boundary", transform=note_axis.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    note_axis.text(
        0.02,
        0.75,
        "The direct role head and the species-to-role mapping disagree on a small but non-zero subset. The one-right / one-wrong subset is the only subset on which expert selection can repair an error. The complete RSG gate raises conditional selection above the HRGV reference in the registered comparisons, while the fusion remains below the true-label oracle diagnostic.",
        transform=note_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        linespacing=1.45,
        color=COLORS["ink"],
        wrap=True,
        bbox={"facecolor": COLORS["note"], "edgecolor": "#C7A965", "boxstyle": "round,pad=0.62"},
    )
    note_axis.text(
        0.02,
        0.15,
        "This is an evidence-structure diagnostic from frozen public-specimen experiments. It does not establish a universal MoE advantage, overall classification superiority, open-set recognition, industrial sorting, grade prediction, or recovery performance.",
        transform=note_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        color=COLORS["muted"],
        wrap=True,
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.06)

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": prefix.with_suffix(".png"),
        "svg": prefix.with_suffix(".svg"),
        "pdf": prefix.with_suffix(".pdf"),
        "tiff": prefix.with_suffix(".tiff"),
        "source_description": prefix.with_name(prefix.name + "_source.json"),
    }
    figure.savefig(outputs["png"], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    _write_source_description(outputs["source_description"], protocols)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RSG expert-complementarity diagnostics.")
    parser.add_argument("--fixed-summary-json", type=Path, required=True)
    parser.add_argument("--holdout-summary-json", type=Path, required=True)
    parser.add_argument("--portability-summary-json", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    generate_rsg_expert_complementarity_figure(
        args.fixed_summary_json,
        args.holdout_summary_json,
        args.portability_summary_json,
        args.prefix,
    )


if __name__ == "__main__":
    main()
