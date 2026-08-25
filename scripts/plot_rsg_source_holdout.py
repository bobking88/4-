from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(summary_path: Path, paired_path: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    reference = summary["hrgv_reference"]
    rsg = summary["rsg_complete"]
    labels = ("Macro F1", "Target recall", "Ti intrusion", "Metallic intrusion")
    metric_keys = (
        "macro_f1",
        "target_recall",
        "ti_to_target_intrusion_rate",
        "metallic_to_target_intrusion_rate",
    )
    colors = ("#1D4ED8", "#0F766E")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)
    positions = list(range(len(metric_keys)))
    width = 0.34
    for offset, (name, color, label) in enumerate(
        (("hrgv_reference", colors[0], "HRGV reference"), ("rsg_complete", colors[1], "RSG-HRGV"))
    ):
        values = [summary[name][key]["mean"] for key in metric_keys]
        errors = [summary[name][key]["sample_std"] for key in metric_keys]
        axes[0].bar(
            [value + (offset - 0.5) * width for value in positions],
            values,
            width=width,
            yerr=errors,
            capsize=4,
            color=color,
            label=label,
        )
    axes[0].set_title("Photographer-held-out role metrics")
    axes[0].set_xticks(positions, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Score / rate")
    axes[0].set_ylim(0, 0.85)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")

    route_labels = ("One-right selection", "Routing regret")
    reference_values = (
        reference["one_right_gate_selection_accuracy"]["mean"],
        reference["mean_routing_regret_nll"]["mean"],
    )
    rsg_values = (
        rsg["one_right_gate_selection_accuracy"]["mean"],
        rsg["mean_routing_regret_nll"]["mean"],
    )
    reference_errors = (
        reference["one_right_gate_selection_accuracy"]["sample_std"],
        reference["mean_routing_regret_nll"]["sample_std"],
    )
    rsg_errors = (
        rsg["one_right_gate_selection_accuracy"]["sample_std"],
        rsg["mean_routing_regret_nll"]["sample_std"],
    )
    for index, (values, errors, color, label) in enumerate(
        ((reference_values, reference_errors, colors[0], "HRGV reference"), (rsg_values, rsg_errors, colors[1], "RSG-HRGV"))
    ):
        axes[1].bar(
            [value + (index - 0.5) * width for value in range(2)],
            values,
            width=width,
            yerr=errors,
            capsize=4,
            color=color,
            label=label,
        )
    regret = paired["routing_regret"]
    axes[1].annotate(
        "Regret difference (RSG-HRGV):\n"
        f"{100 * regret['difference']:.2f} pp, 95% CI "
        f"[{100 * regret['ci_low']:.2f}, {100 * regret['ci_high']:.2f}] pp",
        xy=(1, rsg_values[1]),
        xytext=(0.45, 0.35),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#334155"},
        fontsize=9,
        color="#1F2937",
    )
    axes[1].set_title("RSG routing mechanism")
    axes[1].set_xticks((0, 1), route_labels, rotation=12, ha="right")
    axes[1].set_ylabel("Score / NLL regret")
    axes[1].set_ylim(0, 0.65)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RSG photographer-held-out results.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(args.summary, args.paired, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
