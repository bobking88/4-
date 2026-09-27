"""Generate the independent OOS-RSG confirmation effect figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "ink": "#24343D",
    "muted": "#66757E",
    "grid": "#D9E0E3",
    "benefit": "#177E89",
    "harm": "#C85A54",
    "neutral": "#82939C",
    "panel": "#F5F7F8",
    "pass": "#DCEFE7",
    "fail": "#F5DEDA",
}


def build_layout_contract() -> dict[str, float]:
    return {
        "panel_note_y": 0.95,
        "decision_title_y": 0.20,
        "decision_body_y": 0.04,
    }


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _benefit(summary: dict[str, Any], *, reverse: bool, scale: float = 1.0) -> dict[str, Any]:
    sign = -1.0 if reverse else 1.0
    low = sign * float(summary["ci_low"]) * scale
    high = sign * float(summary["ci_high"]) * scale
    return {
        "benefit": sign * float(summary["difference"]) * scale,
        "ci_low": min(low, high),
        "ci_high": max(low, high),
        "per_seed": [
            sign * float(value) * scale
            for value in summary["per_seed_differences"]
        ],
    }


def build_figure_data(analysis: dict[str, Any]) -> dict[str, Any]:
    comparisons = analysis["comparisons"]
    nll_specs = (
        ("unseen_minus_seen", "Seen gate"),
        ("unseen_minus_equal", "Equal fusion"),
        ("unseen_minus_original_joint_gate", "Original joint gate"),
    )
    nll_benefits = []
    for key, label in nll_specs:
        effect = _benefit(
            comparisons[key]["summary"]["final_nll_nats"],
            reverse=True,
        )
        nll_benefits.append({"comparison": label, **effect})

    equal_summary = comparisons["unseen_minus_equal"]["summary"]
    metric_specs = (
        ("macro_f1", "Macro F1", False),
        ("target_recall", "Target recall", False),
        ("ti_intrusion", "Ti intrusion", True),
        ("metal_intrusion", "Metallic intrusion", True),
    )
    tradeoff = []
    for key, label, reverse in metric_specs:
        effect = _benefit(equal_summary[key], reverse=reverse, scale=100.0)
        tradeoff.append(
            {
                "metric": label,
                "benefit_pp": effect["benefit"],
                "ci_low_pp": effect["ci_low"],
                "ci_high_pp": effect["ci_high"],
                "per_seed_pp": effect["per_seed"],
            }
        )

    return {
        "nll_benefits": nll_benefits,
        "tradeoff_vs_equal": tradeoff,
        "promotion": analysis["promotion_decision"],
        "bootstrap_replicates": analysis["bootstrap_replicates"],
        "expert_seeds": analysis["expert_seeds"],
        "scope": analysis["scope"],
        "evidence_boundary": analysis["evidence_boundary"],
    }


def _plot_interval(
    axis: plt.Axes,
    y: float,
    value: float,
    low: float,
    high: float,
    color: str,
    marker: str = "o",
    size: float = 35,
) -> None:
    axis.errorbar(
        value,
        y,
        xerr=np.array([[value - low], [high - value]]),
        fmt=marker,
        markersize=np.sqrt(size),
        color=color,
        ecolor=color,
        elinewidth=1.35,
        capsize=2.5,
        capthick=1.0,
        zorder=4,
    )


def generate_figure(analysis_path: Path, output_prefix: Path) -> dict[str, Path]:
    _configure_style()
    layout = build_layout_contract()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    data = build_figure_data(analysis)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.25, 3.45), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.05, 1.35, 0.90))
    axis_nll = figure.add_subplot(grid[0, 0])
    axis_tradeoff = figure.add_subplot(grid[0, 1])
    axis_decision = figure.add_subplot(grid[0, 2])
    figure.patch.set_facecolor("white")

    figure.suptitle(
        "Independent OOS routing confirmation",
        x=0.015,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    # Panel a: positive values mean lower final NLL.
    nll_rows = data["nll_benefits"]
    y_nll = np.arange(len(nll_rows))[::-1]
    axis_nll.axvline(0, color=PALETTE["grid"], linewidth=0.9, zorder=0)
    for y, row in zip(y_nll, nll_rows):
        _plot_interval(
            axis_nll,
            y,
            row["benefit"],
            row["ci_low"],
            row["ci_high"],
            PALETTE["benefit"],
        )
        offsets = np.array([-0.11, 0.0, 0.11])
        axis_nll.scatter(
            row["per_seed"],
            y + offsets,
            s=8,
            facecolor="white",
            edgecolor=PALETTE["benefit"],
            linewidth=0.75,
            zorder=5,
        )
    axis_nll.set_yticks(y_nll, [row["comparison"] for row in nll_rows])
    axis_nll.set_xlabel("NLL benefit (nat)")
    axis_nll.set_title("a  Proper-scoring risk", loc="left", fontweight="bold")
    axis_nll.grid(axis="x", color=PALETTE["grid"], linewidth=0.55, alpha=0.8)
    axis_nll.text(
        0.98,
        layout["panel_note_y"],
        "positive = lower NLL; circles = seeds",
        transform=axis_nll.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=PALETTE["muted"],
    )
    equal_row = next(row for row in nll_rows if row["comparison"] == "Equal fusion")
    axis_nll.annotate(
        "3/3 experts",
        xy=(equal_row["benefit"], y_nll[1]),
        xytext=(equal_row["benefit"] + 0.013, y_nll[1] + 0.36),
        arrowprops={"arrowstyle": "-", "color": PALETTE["muted"], "lw": 0.7},
        fontsize=5.8,
        color=PALETTE["muted"],
    )

    # Panel b: all effects are direction-aligned so positive means beneficial.
    trade_rows = data["tradeoff_vs_equal"]
    y_trade = np.arange(len(trade_rows))[::-1]
    axis_tradeoff.axvspan(-1.0, 0.0, color=PALETTE["fail"], alpha=0.18, zorder=0)
    axis_tradeoff.axvline(0, color=PALETTE["grid"], linewidth=0.9, zorder=1)
    for y, row in zip(y_trade, trade_rows):
        value = row["benefit_pp"]
        color = PALETTE["benefit"] if value >= 0 else PALETTE["harm"]
        _plot_interval(
            axis_tradeoff,
            y,
            value,
            row["ci_low_pp"],
            row["ci_high_pp"],
            color,
        )
    axis_tradeoff.set_yticks(y_trade, [row["metric"] for row in trade_rows])
    axis_tradeoff.set_xlabel("Beneficial change vs equal fusion (percentage points)")
    axis_tradeoff.set_title("b  Classification trade-off", loc="left", fontweight="bold")
    axis_tradeoff.grid(axis="x", color=PALETTE["grid"], linewidth=0.55, alpha=0.8)
    target = next(row for row in trade_rows if row["metric"] == "Target recall")
    target_y = y_trade[[row["metric"] for row in trade_rows].index("Target recall")]
    axis_tradeoff.annotate(
        "fails -1 pp guardrail",
        xy=(target["benefit_pp"], target_y),
        xytext=(-4.55, target_y - 0.72),
        arrowprops={"arrowstyle": "->", "color": PALETTE["harm"], "lw": 0.8},
        fontsize=5.8,
        color=PALETTE["harm"],
    )
    axis_tradeoff.text(
        0.98,
        layout["panel_note_y"],
        "positive = benefit; intrusion signs reversed",
        transform=axis_tradeoff.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=PALETTE["muted"],
    )

    # Panel c: the preregistered decision, not a post-hoc narrative.
    axis_decision.set_facecolor(PALETTE["panel"])
    axis_decision.set_title("c  Preregistered decision", loc="left", fontweight="bold")
    axis_decision.set_xticks([])
    axis_decision.set_yticks([])
    for spine in axis_decision.spines.values():
        spine.set_visible(False)
    criteria = data["promotion"]["criteria"]
    checklist = (
        ("NLL CI below zero", criteria["nll_ci_below_zero"]),
        ("2/3 seed direction", criteria["nll_favorable_in_at_least_two_seeds"]),
        ("Macro F1 guardrail", criteria["macro_f1_guardrail"]),
        ("Target recall guardrail", criteria["target_recall_guardrail"]),
        ("Ti intrusion guardrail", criteria["ti_intrusion_guardrail"]),
        ("Metal intrusion guardrail", criteria["metal_intrusion_guardrail"]),
    )
    y = 0.86
    for label, passed in checklist:
        axis_decision.text(
            0.08,
            y,
            "PASS" if passed else "FAIL",
            transform=axis_decision.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=PALETTE["benefit"] if passed else PALETTE["harm"],
        )
        axis_decision.text(
            0.37,
            y,
            label,
            transform=axis_decision.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            color=PALETTE["ink"],
        )
        y -= 0.105
    axis_decision.text(
        0.08,
        layout["decision_title_y"],
        "Not promoted",
        transform=axis_decision.transAxes,
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["harm"],
    )
    axis_decision.text(
        0.08,
        layout["decision_body_y"],
        "NLL and hard-negative control improved,\n"
        "but target protection was not retained.",
        transform=axis_decision.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=PALETTE["muted"],
        linespacing=1.2,
    )

    outputs: dict[str, Path] = {}
    export_specs = (
        ("png", {"dpi": 300}),
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    )
    for extension, kwargs in export_specs:
        path = output_prefix.with_suffix(f".{extension}")
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        if extension == "svg":
            svg = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
        outputs[extension] = path
    plt.close(figure)

    source_path = output_prefix.with_name(output_prefix.name + "_source.json")
    source_payload = {
        "figure_archetype": "quantitative grid",
        "core_conclusion": (
            "Independent out-of-sample routing improved final NLL and hard-negative "
            "intrusion control but failed the preregistered target-recall guardrail."
        ),
        "source_analysis": str(analysis_path),
        "metric_direction": (
            "All plotted values are direction-aligned benefits; intrusion-rate "
            "differences are sign-reversed."
        ),
        "interval": (
            "95% paired two-stage bootstrap interval: expert seeds first, then "
            "split_group_id clusters within mineral-role strata."
        ),
        "figure_data": data,
        "exports": {key: str(value) for key, value in outputs.items()},
    }
    source_path.write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs["source_data"] = source_path
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = generate_figure(args.analysis, args.output_prefix)
    print(
        json.dumps(
            {key: str(value) for key, value in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
