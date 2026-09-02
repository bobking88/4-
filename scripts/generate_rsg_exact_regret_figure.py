from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ink": "#20303C",
    "signal": "#237A8B",
    "accent": "#D46A4C",
    "muted": "#647681",
    "grid": "#D8E0E4",
    "note": "#F6E7B8",
    "curve_1": "#237A8B",
    "curve_2": "#5C8E9F",
    "curve_3": "#D46A4C",
    "curve_4": "#8E6DAE",
}


def _load_summary(summary_path: Path) -> dict[str, object]:
    evidence = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    overall = evidence.get("overall", {})
    required = {
        "run_count",
        "sample_count",
        "exact_decomposition_max_abs_residual",
        "exact_decomposition_violation_count",
    }
    if not required <= set(overall):
        raise ValueError("Exact-decomposition summary is incomplete.")
    if int(overall["exact_decomposition_violation_count"]) != 0:
        raise ValueError("Figure is reserved for zero-violation exact-decomposition checks.")
    return evidence


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.8,
        }
    )


def generate_rsg_exact_regret_figure(
    summary_path: Path, prefix: Path
) -> dict[str, Path]:
    """Render the analytic regret curve and the supporting replay-consistency note."""
    evidence = _load_summary(summary_path)
    overall = evidence["overall"]
    numeric_settings = evidence.get("numeric_settings", {})
    tolerance = float(numeric_settings.get("tolerance", 2e-6))

    _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(8.95, 5.1), constrained_layout=False)
    figure.patch.set_facecolor("white")
    hero_axis, sensitivity_axis = axes

    u = np.linspace(0.0, 0.92, 400)
    regret = -np.log1p(-u)
    hero_axis.plot(u, regret, color=COLORS["signal"], linewidth=2.6)
    hero_axis.fill_between(u, 0.0, regret, color=COLORS["signal"], alpha=0.10)
    hero_axis.axline((0.0, 0.0), slope=1.0, color=COLORS["muted"], linewidth=1.0, linestyle="--")
    hero_axis.text(
        0.44,
        1.84,
        "r = -log(1 - u)",
        color=COLORS["signal"],
        fontsize=10,
        fontweight="bold",
    )
    hero_axis.text(
        0.44,
        1.55,
        "u = delta d / M",
        color=COLORS["ink"],
        fontsize=8,
    )
    hero_axis.annotate(
        "first-order reference r = u",
        xy=(0.28, 0.28),
        xytext=(0.05, 0.75),
        arrowprops={"arrowstyle": "-", "color": COLORS["muted"], "linewidth": 0.8},
        color=COLORS["muted"],
        fontsize=7,
    )
    hero_axis.set_title("Exact convex-fusion decomposition", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    hero_axis.set_xlabel("Normalized probability loss u", fontsize=8)
    hero_axis.set_ylabel("Routing regret r (NLL)", fontsize=8)
    hero_axis.set_xlim(0.0, 0.92)
    hero_axis.set_ylim(0.0, 2.75)
    hero_axis.grid(axis="both", color=COLORS["grid"], linewidth=0.6)
    hero_axis.spines[["top", "right"]].set_visible(False)
    hero_axis.tick_params(labelsize=7)

    delta = np.linspace(0.0, 1.0, 400)
    ratios = (0.15, 0.35, 0.60, 0.85)
    colors = (COLORS["curve_1"], COLORS["curve_2"], COLORS["curve_3"], COLORS["curve_4"])
    for ratio, color in zip(ratios, colors):
        sensitivity_axis.plot(
            delta,
            -np.log1p(-delta * ratio),
            linewidth=2.0,
            color=color,
            label=f"d/M = {ratio:.2f}",
        )
    sensitivity_axis.set_title("Gate-error sensitivity at fixed expert gaps", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    sensitivity_axis.set_xlabel("Gate deviation delta = |g - g_o|", fontsize=8)
    sensitivity_axis.set_ylabel("Routing regret r (NLL)", fontsize=8)
    sensitivity_axis.set_xlim(0.0, 1.0)
    sensitivity_axis.set_ylim(0.0, 2.05)
    sensitivity_axis.grid(axis="both", color=COLORS["grid"], linewidth=0.6)
    sensitivity_axis.spines[["top", "right"]].set_visible(False)
    sensitivity_axis.tick_params(labelsize=7)
    sensitivity_axis.legend(loc="upper left", fontsize=7, frameon=False, title="Expert gap")

    figure.text(0.07, 0.955, "a", fontsize=11, fontweight="bold", color=COLORS["ink"])
    figure.text(0.545, 0.955, "b", fontsize=11, fontweight="bold", color=COLORS["ink"])
    figure.text(
        0.5,
        0.045,
        "mechanism verification only: exact identity checked on "
        f"{int(overall['run_count'])} checkpoint replays / {int(overall['sample_count']):,} image evaluations; "
        f"max residual = {float(overall['exact_decomposition_max_abs_residual']):.2e} "
        f"(tolerance = {tolerance:.1e}), violations = 0.  "
        "not a classification-performance comparison.",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["note"], "edgecolor": "#C7A965", "boxstyle": "round,pad=0.42"},
    )
    figure.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.19, wspace=0.28)

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
    figure.savefig(
        outputs["tiff"],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    outputs["source_description"].write_text(
        json.dumps(
            {
                "figure_type": "rsg-exact-convex-fusion-regret-decomposition",
                "core_conclusion": "The convex-fusion routing regret is exactly -log(1-u), where u equals the product of gate deviation and normalized expert probability gap.",
                "formula": {"u": "delta d / M", "r": "-log(1-u)"},
                "sample_count": int(overall["sample_count"]),
                "run_count": int(overall["run_count"]),
                "exact_decomposition_max_abs_residual": float(overall["exact_decomposition_max_abs_residual"]),
                "exact_decomposition_violation_count": int(overall["exact_decomposition_violation_count"]),
                "tolerance": tolerance,
                "claim_boundary": evidence["claim_boundary"],
                "source_summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact RSG convex-fusion regret figure.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate_rsg_exact_regret_figure(args.summary_json, args.output_prefix)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
