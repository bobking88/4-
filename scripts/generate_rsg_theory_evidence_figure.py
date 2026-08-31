from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


COLORS = {
    "ink": "#20303C",
    "muted": "#647681",
    "line": "#516775",
    "theory": "#DCE9F4",
    "evidence": "#DCEFE7",
    "warning": "#F6E7B8",
    "formula": "#F4F6F7",
    "accent": "#2F7D72",
}


def _load_routing_effect(path: Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    routing = payload["routing_regret"]
    return {
        "difference": float(routing["difference"]),
        "ci_low": float(routing["ci_low"]),
        "ci_high": float(routing["ci_high"]),
        "probability_favorable": float(routing["probability_favorable"]),
        "bootstrap_replicates": int(routing["bootstrap_replicates"]),
    }


def _box(axis, y: float, title: str, body: str) -> None:
    box = FancyBboxPatch(
        (0.035, y),
        0.91,
        0.20,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=COLORS["theory"],
        edgecolor=COLORS["line"],
        linewidth=0.9,
        transform=axis.transAxes,
    )
    axis.add_patch(box)
    axis.text(
        0.065,
        y + 0.151,
        title,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        0.065,
        y + 0.080,
        body,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.1,
        linespacing=1.35,
        color=COLORS["ink"],
    )


def _percent(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def generate_rsg_theory_evidence_figure(
    fixed_test_json: Path,
    photographer_holdout_json: Path,
    prefix: Path,
    portability_json: Path | None = None,
) -> dict[str, Path]:
    """Render formula-level RSG properties with the evidence they directly support."""
    fixed = _load_routing_effect(fixed_test_json)
    holdout = _load_routing_effect(photographer_holdout_json)
    portability = _load_routing_effect(portability_json) if portability_json else None
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )
    figure = plt.figure(figsize=(12.8, 7.2))
    gridspec = figure.add_gridspec(1, 2, width_ratios=(1.24, 1.0), wspace=0.10)
    theorem_axis = figure.add_subplot(gridspec[0, 0])
    effect_axis = figure.add_subplot(gridspec[0, 1])
    figure.patch.set_facecolor("white")

    theorem_axis.set_axis_off()
    theorem_axis.text(
        0.02,
        0.975,
        "RSG-HRGV: theory properties and directly matched evidence",
        transform=theorem_axis.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=COLORS["ink"],
    )
    theorem_axis.text(
        0.02,
        0.925,
        "The statements below characterize expert routing; they do not assert a universal classification gain.",
        transform=theorem_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=COLORS["muted"],
    )
    _box(
        theorem_axis,
        0.665,
        "Theorem B.1  Routing regret upper bound",
        "For a=p_d(y|x), b=p_m(y|x), g_o=I[a>=b], and a,b>=epsilon:\n"
        "-log p_g(y|x) + log max(a,b) <= |g-g_o| |a-b| / epsilon.",
    )
    _box(
        theorem_axis,
        0.410,
        "Theorem B.2  Soft-oracle approximation",
        "Delta=l_m-l_d,  g*=sigmoid(Delta/T_r).  The deviation from hard optimal routing\n"
        "is bounded by exp(-|Delta|/T_r); small gaps retain uncertainty instead of forcing a switch.",
    )
    _box(
        theorem_axis,
        0.155,
        "Theorem B.3  Local gradient isolation",
        "L_reg=w BCE(g,g*),  w=tanh(|Delta|/T_w).  Applying stop-gradient to the regret branch\n"
        "gives dL_reg/dtheta_e=0 for backbone and expert parameters through this branch.",
    )
    formula_box = FancyBboxPatch(
        (0.035, 0.025),
        0.91,
        0.087,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor=COLORS["formula"],
        edgecolor="#C8D1D7",
        linewidth=0.8,
        transform=theorem_axis.transAxes,
    )
    theorem_axis.add_patch(formula_box)
    theorem_axis.text(
        0.49,
        0.071,
        r"p_g=g p_d+(1-g)p_m,   g*=sigmoid((l_m-l_d)/T_r),   w=tanh(|l_m-l_d|/T_w)",
        transform=theorem_axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
        color=COLORS["ink"],
    )

    effect_axis.set_title(
        "Predefined routing-regret effect (RSG-HRGV minus HRGV)",
        fontsize=10.3,
        fontweight="bold",
        color=COLORS["ink"],
        pad=11,
    )
    rows = [("Fixed test", fixed), ("Photographer holdout", holdout)]
    if portability is not None:
        rows.append(("ResNet50 portability", portability))
    y_values = list(reversed(range(len(rows))))
    max_abs = max(abs(effect["ci_low"]) for _, effect in rows) * 100
    max_abs = max(max_abs, max(abs(effect["ci_high"]) for _, effect in rows) * 100, 1.0)
    bound = max_abs * 1.24
    effect_axis.axvline(0, color=COLORS["line"], linewidth=0.9, zorder=1)
    for y, (label, effect) in zip(y_values, rows):
        difference = effect["difference"] * 100
        ci_low = effect["ci_low"] * 100
        ci_high = effect["ci_high"] * 100
        effect_axis.errorbar(
            difference,
            y,
            xerr=[[difference - ci_low], [ci_high - difference]],
            fmt="o",
            color=COLORS["accent"],
            ecolor=COLORS["accent"],
            elinewidth=2.0,
            capsize=4,
            markersize=7,
            zorder=3,
        )
        effect_axis.text(
            difference,
            y + 0.17,
            f"{_percent(effect['difference'])}  [{_percent(effect['ci_low'])}, {_percent(effect['ci_high'])}]",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=COLORS["ink"],
        )
    effect_axis.set_xlim(-bound, bound)
    effect_axis.set_ylim(-0.80, len(rows) - 0.18)
    effect_axis.set_yticks(y_values, [label for label, _ in rows])
    effect_axis.set_xlabel("Difference in mean routing regret (percentage points; lower is better)", fontsize=7.2)
    effect_axis.grid(axis="x", color="#D8E0E4", linewidth=0.7)
    effect_axis.spines[["top", "right", "left"]].set_visible(False)
    effect_axis.tick_params(axis="y", length=0, labelsize=8)
    effect_axis.tick_params(axis="x", labelsize=7)
    all_intervals_below_zero = all(effect["ci_high"] < 0 for _, effect in rows)
    evidence_note = (
        "All displayed 95% bootstrap intervals remain below zero.\n"
        "This confirms reduced predefined routing regret, not classification superiority."
        if all_intervals_below_zero
        else "The displayed intervals define the evidence boundary for routing regret.\n"
        "Classification superiority is not claimed."
    )
    effect_axis.text(
        0.50,
        0.12,
        evidence_note,
        transform=effect_axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["warning"], "edgecolor": "#C7A965", "boxstyle": "round,pad=0.45"},
    )
    effect_axis.text(
        0.50,
        0.035,
        "classification superiority is not claimed",
        transform=effect_axis.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=COLORS["muted"],
    )
    figure.subplots_adjust(left=0.035, right=0.985, top=0.93, bottom=0.14, wspace=0.17)

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
                "figure_type": "theory-properties-and-evidence",
                "properties": ["Theorem B.1", "Theorem B.2", "Theorem B.3"],
                "effect_metric": "mean routing regret, RSG-HRGV minus HRGV; lower is better",
                "fixed_test": fixed,
                "photographer_holdout": holdout,
                "resnet50_portability": portability,
                "display_effects": {
                    "fixed_test": {
                        "difference": _percent(fixed["difference"]),
                        "ci": [_percent(fixed["ci_low"]), _percent(fixed["ci_high"])],
                    },
                    "photographer_holdout": {
                        "difference": _percent(holdout["difference"]),
                        "ci": [_percent(holdout["ci_low"]), _percent(holdout["ci_high"])],
                    },
                    **(
                        {
                            "resnet50_portability": {
                                "difference": _percent(portability["difference"]),
                                "ci": [
                                    _percent(portability["ci_low"]),
                                    _percent(portability["ci_high"]),
                                ],
                            }
                        }
                        if portability is not None
                        else {}
                    ),
                },
                "claim_boundary": "The paired intervals support reduced predefined routing regret only; classification superiority is not claimed.",
                "data_sources": [
                    str(fixed_test_json),
                    str(photographer_holdout_json),
                    *([str(portability_json)] if portability_json else []),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the RSG theory-evidence figure.")
    parser.add_argument("--fixed-test-json", required=True, type=Path)
    parser.add_argument("--photographer-holdout-json", required=True, type=Path)
    parser.add_argument("--portability-json", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate_rsg_theory_evidence_figure(
        args.fixed_test_json,
        args.photographer_holdout_json,
        args.output_prefix,
        portability_json=args.portability_json,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
