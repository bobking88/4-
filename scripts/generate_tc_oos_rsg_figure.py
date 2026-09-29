"""Generate the publication architecture and theorem-evidence figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


EVIDENCE_DIMENSIONS = (
    "q0_role_0",
    "q0_role_1",
    "q0_role_2",
    "q0_role_3",
    "qphi_role_0",
    "qphi_role_1",
    "qphi_role_2",
    "qphi_role_3",
    "abs_delta_role_0",
    "abs_delta_role_1",
    "abs_delta_role_2",
    "abs_delta_role_3",
    "candidate_gate",
    "ti_verifier_probability",
    "metallic_verifier_probability",
    "entropy_q0",
    "entropy_qphi",
    "js_divergence",
)

COLORS = {
    "ink": "#243447",
    "muted": "#607180",
    "line": "#78909C",
    "pale": "#F4F7F9",
    "navy": "#DCE8F2",
    "teal": "#D8EEE8",
    "gold": "#F5E9C8",
    "coral": "#F4D8D1",
    "safe": "#267C6B",
    "accent": "#356C9B",
}


def _box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#6F7F89",
    fontsize: float = 6.4,
    linewidth: float = 0.8,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight=weight,
        transform=axis.transAxes,
    )


def _arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    style: str = "-|>",
    linewidth: float = 0.9,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=linewidth,
            color=color or COLORS["line"],
            connectionstyle=connectionstyle,
            transform=axis.transAxes,
            clip_on=False,
        )
    )


def _panel_label(axis, label: str, title: str, x: float, y: float) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    axis.text(
        x + 0.026,
        y,
        title,
        transform=axis.transAxes,
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )


def _source_payload(invariants: Mapping[str, object]) -> dict[str, object]:
    return {
        "figure_contract": {
            "core_conclusion": (
                "A learned OOS route improves adaptive fusion while a deterministic "
                "target constraint limits per-image target-posterior harm and preserves "
                "an explicit q0 fallback."
            ),
            "archetype": "schematic-led composite",
            "backend": "Python/matplotlib",
            "final_size_mm": [183, 137],
            "claim_boundary": (
                "The bound is on target posterior mass, not a finite-sample hard-recall "
                "or industrial sorting guarantee."
            ),
        },
        "implemented_modules": [
            "frozen shared HRGV feature extractor",
            "direct role posterior",
            "species-to-role mapped posterior",
            "OOS candidate gate",
            "Ti and metallic residual verifiers on both branches",
            "18-dimensional projection evidence",
            "18-64-16-1 projection MLP",
            "deterministic target-risk cap",
            "explicit q0 fallback",
        ],
        "evidence_vector": {
            "dimension_count": len(EVIDENCE_DIMENSIONS),
            "ordered_dimensions": list(EVIDENCE_DIMENSIONS),
        },
        "theorem_labels": [
            "simplex validity",
            "per-image target posterior lower bound",
            "exact convex decomposition",
            "convex NLL upper bound",
        ],
        "numerical_invariants": dict(invariants),
    }


def generate_tc_oos_rsg_figure(
    output_prefix: Path,
    *,
    invariants: Mapping[str, object],
    invariant_output: Path,
) -> dict[str, object]:
    output_prefix = Path(output_prefix)
    invariant_output = Path(invariant_output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    invariant_output.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.02,
        0.975,
        "TC-OOS-RSG: target-constrained out-of-sample risk-supervised gating",
        transform=axis.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    axis.text(
        0.02,
        0.943,
        "Adaptive fusion and deterministic target protection are implemented as separate, auditable operators.",
        transform=axis.transAxes,
        fontsize=6.5,
        color=COLORS["muted"],
        va="top",
    )

    _panel_label(axis, "a", "Frozen verifier-complete expert branches", 0.02, 0.895)
    _box(axis, 0.025, 0.705, 0.085, 0.08, "image\nx", facecolor=COLORS["pale"], weight="bold")
    _box(axis, 0.145, 0.705, 0.105, 0.08, "shared\nfeatures h(x)", facecolor=COLORS["navy"], weight="bold")
    _box(axis, 0.29, 0.775, 0.10, 0.07, "direct role\np_d", facecolor=COLORS["navy"])
    _box(axis, 0.29, 0.655, 0.10, 0.07, "species map\np_m", facecolor=COLORS["navy"])
    _arrow(axis, (0.11, 0.745), (0.145, 0.745))
    _arrow(axis, (0.25, 0.745), (0.29, 0.81), connectionstyle="arc3,rad=-0.12")
    _arrow(axis, (0.25, 0.745), (0.29, 0.69), connectionstyle="arc3,rad=0.12")

    _box(axis, 0.435, 0.775, 0.11, 0.07, "fixed mixture\n0.5 p_d + 0.5 p_m", facecolor=COLORS["gold"], fontsize=5.9)
    _box(axis, 0.435, 0.655, 0.11, 0.07, "OOS gate g_phi\ng p_d + (1-g) p_m", facecolor=COLORS["gold"], fontsize=5.7)
    for y_start, y_end in ((0.81, 0.81), (0.69, 0.81), (0.81, 0.69), (0.69, 0.69)):
        _arrow(axis, (0.39, y_start), (0.435, y_end), connectionstyle="arc3,rad=0.08")

    _box(
        axis,
        0.59,
        0.775,
        0.115,
        0.07,
        "V_Ti + V_Metal\nverifier-complete",
        facecolor=COLORS["teal"],
        fontsize=5.8,
        weight="bold",
    )
    _box(
        axis,
        0.59,
        0.655,
        0.115,
        0.07,
        "V_Ti + V_Metal\nverifier-complete",
        facecolor=COLORS["teal"],
        fontsize=5.8,
        weight="bold",
    )
    _arrow(axis, (0.545, 0.81), (0.59, 0.81))
    _arrow(axis, (0.545, 0.69), (0.59, 0.69))
    _box(axis, 0.75, 0.775, 0.085, 0.07, "q0\nfallback", facecolor=COLORS["coral"], weight="bold")
    _box(axis, 0.75, 0.655, 0.085, 0.07, "q_phi\ncandidate", facecolor=COLORS["coral"], weight="bold")
    _arrow(axis, (0.705, 0.81), (0.75, 0.81))
    _arrow(axis, (0.705, 0.69), (0.75, 0.69))

    _box(
        axis,
        0.86,
        0.688,
        0.115,
        0.115,
        "frozen experts\nno outer-fold update\nexplicit q0 fallback",
        facecolor=COLORS["pale"],
        fontsize=5.7,
        edgecolor=COLORS["accent"],
    )
    _arrow(axis, (0.835, 0.81), (0.86, 0.775), style="-[", color=COLORS["accent"])
    _arrow(axis, (0.835, 0.69), (0.86, 0.715), style="-[", color=COLORS["accent"])

    _panel_label(axis, "b", "Learned 18D risk evidence and route proposal", 0.02, 0.59)
    _box(
        axis,
        0.035,
        0.42,
        0.33,
        0.105,
        "e(x) = [ q0(4), q_phi(4), |q_phi-q0|(4),\n"
        "g_phi(1), V_Ti(1), V_Metal(1), H(q0), H(q_phi), JS(1) ]\n"
        "ordered dimension count = 18",
        facecolor=COLORS["navy"],
        fontsize=6.0,
        weight="bold",
    )
    _box(
        axis,
        0.43,
        0.42,
        0.23,
        0.105,
        "projection MLP\n18 -> 64 -> LN -> SiLU\n-> Dropout(0.10) -> 16 -> 1",
        facecolor=COLORS["gold"],
        fontsize=6.0,
        weight="bold",
    )
    _box(axis, 0.72, 0.438, 0.12, 0.07, "sigmoid\nrho_raw", facecolor=COLORS["teal"], weight="bold")
    _arrow(axis, (0.365, 0.472), (0.43, 0.472))
    _arrow(axis, (0.66, 0.472), (0.72, 0.472))
    axis.text(
        0.87,
        0.475,
        "trained on disjoint\nprojector-fit groups",
        transform=axis.transAxes,
        fontsize=5.8,
        color=COLORS["muted"],
        va="center",
        ha="center",
    )
    _arrow(axis, (0.84, 0.472), (0.86, 0.472), style="-[")

    _panel_label(axis, "c", "Deterministic target-risk projection", 0.02, 0.35)
    _box(axis, 0.035, 0.21, 0.16, 0.08, "d_T = [q0,T - q_phi,T]+", facecolor=COLORS["pale"], fontsize=6.1)
    _box(axis, 0.235, 0.21, 0.16, 0.08, "c_T = min(1, epsilon_T / d_T)\n(c_T=1 if d_T=0)", facecolor=COLORS["pale"], fontsize=5.7)
    _box(axis, 0.435, 0.21, 0.15, 0.08, "rho_tilde = min(rho_raw, c_T)", facecolor=COLORS["coral"], fontsize=6.0, weight="bold")
    _box(axis, 0.63, 0.21, 0.19, 0.08, "q_TC = (1-rho_tilde) q0\n+ rho_tilde q_phi", facecolor=COLORS["teal"], fontsize=6.0, weight="bold")
    _box(axis, 0.86, 0.21, 0.115, 0.08, "safe output\nor q0 fallback", facecolor=COLORS["navy"], fontsize=6.0, weight="bold")
    for start, end in (((0.195, 0.25), (0.235, 0.25)), ((0.395, 0.25), (0.435, 0.25)), ((0.585, 0.25), (0.63, 0.25)), ((0.82, 0.25), (0.86, 0.25))):
        _arrow(axis, start, end, color=COLORS["safe"])

    _panel_label(axis, "d", "Proved properties and numerical audit", 0.02, 0.155)
    theorem_text = (
        "P1  q_TC is a valid probability simplex     "
        "P2  q_TC,T >= q0,T - epsilon_T     "
        "P3  q_TC-q0 = rho_tilde(q_phi-q0)\n"
        "P4  NLL(q_TC) <= (1-rho_tilde)NLL(q0) + rho_tilde NLL(q_phi)"
    )
    _box(axis, 0.035, 0.035, 0.67, 0.09, theorem_text, facecolor=COLORS["pale"], fontsize=5.8, weight="bold")
    passed = bool(invariants.get("all_invariants_pass", False))
    audit_scope = str(invariants.get("scope", "unspecified scope"))
    audit_text = (
        f"numerical audit [{audit_scope}]: {'PASS' if passed else 'CHECK'}\n"
        f"n = {invariants.get('row_count', 'NA')} | safety violations = "
        f"{invariants.get('target_safety_violation_count', 'NA')}\n"
        f"max decomposition residual = {float(invariants.get('max_decomposition_residual', 0.0)):.2e}"
    )
    _box(
        axis,
        0.74,
        0.035,
        0.235,
        0.09,
        audit_text,
        facecolor=COLORS["teal"] if passed else COLORS["gold"],
        edgecolor=COLORS["safe"] if passed else "#A67C00",
        fontsize=5.7,
        weight="bold",
    )

    source = _source_payload(invariants)
    source_path = output_prefix.with_name(output_prefix.name + "_source.json")
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    invariant_output.write_text(
        json.dumps(dict(invariants), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text(
        "\n".join(
            line.rstrip()
            for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)
    return source


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--invariants-input", type=Path, required=True)
    parser.add_argument("--invariant-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    invariants = json.loads(args.invariants_input.read_text(encoding="utf-8"))
    source = generate_tc_oos_rsg_figure(
        args.output_prefix,
        invariants=invariants,
        invariant_output=args.invariant_output,
    )
    print(json.dumps(source, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
