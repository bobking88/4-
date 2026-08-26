from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "ink": "#20303C",
    "muted": "#647681",
    "line": "#516775",
    "backbone": "#DCE9F4",
    "expert": "#DCEFE7",
    "gate": "#F6E7B8",
    "verify": "#F3D8D3",
    "output": "#D9E2F5",
    "training": "#E9E0F1",
    "formula": "#F4F6F7",
}


def _box(
    axis,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    color: str,
    title_size: float = 8.0,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color,
        edgecolor=COLORS["line"],
        linewidth=0.9,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.61,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + height * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=6.3,
        color=COLORS["muted"],
        linespacing=1.15,
        zorder=3,
    )


def _arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    dashed: bool = False,
    connectionstyle: str = "arc3,rad=0",
    color: str | None = None,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        linestyle="--" if dashed else "-",
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=1,
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.018,
            label,
            ha="center",
            va="center",
            fontsize=5.8,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
            zorder=4,
        )


def generate_hrgv_architecture_figure(prefix: Path) -> dict[str, Path]:
    """Draw the HRGV-Net architecture and export a publication bundle."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )
    figure, axis = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    axis.text(
        0.02,
        0.965,
        "RSG-HRGV-Net: regret-supervised hierarchical evidence routing and risk verification",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        0.02,
        0.925,
        "共享视觉表征支持角色/种类双专家；训练期软后悔目标监督门控选择更可靠的证据",
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )

    _box(axis, (0.025, 0.60), 0.095, 0.13, "Input image", "RGB mineral\nspecimen", COLORS["formula"])
    _box(axis, (0.155, 0.57), 0.135, 0.19, "EfficientNet-B0", "shared visual\nbackbone", COLORS["backbone"], 8.8)
    _box(axis, (0.325, 0.60), 0.085, 0.13, "Feature h", "1,280-D shared\nrepresentation", COLORS["backbone"])

    _box(axis, (0.455, 0.765), 0.145, 0.13, "Direct role expert", "p_d(y|x), 4 roles", COLORS["expert"])
    _box(axis, (0.455, 0.555), 0.145, 0.13, "Species expert", "p_s(k|x), 17 minerals", COLORS["expert"])
    _box(axis, (0.635, 0.555), 0.12, 0.13, "Mapping matrix A", "p_m = A p_s\n17 species -> 4 roles", COLORS["expert"], 7.6)
    _box(
        axis,
        (0.635, 0.765),
        0.12,
        0.13,
        "Reliability gate",
        "Regret-supervised gate\nsoft oracle target (train)",
        COLORS["gate"],
        7.4,
    )
    _box(axis, (0.79, 0.69), 0.13, 0.14, "Adaptive fusion", "p_f = g p_d + (1-g) p_m", COLORS["gate"], 8.2)

    _box(axis, (0.455, 0.325), 0.145, 0.13, "Ti-bearing verifier", "target vs Ti-bearing\nv_Ti(x)", COLORS["verify"], 7.6)
    _box(axis, (0.635, 0.325), 0.145, 0.13, "Metallic verifier", "target vs metallic\nv_Met(x)", COLORS["verify"], 7.6)
    _box(axis, (0.80, 0.43), 0.145, 0.155, "Neutral-zone residual correction", "supportive scores: identity\nopposing scores: attenuate target", COLORS["verify"], 7.0)
    _box(axis, (0.855, 0.205), 0.12, 0.12, "Final role posterior", "q(y|x), 4 roles", COLORS["output"], 7.7)
    _box(axis, (0.305, 0.285), 0.13, 0.13, "Role-aware contrastive head", "z(x), training only", COLORS["training"], 6.5)

    _arrow(axis, (0.12, 0.665), (0.155, 0.665))
    _arrow(axis, (0.29, 0.665), (0.325, 0.665))
    _arrow(axis, (0.41, 0.68), (0.455, 0.83), connectionstyle="arc3,rad=0.08")
    _arrow(axis, (0.41, 0.65), (0.455, 0.62), connectionstyle="arc3,rad=-0.05")
    _arrow(axis, (0.60, 0.62), (0.635, 0.62))
    _arrow(axis, (0.60, 0.83), (0.635, 0.83), label="p_d")
    _arrow(axis, (0.755, 0.62), (0.79, 0.73), label="p_m", connectionstyle="arc3,rad=-0.12")
    _arrow(axis, (0.755, 0.83), (0.79, 0.77), label="g")
    _arrow(axis, (0.60, 0.815), (0.635, 0.80), dashed=True, label="uncertainty")
    _arrow(axis, (0.755, 0.66), (0.635, 0.79), dashed=True, connectionstyle="arc3,rad=0.12")

    _arrow(axis, (0.38, 0.60), (0.50, 0.455), connectionstyle="arc3,rad=0.14")
    _arrow(axis, (0.39, 0.61), (0.67, 0.455), connectionstyle="arc3,rad=0.18")
    _arrow(axis, (0.37, 0.60), (0.37, 0.415), dashed=True, label="projection")
    _arrow(axis, (0.855, 0.69), (0.86, 0.585), label="p_f")
    _arrow(axis, (0.60, 0.39), (0.80, 0.49), label="v_Ti", connectionstyle="arc3,rad=-0.08")
    _arrow(axis, (0.78, 0.39), (0.85, 0.43), label="v_Met")
    _arrow(axis, (0.875, 0.43), (0.91, 0.325), label="q")

    axis.text(
        0.02,
        0.235,
        "Training objective",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
    )
    objective = (
        r"$\mathcal{L}=\mathcal{L}_{role}+\lambda_d\mathcal{L}_{direct}+"
        r"\lambda_s\mathcal{L}_{species}+\lambda_c\mathcal{L}_{KL}+"
        r"\lambda_v(\mathcal{L}_{Ti}+\mathcal{L}_{Met})+\lambda_{con}\mathcal{L}_{contrast}+"
        r"\lambda_g\mathcal{L}_{reg}$"
    )
    axis.text(0.02, 0.195, objective, fontsize=8.0, color=COLORS["ink"], ha="left", va="center")
    axis.text(
        0.02,
        0.155,
        "Solid arrows: inference path     Dashed arrows: uncertainty/training cues     "
        "The regret target uses stop-gradient expert evidence; verifier coupling is retained in the formal model.",
        fontsize=6.2,
        color=COLORS["muted"],
        ha="left",
    )

    formula_box = FancyBboxPatch(
        (0.02, 0.035),
        0.955,
        0.085,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        facecolor=COLORS["formula"],
        edgecolor="#C8D1D7",
        linewidth=0.8,
    )
    axis.add_patch(formula_box)
    regret = (
        r"$\Delta=\ell_m-\ell_d,\quad g^*=\sigma(\Delta/T_r),\quad "
        r"w=\tanh(|\Delta|/T_w),\quad \mathcal{L}_{reg}=w\,\mathrm{BCE}(g,g^*)$"
    )
    correction = (
        r"$c_h(v_h)=[\tau_h-v_h]_+/\tau_h,\quad "
        r"q_T\propto p_{f,T}\exp[-\beta_{Ti}c_{Ti}-\beta_{Met}c_{Met}],\quad q_j\propto p_{f,j}$"
    )
    axis.text(0.497, 0.092, regret, ha="center", va="center", fontsize=7.8, color=COLORS["ink"])
    axis.text(0.497, 0.058, correction, ha="center", va="center", fontsize=7.5, color=COLORS["ink"])

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
                "core_conclusion": "A soft regret target supervises hierarchical expert routing before asymmetric hard-negative verification.",
                "archetype": "schematic-led composite",
                "backend": "Python/matplotlib",
                "final_width_mm": 183,
                "evidence_scope": "Architecture and mathematical inference path; no industrial performance claim.",
                "exports": {name: str(path) for name, path in outputs.items() if name != "source_description"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def generate_cgdc_architecture_figure(prefix: Path) -> dict[str, Path]:
    """Draw the CGDC-RSG-HRGV-Net evidence-decomposition architecture."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )
    figure, axis = plt.subplots(figsize=(13.0, 7.2), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.patch.set_facecolor("white")
    axis.text(
        0.02, 0.965,
        "CGDC-RSG-HRGV-Net: decomposed cross-granularity evidence and disagreement calibration",
        ha="left", va="top", fontsize=11.4, fontweight="bold", color=COLORS["ink"],
    )
    axis.text(
        0.02, 0.928,
        "双残差适配器将角色与种类证据显式分解；仅在两专家分歧时对融合后验施加有界校准。",
        ha="left", va="top", fontsize=7.2, color=COLORS["muted"],
    )

    _box(axis, (0.02, 0.62), 0.085, 0.12, "Input image", "RGB mineral\nspecimen", COLORS["formula"])
    _box(axis, (0.13, 0.59), 0.12, 0.18, "EfficientNet-B0", "shared backbone\nh in R^1280", COLORS["backbone"], 8.2)
    _box(axis, (0.29, 0.77), 0.12, 0.115, "Direct residual adapter", "u_d = h + A_d(h)", COLORS["expert"], 7.1)
    _box(axis, (0.29, 0.52), 0.12, 0.115, "Species residual adapter", "u_s = h + A_s(h)", COLORS["expert"], 7.0)
    _box(axis, (0.45, 0.77), 0.11, 0.115, "Role expert", "p_d(y|x)\n4 roles", COLORS["expert"])
    _box(axis, (0.45, 0.52), 0.11, 0.115, "Species expert", "p_s(k|x)\n17 minerals", COLORS["expert"])
    _box(axis, (0.59, 0.52), 0.10, 0.115, "Mapping A", "p_m = A p_s\n17 -> 4", COLORS["expert"])
    _box(axis, (0.60, 0.77), 0.10, 0.115, "RSG gate", "g*, gap-weighted\nsoft routing", COLORS["gate"], 7.2)
    _box(axis, (0.73, 0.68), 0.12, 0.13, "RSG fusion", "p_f = g p_d\n+ (1-g)p_m", COLORS["gate"])

    _box(axis, (0.55, 0.31), 0.14, 0.13, "Jensen-Shannon disagreement", "D_JS(p_d || p_m)\nrho = 1 - exp(-D_JS)", COLORS["training"], 6.7)
    _box(axis, (0.73, 0.42), 0.15, 0.15, "Disagreement-triggered calibrator", "z_c=[u_d,u_s,u_d*u_s,|u_d-u_s|,...]\ntanh-bounded residual s", COLORS["training"], 6.2)
    _box(axis, (0.90, 0.50), 0.08, 0.12, "Calibrated posterior", "p_c=softmax\n(log p_f+rho s)", COLORS["output"], 6.2)
    _box(axis, (0.80, 0.21), 0.15, 0.14, "Residual hard-negative verifiers", "Ti-bearing and metallic\ncontradiction penalties", COLORS["verify"], 6.7)
    _box(axis, (0.90, 0.08), 0.08, 0.10, "Final posterior", "q(y|x)", COLORS["output"], 7.1)
    _box(axis, (0.28, 0.25), 0.16, 0.12, "Adapter decomposition loss", "L_dec=mean cos^2(A_d(h), A_s(h))", COLORS["training"], 6.2)

    _arrow(axis, (0.105, 0.68), (0.13, 0.68))
    _arrow(axis, (0.25, 0.71), (0.29, 0.825), connectionstyle="arc3,rad=0.08")
    _arrow(axis, (0.25, 0.65), (0.29, 0.58), connectionstyle="arc3,rad=-0.07")
    _arrow(axis, (0.41, 0.827), (0.45, 0.827))
    _arrow(axis, (0.41, 0.578), (0.45, 0.578))
    _arrow(axis, (0.56, 0.578), (0.59, 0.578))
    _arrow(axis, (0.56, 0.827), (0.60, 0.827), label="p_d")
    _arrow(axis, (0.69, 0.58), (0.73, 0.71), label="p_m", connectionstyle="arc3,rad=-0.13")
    _arrow(axis, (0.70, 0.83), (0.73, 0.78), label="g")
    _arrow(axis, (0.56, 0.80), (0.55, 0.44), dashed=True, label="p_d")
    _arrow(axis, (0.69, 0.60), (0.66, 0.44), dashed=True, label="p_m")
    _arrow(axis, (0.69, 0.375), (0.73, 0.49), label="rho")
    _arrow(axis, (0.85, 0.74), (0.80, 0.57), label="p_f", connectionstyle="arc3,rad=0.15")
    _arrow(axis, (0.88, 0.50), (0.90, 0.56), label="rho s")
    _arrow(axis, (0.94, 0.50), (0.87, 0.35), label="p_c", connectionstyle="arc3,rad=0.08")
    _arrow(axis, (0.91, 0.21), (0.94, 0.18), label="q")
    _arrow(axis, (0.35, 0.52), (0.36, 0.37), dashed=True, label="A_d,A_s")

    axis.text(0.02, 0.15, "Objective", fontsize=8.3, fontweight="bold", color=COLORS["ink"])
    axis.text(
        0.02, 0.105,
        r"$\mathcal{L}_{CGDC}=\mathcal{L}_{HRGV}+\lambda_{dec}\mathcal{L}_{dec}+\lambda_{cal}\mathcal{L}_{cal},\quad "
        r"\mathcal{L}_{cal}=-\log p_c(y|x)$",
        fontsize=8.0, color=COLORS["ink"], ha="left",
    )
    axis.text(
        0.02, 0.055,
        r"Agreement identity: $p_d=p_m\Rightarrow\rho=0\Rightarrow p_c=p_f$.  "
        r"Bounded shift: $|\log(p_{c,j}/p_{c,k})-\log(p_{f,j}/p_{f,k})|\leq2\rho$.",
        fontsize=7.0, color=COLORS["muted"], ha="left",
    )

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
    outputs["source_description"].write_text(
        json.dumps(
            {
                "core_conclusion": "A disagreement-triggered calibrator corrects the fused role posterior only when decomposed role and species experts disagree.",
                "archetype": "schematic-led composite",
                "backend": "Python/matplotlib",
                "evidence_scope": "Architecture and bounded-posterior mechanism; no industrial sorting, grade, recovery, or OOD claim.",
                "exports": {name: str(path) for name, path in outputs.items() if name != "source_description"},
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the HRGV-Net architecture figure.")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate_hrgv_architecture_figure(args.output_prefix)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
