"""Draw the out-of-sample risk-supervised RSG-HRGV training architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "ink": "#1F303A",
    "muted": "#5D6B74",
    "line": "#425A67",
    "data": "#F3F5F6",
    "split": "#E7EEF2",
    "expert": "#DCEFE7",
    "direct": "#D8E8F5",
    "species": "#E8E1F2",
    "verifier": "#F6E7C7",
    "gate": "#F3DAD2",
    "loss": "#F5E9B8",
    "boundary": "#F0F0F0",
}


def _configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
    })


def _box(axis, xy, width, height, title, body, color, title_size=7.8, body_size=5.8):
    x, y = xy
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor=PALETTE["line"], linewidth=0.85, zorder=2,
    ))
    axis.text(
        x + width / 2, y + height * 0.66, title,
        ha="center", va="center", fontsize=title_size, fontweight="bold",
        color=PALETTE["ink"], zorder=3,
    )
    axis.text(
        x + width / 2, y + height * 0.29, body,
        ha="center", va="center", fontsize=body_size, color=PALETTE["muted"],
        linespacing=1.15, zorder=3,
    )


def _arrow(axis, start, end, label=None, dashed=False, curve=0.0, color=None):
    line_color = color or PALETTE["line"]
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=0.95,
        linestyle="--" if dashed else "-", color=line_color,
        connectionstyle=f"arc3,rad={curve}", shrinkA=2, shrinkB=2, zorder=1,
    ))
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.013,
            label, ha="center", va="center", fontsize=5.2, color=PALETTE["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2}, zorder=4,
        )


def generate_figure(prefix: Path) -> dict[str, Path]:
    """Export a reviewable schematic and its claim metadata."""
    _configure_style()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(13.4, 8.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.02, 0.975,
        "OOS-RSG-HRGV: out-of-sample risk-supervised routing for dual mineral experts",
        ha="left", va="top", fontsize=12.5, fontweight="bold", color=PALETTE["ink"],
    )
    axis.text(
        0.02, 0.941,
        "Training graph: the gate learns final verifier-aware risk on a group-independent supervision subset; deployment graph is unchanged.",
        ha="left", va="top", fontsize=6.9, color=PALETTE["muted"],
    )

    # Data partition row.
    _box(axis, (0.022, 0.745), 0.115, 0.13, "Grouped images", "photo/group IDs\nkept indivisible", PALETTE["data"])
    _box(axis, (0.175, 0.77), 0.115, 0.105, r"$D_E$: expert fit", "4,176 images\nexpert gradients", PALETTE["split"])
    _box(axis, (0.315, 0.77), 0.115, 0.105, r"$D_S$: expert stop", "893 images\nmodel selection", PALETTE["split"])
    _box(axis, (0.455, 0.77), 0.115, 0.105, r"$D_G$: gate unseen", "892 images\n17-stratum match", PALETTE["split"])
    _box(axis, (0.595, 0.77), 0.115, 0.105, r"$D_{seen}$: paired control", "892 images\nfrom expert fit", PALETTE["split"])
    _arrow(axis, (0.137, 0.81), (0.175, 0.825))
    _arrow(axis, (0.137, 0.81), (0.315, 0.825), curve=-0.05)
    _arrow(axis, (0.137, 0.81), (0.455, 0.825), curve=-0.08)
    _arrow(axis, (0.137, 0.81), (0.595, 0.825), curve=-0.10)

    # Frozen expert construction.
    _box(axis, (0.175, 0.555), 0.15, 0.14, "Expert training", r"$\hat\theta=\mathcal{A}(D_E,D_S)$" "\nweighted multi-task loss", PALETTE["expert"], 8.2)
    _box(axis, (0.365, 0.57), 0.135, 0.115, "Shared backbone", "EfficientNet-B0\nfeature h (1,280-d)", PALETTE["expert"])
    _box(axis, (0.535, 0.635), 0.13, 0.105, "Direct-role expert", r"$p_d(r\mid x)$" "\n4 beneficiation roles", PALETTE["direct"])
    _box(axis, (0.535, 0.49), 0.13, 0.105, "Species expert", r"$p_s(k\mid x)$" "\n17 mineral species", PALETTE["species"])
    _box(axis, (0.7, 0.49), 0.12, 0.105, "Fixed mapping A", r"$p_m=A p_s$" "\nspecies -> role", PALETTE["species"], 7.0)
    _box(axis, (0.7, 0.635), 0.12, 0.105, "Two verifiers", r"$v_{Ti},v_M$" "\nhard-negative cues", PALETTE["verifier"])

    _arrow(axis, (0.232, 0.77), (0.232, 0.695), label=r"$D_E$")
    _arrow(axis, (0.372, 0.77), (0.294, 0.695), label=r"$D_S$", dashed=True, curve=-0.12)
    _arrow(axis, (0.325, 0.625), (0.365, 0.625))
    _arrow(axis, (0.5, 0.635), (0.535, 0.685), curve=0.06)
    _arrow(axis, (0.5, 0.61), (0.535, 0.54), curve=-0.05)
    _arrow(axis, (0.665, 0.54), (0.7, 0.54))
    _arrow(axis, (0.5, 0.655), (0.7, 0.685), curve=0.05)

    # Gate supervision and final-risk objective.
    _box(axis, (0.855, 0.55), 0.12, 0.16, "Frozen expert", "no gradients\nfixed parameters and buffers", PALETTE["boundary"], 7.8)
    _arrow(axis, (0.82, 0.69), (0.855, 0.66), label="verifiers")
    _arrow(axis, (0.82, 0.54), (0.855, 0.60), label=r"$p_d,p_m$")

    _box(axis, (0.05, 0.285), 0.19, 0.14, "Gate observations", r"$u=[h,p_d,p_m,H_d,H_m,$" "\n" r"$JS(p_d,p_m),v_{Ti},v_M]$", PALETTE["gate"], 8.0)
    _box(axis, (0.285, 0.285), 0.16, 0.14, "Risk-supervised gate", r"$g_\phi(u)\in(0,1)$" "\n" r"$p_g=g_\phi p_d+(1-g_\phi)p_m$", PALETTE["gate"], 7.7)
    _box(axis, (0.49, 0.285), 0.18, 0.14, "Verifier-aware posterior", r"$p_f=\mathcal{V}(p_g,v_{Ti},v_M)$" "\nfinal decision distribution", PALETTE["verifier"], 7.6)
    _box(axis, (0.715, 0.285), 0.235, 0.14, "Out-of-sample gate objective", r"$\min_\phi\;\mathbb{E}_{D_G}[-\log p_f(y\mid x)]$" "\n" r"$+\lambda\,\mathrm{BCE}(g_\phi,t_{regret})$", PALETTE["loss"], 7.7)

    _arrow(axis, (0.915, 0.55), (0.19, 0.425), label="predictions / features", curve=0.18)
    _arrow(axis, (0.512, 0.77), (0.135, 0.425), label=r"$D_G$ labels", dashed=True, curve=0.12)
    _arrow(axis, (0.652, 0.77), (0.205, 0.425), label=r"paired $D_{seen}$", dashed=True, curve=0.08)
    _arrow(axis, (0.24, 0.355), (0.285, 0.355))
    _arrow(axis, (0.445, 0.355), (0.49, 0.355))
    _arrow(axis, (0.67, 0.355), (0.715, 0.355))
    _arrow(axis, (0.835, 0.285), (0.415, 0.285), label="gate gradients only", dashed=True, curve=-0.18)

    # Theory and evaluation boundaries.
    _box(axis, (0.05, 0.075), 0.255, 0.12, "Proposition 1: conditional unbiasedness",
         r"For fixed $\phi$, group-independent $D_G$ yields" "\n" r"unbiased conditional risk and gradient estimates.",
         PALETTE["boundary"], 7.4, 5.5)
    _box(axis, (0.34, 0.075), 0.255, 0.12, "Proposition 2: gate degeneracy",
         r"If $p_d=p_m$, then $p_g$ is gate-independent," "\n" r"$\nabla_\phi\ell_{final}=0$ and the RSG gap is zero.",
         PALETTE["boundary"], 7.4, 5.5)
    _box(axis, (0.63, 0.075), 0.32, 0.12, "Evidence boundary",
         "One frozen expert and one reused stop set; three gate seeds\nare paired initializations, not independent confirmation.",
         PALETTE["boundary"], 7.4, 5.5)

    axis.text(
        0.022, 0.022,
        "Solid arrows: deployment-compatible computation. Dashed arrows: training or selection only. "
        "Current evidence supports a supervision-source mechanism, not stable overall superiority.",
        ha="left", va="bottom", fontsize=5.7, color=PALETTE["muted"],
    )

    outputs: dict[str, Path] = {}
    for extension, kwargs in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        path = prefix.with_suffix(extension)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[extension] = path
    plt.close(figure)

    metadata = {
        "evidence_type": "training_architecture_and_theory_schematic",
        "figure_archetype": "data-split and network-training graph",
        "core_conclusion": (
            "OOS-RSG-HRGV trains only the routing gate on a group-independent subset using "
            "the final verifier-aware risk while keeping the dual expert frozen."
        ),
        "claim_boundary": (
            "candidate network innovation; one frozen expert; reused stop set; "
            "three paired gate initializations are not independent confirmation"
        ),
        "formulae": [
            r"\hat\theta=\mathcal{A}(D_E,D_S)",
            r"p_g=g_\phi p_d+(1-g_\phi)p_m",
            r"\min_\phi\mathbb{E}_{D_G}[-\log p_f(y\mid x)]+\lambda\mathrm{BCE}(g_\phi,t_{regret})",
            r"p_d=p_m\Rightarrow\nabla_\phi\ell_{final}=0",
        ],
        "exports": {extension: str(path) for extension, path in outputs.items()},
    }
    source_path = prefix.with_suffix(".json")
    source_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs[".json"] = source_path
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_figure(args.output_prefix)
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
