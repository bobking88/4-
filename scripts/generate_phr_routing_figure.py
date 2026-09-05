"""Draw the candidate PHR-Routing-Net architecture without performance claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "ink": "#20303C", "muted": "#61707A", "line": "#415766",
    "input": "#F4F6F7", "backbone": "#D9E8F5", "expert": "#DCEFE7",
    "gate": "#F7E5B5", "pair": "#F4DAD3", "correction": "#E9E1F2",
    "output": "#DCE6F8", "train": "#F2F2F2",
}


def _box(axis, xy, width, height, title, body, color, title_size=8.0):
    x, y = xy
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.009,rounding_size=0.014",
        facecolor=color, edgecolor=PALETTE["line"], linewidth=.85, zorder=2,
    ))
    axis.text(x + width / 2, y + height * .63, title, ha="center", va="center",
              fontsize=title_size, fontweight="bold", color=PALETTE["ink"], zorder=3)
    axis.text(x + width / 2, y + height * .30, body, ha="center", va="center",
              fontsize=6.1, color=PALETTE["muted"], linespacing=1.15, zorder=3)


def _arrow(axis, start, end, label=None, dashed=False, curve=0):
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=.95,
        linestyle="--" if dashed else "-", color=PALETTE["line"],
        connectionstyle=f"arc3,rad={curve}", shrinkA=2, shrinkB=2, zorder=1,
    ))
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .017, label,
                  ha="center", va="center", fontsize=5.5, color=PALETTE["muted"],
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": .3}, zorder=4)


def _configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    })


def generate_phr_routing_architecture(prefix: Path) -> dict[str, Path]:
    """Export a reviewable schematic for the PHR network contract."""
    _configure_style()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12.5, 7.2), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(.02, .965, "PHR-Routing-Net: pairwise hard-negative regret routing (candidate extension)",
              ha="left", va="top", fontsize=12, fontweight="bold", color=PALETTE["ink"])
    axis.text(.02, .928,
              "Two target-versus-negative margins are routed independently; dashed paths and oracle labels are training-only.",
              ha="left", va="top", fontsize=7, color=PALETTE["muted"])

    _box(axis, (.025, .56), .095, .13, "Input x", "public mineral\nspecimen image", PALETTE["input"])
    _box(axis, (.15, .53), .13, .19, "EfficientNet-B0", "shared feature h\n1,280 dimensions", PALETTE["backbone"], 8.5)
    _box(axis, (.315, .73), .135, .125, "Direct-role expert", r"$p_d(T,I,G,M\mid x)$", PALETTE["expert"])
    _box(axis, (.315, .51), .135, .125, "Species expert", r"$p_s(k\mid x),\ k=1,\ldots,17$", PALETTE["expert"])
    _box(axis, (.485, .51), .12, .125, "Fixed mapping A", r"$p_m=A\,p_s$" "\n17 species -> 4 roles", PALETTE["expert"], 7.0)
    _box(axis, (.49, .73), .115, .125, "Global RSG gate", r"$p_0=g_0p_d+(1-g_0)p_m$", PALETTE["gate"], 7.1)
    _box(axis, (.645, .675), .145, .16, "Base posterior", r"$p_0(T,I,G,M)$" "\nlog-odds baseline", PALETTE["gate"], 7.7)

    _box(axis, (.315, .24), .14, .135, "Pair features", "h; p_d, p_m;\npair uncertainties", PALETTE["pair"], 7.6)
    _box(axis, (.495, .31), .13, .12, r"Ti pair gate $g_I$", r"$m_{f,I}=g_I m_{d,I}$" "\n" r"$+(1-g_I)m_{m,I}$", PALETTE["pair"], 7.1)
    _box(axis, (.495, .12), .13, .12, r"Metallic pair gate $g_M$", r"$m_{f,M}=g_M m_{d,M}$" "\n" r"$+(1-g_M)m_{m,M}$", PALETTE["pair"], 6.8)
    _box(axis, (.67, .275), .14, .17, "Constrained correction", r"$\min_a\|a\|_2^2$" "\nsubject to two margins\n" r"$a_G=0$", PALETTE["correction"], 7.2)
    _box(axis, (.855, .34), .115, .14, "PHR posterior", r"$\mathrm{softmax}(\log p_0+a)$" "\nT, I, G, M", PALETTE["output"], 7.4)

    _box(axis, (.10, .085), .175, .115, "Label-directed soft oracle", r"$t_q=\sigma(\Delta_q/\tau)$" "\n" r"$w_q=\tanh(|\Delta_q|/\tau_w)$", PALETTE["train"], 6.6)
    _box(axis, (.315, .065), .15, .115, "Pair-gate objective", r"$\mathrm{BCE}_w(t_q,g_q)$" "\nfeatures/targets detached", PALETTE["train"], 6.9)
    _box(axis, (.67, .075), .23, .11, "Formal scope", "P1-P8 prove local routing/correction properties.\nThey do not claim four-class accuracy gain.", PALETTE["train"], 6.5)

    _arrow(axis, (.12, .625), (.15, .625))
    _arrow(axis, (.28, .64), (.315, .795), curve=.06)
    _arrow(axis, (.28, .60), (.315, .57), curve=-.04)
    _arrow(axis, (.45, .57), (.485, .57))
    _arrow(axis, (.45, .80), (.49, .80), label="p_d")
    _arrow(axis, (.605, .57), (.645, .735), label="p_m", curve=-.12)
    _arrow(axis, (.605, .80), (.645, .78), label="g_0", curve=.08)
    _arrow(axis, (.28, .54), (.315, .31), curve=-.13)
    _arrow(axis, (.455, .31), (.495, .37))
    _arrow(axis, (.455, .27), (.495, .18))
    _arrow(axis, (.625, .37), (.67, .37), label="m_f,I")
    _arrow(axis, (.625, .18), (.67, .31), label="m_f,M", curve=-.13)
    _arrow(axis, (.79, .75), (.855, .43), label="log p_0", curve=-.08)
    _arrow(axis, (.81, .36), (.855, .405), label="a")
    _arrow(axis, (.275, .14), (.315, .12), dashed=True, label="train only")
    _arrow(axis, (.465, .12), (.495, .18), dashed=True)

    axis.text(.025, .015,
              "Roles: T target proxy; I Ti-bearing negative; G gangue; M metallic hard negative. "
              "Candidate architecture: no empirical performance claim.",
              ha="left", va="bottom", fontsize=5.8, color=PALETTE["muted"])

    outputs: dict[str, Path] = {}
    for extension, kwargs in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {}), (".tiff", {"dpi": 600})):
        path = prefix.with_suffix(extension)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[extension] = path
    plt.close(figure)
    source = {
        "evidence_type": "network_contract_schematic",
        "figure_archetype": "schematic-led composite",
        "core_conclusion": "PHR separately routes the two registered target-versus-hard-negative margins and embeds them through a constrained correction.",
        "claim_boundary": "candidate architecture; no empirical performance claim; oracle paths are training-only",
        "formulae": [
            r"m_{f,I}=g_I m_{d,I}+(1-g_I)m_{m,I}",
            r"m_{f,M}=g_M m_{d,M}+(1-g_M)m_{m,M}",
            r"a=A^\top(AA^\top)^{-1}\delta",
            r"p_{\mathrm{PHR}}=\mathrm{softmax}(\log p_0+a)",
        ],
        "exports": {extension: str(path) for extension, path in outputs.items()},
    }
    source_path = prefix.with_suffix(".json")
    source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs[".json"] = source_path
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps({key: str(value) for key, value in generate_phr_routing_architecture(args.output_prefix).items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
