"""Editable formula-grounded IC-RFA schematic, not a performance plot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ic_role_model import ICRoleModel, reference_identity, unit_spec


def render_architecture(output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = unit_spec("C-S-R")
    model = ICRoleModel("C-S-R", seed=20261002)
    identity = reference_identity(model)
    dims = dict(H=1280, h=model.adapters[0].down.in_features, E=model.head[0].in_features-64,
                bottleneck=model.adapters[0].down.out_features,
                readout=[model.head[0].in_features, spec["width"], model.head[2].out_features])
    contract = dict(dimensions=dims, current_parameters=identity["trainable_count"],
                    reference_parameters=identity["fixed_function_count"], delta_divisor=16,
                    reference_input="original h and E, never adapted h", backbone_trainable=False,
                    auxiliary_training_only=True, archetype="schematic-led composite",
                    claim="Identity-centered bounded role adaptation: structure, not real effectiveness",
                    evidence="Actual unit_spec and module dimensions; equations (2)--(9)",
                    export="12 x 7.5 inch source, SVG editable text, vector PDF, 300 dpi PNG")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(12, 7.5))
        ax = fig.add_axes([.02, .02, .96, .96], xlim=(0, 16), ylim=(0, 10))
        ax.set_axis_off()
        nodes, text_boxes = [], []
        def box(name, x, y, w, h, text, *, frozen=False, auxiliary=False):
            patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.05",
                                  facecolor="#eff3f5" if frozen else "#e2f1f0",
                                  edgecolor="#52616b" if frozen else "#25847d", linewidth=1.1,
                                  linestyle="--" if auxiliary else "-")
            patch.set_gid(name)
            ax.add_patch(patch)
            label = ax.text(x+w/2, y+h/2, text, ha="center", va="center", linespacing=1.45)
            nodes.append(name)
            text_boxes.append((label, patch))
        def link(points, *, dashed=False, color="#53636c"):
            xs, ys = zip(*points, strict=True)
            ax.plot(xs, ys, color=color, lw=1.15, ls="--" if dashed else "-", zorder=0)
            ax.annotate("", xy=points[-1], xytext=points[-2],
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.15, linestyle="--" if dashed else "-"))
        ax.text(.35, 9.65, "IC-RFA-B v1 | Identity-centered role feature adaptation", fontsize=15, weight="bold")
        ax.text(.35, 9.22, "Structure / mechanism acceptance only; not evidence of mineral-recognition improvement", fontsize=10)
        box("frozen_backbone", .4, 7.75, 2.8, .95, "Frozen EfficientNet-B0\nvisual feature H1280", frozen=True)
        box("pca_h64", 3.9, 7.75, 2.35, .95, "Train-fold PCA\noriginal h64", frozen=True)
        box("expert_e23", .4, 6.15, 2.8, 1.0, "Frozen expert evidence\nstandardized E23", frozen=True)
        box("anchor", .4, 4.3, 2.8, 1.1, "Fold temperature anchor\n4-class probability p", frozen=True)
        box("routing", 3.9, 5.25, 2.35, 1.5, "Fixed role routing\nb = 4pT(1 - pT)\ngj = b wj\nU control: gj = b/3", frozen=True)
        for j, role in enumerate(("Ti-bearing negative", "Gangue negative", "Metallic negative")):
            y = 7.55-j*1.3
            box(f"adapter_{j}", 7.05, y, 4.0, 1.1,
                f"{role}: 64 - 4 - 64\nCurrent Aj(h); frozen Aj,0(h)\ndelta j = [Aj(h) - Aj,0(h)] / 16")
            link([(6.25, 8.23), (6.65, 8.23), (6.65, y+.55), (7.05, y+.55)])
        box("aggregate", 12.0, 6.15, 3.55, 1.5, "Budgeted adapted feature\nh~ = h + sum(gj delta j)\n||delta j|| <= 1\n||h~ - h|| <= b")
        for j in range(3):
            link([(11.05, 8.1-j*1.3), (11.5, 8.1-j*1.3), (11.5, 6.9), (12., 6.9)])
        link([(6.25, 5.95), (6.6, 5.95), (6.6, 4.65), (13.75, 4.65), (13.75, 6.15)])
        box("reference_head", 3.9, 3.2, 3.0, 1.2, "Frozen F0(h, E)\n87 - 16 - 4\nOriginal inputs only", frozen=True)
        box("current_head", 8.1, 3.2, 3.0, 1.2, "Current Ftheta(h~, E)\n87 - 16 - 4\nShared with auxiliary views")
        box("score_difference", 12., 3.2, 3.55, 1.2, "Centered scores\ns = Ftheta(h~, E) - F0(h, E)\nInitial s = 0")
        box("bounded_output", 12., 1.45, 3.55, 1.15, "Bounded 4-class q\nalpha max = beta max = 0.5\nInitial q = p")
        box("auxiliary_views", 3.9, 1.4, 7.2, 1.05, "Training only: hj = h + delta j (no gj)\nSame current head and same frozen reference F0(h, E)\nNLL + 0.10 pair loss + 0.001/2 ||theta - theta0||^2", auxiliary=True)
        link([(3.2, 8.23), (3.9, 8.23)])
        link([(6.25, 8.5), (6.65, 8.5), (6.65, 8.95), (13.75, 8.95), (13.75, 7.65)])
        link([(3.2, 4.85), (3.55, 4.85), (3.55, 6.), (3.9, 6.)])
        link([(5.05, 7.75), (5.05, 7.25), (3.65, 7.25), (3.65, 3.8), (3.9, 3.8)])
        link([(3.2, 6.65), (3.45, 6.65), (3.45, 3.8), (3.9, 3.8)])
        link([(3.45, 3.8), (3.45, 2.9), (9.6, 2.9), (9.6, 3.2)])
        link([(13.75, 6.15), (13.75, 4.9), (9.6, 4.9), (9.6, 4.4)])
        link([(6.9, 3.8), (7.45, 3.8), (7.45, 4.55), (13.3, 4.55), (13.3, 4.4)])
        link([(11.1, 3.8), (12., 3.8)])
        link([(13.75, 3.2), (13.75, 2.6)])
        link([(3.2, 4.55), (3.3, 4.55), (3.3, 1.1), (13.75, 1.1), (13.75, 1.45)])
        link([(7.5, 2.45), (7.5, 3.), (8.1, 3.5)], dashed=True, color="#25847d")
        ax.text(.4, .57, "3024 current trainable + 3024 frozen function-reference parameters; backbone and preprocessing excluded", fontsize=10)
        ax.text(.4, .14, "Frozen reference uses original h and E; never detach an adapted-input reference to fabricate a gradient", fontsize=9)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        clipped = sum(not patch.get_window_extent(renderer).contains(*label.get_window_extent(renderer).get_points()[i])
                      for label, patch in text_boxes for i in (0, 1))
        if clipped:
            plt.close(fig)
            raise ValueError(f"Architecture text clipping: {clipped}")
        paths = {extension: str(output/f"ic_rfa_architecture.{extension}") for extension in ("svg", "pdf", "png")}
        for extension, path in paths.items(): fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
    caption = ("IC-RFA-B v1 architecture. Frozen EfficientNet-B0 features are reduced using train-fold-only PCA. "
               "Three 64-4-64 centered adapters use fixed role gates. The current 87-16-4 readout subtracts "
               "the same frozen reference evaluated on original h and E, not adapted inputs. Dashed paths show "
               "training-only auxiliary views. Current and reference storage each contains 3024 scalars. "
               "The diagram illustrates formula implementation, not real effectiveness, convergence, grade prediction or industrial validation.\n")
    caption_path = output/"caption.md"
    caption_path.write_text(caption, encoding="utf-8")
    result = dict(paths, caption=str(caption_path), contract=contract, node_ids=nodes,
                  dpi=300, text_clipping_count=clipped, source_dimensions_inches=[12, 7.5])
    (output/"architecture_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(render_architecture(parser.parse_args().output), indent=2))
