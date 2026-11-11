"""Render two manuscript figures from archived aggregate figure summaries only."""

import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "outputs" / "paper_figures_v3"
PROTOCOLS = ["fixed_test", "photographer_holdout", "resnet50_portability"]
LABELS = ["Fixed split", "Photographer\nholdout", "ResNet50"]
COLORS = ["#4175A8", "#238679", "#474747", "#9B4660"]
QA = {}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "text.color": "#202020",
})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sources():
    names = [
        "fig_rsg_theory_evidence_portability_source.json",
        "fig_rsg_expert_complementarity_source.json",
    ]
    original = [json.loads((SOURCE / name).read_text(encoding="utf-8")) for name in names]
    effects = []
    diagnostics = []
    for protocol in PROTOCOLS:
        row = original[0][protocol]
        assert row["ci_low"] <= row["difference"] <= row["ci_high"]
        effects.append({"protocol": protocol, **{
            key: row[key] for key in ["difference", "ci_low", "ci_high", "bootstrap_replicates"]
        }})
        row = original[1]["protocols"][protocol]
        assert all(np.isfinite(value) and 0 <= value <= 1 for value in row.values())
        diagnostics.append({"protocol": protocol, **row})
    for filename, rows in [("routing_effects_nat.csv", effects), ("expert_diagnostics.csv", diagnostics)]:
        with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return effects, diagnostics, {
        name: sha256(SOURCE / name) for name in names
    }


def save_figure(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if not fig.bbox.padded(1).contains(box.x0, box.y0) or not fig.bbox.padded(1).contains(box.x1, box.y1):
            outside.append(artist.get_text())
    assert not outside, (name, outside)
    for extension in ["svg", "pdf", "png"]:
        options = {"dpi": 300} if extension == "png" else {}
        if extension == "pdf":
            options["metadata"] = {"CreationDate": None, "ModDate": None}
        fig.savefig(OUT / f"{name}.{extension}", **options)
        if extension == "svg":
            path = OUT / f"{name}.svg"
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    QA[name] = {
        "width_mm": round(fig.get_figwidth() * 25.4, 2),
        "height_mm": round(fig.get_figheight() * 25.4, 2),
        "text_outside_canvas": outside,
        "files": {extension: sha256(OUT / f"{name}.{extension}") for extension in ["svg", "pdf", "png"]},
    }
    plt.close(fig)


def routing_figure(rows):
    fig = plt.figure(figsize=(183 / 25.4, 120 / 25.4))
    fig.text(.05, .955, "a  Conditional properties of the routing graph", weight="bold", size=8)
    columns = [
        (.05, "Exact decomposition", r"$r=-\log(1-\delta d/M)$", "Two positive experts; convex fusion.\nSensitivity holds at fixed M."),
        (.38, "Soft target", r"$g^*=\sigma(\Delta/T_r)$", "Entropy-regularized oracle target;\nBCE is a surrogate, not fused NLL."),
        (.71, "Local isolation", r"$\nabla_{\theta_e}L_{reg}=0$", "Detached input, target and weight;\nno guarantee of nonzero gate steps."),
    ]
    for x, title, formula, note in columns:
        fig.text(x, .90, title, weight="bold", size=7)
        fig.text(x, .845, formula, size=10)
        fig.text(x, .78, note, size=6.4, linespacing=1.4)
    fig.text(.05, .68, "b  Archived paired effects: RSG-HRGV minus HRGV", weight="bold", size=8)
    ax = fig.add_axes([.33, .20, .59, .40])
    for y, row, label in zip([2, 1, 0], rows, LABELS):
        value, low, high = row["difference"], row["ci_low"], row["ci_high"]
        ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o", color=COLORS[1],
                    markersize=4.2, capsize=3, linewidth=1.3)
        ax.text(value, y + .23, f"{value:.4f}  [{low:.4f}, {high:.4f}]", ha="center", size=6.7)
    ax.set_yticks([2, 1, 0], LABELS)
    ax.set_ylim(-.5, 2.65)
    ax.set_xlim(-.056, .006)
    ax.set_xticks([-.05, -.04, -.03, -.02, -.01, 0])
    ax.axvline(0, color="#686868", linewidth=.7, linestyle="--")
    ax.grid(axis="x", color="#D8D8D8", linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel("Difference in mean routing regret (nat)")
    fig.text(.05, .087, "Points: archived differences; whiskers: paired 95% bootstrap intervals; 3 seeds per protocol.", size=6.5)
    fig.text(.05, .047, "Routing-risk evidence only. Overall classification superiority has not been established.", size=7, weight="bold")
    save_figure(fig, "fig_rsg_routing_evidence_nat")


def complementarity_figure(rows):
    fig = plt.figure(figsize=(183 / 25.4, 125 / 25.4))
    ax_a = fig.add_axes([.16, .25, .36, .49])
    ax_b = fig.add_axes([.62, .25, .13, .49])
    ax_c = fig.add_axes([.85, .25, .12, .49])
    fig.text(.05, .955, "a  Expert and final predictions", weight="bold", size=8)
    fig.text(.59, .955, "b  Subset sizes", weight="bold", size=8)
    fig.text(.81, .955, "c  Gate agreement", weight="bold", size=8)
    y_values = [2.4, 1.2, 0]
    series = [
        ("direct_accuracy", "Direct role expert", "o", .27),
        ("mapped_accuracy", "Mapped species expert", "s", .09),
        ("fused_accuracy", "Final RSG prediction", "^", -.09),
        ("oracle_accuracy", "Hard-choice oracle", "D", -.27),
    ]
    handles = []
    for color, (key, label, marker, offset) in zip(COLORS, series):
        values = [row[key] * 100 for row in rows]
        ys = [y + offset for y in y_values]
        ax_a.scatter(values, ys, c=color, marker=marker, s=13)
        for value, y in zip(values, ys):
            ax_a.annotate(f"{value:.2f}", (value, y), xytext=(4, 0), textcoords="offset points",
                          va="center", color=color, size=6)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="none", label=label, markersize=4))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.05, .91), ncol=2,
               fontsize=6.1, frameon=False, columnspacing=1.0, handletextpad=.4)
    ax_a.set_yticks(y_values, LABELS)
    ax_a.set_xlim(70, 81)
    ax_a.set_xticks([70, 75, 80])
    ax_a.set_xlabel("Accuracy (%)")
    for key, label, marker, offset, color in [
        ("expert_prediction_disagreement_rate", "Disagreement", "o", .14, COLORS[0]),
        ("one_right_one_wrong_rate", "One correct", "s", -.14, COLORS[1]),
    ]:
        values = [row[key] * 100 for row in rows]
        ys = [y + offset for y in y_values]
        ax_b.scatter(values, ys, color=color, marker=marker, s=13, label=label)
        for value, y in zip(values, ys):
            ax_b.annotate(f"{value:.2f}", (value, y), xytext=(-4, 0), textcoords="offset points",
                          va="center", ha="right", size=6, color=color)
    ax_b.legend(loc="lower center", bbox_to_anchor=(.5, 1.04), fontsize=6, frameon=False,
                borderaxespad=0, handletextpad=.2)
    ax_b.set_xlim(0, 8)
    ax_b.set_xticks([0, 4, 8])
    ax_b.set_xlabel("All images (%)")
    values = [row["one_right_gate_selection_accuracy"] * 100 for row in rows]
    ax_c.scatter(values, y_values, color=COLORS[2], marker="o", s=18)
    for value, y in zip(values, y_values):
        ax_c.annotate(f"{value:.2f}", (value, y), xytext=(0, 7), textcoords="offset points", ha="center", size=6.5)
    ax_c.set_xlim(0, 100)
    ax_c.set_xticks([0, 50, 100])
    ax_c.set_xlabel("Conditional (%)")
    fig.text(.91, .815, "True-class\nprobability oracle", ha="center", size=6.2)
    for ax in [ax_a, ax_b, ax_c]:
        ax.set_ylim(-.65, 3.05)
        ax.grid(axis="x", color="#DDDDDD", linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines["left"].set_visible(False)
    ax_b.set_yticks([])
    ax_c.set_yticks([])
    fig.text(.05, .145, "Three-seed means from archived summaries; no uncertainty intervals were supplied for these values.", size=6.3)
    fig.text(.05, .105, "Hard-choice oracle: diagnostic reference, not a continuous-fusion accuracy bound.", size=6.5)
    fig.text(.05, .065, "Gate agreement is conditional on exactly one correct expert. It need not select that correct expert.", size=6.3)
    fig.text(.05, .025, "Final RSG accuracy is measured after verification; routing regret is defined before verification.", size=6.3)
    save_figure(fig, "fig_rsg_expert_diagnostics")


if __name__ == "__main__":
    effects, diagnostics, hashes = read_sources()
    routing_figure(effects)
    complementarity_figure(diagnostics)
    manifest = {
        "source_commit": "aef98c6ff740aca2adbb25b629430c6633997a7b",
        "source_sha256": hashes,
        "renderer_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "scope": "Aggregate transcription and display only; no new statistics, predictions or experiments.",
        "figures": QA,
    }
    (OUT / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
