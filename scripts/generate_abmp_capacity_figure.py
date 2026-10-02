"""Draw the finite-sample capacity diagnosis and its structural equations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_tc_oos_rsg_experiments import _file_sha256, _write_json


ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("gate_stop_projector_fit", "projector_stop")
FORMULAS = {
    "scale": r"$V_s(p)=\frac{D_sp}{Z(p)},\quad D_s=\mathrm{diag}(s,1,1,1),\quad Z(p)=1-(1-s)p_T,\quad 0<s\leq1$",
    "commutation": r"$V_s(gd+(1-g)m)=g'V_s(d)+(1-g')V_s(m),\quad g'=\frac{gZ(d)}{gZ(d)+(1-g)Z(m)}$",
    "monotonicity": r"$\{i:\arg\max V_{s_i}(p_i)=T\}\subseteq\{i:\arg\max p_i=T\},\quad\mathrm{Rec}_T(V_s(p))\leq\mathrm{Rec}_T(p)$",
    "interval": r"$\mathcal{I}_{\eta}(d,m)=\{g\in[0,1]:a_k+gb_k\geq\eta,\ \forall k\ne T\},\quad a_k=m_T-m_k,\quad b_k=(d_T-d_k)-a_k$",
    "capacity": r"$\mathcal{C}_{\eta}(q_0,q_\phi)\subseteq\mathcal{C}_{\eta}(V_s(d),V_s(m)),\quad\mathrm{Rec}_{T,\eta}(q)\leq\frac{1}{N_T}\sum_{i:y_i=T}\mathbf{1}[\mathcal{I}_{\eta,i}\ne\varnothing]$",
    "protected_scale": r"$s_{\min}(p,\delta)=\max_{k\ne T}\frac{p_k+\delta(1-p_T)}{(1-\delta)p_T},\quad s\in[s_{\min},1]\Rightarrow V_s(p)_T-V_s(p)_k\geq\delta$",
}


def build_plot_data(summary):
    if summary.get("outer_images_loaded") is not False:
        raise ValueError("Capacity plot must contain development data only.")
    result = {}
    for name in SUBSETS:
        subset = summary["subsets"][name]
        target_count = subset["class_counts"]["0"]
        observed = round(subset["metrics"]["fixed_verified"]["target_recall"]*target_count)
        result[name] = {"target_count": target_count,
                        "capacity_counts": [observed, *[subset["segment_capacity"][key]["true_target_feasible"] for key in ("restricted_verified", "full_verified", "full_pre")]],
                        "fixed_verifier_removals": [subset["verification_effects"]["fixed"][key] for key in ("true_target_removed", "false_target_removed")]}
        if any(not 0 <= count <= target_count for count in result[name]["capacity_counts"]):
            raise ValueError("Target capacity count outside sample bounds.")
    return result


def generate(summary_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = build_plot_data(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    colors = ("#4C6E81", "#789E96")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.25), gridspec_kw={"width_ratios": [1.65, 1]}, layout="constrained")
    width = .34
    for index, name in enumerate(SUBSETS):
        x = np.arange(4)+(index-.5)*width
        bars = axes[0].bar(x, data[name]["capacity_counts"], width, color=colors[index], label="Policy fit" if index == 0 else "Policy stop")
        axes[0].bar_label(bars, padding=3, fontsize=10)
    axes[0].set_xticks(range(4), ["Fixed\nobserved", "Restricted\nattainable", "Full verified\nattainable", "Full pre-verifier\nattainable"])
    axes[0].set_ylim(0, 66)
    axes[0].set_ylabel("True targets (66 per subset)")
    axes[0].set_title("a  Routing capacity", loc="left", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    for index, name in enumerate(SUBSETS):
        x = np.arange(2)+(index-.5)*width
        bars = axes[1].bar(x, data[name]["fixed_verifier_removals"], width, color=colors[index])
        axes[1].bar_label(bars, padding=3, fontsize=10)
    axes[1].set_xticks(range(2), ["True targets\nremoved", "False targets\nremoved"])
    axes[1].set_ylim(0, 5.3)
    axes[1].set_yticks(range(6))
    axes[1].set_ylabel("Images")
    axes[1].set_title("b  Fixed-fusion verification", loc="left", fontweight="bold")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig_abmp_candidate_capacity.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    folder = output_dir / "capacity_formulas"
    folder.mkdir(parents=True, exist_ok=True)
    for name, formula in FORMULAS.items():
        fig = plt.figure(figsize=(14, .8), facecolor="white")
        artist = fig.text(.5, .5, formula, ha="center", va="center", fontsize=16)
        fig.canvas.draw()
        width_inches = artist.get_window_extent(fig.canvas.get_renderer()).width/fig.dpi
        if width_inches > 13.5:
            artist.set_fontsize(16*13.5/width_inches)
        fig.savefig(folder / f"capacity_{name}.png", dpi=300, bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
    _write_json(output_dir / "fig_abmp_candidate_capacity_source.json", {
        "source_sha256": _file_sha256(summary_path), "source_data": data,
        "contract": {"conclusion": "Wider verified routing has no additional target capacity on stopping data; verification removes two pre-verifier opportunities.",
                     "backend": "Python matplotlib", "archetype": "quantitative grid", "formats": ["PNG 300 dpi", "PDF", "SVG editable labels"],
                     "error_bars": "None: finite deterministic counts, not replicate inference.",
                     "scope": "Fold 0 development only; attainable counts are label-aware upper bounds, not measured network gains."},
        "equations": FORMULAS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/theory/abmp_candidate_capacity_v1/audit_summary.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper_figures_v5")
    args = parser.parse_args()
    generate(args.summary, args.output_dir)
    print("Capacity figure and six structural equations generated.")


if __name__ == "__main__":
    main()
