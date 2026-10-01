"""Export the ABMP architecture and reproducible analytic property checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import torch

from abmp_rsg import AdaptiveBudgetPolicy, apply_abmp_projection, audit_abmp_projection


def run_theory_audit(samples=25000, seed=20260930):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    q0 = torch.rand(samples, 4, dtype=torch.float64)
    q0 /= q0.sum(1, keepdim=True)
    qp = torch.rand_like(q0)
    qp /= qp.sum(1, keepdim=True)
    route = torch.rand(samples, 1, dtype=torch.float64)
    epsilon = .08 * torch.rand(samples, 1, dtype=torch.float64)
    projected = apply_abmp_projection(q0, qp, route, epsilon, tau_p=.4, tau_m=.05, delta=.005)
    audit = audit_abmp_projection(projected, q0, qp, torch.randint(4, (samples,)), .08, .005)
    # Deliberately cross the target decision boundary while respecting posterior budget.
    a = torch.tensor([[.42, .36, .11, .11]], dtype=torch.float64)
    b = torch.tensor([[.38, .42, .10, .10]], dtype=torch.float64)
    full = apply_abmp_projection(a, b, torch.tensor([[.9]], dtype=torch.float64), .04, tau_p=.4, tau_m=.05, delta=.005)
    post = apply_abmp_projection(a, b, torch.tensor([[.9]], dtype=torch.float64), .04, tau_p=.4, tau_m=.05, delta=.005, mode="posterior")
    # Sample the original analytically dominated region, including extreme candidates.
    dominant_q0 = q0.clone()
    dominant_q0[:, :1] = .60 + .39 * torch.rand(samples, 1, dtype=torch.float64)
    dominant_q0[:, 1:] = q0[:, 1:] / q0[:, 1:].sum(1, keepdim=True) * (1 - dominant_q0[:, :1])
    dominant_full = apply_abmp_projection(dominant_q0, qp, route, epsilon, tau_p=.6, tau_m=.05, delta=.005)
    dominant_post = apply_abmp_projection(dominant_q0, qp, route, epsilon, tau_p=.6, tau_m=.05, delta=.005, mode="posterior")
    same_argmax = q0.argmax(1) == qp.argmax(1)
    convex_probabilities = (1 - route) * q0 + route * qp
    identical_prediction_violations = int((same_argmax & (convex_probabilities.argmax(1) != q0.argmax(1))).sum())
    return {
        "purpose": "formula_verification_not_classification_performance",
        "seed": seed, "random_samples": samples, "random_audit": audit,
        "independent_margin_activation_count": int((full["projected_route"] < post["projected_route"] - 1e-7).sum()),
        "boundary_fixture": {"q0": a.tolist(), "qphi": b.tolist(), "raw_route": .9, "epsilon": .04, "full_route": float(full["projected_route"]), "posterior_only_route": float(post["projected_route"]), "full_prediction": int(full["final_probabilities"].argmax()), "posterior_only_prediction": int(post["final_probabilities"].argmax())},
        "dominance_counterexamples": int(((dominant_full["final_probabilities"] - dominant_post["final_probabilities"]).abs().max(1).values > 1e-12).sum()),
        "dominance_condition": "tau_p - epsilon_max > (1 + delta) / 2",
        "same_expert_argmax_convex_fusion": {
            "rows": int(same_argmax.sum()),
            "violations": identical_prediction_violations,
            "assumption": "Unique endpoint argmax; arbitrary sample-wise route in [0,1].",
        },
    }


def generate_figure(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 12})
    fig = plt.figure(figsize=(10.5, 9), facecolor="white")
    ax = fig.add_axes((.02, .27, .96, .68))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    palette = {"frozen": "#ECEFF1", "train": "#DDEFE8", "analytic": "#FFF0CF"}

    def box(x, y, w, h, label, kind="frozen", size=12.5):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=palette[kind], edgecolor="#59646B", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)

    def arrow(start, end, points=None):
        if points:
            coords = [start, *points, end]
            ax.plot(*zip(*coords[:-1]), color="#59646B", lw=1.1)
            start = coords[-2]
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#59646B", linewidth=1.1))

    ax.text(.0, 1.03, "a  ABMP-RSG-Net v2", fontsize=17, fontweight="bold")
    box(.01, .80, .22, .14, "RGB image\nEfficientNet-B0")
    box(.01, .58, .22, .14, "Role / species heads\nRole map + OOS gate")
    box(.01, .34, .22, .16, "Ti / metallic verifiers\nComplete posteriors\n$q_0$ and $q_\phi$")
    arrow((.12, .80), (.12, .72))
    arrow((.12, .58), (.12, .50))
    box(.28, .56, .20, .21, "18D evidence\nPosterior pairs\nDisagreement / entropy\nVerifier outputs / JS", size=12)
    arrow((.23, .43), (.28, .66), points=[(.255, .43), (.255, .66)])
    box(.28, .28, .20, .18, "Shared policy trunk\n" + r"18 $\to$ 64 $\to$ 16" + "\nLayerNorm + SiLU", "train", 12)
    arrow((.38, .56), (.38, .46))
    box(.54, .69, .19, .12, "Route head\n" + r"$\widetilde\rho=\sigma(h_r)$", "train")
    box(.54, .47, .19, .12, "Budget head\n" + r"$\varepsilon_i=\varepsilon_{max}\sigma(h_\varepsilon-\lambda)$", "train", 12)
    arrow((.48, .40), (.54, .75), points=[(.51, .40), (.51, .75)])
    arrow((.48, .34), (.54, .53), points=[(.52, .34), (.52, .53)])
    box(.78, .47, .21, .12, "Posterior cap\n$c_i^{post}$", "analytic")
    arrow((.73, .53), (.78, .53))
    box(.78, .27, .21, .12, "Anchor margin cap\n$c_i^{margin}$", "analytic", 12.5)
    ax.text(.885, .415, "$q_0,q_\phi$ + anchors", fontsize=12, ha="center")
    box(.54, .05, .45, .12, r"$\widehat\rho_i=\min(\widetilde\rho_i,c_i^{post},c_i^{margin})$", "analytic", 14)
    arrow((.88, .47), (.88, .17), points=[(.755, .47), (.755, .205), (.88, .205)])
    arrow((.88, .27), (.88, .17))
    arrow((.635, .69), (.635, .17), points=[(.745, .69), (.745, .205), (.635, .205)])
    box(.01, .02, .40, .14, "Protected convex fusion\n" + r"$q=(1-\widehat\rho)q_0+\widehat\rho q_\phi$", "analytic", 14)
    arrow((.54, .11), (.41, .11))
    for index, (label, color) in enumerate(palette.items()):
        ax.add_patch(Rectangle((.29 + .10 * index, .92), .018, .022, facecolor=color, edgecolor="#59646B", lw=.6))
        ax.text(.314 + .10 * index, .931, label, fontsize=11, va="center")
    proof = fig.add_axes((.03, .04, .94, .18))
    proof.axis("off")
    proof.text(0, 1.0, "b  Analytic properties and non-redundancy", fontsize=15, weight="bold", va="top")
    proof.text(.00, .58, r"$q_T\geq q_{0,T}-\varepsilon_i$" + "\nPosterior loss bound", fontsize=13, va="center")
    proof.text(.45, .58, r"$A_i=1\ \Rightarrow\ q_T-q_k\geq\delta$" + "\nAnchor margin retention", fontsize=13, va="center")
    proof.text(.00, .06, r"$\widehat\rho_i=\mathrm{Proj}_{[0,\min(c_i^{post},c_i^{margin})]}(\widetilde\rho_i)$", fontsize=13, va="bottom")
    proof.text(.61, .08, "Avoid redundant caps:\n" + r"$\tau_p-\varepsilon_{max}>(1+\delta)/2$", fontsize=12, va="bottom")
    source = {
        "figure_contract": {"conclusion": "A learned sample-wise budget and target-margin constraints define an auditable projection of the raw route.", "archetype": "schematic-led composite", "backend": "python", "dimensions_inches": [10.5, 9], "exports": ["editable SVG", "PDF", "300 dpi PNG"], "claim_boundary": "No performance or population recall guarantee is implied by this schematic."},
        "evidence_dimension": 18, "policy_parameter_count": sum(p.numel() for p in AdaptiveBudgetPolicy().parameters()),
        "components": ["frozen EfficientNet-B0", "direct role and species heads", "species-role map", "frozen out-of-sample candidate gate", "Ti and metallic verifiers", "complete q0/qphi", "18D evidence", "shared policy trunk", "route head", "budget head", "posterior cap", "anchor margin cap", "minimum route", "convex fusion"],
    }
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output_dir / f"fig_abmp_rsg_architecture.{extension}", dpi=300)
    plt.close(fig)
    (output_dir / "fig_abmp_rsg_architecture_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    return source


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/paper_figures_v5")
    args = parser.parse_args()
    source = generate_figure(args.output_dir)
    invariants = run_theory_audit()
    path = root / "outputs/theory/abmp_rsg_invariants.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(invariants, indent=2), encoding="utf-8")
    print(json.dumps({"policy_parameters": source["policy_parameter_count"], "audit": invariants["random_audit"]}, indent=2))


if __name__ == "__main__":
    main()
