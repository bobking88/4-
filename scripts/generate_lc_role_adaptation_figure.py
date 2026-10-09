"""Draw the implemented LC-RFA-B architecture from inspected PyTorch modules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import torch

import lc_role_adapter
from lc_role_adapter import RoleResidualModel

STATE = "IMPLEMENTED_EFFECT_UNVERIFIED"
COLORS = dict(ink="#243138",line="#667279",frozen="#ECEFF1",train="#DDEFE8",
              analytic="#FFF0CF",auxiliary="#E5EDF5",training_line="#527899")


def architecture_manifest(model: RoleResidualModel) -> dict:
    if (model.arm != "R1" or len(model.adapters) != 3 or model.head[0].in_features != 87
            or model.head[0].out_features != 16 or model.head[2].out_features != 4
            or any(a.down.in_features != 64 or a.down.out_features != 4
                   or a.up.out_features != 64 or a.up.bias is not None for a in model.adapters)):
        raise ValueError("This registered diagram requires the implemented R1 architecture.")
    modules = {}
    for name,module in model.named_modules():
        entry = dict(type=type(module).__name__)
        if hasattr(module,"weight"):
            entry.update(weight_shape=list(module.weight.shape),
                         bias_shape=None if module.bias is None else list(module.bias.shape))
        modules[name] = entry
    def count(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    with torch.no_grad():
        sample = model(torch.zeros(2,64,dtype=torch.float64),torch.zeros(2,23,dtype=torch.float64),
                       torch.full((2,4),-torch.log(torch.tensor(4.,dtype=torch.float64)),dtype=torch.float64),auxiliary=True)
    nodes = [dict(id=i,kind=k,**({"module_path":p} if p else {})) for i,k,p in (
        ("H1280","frozen",None),("PCA64","fit_preprocessing",None),("E23","frozen",None),
        ("E_scaled","fit_preprocessing",None),("p","frozen",None),("anchor","fit_preprocessing",None),
        ("budget","analytic",None),("delta1","trainable","adapters.0"),
        ("delta2","trainable","adapters.1"),("delta3","trainable","adapters.2"),
        ("adapted_h","analytic",None),("head","trainable","head"),
        ("bounded","analytic",None),("q","output",None),("auxiliary","training",None))]
    edges = []
    for source,target in (("H1280","PCA64"),("E23","E_scaled"),("p","anchor"),("anchor","budget"),
                          ("PCA64","adapted_h"),("budget","adapted_h"),("budget","bounded"),
                          ("anchor","bounded"),("adapted_h","head"),("E_scaled","head"),("head","bounded"),("bounded","q")):
        edges.append(dict(source=source,target=target,style="solid",kind="inference"))
    for j in range(1,4):
        edges.extend([dict(source="PCA64",target=f"delta{j}",style="solid",kind="inference"),
                      dict(source=f"delta{j}",target="adapted_h",style="solid",kind="inference"),
                      dict(source=f"delta{j}",target="auxiliary",style="dashed",kind="training")])
    edges.append(dict(source="auxiliary",target="head",style="dashed",kind="training"))
    files = [Path(lc_role_adapter.__file__),Path(__file__)]
    return dict(name="LC-RFA-B v1 / R1",state=STATE,arm=model.arm,
                trainable_parameter_count=count(model),adapter_parameter_count=[count(a) for a in model.adapters],
                shared_head_parameter_count=count(model.head),modules=modules,nodes=nodes,edges=edges,
                example_shapes={k:list(v.shape) for k,v in sample.items()},
                code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                figure_contract=dict(conclusion="Three role-conditioned bounded feature adapters feed one shared readout.",
                    archetype="schematic-led composite",backend="Python/matplotlib",size_inches=[7.2,6.8],
                    dpi=300,editable_text=True,statistical_claims=False,
                    panels=["main implemented graph","single adapter detail","matched S1/R1 comparison"]),
                formulas=dict(adapter="delta_j=tanh(U_j SiLU(V_j h+b_j))/sqrt(64)",
                    routing="g_j=4*pbar_T*pbar_j; b=sum_j(g_j)",adapted_h="h_tilde=h+sum_j(g_j*delta_j)",
                    auxiliary="h+delta_j; same head and bounded map; no g_j factor",head="F:87->16->4 (SiLU)",
                    budgets="alpha=alpha_max*b; beta=beta_max*b",output="q_T=sigmoid(logit(t)+u); q_j=(1-q_T)*softmax(log(w)+v)_j"))


def render_architecture(manifest: dict, output_dir: Path) -> dict:
    if (manifest.get("state") != STATE or manifest.get("trainable_parameter_count") != 3024
            or manifest.get("adapter_parameter_count") != [516]*3 or manifest.get("shared_head_parameter_count") != 1476):
        raise ValueError("Unsupported diagram structure, count, or effectiveness claim.")
    output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"svg.fonttype":"none","pdf.fonttype":42})
    fig = plt.figure(figsize=(7.2,6.8),facecolor="white")
    fig.text(.035,.970,"LC-RFA-B: lightweight role-conditioned feature adaptation",fontsize=10.3,weight="bold",color=COLORS["ink"])
    fig.text(.035,.945,"Implemented architecture  |  Effectiveness awaits real-data evaluation",fontsize=7.1,color=COLORS["line"])
    ax = fig.add_axes([.025,.36,.95,.56])
    ax.set(xlim=(0,100),ylim=(-3,83))
    ax.axis("off")
    box_text = []
    def box(axis,xy,w,h,label,kind="frozen",size=6.9):
        patch = Rectangle(xy,w,h,facecolor=COLORS[kind],edgecolor=COLORS["line"],linewidth=.7,zorder=3)
        axis.add_patch(patch)
        text = axis.text(xy[0]+w/2,xy[1]+h/2,label,ha="center",va="center",fontsize=size,color=COLORS["ink"],zorder=4,linespacing=1.35)
        box_text.append((patch,text,label))
    def arrow(axis,points,training=False):
        color = COLORS["training_line"] if training else COLORS["line"]
        style = "--" if training else "-"
        if len(points)>2:
            axis.plot(*zip(*points[:-1]),color=color,lw=.7,ls=style,zorder=1)
        axis.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=6,
                                       linewidth=.7,linestyle=style,color=color,zorder=2))
    ax.text(0,81,"a",weight="bold",fontsize=10)
    box(ax,(2,65),17,11,"Frozen expert\n4-class posterior $p$")
    box(ax,(25,65),18,11,"Fit-only temperature\n"+r"$\bar p=\mathrm{softmax}(\log p/T_0)$",size=6.4)
    box(ax,(69,65),28,11,r"$g_j=4\bar p_T\bar p_j,\quad b=\sum_j g_j$"+"\n"+r"$\alpha=\alpha_{\max}b,\quad\beta=\beta_{\max}b$","analytic",7.2)
    box(ax,(2,44),17,11,"Existing EfficientNet-B0\nFrozen $H$: 1280",size=6.4)
    box(ax,(25,44),18,11,"Fit-only PCA + scale\n$h$: 64")
    for j,y in enumerate((49,36,23),1):
        box(ax,(47,y),16,10,f"Adapter {j}\n"+rf"$\delta_{j}$: 64  |  516 params","train",6.5)
    box(ax,(69,44),28,12,r"$\widetilde h=h+\sum_j g_j\delta_j$"+"\nBounded feature adaptation","analytic",7.1)
    box(ax,(2,24),17,10,"Frozen evidence\n$E$: 23")
    box(ax,(25,24),18,10,"Fit-only standardization\n$e$: 23",size=6.4)
    box(ax,(69,25),28,11,r"ONE shared head $F([\widetilde h,e])$"+"\n"+r"87 $\rightarrow$ 16 $\rightarrow$ 4, SiLU | 1476 params","train",6.5)
    box(ax,(69,1),28,18,r"$u=\alpha\tanh a_T,\quad v_j=\frac{\beta}{2}\tanh a_j$"+"\n"+
        r"$q_T=\sigma(\mathrm{logit}\,t+u)$"+"\n"+r"$q_j=(1-q_T)\mathrm{softmax}(\log w+v)_j$","analytic",6.35)
    box(ax,(2,1),61,16,r"Training only: three views $h+\delta_j$, without route scaling"+"\n"+
        "Reuse the SAME head $F$ and bounded output map for pair NLL\n"+
        r"$\mathcal{L}=\mathrm{NLL}(q,y)+0.10\mathcal{L}_{\mathrm{pair}}+\frac{\lambda}{2}\Vert\theta\Vert_2^2$","auxiliary",6.8)
    arrow(ax,[(19,70.5),(25,70.5)])
    arrow(ax,[(43,70.5),(69,70.5)])
    arrow(ax,[(19,49.5),(25,49.5)])
    for y in (54,41,28):
        arrow(ax,[(43,49.5),(45,49.5),(45,y),(47,y)])
        arrow(ax,[(63,y),(66,y),(66,50),(69,50)])
    arrow(ax,[(34,55),(34,60.5),(78,60.5),(78,56)])
    ax.text(51,61.6,"identity path $h$",fontsize=6.1,ha="center")
    arrow(ax,[(83,65),(83,56)])
    arrow(ax,[(97,70),(99,70),(99,10),(97,10)])
    arrow(ax,[(19,29),(25,29)])
    arrow(ax,[(43,29),(44,29),(44,20),(66,20),(66,30),(69,30)])
    arrow(ax,[(83,44),(83,36)])
    arrow(ax,[(83,25),(83,19)])
    ax.text(85,21.8,"scores $a$",fontsize=6.2)
    for y in (54,41,28):
        ax.plot([63,64.5],[y,y],ls="--",lw=.7,color=COLORS["training_line"],zorder=2)
    arrow(ax,[(64.5,54),(64.5,18.5),(59,18.5),(59,17)],True)
    arrow(ax,[(63,8),(67.5,8),(67.5,27),(69,27)],True)
    ax.text(84,-1.8,"Four-class probabilities $q$",fontsize=6.5,ha="center")
    ax.text(2,-1.8,"Solid: inference     Dashed: training-only auxiliary reuse",fontsize=6.1,color=COLORS["line"])

    detail = fig.add_axes([.035,.11,.45,.20])
    detail.set(xlim=(0,100),ylim=(0,50)); detail.axis("off")
    detail.text(0,49,"b  Each bounded adapter",weight="bold",fontsize=8.2)
    box(detail,(0,22),28,17,r"Linear 64 $\to$ 4"+"\nwith bias","train",6.5)
    box(detail,(36,22),28,17,"SiLU then\n"+r"Linear 4 $\to$ 64"+"\nno bias","train",6.2)
    box(detail,(72,22),28,17,"tanh / 8\n"+r"$\|\delta_j\|_2\leq1$","analytic",6.5)
    arrow(detail,[(28,30.5),(36,30.5)])
    arrow(detail,[(64,30.5),(72,30.5)])
    detail.text(0,14,r"$\delta_j=\frac{1}{\sqrt{64}}\tanh[U_j\mathrm{SiLU}(V_jh+b_j)]$",fontsize=7.6)
    detail.text(0,5,"Parameters: (64 x 4 + 4) + (4 x 64) = 516",fontsize=6.7)

    comparison = fig.add_axes([.53,.11,.435,.20])
    comparison.set(xlim=(0,100),ylim=(0,50)); comparison.axis("off")
    comparison.text(0,49,"c  Matched routing comparison",weight="bold",fontsize=8.2)
    box(comparison,(0,25),100,13,r"S1: uniform $g_j=b/3$    |    R1: role $g_j=4\bar p_T\bar p_j$","analytic",7)
    comparison.text(0,16,"Same three adapters, shared head, pair loss and budget.",fontsize=6.7)
    comparison.text(0,7,"R1 trainable parameters: 3 x 516 + 1476 = 3024",fontsize=6.7)
    fig.text(.035,.065,r"$t=\bar p_T,\ w_j=\bar p_j/(1-t),\ j\in\{\mathrm{Ti},\mathrm{G},\mathrm{M}\}$;  $T$: target proxy; Ti/G/M: three non-target roles.",fontsize=6.8)
    fig.text(.035,.039,"Frozen image features and evidence are inputs. Labels enter the training loss only. Bounds do not guarantee accuracy or recall.",fontsize=6.5,color=COLORS["line"])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for patch,text,label in box_text:
        bounds = patch.get_window_extent(renderer)
        text_bounds = text.get_window_extent(renderer)
        if (text_bounds.x0 < bounds.x0+1 or text_bounds.x1 > bounds.x1-1
                or text_bounds.y0 < bounds.y0+1 or text_bounds.y1 > bounds.y1-1):
            overflow.append(label)
    if overflow:
        plt.close(fig)
        raise ValueError("Diagram box text overflow: "+repr(overflow))
    base = output_dir/"lc_rfa_b_architecture"
    paths = {ext:str(base.with_suffix("."+ext)) for ext in ("svg","pdf","png")}
    for ext,path in paths.items():
        fig.savefig(path,dpi=300,facecolor="white")
    plt.close(fig)
    source = base.with_suffix(".json")
    source.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return dict(paths,source=str(source),qa=dict(text_overflow=[],size_inches=[7.2,6.8],png_dpi=300,state=STATE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = RoleResidualModel("R1",seed=20261002,alpha_max=.5,beta_max=.5)
    print(json.dumps(render_architecture(architecture_manifest(model),args.output_dir)))


if __name__ == "__main__":
    main()
