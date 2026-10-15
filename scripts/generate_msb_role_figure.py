"""Python-only candidate architecture; no image loading or performance claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from msb_role_data import file_digest
from msb_role_model import MSBRoleModel, unit_spec

STATUS_LABEL = "MSB-RRE v1：合成验收，真实效果未验证"


def connector_intersections(connectors, boxes):
    hits = []
    for i,points in enumerate(connectors):
        for j,(x,y,w,h) in enumerate(boxes):
            for (x1,y1),(x2,y2) in zip(points,points[1:]):
                horizontal = abs(y1-y2)<1e-10 and y+1e-8<y1<y+h-1e-8 and max(min(x1,x2),x)<min(max(x1,x2),x+w)-1e-8
                vertical = abs(x1-x2)<1e-10 and x+1e-8<x1<x+w-1e-8 and max(min(y1,y2),y)<min(max(y1,y2),y+h)-1e-8
                if horizontal or vertical:
                    hits.append((i,j)); break
    return hits


def figure_spec() -> dict:
    model = MSBRoleModel("MAIN",seed=20261002)
    settings = unit_spec("MAIN")
    branch = model.branches[0]
    return dict(unit="MAIN",roles=["Ti","G","M"],rank=settings["rank"],gamma=settings["gamma"],
                tau=settings["tau"],trainable_parameters=sum(p.numel() for p in model.parameters()),
                fixed_reference_parameters=sum(p.numel() for p in model.buffers()),
                U_shape=list(branch.U.shape),V_shape=list(branch.V.shape),
                evidence_dimensions=dict(E=18,log_base=4,contradiction=1,total=branch.V.shape[1]),
                frozen_backbone="EfficientNet-B0",frozen_backbone_width=1280,
                trainable_shared_readout=False,real_effect_verified=False,
                mechanisms=["Hadamard","fixed_initial_reference","margin_budget","common_softmax","training_loss"],
                relative_weight=settings["relative_weight"],pair_weight=settings["pair_weight"],sp_weight=settings["sp_weight"],
                formulas=dict(center="s_j = f_theta_j(h,e) - f_theta0_j(h,e)",
                              budget="R_ref_j = 2*abs(log_anchor_T-log_anchor_j)+0.25",
                              correction="d_j=R_ref_j*tanh(s_j)",
                              decoder="softmax([log_anchor_T, log_anchor_non-d])"))


def build_architecture_figure(output: Path, *, status_label: str) -> dict:
    output = Path(output)
    if status_label != STATUS_LABEL:
        raise ValueError("Candidate synthetic/unverified status is mandatory.")
    resolved = output.resolve()
    if tuple(resolved.parts[-3:]) != ("outputs","msb_role_synthetic_v1","assets"):
        raise ValueError("Only the new MSB synthetic assets directory may be written.")
    if output.exists(): raise FileExistsError("Candidate figure exports are exclusive.")
    for parent in (output,*output.parents):
        if parent.is_symlink(): raise ValueError("Figure export symlink forbidden.")
    output.mkdir(parents=True,exist_ok=False)
    spec = figure_spec()
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((f for f in ("Microsoft YaHei","SimHei","Noto Sans CJK SC") if f in fonts),None)
    if font is None: raise RuntimeError("A Chinese-capable font is required for verified figure labels.")
    style = {"font.family":"sans-serif","font.sans-serif":[font,"DejaVu Sans"],
             "font.size":10,"svg.fonttype":"none","pdf.fonttype":42,
             "text.usetex":False,"axes.unicode_minus":False,"mathtext.fontset":"dejavusans"}
    with plt.rc_context(style):
        fig,ax = plt.subplots(figsize=(13.2,8.8))
        fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985)
        ax.set(xlim=(0,18),ylim=(0,12)); ax.axis("off")
        blue,teal,gray = "#52758A","#4D807A","#687178"
        boxes = []
        connectors = []

        def text(x,y,label,**kwargs):
            item = ax.text(x,y,label,color="#20272B",va="center",ha="center",**kwargs)
            return item

        def box(x,y,w,h,label,*,fill="#F0F3F4",edge=gray,size=10,dashed=False):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=fill,edgecolor=edge,linewidth=.9,linestyle="--" if dashed else "-"))
            item=text(x+w/2,y+h/2,label,fontsize=size,linespacing=1.55)
            boxes.append((item,(x,y,w,h)))

        def arrow(points,*,color=gray,dashed=False):
            connectors.append(points)
            for a,b in zip(points[:-2],points[1:-1]):
                ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.9,ls="--" if dashed else "-")
            ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=10,
                         linewidth=.9,color=color,linestyle="--" if dashed else "-",shrinkA=1,shrinkB=3))

        ax.text(.35,11.65,"MSB-RRE v1 | 间隔缩放预算与独立角色残差证据",ha="left",va="center",fontsize=15,weight="bold")
        ax.text(.35,11.12,status_label,ha="left",va="center",fontsize=11,color=teal)
        ax.text(.35,10.72,"a  冻结输入与三个独立角色分支",ha="left",va="center",fontsize=11,weight="bold")
        box(.4,9.35,3.65,1.05,"冻结 EfficientNet-B0\n"+r"$H_{1280}\ \longrightarrow\ \mathrm{PCA}_{\mathrm{fit}}\ \longrightarrow\ h_{64}$",edge=blue)
        box(.4,7.85,3.65,1.05,"已有标准化预测证据\n"+r"$e_{23}=[E_{18},\log p_{\rm base,4},c_1]$",edge=blue)
        box(.4,6.35,3.65,1.05,"冻结概率锚点（既有折内温度）\n"+r"$\ell=\log p_{\rm anchor}=\log T_0(p_{\rm base})$",size=9)
        for j,(role,y) in enumerate(zip(spec["roles"],(9.55,8.15,6.75))):
            box(5.0,y,3.35,.9,f"独立角色 {role}  ·  rank {spec['rank']}\n"+r"$s_j=f_{\theta_j}(h,e)-f_{\theta^0_j}(h,e)$",fill="#EAF2F0",edge=teal,size=10)
            arrow([(4.05,9.875),(4.35,9.875),(4.35,y+.58),(5,y+.58)],color=blue)
            arrow([(4.05,8.375),(4.65,8.375),(4.65,y+.25),(5,y+.25)],color=gray)
            arrow([(8.35,y+.45),(8.75,y+.45),(8.75,9.7),(9.25,9.7)],color=teal)
        box(9.25,8.35,3.4,2.05,"间隔缩放预算\n"+r"$m_j=\ell_T-\ell_j$"+"\n"+
            rf"$R_j={spec['gamma']:g}|m_j|+{spec['tau']:g}$"+"\n"+r"$d_j=R_j\tanh(s_j)$",fill="#EAF2F0",edge=teal,size=11)
        arrow([(4.05,6.875),(4.2,6.875),(4.2,6.1),(8.95,6.1),(8.95,8.75),(9.25,8.75)])
        box(13.25,8.35,4.2,2.05,"角色坐标修正，无可训练共享读出\n"+
            r"$\widetilde\ell=[\ell_T,\ell_{Ti}-d_{Ti},$"+"\n"+r"$\ell_G-d_G,\ell_M-d_M]$",size=10)
        arrow([(12.65,9.35),(13.25,9.35)],color=teal)
        arrow([(4.05,6.55),(4.12,6.55),(4.12,5.95),(12.95,5.95),(12.95,8.75),(13.25,8.75)])
        box(13.25,6.6,4.2,1.0,"共同 softmax\n"+r"$q=\mathrm{softmax}(\widetilde\ell)$",fill="#EAF2F0",edge=teal,size=11)
        arrow([(15.35,8.35),(15.35,7.6)])
        text(15.35,6.1,"固定目标 logit 不等于固定目标概率",fontsize=9)
        ax.plot([.35,17.6],[5.7,5.7],color="#CAD0D3",lw=.7)
        ax.text(.35,5.35,"b  单个角色 j 的非线性证据分支（上图三个角色各自独立）",ha="left",va="center",fontsize=11,weight="bold")
        box(.4,3.65,4.25,1.25,r"$z_h=\tanh(U_jh+u_j),\quad U_j:4\times64$"+"\n"+
            r"$z_e=\tanh(V_je+v_j),\quad V_j:4\times23$",edge=blue,size=10)
        box(5.3,3.65,2.2,1.25,"Hadamard 乘积\n"+r"$z_h\odot z_e\in\mathbb{R}^{4}$",fill="#EAF2F0",edge=teal)
        box(8.15,3.65,3.75,1.25,"当前独立线性读出\n"+r"$f_{\theta_j}=a_j^T(z_h\odot z_e)/\sqrt{4}+\zeta_j$",fill="#EAF2F0",edge=teal,size=9)
        box(12.55,3.65,4.9,1.25,"同输入、同结构的初始参照相减\n"+r"$s_j=f_{\theta_j}(h,e)-f_{\theta^0_j}(h,e)$",fill="#EAF2F0",edge=teal,size=11)
        arrow([(4.65,4.275),(5.3,4.275)],color=blue)
        arrow([(7.5,4.275),(8.15,4.275)],color=teal)
        arrow([(11.9,4.275),(12.55,4.275)],color=teal)
        box(.4,2.0,11.5,1.05,"固定初始参照 "+r"$f_{\theta^0_j}(h,e)$"+"：独立缓冲区，不参与更新；与当前分支使用相同输入\n"+
            "初始化残差为零；训练与推理均需计算当前分支和固定参照",dashed=True,size=10)
        arrow([(11.9,2.525),(15,2.525),(15,3.65)],dashed=True)
        text(15.0,2.05,f"{len(spec['roles'])} × {spec['trainable_parameters']//3} = {spec['trainable_parameters']} 可训练参数\n"+
             f"另存 {spec['fixed_reference_parameters']} 个固定参照参数",fontsize=10,linespacing=1.7)
        box(.4,.4,17.05,1.1,"仅训练损失： "+
            r"$\mathcal{L}=\mathcal{L}_{\rm NLL}+0.1\mathcal{L}_{\rm pair}$"+
            r"$+0.01\,\mathbb{E}_{i,j}(d_{ij}/R_{ij})^2+\frac{0.001}{2}\|\theta-\theta^0\|_2^2$"+
            "\n候选适配网络，不改造 CNN 主干；预测证据不是新增物理模态；无性能或工业效果宣称",dashed=True,size=10)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overflow = []
        for item,(x,y,w,h) in boxes:
            bounds = item.get_window_extent(renderer).transformed(ax.transData.inverted())
            if bounds.x0<x-.03 or bounds.x1>x+w+.03 or bounds.y0<y-.03 or bounds.y1>y+h+.03:
                overflow.append(item.get_text())
        if overflow:
            plt.close(fig)
            raise ValueError("Figure text overflows its formula box: "+repr(overflow))
        crossings=connector_intersections(connectors,[bounds for _,bounds in boxes])
        if crossings:
            plt.close(fig)
            raise ValueError("Figure connector crosses formula box: "+repr(crossings))
        stem = output/"msb_rre_v1_architecture"
        for extension in ("svg","pdf","png"):
            fig.savefig(stem.with_suffix("."+extension),dpi=300,facecolor="white")
        plt.close(fig)
    result = dict(status=status_label,real_effect_verified=False,font=font,text_overflow=[],connector_box_intersections=[],spec=spec,
                  artifacts={stem.with_suffix("."+extension).name:file_digest(stem.with_suffix("."+extension))
                             for extension in ("svg","pdf","png")})
    with (output/"architecture_metadata.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build_architecture_figure(args.output,status_label=STATUS_LABEL),ensure_ascii=False))
