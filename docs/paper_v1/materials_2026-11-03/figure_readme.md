# RSG 结构图：R1–R12排版源与图注

日期：2026-11-03。该图只排版正文已有计算关系，不执行网络，不读取图像、预测或权重，不引入候选模块。

## 图件约定

- 结论：共享CNN之上的双专家、完整门控输入隔离、双验证器和训练期后悔监督是相互区分的计算路径。
- 类型：机制示意图，不是性能结果图。上面板a为推理依赖，下面板b为训练监督。
- 标注：R1–R12与正文编号一致；重复出现的同名张量指同一数值，不表示新增分支或复制网络。
- 尺寸：183 × 184 mm；英文标签与数学符号，中文图注在下方。PNG为300 dpi预览，PDF和SVG为矢量输出。SVG保留文本，PDF嵌入TrueType字体。归档时仅清除SVG的行尾空白，不改变路径、文字或布局。
- 后端：沿用用户选择的Python，仅用matplotlib绘制常量框、线和公式。以下文档代码块是绘图排版源，不修改项目模型或实验代码。
- 维度：主实验EfficientNet-B0特征1280，拼接1283，门控隐藏层128，对比嵌入128。后两项来自18份运行记录；ResNet50替换实验不混画在本主图。
- 输入分辨率和专家Dropout概率未由本轮配置记录确认，因此图中不填推测值。
- 图件不得用来声称网络创新效果已经得到额外验证。

## 图注

**图1｜RSG-HRGV-Net的计算结构与公式对应。** a，共享EfficientNet-B0和全局池化得到特征；训练期Dropout后的特征供四角色与十七矿种分类头使用，固定映射将矿种概率聚合为角色概率。门控基于未Dropout的池化特征、两个归一化熵及JS分歧，完整拼接输入先停止梯度，再产生直接角色权重。两路角色概率凸融合后，由目标—含钛及目标—金属光泽验证器的中性区惩罚校正。b，真实标签只参与训练：专家真值类损失差产生停止梯度的软目标和差距权重，按批次权重和归一化的BCE训练门控；对比投影及其余辅助损失参与联合训练。图中实线表示前向依赖，虚线表示仅训练关系；同名张量重复标注以减少交叉线。局部停止梯度不冻结整个主干，主配置的验证器特征仍与主干耦合。该图说明实现关系，不证明总体分类优势或工业应用能力。

## 排版源

只在本目录生成三个图件；不导入项目或深度学习库。输出为图件导出，不是模型前向或数值实验。可在本目录用装有matplotlib的Python执行下面代码块。

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.8,
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "svg.hashsalt": "rsg-r1-r12-20261103",
})
fig = plt.figure(figsize=(183 / 25.4, 184 / 25.4), facecolor="white")
ax = fig.add_axes([.025, .025, .95, .95])
ax.set(xlim=(0, 100), ylim=(0, 100))
ax.axis("off")
ink = "#263238"
expert = "#E6EEF2"
gate = "#E4F1EC"
verify = "#F3E6EA"
train = "#F0F0F0"
text_boxes = []

def box(x, y, w, h, text, face, size=6.8):
    patch = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15,rounding_size=0.7",
                          linewidth=.65,edgecolor=ink,facecolor=face,zorder=2)
    ax.add_patch(patch)
    t = ax.text(x+w/2,y+h/2,text,ha="center",va="center",
                fontsize=size,color=ink,linespacing=1.4,zorder=3)
    text_boxes.append((patch,t))

def arrow(points, dashed=False):
    style = (0,(3,2)) if dashed else "-"
    for a,b in zip(points[:-2],points[1:-1]):
        ax.add_line(Line2D([a[0],b[0]],[a[1],b[1]],color=ink,lw=.7,ls=style,zorder=1))
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",
                                mutation_scale=6,lw=.7,color=ink,linestyle=style,zorder=1))

def label(x,y,text,size=6.5,ha="center"):
    ax.text(x,y,text,fontsize=size,ha=ha,va="center",color=ink,zorder=3)

label(1,98,"a  Inference graph: shared visual evidence and role decision",8.5,"left")
box(1,84,11,8,"RGB\n$x$",train)
box(16,81,20,14,"EfficientNet-B0\nGlobal avg. pool\n$h\\in\\mathbb{R}^{1280}$  [R1]",expert)
box(40,84,15,8,"Dropout\n$\\widetilde{h}$  [R1]",expert)
box(60,88,17,8,"Affine + softmax\n$p_d\\in\\mathbb{R}^{4}$  [R2]",expert,6.6)
box(60,75,17,8,"Affine + softmax\n$p_s\\in\\mathbb{R}^{17}$  [R2]",expert,6.6)
box(81,75,17,8,"Fixed $A_{4\\times17}$\n$p_m=Ap_s$  [R3]",expert,6.6)
arrow([(12,88),(16,88)])
arrow([(36,88),(40,88)])
arrow([(55,88),(57.5,88),(57.5,92),(60,92)])
arrow([(55,88),(57.5,88),(57.5,79),(60,79)])
arrow([(77,79),(81,79)])

box(1,59,37,12,"Reuse $h,p_d,p_m$ from above\n$z=[h,\\bar H(p_d),\\bar H(p_m),D_{JS}]$\nComplete input: $\\mathrm{sg}(z)$  [R4]",gate,6.6)
box(43,59,23,12,"Gate: $1283\\to128\\to1$\nReLU / Dropout(0.10)\n$g=\\sigma(G_\\phi(\\mathrm{sg}(z)))$  [R4]",gate,6.5)
box(72,59,26,12,"Convex fusion  [R5]\n$p_f=gp_d+(1-g)p_m$\n$g$: direct-role weight",gate,6.7)
arrow([(38,65),(43,65)])
arrow([(66,65),(72,65)])
arrow([(77,92),(99.2,92),(99.2,65),(98,65)])
arrow([(89.5,75),(89.5,71)])
label(85,73,"$p_m$",6.2)

box(1,43,31,10,"Reuse $h$: two affine $1280\\to2$ heads\nTarget vs Ti / Target vs metallic\nSoftmax target probabilities $v_{Ti},v_{Met}$",verify,6.3)
box(37,43,26,10,"Neutral-zone penalty  [R6]\n$c_k=[\\tau_k-v_k]_+/\\tau_k$\n$\\eta=\\beta_{Ti}c_{Ti}+\\beta_{Met}c_{Met}$",verify,6.5)
box(72,43,26,10,"Target-odds correction\n$q\\propto(p_{f,T}e^{-\\eta},p_{f,\\neg T})$\nNormalize to four roles  [R7, R8]",verify,6.3)
arrow([(32,48),(37,48)])
arrow([(63,48),(72,48)])
arrow([(85,59),(85,53)])
label(1,39.7,"Named tensors are reused, not extra inputs. No labels enter inference.",6.5,"left")
ax.axhline(37.5,xmin=.01,xmax=.99,color="#B0BEC5",lw=.6)

label(1,35.5,"b  Training-only supervision: labels do not enter panel a",8.5,"left")
box(1,23,23,9,"$p_d,p_m$ + true role $y$\n$\\ell_d=-\\log p_d(y)$\n$\\ell_m=-\\log p_m(y)$  [R10]",train,6.5)
box(29,23,15,9,"Loss gap\n$\\Delta=\\ell_m-\\ell_d$\n[R10]",train,6.6)
box(49,23,23,9,"$g^*=\\sigma(\\Delta/T_r)$\n$w=\\tanh(|\\Delta|/T_w)$\n$\\mathrm{sg}(g^*),\\mathrm{sg}(w)$  [R11]",gate,6.5)
box(77,23,21,9,"$g$ from panel a\nWeighted BCE / $\\sum_i w_i$\n$\\mathcal{L}_{reg}$  [R12]",gate,6.2)
arrow([(24,27.5),(29,27.5)],True)
arrow([(44,27.5),(49,27.5)],True)
arrow([(72,27.5),(77,27.5)],True)

box(1,8,32,10,"Reuse $h$: contrast projection\n$1280\\to1280\\to128$; ReLU; L2 norm\nRole labels supervise contrast loss",train,6.5)
box(38,8,60,10,"Joint objective; role/species labels  [R9]\nRole + weighted direct/species/KL/verifier losses\n$+\\;\\lambda_{con}\\mathcal{L}_{contrast}+\\lambda_g\\mathcal{L}_{reg}$",train,6.8)
arrow([(33,13),(38,13)],True)
arrow([(87.5,23),(87.5,18)],True)
label(1,4.4,"Solid: forward dependency. Dashed: training only. sg: stop-gradient.",6.5,"left")
label(1,1.4,"Only the regret branch is locally isolated; verifier features remain coupled.",6.5,"left")

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
overflow = []
for patch,t in text_boxes:
    pb,tb = patch.get_window_extent(renderer),t.get_window_extent(renderer)
    if tb.x0 < pb.x0 or tb.x1 > pb.x1 or tb.y0 < pb.y0 or tb.y1 > pb.y1:
        overflow.append(t.get_text())
if overflow:
    raise RuntimeError("Text exceeds boxes: " + repr(overflow))
out = Path.cwd()
for ext in ("svg","pdf","png"):
    kw = {"dpi":300} if ext=="png" else {}
    metadata = {"Creator":"RSG manuscript material layout"} if ext=="pdf" else {}
    if ext=="svg":
        metadata={"Date":None,"Creator":"RSG manuscript material layout"}
    fig.savefig(out / ("fig_rsg_r1_r12."+ext),metadata=metadata,**kw)
print("Exported SVG/PDF/PNG; all box labels fit.")
plt.close(fig)
```

## 图与公式核对

| 编号 | 图中位置 | 应检查的限定 |
|---|---|---|
| R1 | a第一行共享特征与Dropout | 只有专家分类头使用Dropout后的特征 |
| R2 | a第一行两个affine + softmax | affine包含偏置；四角色与十七矿种分开 |
| R3 | a右上固定映射 | 不是可学习映射；不产生新标签 |
| R4 | a第二行拼接、sg、门控 | sg覆盖完整1283维输入；g是直接专家权重 |
| R5 | a第二行右侧融合 | 验证前后验为两专家凸组合 |
| R6 | a第三行中性惩罚 | 两个验证概率；非负惩罚，不能主动提高目标赔率 |
| R7–R8 | a右下后验 | 目标缩放后对全部四角色归一化 |
| R9 | b下方联合目标 | 其他任务仍训练主干；对比投影仅训练需要 |
| R10 | b上方前两框 | 真值类损失差只在训练可用 |
| R11 | b上方第三框 | 目标和差距权重均停止梯度 |
| R12 | b右上 | 权重和归一化；完整零权重与截断约定见正文 |

图中为节省空间缩写R12为“weighted BCE / sum weights”；其零权重分支和数值截断未画全，必须配合正文R12阅读。图中L2 norm指对比表示归一化，不是额外的参数正则化。颜色只是分组辅助，不表达性能好坏。
