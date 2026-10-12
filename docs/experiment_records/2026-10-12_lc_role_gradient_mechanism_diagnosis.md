# LC-RFA-B v2：梯度机制诊断与网络理论改进候选

## Material Passport

- 日期：2026-10-12。
- 类型：已有实验的只读机制分析；`Verification Status: ANALYZED`。
- 原始结果提交：`29769e01cb01c70762ec7fe5e484aac0992349a7`。
- 实验：`lc_role_adaptation_v2/development_fold_0`，授权已消耗。
- 材料：锁定参数状态、前五次更新前的梯度范数、完整训练历史及既有评价摘要。
- 本轮没有模型前向、自动求导、优化器更新、训练重放、重新筛选或 stop 评价；没有读取原图或冻结视觉缓存。
- 没有修改冻结代码、协议、失败目录或正式技术报告。网络改进仅是待讨论候选，不是已批准规格或已实现方法。
- 本文由作者侧分析产生，不是独立审稿或独立训练复现。

## 1. 结论与证据层级

完整重跑仍是负结果：R1 与同预算 S1 在三个种子上的 340 张已暴露开发 stop 图像具有完全相同的 argmax；R1 对目标类的正确数为 40/41/40，低于 T0 的 43。注册筛选另外因十二个截断对照条目而失败。这里不重新解释或放宽该筛选。

本轮找到一个**可以由代码证明、又得到全部存储轨迹支持的结构机制**：适配器上投影零初始化与共享读出末层零初始化形成串联梯度阻断。随后，下投影的数据梯度很小，而显式 L2 正则化对随机下投影立即产生更新。

证据能够支持“存在初始梯度阻断和明显的参数范数收缩”，但尚不能支持“消除该机制就能提高目标召回”或“它是负结果的唯一原因”。前五步以后的梯度没有存储，不能虚构后期梯度结论。

论文理论主线应从“有界公式保证更好识别”修正为：**联合分析有界角色适配的稳定性、可学习性及二者的冲突，并设计能保持锚点恒等起点、又不结构性关闭分支 Jacobian 的网络。** 这仍是候选贡献，不能把通用初始化技巧直接称为原创理论。

## 2. 全量轨迹诊断

锁定记录包含 392 个唯一拟合状态：372 条神经网络轨迹和 20 个凸对照状态；重复使用的 matched-S1 状态按路径去重，不增加重复实验数。神经网络计数为 H0/F0/S0/S1/R0/R1 各 57 条、U1 30 条。

派生证据目录为 `outputs/lc_role_gradient_diagnosis_v2`。读取的状态文件和存档 CSV 均逐文件核对原始 manifest 的 SHA-256；梯度和历史 CSV 与状态中嵌入的记录逐项相等。总计：

| 材料 | 数量 | 范围 |
| --- | ---: | --- |
| 神经网络轨迹 | 372 | 351 条 CV + 21 条最终拟合 |
| 梯度记录 | 19,050 | 各轨迹前五次更新前 |
| 历史记录 | 146,472 | 包括没有被选择的轨迹与截断控制 |
| 检查点参数范数记录 | 26,448 | 原始参数及三个下投影合并范数 |
| 第一步梯度恒等核对 | 3,810 | 梯度范数与初始参数范数 |

有适配器的轨迹共 258 条，每条有三个下投影权重、三个下投影偏置和三个上投影权重。

| 可直接验证的观察 | 分母 | 结果 |
| --- | ---: | --- |
| 第一步下投影数据梯度为零 | 1,548 个参数块 | 全部为零 |
| 第二步下投影数据梯度为零 | 1,548 个参数块 | 全部为零 |
| 第一步上投影数据梯度为零 | 774 个参数块 | 全部为零 |
| 第一步读出首层权重/偏置数据梯度为零 | 各 372 个参数块 | 全部为零 |
| 第一步数据梯度为零的块满足总梯度范数 = lambda × 初始参数范数 | 3,066 个块 | 误差 < 1e-14 |

最后一行不包括 744 个读出末层权重/偏置块，它们的数据梯度可非零。合计为 3,810 个第一步记录。

定义方向无关的诊断比值

$$
r_G=\frac{\|\nabla L_{\mathrm{data}}\|_2}
{\|\nabla L_{\mathrm{total}}\|_2}.
$$

R1 的全部 57 条轨迹在第五步有 342 个下投影参数块，$r_G$ 中位数为 $2.052\times10^{-6}$，最大值为 $8.205\times10^{-5}$。最终三种子 R1 的 18 个块，中位数为 $1.678\times10^{-5}$、最大值为 $6.437\times10^{-5}$。这些是描述性参数块统计，不是独立统计样本或显著性检验。

不能把总梯度范数减去数据梯度范数当作精确正则化梯度。原始记录不包含向量方向，只能使用

$$
\max(0,\|g_{\mathrm{total}}\|-\|g_{\mathrm{data}}\|)
\leq \|g_{\mathrm{reg}}\|
\leq \|g_{\mathrm{total}}\|+\|g_{\mathrm{data}}\|.
$$

因此上述小比值说明第五步下投影更新的梯度范数主要受正则化项影响，但不能推断每个坐标的方向或每次 Adam 更新的贡献比例。

把下投影偏置并入矩阵，记 $\bar D_j=[D_j,d_j]$。最终 R1 三种子共九个分支在 100 次更新后，$\|\bar D_j\|_F/\|\bar D_{j,0}\|_F$ 为 0.21660--0.22995；S1 的范围为 0.21630--0.22991。已有评价中，R1 分支输出非零但最大范数仅约 $3.28\times10^{-4}$、$5.26\times10^{-4}$、$3.81\times10^{-4}$，且逐分支固定状态移除均改变零个 argmax。

R1 所有 CV 轨迹的下投影合并范数保留比例中位数如下。每格为 54 个分支块，混合两个注册预算、三个种子及三个内层折；不是重新选择的超参数。

| lambda | 20 步 | 50 步 | 100 步 | 200 步 | 400 步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| .001 | .770735 | .497999 | .220447 | .028713 | .00005044 |
| .01 | .770709 | .497937 | .220106 | .028672 | .00005018 |
| .1 | .770706 | .497932 | .220101 | .028670 | .00005016 |

不同 lambda 下的下投影收缩非常相近，不能据此建议“把 lambda 降一点就会解决”。该现象与 Adam 对梯度尺度的归一化及早期数据梯度弱相容，但没有执行去正则化或改变初始化的因果对照。

## 3. 命题一：串联零初始化的梯度阻断

### 定义

冻结的视觉特征为 $h\in\mathbb R^{64}$，辅助证据为 $e\in\mathbb R^{23}$，$\bar h=[h;1]$。三个适配器和共享读出为

$$
\delta_j(h)=\frac{\varepsilon_f}{\sqrt d}
\tanh\!\left(U_j\phi(\bar D_j\bar h)\right),\qquad d=64,
$$

$$
\widetilde h=h+\sum_{j=1}^3g_j\delta_j(h),\qquad
F(\widetilde h,e)=W_2\phi(W_1[\widetilde h;e]+c_1)+c_2,
$$

其中 $\phi$ 为 SiLU，当前实现 $\varepsilon_f=1$、瓶颈宽度为 4、共享读出宽度为 16。初始化为 $U_j=0,W_2=0,c_2=0$；$\bar D_j,W_1,c_1$ 非零随机初始化。

### 命题与证明

对任意经过该读出得到的可微数据损失，包括主任务 NLL 和现有无缩放辅助视图的 pair loss，初始时

$$
\nabla_{U_j}L_{\mathrm{data}}=
\nabla_{\bar D_j}L_{\mathrm{data}}=
\nabla_{W_1}L_{\mathrm{data}}=
\nabla_{c_1}L_{\mathrm{data}}=0.
$$

证明：读出对特征的 Jacobian 为

$$
J_hF=W_2\operatorname{diag}(\phi'(\cdot))W_{1,h}=0,
$$

故适配器所有数据梯度被链式法则中的 $W_2$ 阻断；$W_1,c_1$ 的数据梯度也含该因子。$W_2,c_2$ 的数据梯度不必为零，网络不是永远无法学习。

当前显式正则化为 $L_{\mathrm{reg}}=\lambda\|\theta\|_2^2/2$。第一步 $U_j$ 的总梯度也为零，因此从零 Adam 状态出发，上投影在第一次更新后仍为零。第二步下投影数据梯度还含 $U_j$，故仍为零；上投影则可以在第二步开始收到数据梯度。后续是否非零取决于样本、参数和优化轨迹，不能无条件保证。

该命题是当前结构的数学诊断，不应单独宣称新的一般深度学习定理。

## 4. 命题二：稳定界同时限制适配分支梯度

### 明确假设

以下分析针对 R1 主路径及现有 pair 辅助损失。锚点概率和特征冻结，不通过它们反向传播；$0<t<1$。令

$$
\rho_j=\frac{p_j}{1-t},\quad b=4t(1-t),\quad g_j=b\rho_j,
\quad \kappa=\sqrt{\bar\alpha^2+\bar\beta^2/2}.
$$

概率修正仍为 $u=\bar\alpha b\tanh F_T$、$v_j=\bar\beta b\tanh F_j/2$。$L_F$ 是读出对视觉输入的 Lipschitz 上界，可取

$$
L_F=\|W_2\|_2L_\phi\|W_{1,h}\|_2,
\qquad L_\phi=1+e^{-1}.
$$

该 SiLU 导数界是保守上界，不是紧常数。

### 单样本梯度界

主任务 NLL 对四个读出分数的梯度范数满足

$$
\|\nabla_F\ell_{\mathrm{main}}\|_2\leq b\kappa.
$$

理由：目标 logit 的导数绝对值不超过 $\bar\alpha b$；非目标 softmax 交叉熵的梯度范数不超过 $\sqrt2$，再乘 $\bar\beta b/2$。对目标样本非目标项为零，仍满足该界。令 $a_j=\phi(\bar D_j\bar h)$，由链式法则和 $|\tanh'|\leq1$，

$$
\|\nabla_{U_j}\ell_{\mathrm{main}}\|_F
\leq \frac{\varepsilon_f}{\sqrt d}\kappa L_F
b^2\rho_j\|a_j\|_2,
$$

$$
\|\nabla_{\bar D_j}\ell_{\mathrm{main}}\|_F
\leq \frac{\varepsilon_f}{\sqrt d}\kappa L_F
b^2\rho_j\|U_j\|_2L_\phi\|\bar h\|_2.
$$

同样的 $b^2$ 因子来自两处：特征适配路由 $g_j=b\rho_j$ 和概率预算 $b$。它既产生此前的主路径稳定证书，也压低主路径可传入分支的梯度上界。下投影还受 $\|U_j\|_2$ 限制；初始 $L_F=0$，与命题一一致。

辅助视图为 $h+\delta_j$，没有 $g_j$。对目标与第 $j$ 类的条件二分类损失，同样可由二分类 log-odds 导数得到 $\|\nabla_F\ell_{\mathrm{pair},j}\|\leq b\kappa$。因此它的对应界是 **$b$ 而不是 $b^2\rho_j$**。对当前数据目标 $L_{\mathrm{main}}+.1\sum_j L_{\mathrm{pair},j}/3$，上投影界为

$$
\|\nabla_{U_j}L_{\mathrm{data}}\|_F
\leq\frac{\varepsilon_f\kappa L_F}{\sqrt d}
\left[
\mathbb E_{\mathrm{fit}}b^2\rho_j\|a_j\|_2+
\frac{.1}{3}\mathbb E_{T,j}b\|a_j\|_2
\right].
$$

其中第二个平均仅覆盖目标/第 $j$ 类样本。不能把辅助视图也描述成具有主路径 $b^2$ 保持界。以上是**梯度上界，不是非零梯度下界、训练收敛保证、真实标签风险保证或性能因果解释**。

### 优化器尺度的补充

在第一步数据梯度为零的参数坐标 $d$ 上，显式正则化梯度是 $\lambda d$。对初始状态为零、带偏差修正的 Adam，第一步为

$$
d^{(1)}=d^{(0)}-\eta
\frac{\lambda d^{(0)}}{|\lambda d^{(0)}|+\epsilon_{\mathrm{Adam}}}.
$$

当 $|\lambda d|\gg\epsilon_{\mathrm{Adam}}$ 时，更新约为 $-\eta\operatorname{sign}(d)$，并不随 lambda 线性变小。当前 $\eta=.001$，显式 L2 加入损失，Adam 的 `weight_decay=0`；不能把它与 AdamW 的解耦权重衰减混为一谈。该首步公式也不能外推成整个 400 步都独立于 lambda。

## 5. 下一轮网络候选：恒等起点与可学习性解耦

这是供下一步讨论的数学候选，**不是已批准的网络规格或实验**。不是通过放大预算、去掉风险约束或换个模型名称来获得增益。

可以研究固定初始函数参照的有界适配器：

$$
\delta_j^\star(h)=\frac{\varepsilon_f}{2\sqrt d}
\left[
\tanh(U_j\phi(\bar D_j\bar h))-
\tanh(U_{j,0}\phi(\bar D_{j,0}\bar h))
\right].
$$

初始参数取相同非零参照，参照分支冻结。于是 $\delta_j^\star=0$，但参数 Jacobian 不再被零上投影结构性关闭；同时差值每个坐标不超过 2，仍有 $\|\delta_j^\star\|_2\leq\varepsilon_f$。**非零参数不能保证每个样本或整个 batch 的梯度非零**，仍需要非退化 Jacobian、非零预算和损失方向不抵消的条件。

共享读出候选为

$$
s^\star=F_\theta(\widetilde h,e)-F_{\theta_0}(h,e),
\qquad \widetilde h=h+\sum_jg_j\delta_j^\star(h).
$$

读出初始末层非零，但参数和适配器处在参照状态时 $s^\star=0$，因此初始概率仍等于锚点。**参照读出必须在原始冻结输入 $h$ 上计算，不能同样使用 $\widetilde h$。** 若错误地使用 $F_\theta(\widetilde h,e)-F_{\theta_0}(\widetilde h,e)$，初始的两个输入 Jacobian 相消，适配器仍然被阻断；“都初始化非零再做中心化”并不自动解决问题。

保持原概率预算后，主路径相对于同一权重、适配关闭的读出仍满足

$$
\eta(x)=(\bar\alpha+\bar\beta)\varepsilon_fL_Fb(x)^2.
$$

参照读出不依赖适配输入，因此不改变该差值证明，但需要重新检查辅助视图、函数参照的储存与实现一致性。参照权重、额外计算和缓存都必须计入参数/内存/速度公平性，不能仅计可训练参数。

另一个需要独立消融的候选是初始参数中心的正则化 $\lambda\|\theta-\theta_0\|^2/2$，避免随机分支在尚未接收到数据梯度时立即被向零压缩。它是已有的锚定正则化思想，不应当作新算法；也不能与中心化结构同时改变后把所有增益归因于角色路由。

## 6. 已有研究与创新边界

截至本次核对，初始化、上下投影学习率与多头随机化均已有明确前例：

| 已核对原始来源 | 时间 | 与本候选的关系 |
| --- | --- | --- |
| [LoRA+](https://proceedings.mlr.press/v235/hayou24a.html) | ICML 2024 | 分析上下投影同学习率的特征学习限制；分开学习率不是新点 |
| [The Impact of Initialization on LoRA Finetuning Dynamics](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract-Conference.html) | NeurIPS 2024 | 零初始化方向影响动态，恒等输出并不意味着动态等价 |
| [LoRA-GA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/62c4718cc334f6a0a62fb81c4a2095a1-Abstract-Conference.html) | NeurIPS 2024 | 梯度对齐初始化已被研究，不能宣称首次改进初始梯度 |
| [Beyond Zero Initialization](https://proceedings.mlr.press/v267/li25bm.html) | ICML 2025 | 研究上下投影同时非零初始化；原文第 4.3 节比较是否减去初始低秩项 |
| [R-LoRA](https://aclanthology.org/2025.findings-emnlp.35/) | Findings EMNLP 2025，11 月 | 多头随机化及减去初始低秩更新已有研究，不能宣称首次让分支多样化 |

这些工作多数针对语言模型中的线性权重适配；当前实现是冻结 CNN 特征上的非线性有界适配，不能直接搬用其无限宽定理或数字，也不能因为应用对象不同就认为理论新颖。

**有望保留的研究问题**是：在角色概率路由和受控概率修正的双重预算下，怎样同时满足恒等起点、有限扰动证书和可用的分支学习路径；该结构是否在相同参照、参数量、优化器与正则化控制下产生增量目标纠错。原创性尚需继续查新和严格对照，不得写“已证明稳定提高召回”。

## 7. 下一步顺序与可证伪标准

1. 先讨论是否值得写新的网络规格；保持当前正式报告不动，保留 v2 负结果。
2. 若同意，规格必须冻结参照输入、初始化尺度、正则化中心、全部控制和理论命题。不能只有一个“改良版”与旧零初始化网络比较。
3. 首先用合成验收检验：恒等初始概率、有效 Jacobian、错误参照输入的阻断反例、全路径扰动界、参照不可训练和普通适配公平对照。不据此宣称真实识别提高。
4. 真实实验须另行登记与批准；先做训练 fit 范围的机制对照，不能立即再次查看 stop 或读取未见分组来寻找正结果。
5. 结构中心化、正则化中心和角色路由必须可以分别关闭；普通路由对照应获得相同初始化/参照改进。若新 R1 与新 S1 仍无增量差异，否定角色路由收益，不以“比旧 T0 好”替代该结论。
6. 实际有效 Jacobian、分支作用、目标净纠错、两类误入、NLL、校准与计算开销都要报告；收敛资格和停止规则不能在看到结果后放宽。
7. 只有通过理论核验和受控实验的结论才进入正式报告及论文；阶段条件化 C 方向仍作为后续研究，不制造真实流程标签或工艺代价。

## 8. 解释风险核对

已按 11/11 类型核对：

| 类型 | 本分析处理 |
| --- | --- |
| Simpson 反转 | 保留模型、种子、lambda、更新步数及 CV/final 分层；聚合比值不推断类别收益 |
| 生态谬误 | 参数块和轨迹不是独立图像/矿石样本，不能外推到现场颗粒 |
| Berkson/选择偏差 | 仅既有选择后的开发数据；全轨迹保留，但没有来源外代表性 |
| Collider 偏差 | 不以 stop 表现筛选轨迹或建立新的因果回归 |
| 基率忽视 | 目标正确数保留 66 分母；不用参数块统计替代目标召回 |
| 回归到均值 | 本轮没有新前后干预实验，不把后验机制解释成改进效果 |
| 幸存者偏差 | 包含所有锁定状态和截断对照；未丢弃负结果或重复 matched 状态 |
| 多处搜索效应 | 诊断是探索性，无 p 值、无显著性或最优模型声明 |
| 分析路径自由度 | 记录新增后验分析范围，不伪称为预注册主终点 |
| 相关不等于因果 | 代码证明初始阻断；长期小分支/负结果的因果解释仍需干预对照 |
| 反向因果 | 仅初始化的时间顺序和公式可确定；不宣称其解释所有后期动态 |

当前总体判断：`CAUTION`。参数/梯度与公式的对应关系可分析，网络实际增益和新颖性均未完成验证。

## 9. 复核派生表的最小只读配方

以下不是训练脚本。用已发布 CSV 可独立核对主要描述性数字；检查点范数从本地原始 `.pt` 提取，第三方若没有这些权重只能核对已发布的派生范数表及哈希，不能称完整训练复现。

```python
import csv
import statistics
from pathlib import Path

root = Path("outputs/lc_role_gradient_diagnosis_v2")
with (root / "gradient_observations.csv").open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 19050
for step in ("1", "2"):
    down = [r for r in rows if r["step"] == step
            and r["parameter_group"] == "down"]
    assert len(down) == 1548
    assert all(float(r["data_gradient_l2"]) == 0 for r in down)
r1 = [r for r in rows if r["arm"] == "R1" and r["step"] == "5"
      and r["parameter_group"] == "down"]
assert len(r1) == 342
ratios = [float(r["data_over_total"]) for r in r1]
print("R1 step 5", statistics.median(ratios), max(ratios))
with (root / "checkpoint_parameter_norms.csv").open(encoding="utf-8") as f:
    norms = list(csv.DictReader(f))
retained = [float(r["retention"]) for r in norms
            if r["scope"] == "final" and r["arm"] == "R1"
            and r["step"] == "100"
            and r["parameter"].endswith("down.combined")]
assert len(retained) == 9
print("Final R1 down retention", min(retained), max(retained))
```

`diagnosis_manifest.json` 包含完整输入和输出 SHA-256，以及没有执行训练/推理的分析范围。派生 CSV 的压缩归档不包含图像、冻结缓存或权重。
