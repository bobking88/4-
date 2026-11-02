# 附录 C：未验证的间隔缩放角色残差证据网络

初稿日期：2026-10-30；附录整合：2026-11-02。结构与参数沿用已归档 MSB-RRE v1，本文件不是新的网络规格或可执行协议。2026-11-01的一次[独立静态数学与创新性有限复核](reviews/2026-11-01_msb_math_novelty_review.md)未发现 C1/C2 的致命数学错误，但未建立角色网络优势，建议保留附录。用户已采纳该写作取舍：第一篇论文不追加 MSB 实验，不将它纳入主方法或性能摘要。该复核不是外部同行审稿、完整查新或真实效果验证；本轮不读取真实数据、不运行模型。

文献与表述补核：2026-10-31。补核只修订本草稿，未改模型、证明所用预算、实验矩阵或正式报告。

## 1. 研究问题与设计依据

已有研究区分出三个问题：监督目标是否与最终风险一致、输出空间是否允许纠错、视觉证据是否能学会正确修正。RSG 的标签相关软目标 BCE 不必最小化条件融合 NLL，见[理论附录 A.4b](theoretical_appendix.md)。TC 与 ABMP 的开发则说明，即使保护约束正确，固定端点之间也可能没有新增目标命中的容量。IC 的离线审计进一步发现旧预算排除了 108 个锚点错误中的 92 个。这个计数只对应已暴露的 340-fit 开发组，不是原固定测试集性能。

MSB-RRE 因此研究一个有限问题：在既有冻结视觉表示上，能否通过角色相关的非线性证据修正成对判别间隔，并使实际净纠错优于可靠普通修正头？它是后端网络候选，不改造 EfficientNet-B0 的 MBConv，不新增传感器或真实选矿阶段变量。低秩双线性融合已有先例 [1]，受约束的概率后处理也已有研究 [2]；本节不以其中任一构件单独宣称首创。

## 2. 计算图与训练目标

四角色依次为目标代理类 $T$、含钛干扰 $Ti$、脉石 $G$、金属光泽干扰 $M$。输入为冻结视觉特征 $h\in\mathbb R^{64}$、既有专家证据 $e\in\mathbb R^{23}$ 和严格正的四类锚点后验 $\bar p$。$h$ 来自既有 CNN 池化表示及按折 PCA，$e$ 来自既有 E18、四列 log 概率和一列矛盾量；这些是同一图像的表征，不是新的物理多模态数据。PCA、标准化和温度锚点均复用，不在本轮拟合。

对 $j\in\{Ti,G,M\}$，使用三个不共享可训练参数的秩 $r=4$ 分支：

$$
f_{\theta_j}(h,e)=\frac{1}{\sqrt r}a_j^\top
\big[\tanh(U_jh+u_j)\odot\tanh(V_je+v_j)\big]+\zeta_j,
\qquad s_j=f_{\theta_j}(h,e)-f_{\theta_{j,0}}(h,e).
\tag{C1}
$$

$U_j\in\mathbb R^{4\times64}$、$V_j\in\mathbb R^{4\times23}$，其余向量宽度为 4，$\zeta_j$ 为标量。$f_{\theta_{j,0}}$ 是初始化时复制的固定函数，不是对当前函数作停止梯度。初始 $s_j=0$，但当前参数的导数没有因此消失。每分支 361 个参数，共 1083 个可训练参数及 1083 个固定参照标量；该计数不包含冻结 CNN 或缓存。

定义间隔与预算：

$$
m_j=\log\frac{\bar p_T}{\bar p_j},\quad
R_j=\gamma|m_j|+\tau,\quad
d_j=R_j\tanh s_j,\qquad \gamma=2,\ \tau=0.25.
\tag{C2}
$$

这里 $m_j,R_j,d_j$ 均为对数赔率尺度；给定输入时 $R_j$ 固定，不是另一个学习头。统一输出为

$$
z_T=\log\bar p_T,\quad z_j=\log\bar p_j-d_j,\quad q=\operatorname{softmax}(z),
\quad M_j:=\log\frac{q_T}{q_j}=m_j+d_j.
\tag{C3}
$$

三个分支通过一次共同归一化形成四类后验，不能将三个独立二分类概率直接拼接。

```mermaid
flowchart LR
  H0["冻结 CNN 池化表示 H1280"] --> P["已有按折 PCA / 标准化"]
  P --> H["h64"]
  E0["既有 E18 + log p4 + 矛盾量"] --> E["已有按折标准化: e23"]
  H --> B["三个独立角色分支: 式 C1 / rank 4"]
  E --> B
  H --> F0["冻结函数参照 f_theta0"]
  E --> F0
  B --> S["s_j = f_theta - f_theta0"]
  F0 --> S
  S --> C["tanh s_j"]
  A["冻结 log anchor4"] --> R["m_j; R_j = 2 abs m_j + 0.25"]
  R --> D["d_j = R_j tanh s_j"]
  C --> D
  A --> Q["竞争类 logit 减 d_j / 一次 LogSoftmax"]
  D --> Q
  Q --> OUT["四角色 log q"]
  OUT -.-> L["仅训练: NLL + pair + 修正 / 参数正则"]
  Y["真实角色: 仅训练"] -.-> L
```

图 C1 是与公式对应的 Markdown/Mermaid 结构源，不是新生成或本轮已渲染验收的论文位图。`scripts/msb_role_model.py` 的 `RoleBranch._score` 对应 C1，`anchor_margins` 对应 C2，`MSBRoleModel.forward` 与 `decode_role` 对应 C3。图中冻结 CNN 不是本轮重新执行的模块。

对 $I_j=\{i:y_i\in\{T,j\}\}$，令目标标签 $a_{ij}=1$、竞争角色标签 $a_{ij}=-1$。在每个 $I_j$ 非空时，既有损失为

$$
L_{pair}=\frac13\sum_j\frac1{|I_j|}\sum_{i\in I_j}
\operatorname{softplus}(-a_{ij}M_{ij}),
$$
$$
L=-\frac1N\sum_i\log q_{i,y_i}+0.1L_{pair}
+\frac{0.01}{3N}\sum_{i,j}\left(\frac{d_{ij}}{R^{ref}_{ij}}\right)^2
+\frac{0.001}{2}\sum_j\|\theta_j-\theta_{j,0}\|_2^2,
\quad R^{ref}_{ij}=2|m_{ij}|+0.25.
\tag{C4}
$$

成对损失使用主输出的相同间隔，不提供额外真值信息。普通对照和半径消融使用相同 $R^{ref}$ 定义修正惩罚，避免同时改变预算和惩罚尺度。以上系数沿用既有规格，不因本次推导而调参。

## 3. 命题 C1：跨界条件与统一内部控制

**条件。** 固定输入，$m\in\mathbb R$、$R>0$，分数 $s$ 可取任意有限实数。以下是输出坐标的可达性分析，不预设共享网络可以同时实现所有样本的分数。

对 $M=m+R\tanh s$，其值域为开区间 $(m-R,m+R)$。因此，有限分数能够使 $M$ 分别取严格正值和严格负值，当且仅当

$$
R>|m|.
\tag{C5}
$$

在 $R=\gamma|m|+\tau$、$\gamma\ge0$、$\tau>0$ 下，该条件对任意有限 $m$ 成立，当且仅当 $\gamma\ge1$。若 $0\le\gamma<1$，取 $|m|\ge\tau/(1-\gamma)$ 即得到不能严格跨界的反例。

更强地，要求存在一个与 $m$ 无关的有限控制上界 $S$，对任意 $m$ 都能在 $|s|\le S$ 内取得两种符号，则其充要条件是 $\gamma>1$。证明如下。

当 $\gamma>1$，取 $c=\tanh s=\pm1/\gamma$，则

$$
m+R/\gamma=m+|m|+\tau/\gamma\ge\tau/\gamma,
\qquad m-R/\gamma\le-\tau/\gamma.
\tag{C6}
$$

故 $S=\operatorname{atanh}(1/\gamma)$ 足够。反之，任意有限 $S$ 对应 $\kappa=\tanh S<1$。若 $\gamma\le1$，则 $\kappa\gamma<1$；当 $u>\kappa\tau/(1-\kappa\gamma)$ 时，$u-\kappa(\gamma u+\tau)>0$，以 $m=u$ 为例就无法在给定控制范围内取得负值。证毕。

特别地，$\gamma=1$ 仍允许每个有限输入跨界，但边界分数
$s^*=\operatorname{atanh}[-m/(|m|+\tau)]$ 的绝对值随 $|m|$ 趋于无穷，因而没有统一有限控制上界。$\tau=0,m=0$ 时严格正间隔见证失效，故不能省略正偏置条件。

**四类推论。** 当 $\gamma>1$ 时，令三个 $c_j=1/\gamma$，目标类同时以至少 $\tau/\gamma$ 的间隔胜出。若指定非目标类 $j$，令 $c_j=-1/\gamma$、其余 $c_k=1/\gamma$，则

$$
\log(q_j/q_T)=-m_j+|m_j|+\tau/\gamma\ge\tau/\gamma,
$$
$$
\log(q_j/q_k)=m_k-m_j+|m_j|+|m_k|+2\tau/\gamma
\ge2\tau/\gamma.
\tag{C7}
$$

这说明逐图输出空间没有预先排除任何类别。三个常数 bias 就能实现“将所有图像推向同一指定类”的见证，因此该推论不能证明视觉交互有用，也不能证明不同标签可被同一有限网络同时正确拟合。推理时不使用真值选择符号。

## 4. 命题 C2：边界控制导数的紧下界

对实数模型、固定输入、$\gamma\ge1,\tau>0$，在 $M=0$ 处有 $\tanh s^*=-m/R$。令 $u=|m|$，则

$$
\left.\frac{\partial M}{\partial s}\right|_{M=0}
=R-\frac{m^2}{R}
=\tau+\frac{(\gamma^2-1)u^2+\gamma\tau u}{\gamma u+\tau}
\ge\tau.
\tag{C8}
$$

证明由代入 $R=\gamma u+\tau$ 并通分直接得到。分子两项非负，且 $u=0$ 时取等号，所以 $\tau$ 是对全部有限 $m$ 的紧下界。当前 $\gamma=2,\tau=0.25$ 下，C8 将原规格的较松下界 $\tau(1-\gamma^{-2})=0.1875$ 收紧为 $0.25$；未改变网络或预算。

因 $\partial s_j/\partial\zeta_j=1$，对单个成对样本的未缩放 logistic 损失有

$$
\left|\frac{\partial\operatorname{softplus}(-aM)}{\partial\zeta_j}\right|_{M=0}
=\frac12\left(R-\frac{m^2}{R}\right)\ge\tau/2.
\tag{C9}
$$

**适用边界。** C8/C9 不是全部参数的梯度下界。批次中异号梯度、其他损失和正则项可以抵消；成对相等也可能发生在第三类占优的区域，不一定对应四类获胜边界。浮点饱和、下溢或非法概率不由实数证明排除。该命题不保证优化速度、收敛、召回或来源外泛化。

下界还依赖所选分数坐标。若改用 $t=\kappa s$、$\kappa>0$，则 $\partial M/\partial t=(\partial M/\partial s)/\kappa$，相应边界下界为 $\tau/\kappa$。因此不能仅以导数数值更大宣称优化更快；这里的 $\kappa$ 是坐标缩放系数，与 C9 的标签 $a\in\{-1,1\}$ 不同。C9 只对应单样本、未缩放 logistic 项，实际损失权重和分组平均还会缩放该项梯度。

## 5. 可解释功能量与不可混淆的创新主张

给定锚点后验，修正量由输出唯一确定：

$$
d_j=\log(q_T/q_j)-\log(\bar p_T/\bar p_j).
\tag{C10}
$$

独立分支意味着固定输入时 $\partial M_k/\partial\theta_j=0$（$k\ne j$），但共同归一化仍会改变全部绝对概率。C10 只识别输出功能量，不识别唯一参数、因果机制或真实矿物属性；普通残差头也具有该恒等式，因此不列为第三项独占原创定理。

**固定目标 logit 不是额外的表达能力。** 对任意普通残差头 $q=\operatorname{softmax}(\log\bar p+\delta)$，从全部 logits 减去同一个标量 $\delta_T$ 后概率不变。令 $d_j=\delta_T-\delta_j$，就得到 C3 的形式。因此，目标 logit 保持不变只是选择相对坐标，不能单独解释为新结构优势。MSB 与一般残差头的区别只能来自分支参数约束、可用输入和 $|d_j|<R_j$ 的范围约束。反过来，给定 C3 的 $d$，取 $\delta_T=0,\delta_j=-d_j$ 即可构造同输出的普通残差头；这不表示任意给定宽度和训练程序都能学到该函数。

共享双线性对照可通过分块参数嵌入完整表示主模型。故本文不能主张角色分支比该对照表达能力更强。待验证问题仅是相同输入与公平训练条件下，角色分支的结构约束是否带来额外净纠错。相对于线性/可分加性头的混合导数差异，也不能替代与普通非线性网络的比较。

扩大半径同时扩大有益和有害修正的可达范围。C6/C7 的同一见证既能纠正错误，也能把原本正确的图像推向错误类别，不能据此宣称“保护一切高置信正确预测且纠正任意高置信错误”。

### 5.1 原始文献对照与可保留主张

低秩交互本身不是新算子。Kim 等 [1] 第 3.2 节式 (5) 已使用两路投影后的非线性激活、Hadamard 乘积和线性读出，并明确允许 tanh；第 3.3 节还讨论了 shortcut。C1 采用这类既有交互构件，角色拆分和固定函数参照并不能自动赋予它首创性。

输出后处理也不能笼统作为新意。Dirichlet calibration [3] 将 log 概率经线性层和 softmax 映射为新后验。它不等同于本节输入依赖的受限残差，但说明“在 log 概率上加学习层”并非新的问题定义。

Atarashi 等 [2] 的 v2 第 3.3 节已经讨论输入相关的概率边界，第 4.1 节讨论 logit bounding。因此不能将相关工作概括为“只使用固定边界”。本文的 $R_j=2|m_j|+0.25$ 是相对目标赔率的确定性修正半径，并非该文的概率区间投影；这里的差别尚不足以证明整体方法原创或更优。

可保留的候选贡献是：以角色分离的视觉证据驱动相对判别间隔修正，用间隔缩放预算消除旧预算的部分不可达区域，并给出跨界与边界导数的明确条件。C1/C2 是该设计的性质分析，不称为新的通用学习理论；它们的证明不依赖矿物领域，领域价值仍需由角色结构相对公平普通修正头的收益支持。未见完全相同公式不能替代查新，证明正确也不能替代效果证据。

## 6. 证据与附录定位

| 主张 | 当前支持 | 尚缺证据 |
|---|---|---|
| 固定凸路由存在分类容量限制 | 已有解析论证及 ABMP 开发记录 | 不外推为所有融合网络的限制 |
| MSB 允许统一有限控制跨界 | C1 的实数证明、既有结构实现 | 不说明训练能实现有益的样本相关分数 |
| 边界导数下界可收紧至 $\tau$ | C2 解析证明及2026-11-01有限静态复核 | 不是重参数化不变的优化保证，未验证学习速度或效果 |
| 角色视觉证据优于普通修正 | 尚无真实 MSB 效果证据 | 公平强对照下的净纠错、误入与总体性能 |
| 泛化与应用价值 | 既有 RSG 结果另按其协议报告 | MSB 独立确认及真实矿石验证均不能预设已完成 |

本次整合不生成或修改运行协议。既有11单元、最多75次轻量拟合及7200秒上限是历史候选规格，不是第一篇论文继续执行的清单；本轮不运行它，也不撤改历史协议。仅有 NLL 微动、非零梯度或胜过旧受限预算不足以证明角色机制。若以后另行研究，仍须重新明确授权和未暴露确认数据，不能将既有340-fit或已暴露stop改称独立测试。

本附录保留的是“某种参数化允许何种输出变化”的分析，不是“该网络已经学会正确变化”的结论。三个分支的独立参数、固定目标logit及边界导数下界均不自动构成新的通用学习理论。将候选放入附录也不表示它必然无效，而是依据当前证据停止其在第一篇论文中的扩张。

## 来源与文献边界

- [MSB 原规格](../superpowers/specs/2026-10-14_margin_scaled_bilinear_role_evidence_design.md)：本节继承结构与系数，新增的是 C1 条件分类及 C2 收紧证明；原规格不在本轮改写。
- [IC 离线理论审计](../experiment_records/2026-10-14_ic_role_offline_theory_audit.md)：旧输出预算的能力限制，不能当作 MSB 的效果。
- [ABMP 开发与理论诊断](../experiment_records/2026-10-01_abmp_rsg_v2_development.md)：固定凸路由容量证据。
- [1] Kim, J.-H., et al. *Hadamard Product for Low-rank Bilinear Pooling*. ICLR 2017；arXiv 首发 2016-10-14，v4 修订 2017-03-26。[原文第 3.1--3.3 节，式 (4)--(7)](https://arxiv.org/pdf/1610.04325)。2026-10-31 已核对上述方法段；不据此声称与本网络整体等价。
- [2] Atarashi, K., Oyama, S., Arai, H., & Kashima, H. *Probability Bounding: Post-Hoc Calibration via Box-Constrained Softmax*. arXiv:2506.10572v2；首发 2025-06-12，v2 2026-02-23。[元数据](https://arxiv.org/abs/2506.10572)、[v2 第 3.3、4.1 节](https://arxiv.org/html/2506.10572v2)。2026-10-31 从前一轮摘要核对推进到上述方法段核对，未完成该文全部定理及实验的复核；版本日期以 arXiv 提交记录为准。
- [3] Kull, M., et al. *Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration*. NeurIPS 2019。[会议原始条目与摘要](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html)。2026-10-31 核对其中 log 概率、线性层和 softmax 的结构描述，不引用未检查的性能数值。

本节不使用“首次”“一般风险一致性”“全局收敛”或“已验证识别提升”。明确的公式、成立条件和反例提升论证质量，但其文献原创性与实际角色学习效果仍是不同的待证问题。
