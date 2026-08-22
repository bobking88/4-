# 后悔值监督门控 HRGV 网络设计规范

## 1. 研究动机

现有 HRGV-Net 使用直接选矿角色专家和矿物种类映射专家，并由一个标量门控融合两条后验。三随机种子测试结果显示：

- 直接角色专家平均准确率为 75.57%；
- 种类映射专家平均准确率为 76.51%；
- 若逐样本选择两个专家中预测正确者，理想选择准确率为 78.63%；
- 在“两专家一对一错”的测试样本上，当前门控选择正确专家的比例仅为 46.50%。

这说明两专家存在可利用的互补性，但现有门控主要通过最终分类损失间接学习，尚未形成可靠的专家选择能力。下一版网络应直接监督门控的路由行为，而不是继续增加通用注意力或更深主干。

## 2. 方法定位与命名

新模型命名为后悔值监督门控层级风险验证网络（Regret-Supervised Gating Hierarchical Risk-Gated Verification Network, RSG-HRGV）。

RSG-HRGV 保留以下 HRGV 组件：

1. EfficientNet-B0 共享视觉主干；
2. 四类直接角色专家；
3. 十七类矿物种类专家及固定种类—角色映射；
4. 目标—含钛干扰和目标—金属光泽干扰验证器；
5. 中性区残差后验校正；
6. 角色感知对比约束。

新增组件只有一个：以两个专家在训练标签上的反事实损失差构造软路由目标，并显式监督可靠性门控。

本设计不宣称首次提出混合专家、专家路由、learning-to-defer 或一般后悔值最小化。创新边界是：针对矿物种类专家与选矿角色专家之间的跨粒度互补关系，构造训练期后悔值监督，并将其与钒钛矿相关矿物的双困难负样本风险验证统一。

## 3. 网络结构

### 3.1 双专家后验

设共享特征为

$$
h=f_\theta(x)\in\mathbb R^{1280}.
$$

直接角色专家和矿物种类专家分别产生

$$
p_d(y\mid x)=\operatorname{softmax}(W_dh),
\qquad
p_s(k\mid x)=\operatorname{softmax}(W_sh).
$$

由固定映射矩阵 $A\in\{0,1\}^{R\times K}$ 得到种类映射角色后验

$$
p_m(y\mid x)=Ap_s(k\mid x).
$$

### 3.2 推理门控

推理门控继续使用图像特征、两个专家的归一化熵和 Jensen-Shannon 分歧：

$$
z_g=[h,\bar H(p_d),\bar H(p_m),D_{JS}(p_d\Vert p_m)],
$$

$$
g(x)=\sigma(\operatorname{MLP}(\operatorname{stopgrad}(z_g))).
$$

门控输入默认停止梯度，使门控路径不能为了降低门控损失而反向改变共享表征或两个专家；最终角色损失仍可通过 $p_d$ 和 $p_m$ 的直接概率路径训练主干和专家。

融合后验为

$$
p_f(y\mid x)=g(x)p_d(y\mid x)+[1-g(x)]p_m(y\mid x).
$$

### 3.3 训练期反事实后悔值目标

对训练样本真实角色 $y$，计算两个专家的逐样本负对数似然：

$$
\ell_d(x,y)=-\log p_d(y\mid x),
\qquad
\ell_m(x,y)=-\log p_m(y\mid x).
$$

构造停止梯度的专家优势差

$$
\Delta(x,y)=\operatorname{stopgrad}[\ell_m(x,y)-\ell_d(x,y)].
$$

当直接角色专家损失更低时，$\Delta>0$。软路由目标定义为

$$
g^*(x,y)=\sigma\left(\frac{\Delta(x,y)}{T_r}\right),
$$

其中 $T_r>0$ 控制软目标逼近硬最优专家的速度。

当两个专家损失接近时，强行指定专家会放大随机波动。为此定义差距权重

$$
w(x,y)=\tanh\left(\frac{|\Delta(x,y)|}{T_w}\right),
$$

其中 $T_w>0$。门控监督损失为

$$
\mathcal L_{RSG}
=-
\frac{\sum_i w_i[g_i^*\log g_i+(1-g_i^*)\log(1-g_i)]}
{\sum_i w_i+\epsilon}.
$$

若一个批次中所有权重均接近零，分母中的 $\epsilon$ 保证数值稳定，损失保持可微。

### 3.4 完整训练目标

$$
\mathcal L_{RSG\text{-}HRGV}
=\mathcal L_{HRGV}+\lambda_g\mathcal L_{RSG},
$$

其中

$$
\mathcal L_{HRGV}
=\mathcal L_{role}
+\lambda_d\mathcal L_{direct}
+\lambda_s\mathcal L_{species}
+\lambda_c\mathcal L_{KL}
+\lambda_v(\mathcal L_{Ti}+\mathcal L_{Met})
+\lambda_{con}\mathcal L_{contrast}.
$$

推理阶段不使用真实标签，也不计算 $g^*$、$w$ 或 $\mathcal L_{RSG}$，因此相对 HRGV 不增加推理分支。

## 4. 可证明性质

### 命题 1：门控路由后悔值上界

设真实角色对应的两个专家概率为

$$
a=p_d(y\mid x),\qquad b=p_m(y\mid x),
$$

且 $a,b\ge\varepsilon>0$。定义硬最优门控

$$
g^o=\mathbb I[a\ge b],
$$

融合真实类概率为 $p_g=ga+(1-g)b$。则

$$
0\le -\log p_g+\log\max(a,b)
\le
\frac{|g-g^o|\,|a-b|}{\varepsilon}.
$$

**证明。** $p_{g^o}=\max(a,b)$，故左侧非负。函数 $-\log t$ 在 $[\varepsilon,1]$ 上为 $1/\varepsilon$-Lipschitz，且

$$
|p_g-p_{g^o}|=|g-g^o|\,|a-b|.
$$

代入即得上界。

该命题说明门控造成的额外损失同时取决于路由偏差和两专家真实类概率差距。仅提高门控硬选择准确率仍不充分，实验还应报告概率差距加权的路由后悔值。

### 命题 2：软最优目标对硬最优门控的逼近界

设 $\Delta=\ell_m-\ell_d$ 且 $\Delta\ne0$。软目标

$$
g^*=\sigma(\Delta/T_r)
$$

满足

$$
|g^*-g^o|\le \exp(-|\Delta|/T_r).
$$

**证明。** 当 $\Delta>0$ 时，$g^o=1$，且

$$
1-g^*=\frac{1}{1+e^{\Delta/T_r}}\le e^{-\Delta/T_r}.
$$

当 $\Delta<0$ 时对称成立。

结合命题 1 可得软目标相对硬最优专家的逐样本后悔值界：

$$
\ell(g^*)-\min(\ell_d,\ell_m)
\le
\frac{e^{-|\Delta|/T_r}|a-b|}{\varepsilon}.
$$

### 命题 3：门控监督梯度隔离

当 $z_g$、$\ell_d$ 和 $\ell_m$ 在构造 $\mathcal L_{RSG}$ 时均停止梯度，门控监督损失满足

$$
\frac{\partial\mathcal L_{RSG}}{\partial\theta}
=
\frac{\partial\mathcal L_{RSG}}{\partial W_d}
=
\frac{\partial\mathcal L_{RSG}}{\partial W_s}=0,
$$

而

$$
\frac{\partial\mathcal L_{RSG}}{\partial\phi_g}
\ne0
$$

一般成立，其中 $\phi_g$ 为门控网络参数。该性质保证路由监督只训练门控，不通过目标构造反向操纵专家，使“专家能力”和“专家选择”可以分别审计。

### 推论：期望路由后悔值

对数据分布取期望，有

$$
\mathbb E[\ell(g)-\min(\ell_d,\ell_m)]
\le
\frac{1}{\varepsilon}
\mathbb E[|g-g^o|\,|a-b|].
$$

实验中的平均概率差距加权门控误差，是该上界右侧的可观测代理量。

## 5. 训练与输出接口

### 5.1 新增参数

- `--lambda-gate-regret`：$\lambda_g$；命令行兼容默认值为 0，RSG 正式实验显式传入候选值 0.10；
- `--gate-regret-temperature`：$T_r$，默认候选值 0.20；
- `--gate-gap-temperature`：$T_w$，默认候选值 0.50；
- `--disable-gate-regret`：即使已给出正的 $\lambda_g$，也强制复现原始 HRGV；
- `--hard-gate-target`：硬最优门控目标消融；
- `--unweighted-gate-regret`：取消差距权重的消融；
- `--detach-gate-features`：启用 RSG 推荐的门控输入梯度隔离；不传该参数时保持原始 HRGV 计算图；
- `--couple-gate-features`：仅供 RSG 消融脚本显式覆盖梯度隔离设置。

RSG 实验参数只作为单随机种子试验起点，不能在测试集上调参。候选组合由验证集 Macro F1、目标召回和两类误入率共同选择。旧命令不显式传入 RSG 参数时，输出必须与原始 HRGV 保持一致。

### 5.2 新增逐图输出

预测表增加：

- `direct_true_probability`；
- `mapped_true_probability`；
- `fused_true_probability`；
- `hard_oracle_gate`；
- `soft_oracle_gate`；
- `gate_gap_weight`；
- `gate_selection_correct`；
- `routing_regret_nll`；
- `weighted_gate_error`。

测试阶段的 `hard_oracle_gate` 和相关诊断只用于离线分析，不参与模型推理。

## 6. 实验矩阵

### 6.1 单随机种子可行性试验

固定随机种子 20260728，保持数据划分、增强、优化器、早停和主损失不变：

所有配置均保持当前正式 HRGV 的验证器特征耦合设置；实验中只改变门控后悔监督及其目标、权重或门控梯度路径，避免把验证器梯度隔离混入 RSG 效应。

1. 原始 HRGV；
2. RSG-HRGV 完整版；
3. RSG-HRGV 硬门控目标；
4. RSG-HRGV 无差距权重；
5. RSG-HRGV 门控特征耦合。

可行性试验只用于选择结构，不作为最终统计结论。

### 6.2 三随机种子正式实验

若完整 RSG-HRGV 相对原始 HRGV 满足以下任一条件，则运行 20260727、20260728 和 20260729 三随机种子：

1. Macro F1 提高至少 0.5 个百分点且两类误入率不同时恶化；
2. 门控选择正确率提高至少 5 个百分点且平均路由后悔值下降至少 10%；
3. 在 Accuracy 基本不变（绝对变化不超过 0.3 个百分点）时，目标召回和至少一种困难负样本误入率形成更优风险取舍。

正式比较继续使用固定测试集和成对簇 Bootstrap。

### 6.3 主要指标

分类指标：

- Accuracy；
- Macro F1；
- 目标召回率；
- 含钛干扰误入目标率；
- 金属光泽干扰误入目标率。

路由指标：

- 两专家分歧率；
- 一对一错样本上的门控选择正确率；
- 平均路由后悔值；
- 概率差距加权门控误差；
- 相对理想专家选择上限的互补信息恢复比例。

## 7. 论文主张门槛

只有三随机种子和成对统计均完成后，才能在论文中使用以下表述：

- “后悔值监督提高了门控对跨粒度专家互补性的利用”；
- “路由后悔值界与经验路由误差变化一致”；
- “RSG-HRGV 在总体性能和困难负样本风险之间形成更优取舍”。

若只提高路由指标但未改善分类或风险指标，应将方法写成机制分析或负结果，不能宣称网络性能提升。若完整 RSG-HRGV 未通过单种子门槛，则保留理论分析和门控失效诊断，但不替换当前 HRGV 主模型。

## 8. 测试要求

实现必须采用测试先行流程，至少覆盖：

1. 软门控目标方向正确；
2. 差距越大，权重越接近 1；
3. 两专家相同时门控损失保持数值稳定；
4. 命题 1 的逐样本不等式在随机概率上成立；
5. 命题 2 的软目标逼近界成立；
6. 门控监督损失只更新门控参数；
7. 关闭新损失时精确复现原始 HRGV 接口；
8. 预测表包含全部路由诊断字段；
9. 单批次前向、反向和烟雾训练通过。

## 9. 报告与论文更新

若正式实验成立，需同步更新：

- HRGV 网络结构图，增加训练期反事实监督虚线支路；
- 技术报告方法、理论命题、实验结果、摘要和结论；
- 论文核心初稿和证据映射；
- GitHub 代码、测试、配置、逐图预测和统计结果。

阶段条件化选矿决策图和 OOD 数据仍作为后续增强，不与本次门控创新混合，以避免第一篇论文主线失焦。
