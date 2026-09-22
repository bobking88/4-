# 样本外风险监督路由：理论定位与网络改进边界

## 1. 候选方法

在 RSG-HRGV 的共享主干、直接角色专家、矿物种类映射专家和残差验证器之外，引入按重复组隔离的三段训练数据：专家拟合集 (D_E)、专家停止集 (D_S) 与门控监督集 (D_G)。先由 (D_E,D_S) 得到固定专家参数

\[
\hat\theta=A(D_E,D_S),
\]

再在专家未拟合、未用于专家选择的 (D_G) 上训练门控参数 \(\phi\)。推理计算图不增加主干或分类头，变化发生在训练图：门控只接收固定专家在样本外图像上的共享特征与后验统计。该候选方法暂称 **OOS-RSG-HRGV**（out-of-sample risk-supervised HRGV）。样本切分或交叉拟合本身是标准方法，不能作为一般性首创；本研究可写的贡献是把它与矿物种类—选矿角色双专家、验证器后的最终风险和路由后悔监督结合，并明确审计其适用条件。

设直接角色后验为 \(p_d\)，矿物种类映射后验为 \(p_m=Mp_s\)，门控输入为 \(u_{\hat\theta}(x)\)，门控为

\[
g_\phi(x)=\sigma\!\left(r_\phi(u_{\hat\theta}(x))\right),
\]

验证前融合为

\[
m_{\hat\theta,\phi}(x)=g_\phi(x)p_d(x)+[1-g_\phi(x)]p_m(x).
\]

令固定残差验证映射为 \(V_{\hat\theta}\)，最终后验和风险为

\[
q_{\hat\theta,\phi}(x)=V_{\hat\theta}\!\left(m_{\hat\theta,\phi}(x),x\right),
\qquad
R(\phi\mid\hat\theta)=
\mathbb E_{(X,Y)\sim P}\left[-\log q_{\hat\theta,\phi,Y}(X)\right].
\]

样本外门控的经验目标为

\[
\widehat R_G(\phi\mid\hat\theta)=
-\frac{1}{|D_G|}\sum_{(x_i,y_i)\in D_G}
\log q_{\hat\theta,\phi,y_i}(x_i).
\]

可选的后悔正则仍使用两专家真值类损失差构造软目标：

\[
t_i^*=\sigma\!\left(\frac{\ell_{m,i}-\ell_{d,i}}{T_r}\right),
\qquad
w_i=\tanh\!\left(\frac{|\ell_{m,i}-\ell_{d,i}|}{T_w}\right),
\]

\[
\mathcal L_{\mathrm{OOS}}=
\widehat R_G(\phi\mid\hat\theta)
+\lambda_g\frac{\sum_i w_i\operatorname{BCE}(g_\phi(x_i),t_i^*)}{\sum_i w_i}.
\]

## 2. 命题 1：固定专家下的条件无偏性

若 (D_G) 由与 (D_E,D_S) 独立的重复组抽样得到，且 (\phi) 在取期望时固定，则

\[
\mathbb E\!\left[\widehat R_G(\phi\mid\hat\theta)\mid D_E,D_S\right]
=R(\phi\mid\hat\theta).
\]

若损失可微且梯度可由可积函数支配，则同样有

\[
\mathbb E\!\left[\nabla_\phi\widehat R_G(\phi\mid\hat\theta)\mid D_E,D_S\right]
=\nabla_\phi R(\phi\mid\hat\theta).
\]

证明直接来自条件于 (D_E,D_S) 后 (\hat\theta) 固定，以及 (D_G) 中各重复组服从同一目标分布，随后使用期望线性性；梯度结论再交换梯度与期望。该命题只针对固定 (\phi) 的风险/梯度估计。由同一个 (D_G) 优化得到的 (\hat\phi(D_G)) 不能再用同一数据给出无偏泛化风险；本实验另用 expert_stop 选门控轮次，但该集合已用于选择专家，仍不是完全独立的最终评价集。

## 3. 命题 2：专家共同插值导致门控不可识别

对某个样本，若两个专家给出完全相同的后验

\[
p_d(\cdot\mid x)=p_m(\cdot\mid x),
\]

则对任意 (g\in[0,1])，有

\[
m_g(x)=g p_d(x)+(1-g)p_m(x)=p_d(x).
\]

因此任何只通过 (m_g) 并使用固定验证证据的最终映射均与 (g) 无关：

\[
\frac{\partial}{\partial z}
\left[-\log q_y(x)\right]=0,
\qquad g=\sigma(z).
\]

同时两专家真值类损失差为 0，故 RSG 的差距权重 (w=\tanh(0)=0)。此时训练样本既不给最终风险门控提供梯度，也不给后悔分支提供有效监督，门控在这些样本上不可识别。完全相等是充分而非必要条件；接近共同插值时，梯度和差距权重也可能很小。

数值核验使用 float64 和实际残差验证函数：相同专家后验下最终 NLL 对门控 logit 的梯度精确为 0，差距权重为 0；将映射专家改为不同后验后，两者分别为 -0.308964 和 0.986760。核验脚本及机器可读结果为 `scripts/verify_oos_gate_degeneracy.py` 与 `outputs/theory/oos_gate_degeneracy.json`。该数值例只验证代数实现，不构成真实数据上的性能证据。

该命题解释了为何专家训练内图像可能不适合作为门控监督，却不保证样本外训练一定提高分类性能。样本外图像只有在两个专家呈现具有标签相关性的差异时才增加可路由信息。

## 4. 当前实验对应

单专家受控诊断中，seen/unseen 各 892 张且逐矿物—角色计数相同。仅优化验证器后最终 NLL 时，unseen 相对 seen 在三个门控初始化中均降低停止集 NLL；平均差为 −0.012139 nat，Macro F1 和目标召回平均分别增加 0.004771 与 0.013487。seen 门控自身训练批损失均值为 0.017744，而 unseen 为 0.937503，符合专家训练内样本风险过度乐观的预期。

但 unseen 门控没有同时优于等权融合的 NLL、目标召回及两类误入率；加入后悔 BCE 后，unseen−seen 的 NLL 均值为 +0.000153 nat，方向不一致。三个初始化还共享同一专家和同一停止集。因此当前证据支持“监督来源影响门控学习”的机制判断，不支持“OOS-RSG-HRGV 已稳定提高最终分类性能”。

## 5. 对第一篇论文的使用建议

当前技术报告可以加入命题 1、命题 2、三段训练图和探索结果，作为 RSG-HRGV 的训练机制改进。第一篇论文是否把 OOS-RSG-HRGV 升为主方法，必须以多个从头训练的专家种子和不参与模型选择的最终评价完成确认。若确认失败，应保留为机制性负结果或训练建议，而不削弱已完成的 RSG-HRGV 主线。

稳妥贡献表述是：提出并审计一种面向跨粒度矿物专家的样本外最终风险路由训练方案，给出固定专家下风险/梯度条件无偏性及共同插值时门控不可识别的充分条件。不能表述为首次提出样本切分、交叉拟合、混合专家或条件无偏估计。
