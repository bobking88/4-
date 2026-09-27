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

随后按照预先冻结的五段式协议重新训练三个独立专家。原训练集按图像组分为 `expert_fit` 3,294 张、`expert_stop` 599 张、`gate_unseen` 597 张、`gate_stop` 597 张和 `final_eval` 897 张；`gate_seen` 从 `expert_fit` 中按 17 个矿种-角色分层精确匹配 597 张。所有专家和门控选择均在读取 `final_eval` 前完成。

在三个独立专家上，unseen 门控相对 seen 门控的最终 NLL 平均下降 0.035656 nat，95% 两阶段成对簇 Bootstrap 区间为 [0.023217, 0.050196] nat；相对等权融合平均下降 0.014040 nat，95% 区间为 [0.008724, 0.019431] nat，且三个种子方向一致。含钛与金属光泽困难负样本误入目标相对等权平均分别下降 1.170 和 1.600 个百分点。

但是，目标召回相对等权平均下降 2.476 个百分点，95% 区间为 [0.571, 4.381] 个百分点，超过预注册的 1 个百分点伤害上限。Macro F1 平均变化为 -0.047 个百分点，区间跨 0。故独立确认支持“样本外监督改善最终概率风险并重分配错误类型”，不支持“当前 OOS-RSG 同时改善总体分类和目标保护”。预注册晋级判定失败，当前方法不替代 RSG-HRGV 主线。

## 5. 对第一篇论文的使用建议

当前技术报告可以加入命题 1、命题 2、五段式训练图和独立确认结果，作为 RSG-HRGV 的训练机制审计。确认实验已经完成，但因目标召回护栏失败，OOS-RSG-HRGV 不升为第一篇论文主方法；它应作为受控机制结果和下一代约束路由器的经验依据。

稳妥贡献表述是：提出并审计一种面向跨粒度矿物专家的样本外最终风险路由训练方案，给出固定专家下风险/梯度条件无偏性及共同插值时门控不可识别的充分条件。不能表述为首次提出样本切分、交叉拟合、混合专家或条件无偏估计。

## 6. 目标召回约束缺口与下一理论假设

独立确认揭示了一个稳定的 Pareto 冲突：无约束最终 NLL 使门控更保守地拒绝困难负样本，同时也降低目标矿物召回。下一候选方法不是继续调整普通损失权重，而是在样本外监督中显式加入目标保护约束：

\[
\min_{\phi} R_{\mathrm{NLL}}(\phi)
\quad\mathrm{s.t.}\quad
R_{T,\mathrm{miss}}(\phi)
\le R_{T,\mathrm{miss}}(g\equiv 1/2)+\epsilon.
\]

其拉格朗日形式为

\[
\mathcal L(\phi,\lambda)=R_{\mathrm{NLL}}(\phi)
+\lambda\left[R_{T,\mathrm{miss}}(\phi)-\tau\right],
\qquad \lambda\ge 0.
\]

这里的目标漏识风险必须在独立于专家拟合、门控拟合和门控选择的数据上估计。若使用可微软漏识代理，还需给出它与硬目标召回之间的关系和有限样本安全裕量；仅在训练集满足软约束不能宣称测试召回有保证。

一个可审计的安全回退是把等权最终后验 $q_{1/2}$ 与无约束 OOS 后验 $q_{\phi}$ 做最终概率插值：

\[
q_{\rho}=(1-\rho)q_{1/2}+\rho q_{\phi},
\qquad 0\le\rho\le1.
\]

对目标软漏识风险 $S_T(q)=\mathbb E[1-q_T(X)\mid Y=T]$，有精确恒等式

\[
S_T(q_\rho)=S_T(q_{1/2})
+\rho\left[S_T(q_\phi)-S_T(q_{1/2})\right].
\]

同时由 $-\log$ 的凸性，最终 NLL 满足

\[
R_{\mathrm{NLL}}(q_\rho)
\le (1-\rho)R_{\mathrm{NLL}}(q_{1/2})
+\rho R_{\mathrm{NLL}}(q_\phi).
\]

因此 $\rho=0$ 始终保留等权可行回退；若 $S_T(q_\phi)>S_T(q_{1/2})$，软风险约束给出

\[
\rho\le
\min\!\left\{1,
\frac{\epsilon}{S_T(q_\phi)-S_T(q_{1/2})}
\right\}.
\]

这些等式给出了“风险改善与目标保护”之间可计算的连续路径，但它们只保证软概率风险，不直接保证离散 argmax 目标召回。该候选网络暂记为 **TC-OOS-RSG**，其训练、选择与新的外层确认协议尚待实现；由于现有 `final_eval` 已读取，不能再把它用于该新假设的确认性验证。
