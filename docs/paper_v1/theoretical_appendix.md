# 附录：RSG-HRGV 门控后悔理论证明

本附录只证明 RSG-HRGV-Net 已实现计算图中的三个结论。它们说明后悔监督优化的对象、连续软目标的合理性及局部梯度隔离的范围；不蕴含网络在任何数据集上必然提高 Accuracy 或 Macro F1。

## A.1 记号与假设

对带真实角色标签 $y$ 的图像 $x$，令两位专家对真实角色的后验分别为

$$
a=p_d(y\mid x),\qquad b=p_m(y\mid x),\qquad a,b\in[\varepsilon,1],\quad \varepsilon>0.
$$

门控 $g\in[0,1]$ 对应直接角色专家的权重，融合真实类概率为

$$
p_g=ga+(1-g)b.
$$

令 $M=\max(a,b)$，并取硬最优门控 $g_o=\mathbb I(a\ge b)$。训练期的专家损失差、软目标和差距权重为

$$
\Delta=-\log b+\log a=\log(a/b),\qquad
g^*=\sigma(\Delta/T_r),\qquad
w=\tanh(|\Delta|/T_w),
$$

其中 $T_r,T_w>0$。

## A.2 引理 A.0：凸融合路由后悔的精确分解

令

$$
d=|a-b|,\qquad \delta=|g-g_o|.
$$

**引理。** 在 A.1 的假设下，融合真值类概率和相对更优专家的路由后悔满足

$$
p_g=M-\delta d,
$$

$$
r(g;a,b)=-\log p_g+\log M
=-\log\left(1-\frac{\delta d}{M}\right).
$$

**证明。** 当 \(a\ge b\) 时，\(M=a\)、\(g_o=1\)，从而

$$
p_g=ga+(1-g)b=a-(1-g)(a-b)=M-\delta d.
$$

当 \(a<b\) 时，\(M=b\)、\(g_o=0\)，同理有

$$
p_g=ga+(1-g)b=b-g(b-a)=M-\delta d.
$$

将该等式代入 \(r=-\log p_g+\log M\) 即得。证毕。

**含义。** 该式不是仅给出松弛界，而是表明在两位专家、凸融合和真值类概率比较的条件下，路由后悔精确由门控偏离 \(\delta\) 与专家差距 \(d\) 的乘积决定。对固定 \(M\) 和可行的 \(\delta d<M\)，有

$$
\frac{\partial r}{\partial\delta}=\frac{d}{M-\delta d}\ge0,
\qquad
\frac{\partial r}{\partial d}=\frac{\delta}{M-\delta d}\ge0.
$$

因此，门控偏离或专家差距任一增大都会提高该定义下的路由后悔。这为后续局部上界分层诊断提供了直接机制解释，但不代表门控误差能单独解释总体类别错误。

## A.3 定理 A.1：门控误差的路由后悔上界

**定理。** 在上述假设下，

$$
0\le -\log p_g+\log M
\le \frac{|g-g_o|\,|a-b|}{\varepsilon}.
$$

**证明。** 由引理 A.0，$M-p_g=|g-g_o|\,|a-b|$，且 $p_g\le M$，左侧非负。又因 $p_g,M\in[\varepsilon,1]$，函数 $f(t)=-\log t$ 在该区间满足 $|f'(t)|=1/t\le1/\varepsilon$。由均值定理，

$$
-\log p_g+\log M=f(p_g)-f(M)
\le \frac{M-p_g}{\varepsilon}
=\frac{|g-g_o|\,|a-b|}{\varepsilon}.
$$

证毕。

**含义。** 当专家对真实类的证据几乎相同时，门控即使偏离硬选择，也不会产生较大路由后悔；当差距较大时，错误选择代价上升。因此 $w$ 不是事后调参，而是对上界中风险放大项的有界代理。

## A.4 定理 A.2：软目标的指数逼近

**定理。** 当 $\Delta\ne0$ 时，

$$
|g^*-g_o|\le \exp(-|\Delta|/T_r).
$$

**证明。** 当 $\Delta>0$，$g_o=1$，于是

$$
|g^*-g_o|=1-\sigma(\Delta/T_r)
=\frac{1}{1+e^{\Delta/T_r}}
\le e^{-\Delta/T_r}.
$$

当 $\Delta<0$，$g_o=0$，于是

$$
|g^*-g_o|=\sigma(\Delta/T_r)
=\frac{1}{1+e^{|\Delta|/T_r}}
\le e^{-|\Delta|/T_r}.
$$

证毕。

**含义。** 软目标在专家优劣明确时接近硬选择，在专家接近时保留不确定性。它避免把近似等价的两条证据强行编码成不连续的 0/1 监督。

## A.5 定理 A.3：后悔分支的局部梯度隔离

令 $z=[h,\bar H(p_d),\bar H(p_m),D_{JS}(p_d\Vert p_m)]$，门控实现为

$$
g=\sigma\{f_\phi(\operatorname{stopgrad}(z))\}.
$$

软目标 $g^*$ 和权重 $w$ 同样由 $\operatorname{stopgrad}(p_d,p_m)$ 构造，后悔损失为

$$
\mathcal L_{reg}=w\,\operatorname{BCE}(g,g^*).
$$

**定理。** 对产生 $h,p_d,p_m$ 的任意专家或共享主干参数 $\theta_e$，

$$
\frac{\partial\mathcal L_{reg}}{\partial\theta_e}=0,
$$

而通常 $\partial\mathcal L_{reg}/\partial\phi\ne0$。

**证明。** 在后悔分支中，$\partial\operatorname{stopgrad}(z)/\partial\theta_e=0$，并且 $\partial g^*/\partial\theta_e=\partial w/\partial\theta_e=0$。对链式法则展开可得

$$
\frac{\partial\mathcal L_{reg}}{\partial\theta_e}
=\frac{\partial\mathcal L_{reg}}{\partial g}
\frac{\partial g}{\partial\operatorname{stopgrad}(z)}
\frac{\partial\operatorname{stopgrad}(z)}{\partial\theta_e}
+\frac{\partial\mathcal L_{reg}}{\partial g^*}
\frac{\partial g^*}{\partial\theta_e}
+\frac{\partial\mathcal L_{reg}}{\partial w}
\frac{\partial w}{\partial\theta_e}=0.
$$

若 $0<g<1$、$w>0$ 且 $g\ne g^*$，二元交叉熵对 $g$ 的导数非零，门控参数 $\phi$ 仍接受梯度。证毕。

**边界。** 上式只针对 $\mathcal L_{reg}$ 分支。角色、种类、一致性、验证器和对比损失仍按主模型的耦合计算图更新共享特征，因此不能把该结论误读为“RSG 不训练主干”。

## A.6 推论 A.1：逐图局部路由后悔上界

对第 $i$ 个样本，取

$$
\varepsilon_i=\min\{a_i,b_i\},\qquad
r_i=-\log\{g_i a_i+(1-g_i)b_i\}+\log\max(a_i,b_i).
$$

**推论。** 在 $a_i,b_i>0$ 且 $g_i\in[0,1]$ 时，

$$
r_i\leq B_i^{\mathrm{loc}}
=\frac{|g_i-g_{o,i}|\,|a_i-b_i|}{\varepsilon_i}.
$$

**证明。** 将定理 A.1 中的统一下界替换为样本自身的 \(\varepsilon_i\)。由于 \(a_i,b_i\geq\varepsilon_i\)，定理 A.1 的全部前提仍成立，故结论直接得到。证毕。

**作用与边界。** 该式是定理 A.1 的逐图收紧，不是新的训练目标、网络模块或泛化定理。它仅用于按理论路由难度对冻结模型输出分层，以避免使用全体样本的极小概率下界而得到过松的诊断数值。

## A.7 高精度重放的数值一致性

为检验理论量与实现是否一致，使用保存的最佳检查点对固定测试、摄影者留出和 ResNet50 主干替换三种协议进行只读高精度重放。9 次重放共覆盖 10,242 张图像，其中 ResNet50 子集覆盖 3,852 张图像。使用 float32 概率下界 \(1.19\times10^{-7}\) 和数值容差 \(2\times10^{-6}\) 时，引理 A.0 的融合/后悔精确分解最大绝对残差为 \(8.44\times10^{-7}\)，定理 A.1 的局部上界最大残差为 \(9.41\times10^{-8}\)，定理 A.2 的最大残差为 \(5.94\times10^{-8}\)，三者违反计数均为 0。精确分解残差来自保存的 float32 概率与后悔值的有限精度导出，故报告为容差内一致而非数学意义上的零误差。

按 \(B_i^{\mathrm{loc}}\) 分为三个等量层后，固定测试、摄影者留出和 ResNet50 中的平均路由后悔分别由最低层的 0.00038、0.00141、0.00013 上升至最高层的 0.22702、0.22066、0.24624。按 \(|\log a_i-\log b_i|\) 分层时，软目标--硬最优门控平均偏差分别由 0.49895、0.49622、0.49967 下降至 0.13979、0.13381、0.14212。上述分层现象与定理方向一致，但它们均来自同一批冻结检查点的描述性诊断，不能作为新的分类显著性检验或外部泛化结论。

## A.8 可证伪实验对应

三个理论结论分别对应可观测量：

| 理论结论 | 可观测量 | 判据 |
|---|---|---|
| 引理 A.0 | \(\delta\)、\(d\) 与逐图路由后悔 | 在冻结输出上验证精确分解残差；并以局部上界分层诊断其单调方向 |
| 定理 A.1 | 平均路由后悔 | 完整 RSG 相对 HRGV 的成对差异与 Bootstrap 区间 |
| 定理 A.2 | 一对一错路由正确率；软/硬目标消融 | 软目标不应依赖硬选择的任意并列处理 |
| 定理 A.3 | 取消局部隔离的消融 | 主任务指标与两类误入率的风险取舍应被显式报告 |
| 推论 A.1 | 按 \(B_i^{\mathrm{loc}}\) 分层的平均路由后悔 | 局部上界较高层应呈现更高的已观测路由难度；只作描述性诊断 |
| 引理 A.0、定理 A.1--A.2 的实现检查 | 高精度重放残差和违反数 | 在预设数值容差下逐图检查精确分解、局部上界和软目标界，不作为性能比较 |

固定划分中，RSG 的平均路由后悔差异为 −1.77 个百分点，95%区间为 [−2.86, −0.69]；摄影者留出确认中为 −3.53 个百分点，95%区间为 [−4.75, −2.17]；严格匹配的 ResNet50 主干替换中为 −3.33 个百分点，95%区间为 [−4.86, −1.98]。这些结果支持 RSG 在当前两专家概率契约下降低已定义风险，并不将数学结论扩大为任意网络的通用分类精度定理。
