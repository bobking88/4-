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

## A.2 定理 A.1：门控误差的路由后悔上界

**定理。** 在上述假设下，

$$
0\le -\log p_g+\log M
\le \frac{|g-g_o|\,|a-b|}{\varepsilon}.
$$

**证明。** $p_g$ 是 $a$ 与 $b$ 的凸组合，因此 $p_g\le M$，左侧非负。若 $a\ge b$，则 $g_o=1$，且

$$
M-p_g=a-[ga+(1-g)b]=(1-g)(a-b)=|g-g_o|\,|a-b|.
$$

若 $a<b$，则 $g_o=0$，并且

$$
M-p_g=b-[ga+(1-g)b]=g(b-a)=|g-g_o|\,|a-b|.
$$

两种情形均成立。又因 $p_g,M\in[\varepsilon,1]$，函数 $f(t)=-\log t$ 在该区间满足 $|f'(t)|=1/t\le1/\varepsilon$。由均值定理，

$$
-\log p_g+\log M=f(p_g)-f(M)
\le \frac{M-p_g}{\varepsilon}
=\frac{|g-g_o|\,|a-b|}{\varepsilon}.
$$

证毕。

**含义。** 当专家对真实类的证据几乎相同时，门控即使偏离硬选择，也不会产生较大路由后悔；当差距较大时，错误选择代价上升。因此 $w$ 不是事后调参，而是对上界中风险放大项的有界代理。

## A.3 定理 A.2：软目标的指数逼近

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

## A.4 定理 A.3：后悔分支的局部梯度隔离

令 $z=[h,\bar H(p_d),\bar H(p_m),D_{JS}(p_d\Vert p_m)]$，门控实现为

$$
g=\sigma\{f_\phi(\operatorname{stopgrad}(z))\}.
$$

软目标 $g^*$ 和权重 $w$ 同样由 $operatorname{stopgrad}(p_d,p_m)$ 构造，后悔损失为

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

## A.5 可证伪实验对应

三个理论结论分别对应可观测量：

| 理论结论 | 可观测量 | 判据 |
|---|---|---|
| 定理 A.1 | 平均路由后悔 | 完整 RSG 相对 HRGV 的成对差异与 Bootstrap 区间 |
| 定理 A.2 | 一对一错路由正确率；软/硬目标消融 | 软目标不应依赖硬选择的任意并列处理 |
| 定理 A.3 | 取消局部隔离的消融 | 主任务指标与两类误入率的风险取舍应被显式报告 |

固定划分中，RSG 的平均路由后悔差异为 −1.77 个百分点，95%区间为 [−2.86, −0.69]；摄影者留出确认中为 −3.53 个百分点，95%区间为 [−4.75, −2.17]。这支持 RSG 降低已定义风险，并不将该数学结论扩大为通用分类精度定理。
