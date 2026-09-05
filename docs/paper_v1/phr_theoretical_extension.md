# PHR 网络理论补充：成对路由、训练误差与受约束修正

日期：2026-09-05。状态：解析推导与数值核验；尚不是 PHR 正式训练结果。
本文是 B 主线中 RSG-HRGV 的候选扩展，不替代已有正式模型结论，C 阶段决策仍属后续研究。

## 1. 网络契约与假设

共享主干输出特征 \(h=f_\theta(x)\)。直接角色专家输出 \(p_d\)，矿物种类专家经固定种类到角色矩阵聚合为 \(p_m\)。全局 RSG 门控产生基准概率
\[
p_0=g_0p_d+(1-g_0)p_m.
\]
角色顺序为 \(T,I,G,M\)，分别对应目标代理类、含钛干扰、脉石和金属光泽干扰。
PHR 用两个可学习门控代替原最终验证器后处理；验证器的辅助损失仍保留。
因此 PHR 与 RSG 的总体差异不只是“增加两个门”，消融解释必须包含后处理替换。

以下结论条件于一个已给定的两专家输出。所有对数为自然对数，概率为正。
实现中按机器精度截断概率，公式严格对应截断后的对数值；不对截断前极小概率宣称完全相同的恒等式。
涉及真实标签的 oracle 仅用于训练监督或离线诊断，不得输入推理网络。

对边 \(q\in\{I,M\}\)，
\[
m_{d,q}=\log(p_{d,T}/p_{d,q}),\quad
m_{m,q}=\log(p_{m,T}/p_{m,q}),\quad
m_{f,q}=g_qm_{d,q}+(1-g_q)m_{m,q}.
\]
只在真实标签 \(y\in\{T,q\}\) 的样本上定义该边风险。
令 \(s=1\)（\(y=T\)）或 \(-1\)（\(y=q\)），
\[
u_d=sm_{d,q},\quad u_m=sm_{m,q},\quad
\Delta=u_d-u_m,\quad u_f=g_qu_d+(1-g_q)u_m.
\]

## 2. 性质 P1：精确路由遗憾及二分类损失界

令 \(g^*=\mathbf1[\Delta\ge0]\)，则
\[
r_q=\max(u_d,u_m)-u_f=|\Delta|\,|g_q-g^*|.\tag{P1}
\]
证明：分别代入 \(\Delta\ge0\) 和 \(\Delta<0\)；并列时两边均为零。
对二分类 logistic 损失 \(\phi(u)=\log(1+e^{-u})\)，由于它单调递减且 \(|\phi'(u)|\le1\)，
\[
0\le\phi(u_f)-\phi(\max(u_d,u_m))\le r_q.\tag{P2}
\]
若两专家边际严格同正或同负，凸组合保留其符号；允许零时是弱符号结论，不保证打破并列。
这解释为何需要区分两条困难负样本边，不能用一个四类平均指标代替两条边的诊断。

## 3. 性质 P2：软目标训练误差到边际风险的上界

设软 oracle 为 \(t=\sigma(\Delta/\tau)\)，\(\tau>0\)，预测门控 \(g\in(0,1)\)，
\[
D=\mathrm{KL}(\mathrm{Bern}(t)\Vert\mathrm{Bern}(g))
 =\mathrm{BCE}(t,g)-H(t).
\]
使用 Bernoulli 情形的 Pinsker 不等式 \(|g-t|\le\sqrt{D/2}\)，以及
\[
|t-g^*|=\frac1{1+e^{|\Delta|/\tau}}\le e^{-|\Delta|/\tau},
\]
由三角不等式和 P1 得
\[
r_q\le|\Delta|\sqrt{D/2}+|\Delta|e^{-|\Delta|/\tau}.\tag{P3}
\]
进一步由 Cauchy-Schwarz 与 \(\max_{d\ge0}de^{-d/\tau}=\tau/e\)，
\[
\mathbb E[r_q]\le
\sqrt{\tfrac12\mathbb E[\Delta^2]\mathbb E[D]}+\tau/e.\tag{P4}
\]
期望可取有资格样本的经验均值；总体期望版本需相应可积条件。
P4 不是对训练算法收敛或未见数据泛化的保证。

**与当前加权损失对齐。** 代码使用 \(w=\tanh(|\Delta|/\tau_w)\)。
定义 \(0^2/0=0\)，非零间隔处 \(w>0\)。若 \(\mathbb E[w]>0\)，
\[
\mathbb E[r_q]\le
\sqrt{\tfrac12\mathbb E[\Delta^2/w]\mathbb E[wD]}+\tau/e,\tag{P5}
\]
\[
\mathbb E[wD]=\mathbb E[w]\bigl(L_w-H_w\bigr),\quad
L_w=\frac{\mathbb E[w\,\mathrm{BCE}(t,g)]}{\mathbb E[w]},\quad
H_w=\frac{\mathbb E[wH(t)]}{\mathbb E[w]}.
\]
证明：对 \(\mathbb E[|\Delta|\sqrt{D/2}]\) 写成
\(\mathbb E[(|\Delta|/\sqrt w)\sqrt{wD/2}]\)，应用 Cauchy-Schwarz。
所有间隔均零时遗憾为零，不计算归一化损失商。
加权 BCE 本身并非“遗憾”，必须减去软目标熵后才对应 KL 项。
联合训练时专家和目标仍会变化；即使某次 BCE 下降，也不能单凭该现象断言所有上界因子或实际风险下降。

Pinsker 是既有基础不等式，不能称为本项目原创。
来源：[Canonne，作者维护的 Pinsker 推导笔记](https://github.com/ccanonne/probabilitydistributiontoolbox/blob/master/pinskers-and-beyond.tex)。
本节贡献定位是针对所实现门控与加权训练目标的可核验推导，不是首次提出该通用不等式。

## 4. 性质 P3：两边同时满足的最小范数修正

令 \(\ell_0=\log p_0\)，
\(\delta_q=m_{f,q}-(\ell_{0,T}-\ell_{0,q})\)，在 \(T,I,M\) 三节点求
\[
\min_a \tfrac12\|a\|_2^2,\quad
Aa=\delta,\qquad
A=\begin{bmatrix}1&-1&0\\1&0&-1\end{bmatrix}.
\]
因 \(A\) 满行秩，严格凸目标的唯一解为
\[
a=A^\top(AA^\top)^{-1}\delta
=\frac13\begin{bmatrix}
\delta_I+\delta_M\\
-2\delta_I+\delta_M\\
\delta_I-2\delta_M
\end{bmatrix}.\tag{P6}
\]
证明：拉格朗日一阶条件给出 \(a=A^\top\lambda\)，代入约束求 \(\lambda\)。
令 \(a_G=0\)，则 \(p_{\rm PHR}=\mathrm{softmax}(\ell_0+a)\) 同时满足两条指定对数几率。
单边消融令未启用边的 \(\delta_q=0\)，表示**保持该边基准几率**，不是移除这一约束。

进一步有
\[
\|a\|_2^2=\tfrac23(\delta_I^2-\delta_I\delta_M+\delta_M^2)
\le\|\delta\|_2^2.\tag{P7}
\]
对任意真实类别 \(y\)，令 \(\ell_y(z)=-z_y+\log\sum_ke^{z_k}\)，
\[
|\ell_y(\ell_0+a)-\ell_y(\ell_0)|
\le\max_k a_k-\min_k a_k
\le\sqrt2\|a\|_2
\le\sqrt2\|\delta\|_2.\tag{P8}
\]
第一步由 log-sum-exp 增量介于 \(\min a\) 与 \(\max a\) 之间得到；
第二步用 \(|a_j-a_k|\le\sqrt2\|a\|_2\)。
这是一条扰动大小界，**并不是损失一定降低的单调性结论**。

## 5. 必须保留的反例与创新边界

取均匀基准 \(p_0=(1/4,1/4,1/4,1/4)\)，两条目标边际均改为 3，
则四维修正为 \((2,-1,0,-1)\)。
脉石 logit 不变，但其 softmax 概率下降，真实脉石样本的负对数似然反而增加。
因此不能写“PHR 不影响脉石”“两条边更好必然提高四分类 F1”。

边际融合满足
\[
\exp(m_f)=(p_{d,T}/p_{d,q})^g(p_{m,T}/p_{m,q})^{1-g},
\]
本质是几率的几何融合；公式本身不足以证明新颖性。
多分类中的成对概率耦合至少已有
[Hastie 与 Tibshirani，NIPS 1997](https://papers.nips.cc/paper_files/paper/1997/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)。
本方案可争取的方法贡献是：角色定义的两条稀疏困难边、标签方向感知且隔离梯度的门控监督、保持另一边的受约束修正及完整风险验证。
这仍需正式查新和消融支持，不能称为已经成立的通用网络理论创新。

## 6. 验证层级与待完成实验

1. 解析证明：P1--P8，注明概率截断、边资格和有限矩条件。
2. 数值核验：固定种子、双精度随机数据及并列边界；核对遗憾、KL 界、伪逆独立解、扰动界和反例。脚本为 scripts/verify_phr_theory.py，输出为 outputs/theory/phr_properties_v2.json。数值核验不能替代证明或真实数据实验。
3. 真实验证集筛选：修正后的 RSG 对照与 7 个 PHR 配置，不能使用测试集选配置。
4. 正式多种子实验：只有通过预先规定的验证集标准才启动；报告 Macro F1、目标召回、两条误入率、两条边遗憾及分组置信区间。
5. 将 P3/P5 的预测量在冻结真实输出上复算；理论上界即使成立但过松，也必须如实报告。
6. 正式报告只可将本节列为“候选网络理论及实现验证”；在实际多种子结果完成前不得写“PHR 显著优于 RSG”。
