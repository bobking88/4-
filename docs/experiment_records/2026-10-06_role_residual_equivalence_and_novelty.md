# 角色残差：精确等价、约束几何与创新查核

## Material Passport

- 日期：2026-10-06；类型：设计前的只读理论与文献核查。
- 上一来源提交：`2aedce14b10ca6523c24a2b87ff4fa2c560294d4`。
- 范围：合成代数/自动微分检查、既有两内层子集的 10 张主预测表；不读取图片、视觉特征或外层清单，不训练、不选模。
- 状态：VERIFIED，保存的代码与数值结果精确重放，验证记录另存；不代表新网络效果已验证。
- 单位：输出函数与约束集合；不是新的网络实验、真实标签风险估计或独立测试。
- 学习模块尚待人工确认方向与书面规格，不将本轮核查作为开发/训练授权。
- 推导和文献范围为作者自审，不是第三方查新或同行审稿。

## 1. 关键结论

**两条概率通道本身不扩大无约束输出能力。** 它们是普通四类残差的精确重参数化。数学公式可以帮助设计，但不能用“有公式、有两头”替代新学习机制和强对照。

上一记录中的有限预算容量、损失/KL 界仍成立。补充结论是：预算形成特定耦合可行域；普通残差采用同一可行域时，也具有同样的输出容量与这些数学界。因此理论性质不能被独占地归因于某个网络名称。

## 2. 双通道与普通残差的精确映射

令 $p$ 严格正，$t=p_T$、$w_j=p_j/(1-t)$。双通道定义

$$
q_T=\sigma(\operatorname{logit}t+u),\quad
r_j=\frac{w_je^{v_j}}{Z_w(v)},\quad
Z_w(v)=\sum_{j\ne T}w_je^{v_j},\quad q_j=(1-q_T)r_j.
\tag{1}
$$

普通后验残差为 $q=\operatorname{softmax}(\log p+d)$。取

$$
d_T=u+\log Z_w(v),\qquad d_j=v_j\quad(j\ne T),
\tag{2}
$$

即可精确还原双通道。反向取

$$
v_j=d_j,\qquad u=d_T-\log Z_w(d_{\neg T})
\tag{3}
$$

也成立。证明：普通残差的目标/整体非目标 odds 为

$$
\frac{q_T}{1-q_T}
=\frac{p_Te^{d_T}}{\sum_{j\ne T}p_je^{d_j}}
=\frac{t}{1-t}e^{d_T-\log Z_w(d_{\neg T})}.
\tag{4}
$$

条件非目标分配则为 $w_je^{d_j}/Z_w(d_{\neg T})$。代入两映射即得。

两者都有 $K-1$ 个逐点概率自由度；共同平移 $d+c\mathbf1$ 不改变输出，对应 $u$ 不变和 $v+c\mathbf1$ 的冗余。

### 对训练比较的含义

若仍使用同一参数头输出 $(u_\theta,v_\theta)$，只将最后的重建改为式 (2) 加普通 softmax，那么两个计算图的输出、任意相同输出损失、关于同一 $\theta$ 的梯度在精确算术下相同。**这种重写不需要再训练一个“强对照”来创造不同结果，精确重放即可核验。**

这不证明以下模型训练等价：

- 分别使用普通线性 $d_\theta$ 与线性 $(u_\theta,v_\theta)$ 的固定容量头；式 (2) 含 log-sum-exp，参数函数族可能不同。
- 不同损失、权重、正则、预算参数化、优化器、初始化或随机训练过程。
- 给另一个模型加入不同特征或更大有效输入维数。

若出现差异，必须隔离上述因素；不能自动归因为“矿物角色理论”。二分类目标也已包含在四分类交叉熵链式分解中，重复加入辅助 BCE 改变权重，不是自动获得新标签信息。

## 3. 精确可行域是耦合凸多面体

对固定输入 $p$ 和非负有限常数 $\alpha,\beta$，定义

$$
l=\sigma(\operatorname{logit}p_T-\alpha),\qquad
h=\sigma(\operatorname{logit}p_T+\alpha).
\tag{5}
$$

双预算族 $|u|\le\alpha$、$\operatorname{span}(v)\le\beta$ 等价于概率空间集合

$$
\mathcal F_{\alpha,\beta}(p)
=\left\{
q\in\Delta^{K-1}:
l\le q_T\le h,\quad
q_jp_k\le e^\beta q_kp_j\quad
\forall j,k\ne T
\right\}.
\tag{6}
$$

证明：目标区间与 odds 区间互逆；非目标内部有

$$
\log(r_j/w_j)-\log(r_k/w_k)
=\log(q_j/p_j)-\log(q_k/p_k).
\tag{7}
$$

故 span 约束恰等价于所有成对线性不等式。反向使用 $u=\operatorname{logit}q_T-\operatorname{logit}p_T$、$v_j=\log(r_j/w_j)$ 还原式 (1)。

$p,\alpha,\beta$ 固定时式 (6) 是有限线性不等式与概率单纯形的交集，所以为闭有界凸多面体；$p$ 本身可行。$0<l\le h<1$，有限比例约束迫使每个非目标概率严格正。比如

$$
q_j\ge(1-h)\frac{w_j}{w_j+e^\beta(1-w_j)}>0.
\tag{8}
$$

这不证明**参数训练目标**为凸，也不证明学习出的样本相关预算或骨干训练为凸。凸可行域不自动保证分类精度、真实召回或泛化。

### 与固定概率盒不相同的见证

取三类非目标 $w=(1/3,1/3,1/3)$、$\beta=\log2$。比例可行域各坐标的最小/最大可达值分别为 $1/5$ 与 $1/2$，在 $(0.2,0.4,0.4)$ 和 $(0.5,0.25,0.25)$ 的排列达到。

但 $r=(0.5,0.3,0.2)$ 满足这个最小包围盒及总和 1，却有 $r_{\max}/r_{\min}=2.5>2$，不满足比例可行域。任何包含完整比例域的固定坐标盒也包含该见证，所以不能精确替代该比例域。

这是**固定输入可行集的几何区别**，不是与所有实例相关 PB 网络不可等价的证明，更不是性能优势证据。

## 4. 对照预算不能只看同一个数字

令 $\mathcal S_\delta$ 为普通残差 $\operatorname{span}(d)\le\delta$ 的逐点输出集合。由式 (3)，

$$
|u|\le\operatorname{span}(d),\qquad
\operatorname{span}(v)\le\operatorname{span}(d).
\tag{9}
$$

反向有 $\operatorname{span}(d)\le |u|+\operatorname{span}(v)$。所以

$$
\mathcal S_\delta\subseteq
\mathcal F_{\delta,\delta}(p)
\subseteq\mathcal S_{2\delta}.
\tag{10}
$$

第一包含一般严格。取 $p=(0.5,1/6,1/6,1/6)$、$u=1$、$v=(-1,0,0)$，有 $\alpha=\beta=1$，等效 $d=(0.763383,-1,0,0)$，其 span 为 1.763383，大于 1。

因此不能将“双预算 0.5/0.5”与“普通 span 0.5”直接称为同容量对照。两者相同数值对应不同可行域；若比较性能，需要明确比较的是结构、可行集合还是总漂移规模。式 (10) 是结构分析，不是学习复杂度或样本复杂度定理。

## 5. 数值检查

| 检查 | 数量 | 结果 |
| --- | ---: | --- |
| 双通道至 flat 输出 | 5000 | 最大概率误差 $4.44\times10^{-16}$ |
| 任意 flat 至双通道输出 | 5000 | 最大误差 $3.33\times10^{-16}$ |
| 共同平移不变性 | 5000 | 最大 $u$ 误差 $2.00\times10^{-15}$ |
| 两种预算与多面体成员关系 | 5000 | 无不一致 |
| 同潜变量 NLL 梯度 | 128 | 最大误差 $2.17\times10^{-18}$ |
| Hessian-vector product | 128 | 最大误差 $1.25\times10^{-17}$ |
| 真实已有预测输出重建 | 10 表，共 3400 行 | 最大误差 $3.33\times10^{-16}$，类别变化 0 |

真实重建不使用真实标签；合成梯度检查使用合成类别。没有优化器步骤，没有更新任何头或骨干。自动微分检查只针对相同潜变量加精确映射，不能外推为不同固定网络架构的参数训练等价。

## 6. 补充查新：网络创新的边界

### PB / BCSoftmax

[Atarashi 等的 PB](https://arxiv.org/abs/2506.10572) 首次提交日期为 2025-06-12，v2 为 2026-02-23，当前证据按预印本处理，不推断审稿状态。[v2 方法第 3 节](https://arxiv.org/html/2506.10572v2) 定义盒约束 softmax；第 3.3 节已有用神经网络预测逐样本边界的扩展。因此硬概率限制、边界可学习、优化层及其 Jacobian 本身不作为我们的首次创新。

网页正文日期另显示 2026-08-24，与 arXiv 版本元数据不同；本轮优先记录提交历史，不用网页排版日期证明首发时间。几何不同不代表已排除其所有变体。

### REPAIR

[Wang 等的 REPAIR](https://arxiv.org/abs/2604.01506) 于 2026-04-02 首次提交。[正文第 4–5 节](https://arxiv.org/html/2604.01506v1) 将纠错分为固定类别项与依赖竞争标签的成对项，并讨论固定偏置失效条件及轻量重排序。故成对难负样本纠错、类别项/成对项分解及单纯“胜过固定偏置”不能单独作为新理论。

其对象是固定 top-k 候选列表的长尾重排；本项目是四类矿物角色代理任务及有界修正。对象差异不自动构成方法创新。本轮没有运行 REPAIR/PB，也不声称超过它们。

### 本轮查新范围

检索词包含 PB/BCSoftmax、bounded likelihood ratio/minimax、residual calibration/log odds、REPAIR、softmax conditional reparameterization。检索结果只从作者论文/官方论文页支持方法判断，未使用二手摘要作为技术证据。式 (4) 是标准 softmax 代数关系，不作为查新候选；有限预算专用容量命题仍未完成系统查新，**不能据本轮检索宣称首次**。

## 7. 对下一网络规格的实质修改

原“有界双通道”继续作为待讨论的结构坐标，但不能再作为独立新能力主张。下一规格至少必须分清四层贡献：

1. **坐标层**：双通道与普通残差的映射已等价，仅作实现/解释。
2. **可行域层**：耦合比例约束与目标 odds 区间的效果，必须有相同可行域及简单边界对照。
3. **学习层**：角色关系是否改变所学表示、样本修正分配或优化偏置，需要同特征、同参数能力、同监督及同训练预算的消融。
4. **标签风险层**：在冻结后的未暴露数据上超过最强对照，同时检查召回/误入/NLL，而非只看理论上可达。

任何二分类辅助、角色惩罚或特征分支的增加均须说明它的实际独立作用；若一个共享普通头加相同约束与损失就可复现收益，则不能把坐标分解包装成新网络。

候选网络设计仍待人工选择；本轮不写已经批准的网络规格、不实施新网络，不扩展训练、其他真实折或正式报告修订。原研究目标不缩小为“只做校准”或“仅完成这些检查”。

## 8. 完整只读重放

在仓库根目录使用固定 Python。以下命令执行本文件唯一的 Python 代码块并与保存的 JSON 精确比较，不写文件：

~~~powershell
$replay = @'
import contextlib,io,json,re
from pathlib import Path
doc=Path('docs/experiment_records/2026-10-06_role_residual_equivalence_and_novelty.md').read_text(encoding='utf-8')
codes=re.findall(r'\x60\x60\x60python\n(.*?)\n\x60\x60\x60',doc,re.S)
assert len(codes)==1
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(codes[0],'role_residual_equivalence','exec'),{})
actual=json.loads(buf.getvalue())
expected=json.loads(Path('outputs/theory/abmp_role_residual_equivalence_v1/audit_summary.json').read_text(encoding='utf-8'))
assert actual==expected
print('EXACT_REPLAY_PASS')
'@
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -c $replay
~~~

```python
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
rng=np.random.default_rng(20261006)
def softmax(z):
    z=z-z.max(axis=1,keepdims=True)
    out=np.exp(z); return out/out.sum(axis=1,keepdims=True)
def lse_w(w,v):
    top=v.max(1)
    return top+np.log((w*np.exp(v-top[:,None])).sum(1))
def forward_dual(p,u,v):
    t=p[:,0]; mass=p[:,1:].sum(1); w=p[:,1:]/mass[:,None]
    logit=np.log(t/mass)+u
    qt=1/(1+np.exp(-logit))
    r=softmax(np.log(w)+v)
    return np.column_stack([qt,(1-qt)[:,None]*r])
def dual_to_flat(p,u,v):
    w=p[:,1:]/p[:,1:].sum(1)[:,None]
    return np.column_stack([u+lse_w(w,v),v])
def flat_to_dual(p,d):
    w=p[:,1:]/p[:,1:].sum(1)[:,None]
    v=d[:,1:].copy(); u=d[:,0]-lse_w(w,v)
    return u,v
p=softmax(rng.normal(0,2,size=(5000,4)))
u=rng.uniform(-2,2,5000); v=rng.uniform(-1.5,1.5,(5000,3))
q_dual=forward_dual(p,u,v); d=dual_to_flat(p,u,v)
q_flat=softmax(np.log(p)+d)
forward_error=float(abs(q_dual-q_flat).max())
assert forward_error<1e-12
recovered_u,recovered_v=flat_to_dual(p,d)
inverse_error=float(abs(recovered_u-u).max())
assert inverse_error<1e-12 and np.array_equal(recovered_v,v)
arbitrary_d=rng.normal(0,2,size=(5000,4))
u2,v2=flat_to_dual(p,arbitrary_d)
back_q=forward_dual(p,u2,v2)
reverse_error=float(abs(back_q-softmax(np.log(p)+arbitrary_d)).max())
assert reverse_error<1e-12
shift=rng.normal(0,3,size=(5000,1))
u3,v3=flat_to_dual(p,arbitrary_d+shift)
gauge_u_error=float(abs(u2-u3).max())
gauge_v_span_error=float(abs(np.ptp(v2,axis=1)-np.ptp(v3,axis=1)).max())
assert gauge_u_error<1e-12 and gauge_v_span_error<1e-12
span_d=np.ptp(arbitrary_d,axis=1)
assert (abs(u2)<=span_d+1e-12).all()
assert (np.ptp(v2,axis=1)<=span_d+1e-12).all()
assert (np.ptp(d,axis=1)<=abs(u)+np.ptp(v,axis=1)+1e-12).all()
base=np.array([[.5,1/6,1/6,1/6]])
witness_u=np.array([1.]); witness_v=np.array([[-1.,0.,0.]])
witness_d=dual_to_flat(base,witness_u,witness_v)
witness_span=float(np.ptp(witness_d))
assert witness_span>1+1e-12
alpha_fixed=2.; beta_fixed=3.
q_any=softmax(np.log(p)+arbitrary_d)
odds_base=np.log(p[:,0]/p[:,1:].sum(1))
lo=1/(1+np.exp(-odds_base+alpha_fixed))
hi=1/(1+np.exp(-odds_base-alpha_fixed))
coordinate_member=(abs(u2)<=alpha_fixed)&(np.ptp(v2,axis=1)<=beta_fixed)
polytope_member=(q_any[:,0]>=lo)&(q_any[:,0]<=hi)
for j in [1,2,3]:
    for k in [1,2,3]:
        polytope_member&=q_any[:,j]*p[:,k]<=np.exp(beta_fixed)*q_any[:,k]*p[:,j]+1e-14
polytope_mismatches=int((coordinate_member!=polytope_member).sum())
assert polytope_mismatches==0
box_r=np.array([.5,.3,.2])
assert (box_r>=.2).all() and (box_r<=.5).all() and abs(box_r.sum()-1)<1e-14
assert box_r.max()/box_r.min()>2.
# Same latent outputs and the deterministic exact map: compare derivatives, not fitted networks.
p_t=torch.tensor(p[:128],dtype=torch.float64)
u_t=torch.tensor(u[:128],dtype=torch.float64,requires_grad=True)
v_t=torch.tensor(v[:128],dtype=torch.float64,requires_grad=True)
w_t=p_t[:,1:]/p_t[:,1:].sum(1,keepdim=True)
qt=torch.sigmoid(torch.log(p_t[:,0]/p_t[:,1:].sum(1))+u_t)
rt=torch.softmax(torch.log(w_t)+v_t,1)
dual_t=torch.cat([qt[:,None],(1-qt)[:,None]*rt],1)
zt=torch.logsumexp(torch.log(w_t)+v_t,1)
dt=torch.cat([(u_t+zt)[:,None],v_t],1)
flat_t=torch.softmax(torch.log(p_t)+dt,1)
labels=torch.arange(128)%4
rows=torch.arange(128)
loss_dual=-torch.log(dual_t[rows,labels]).mean()
loss_flat=-torch.log(flat_t[rows,labels]).mean()
grad_dual=torch.autograd.grad(loss_dual,[u_t,v_t],create_graph=True)
grad_flat=torch.autograd.grad(loss_flat,[u_t,v_t],create_graph=True)
gradient_error=float(max((a-b).abs().max().item() for a,b in zip(grad_dual,grad_flat)))
assert gradient_error<1e-12
directions=[torch.tensor(rng.normal(size=x.shape),dtype=torch.float64) for x in [u_t,v_t]]
hvp_dual=torch.autograd.grad(sum((a*b).sum() for a,b in zip(grad_dual,directions)),[u_t,v_t])
hvp_flat=torch.autograd.grad(sum((a*b).sum() for a,b in zip(grad_flat,directions)),[u_t,v_t])
hvp_error=float(max((a-b).abs().max().item() for a,b in zip(hvp_dual,hvp_flat)))
assert hvp_error<1e-12
root=Path.cwd()
out=root/'outputs/training/abmp_anchor_linear_controls_v1/development_fold_0'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
expected='2e018dbe2e223008b1adaf82680230e110dc98edad1fe557ff6138cff7331c32'
assert sha(out/'development_summary.json')==expected
source=json.loads((out/'development_summary.json').read_text(encoding='utf-8'))
used={}; real=[]
for subset in ['fit','stop']:
    bpath=out/('T0_'+subset+'_predictions.csv')
    assert sha(bpath)==source['artifact_sha256'][bpath.name]
    used[bpath.name]=sha(bpath)
    with bpath.open(encoding='utf-8-sig',newline='') as f: base_rows=list(csv.DictReader(f))
    base=np.array([[float(x['prob_'+str(j)]) for j in range(4)] for x in base_rows])
    base_ids=[(x['image_id'],x['split_group_id']) for x in base_rows]
    for arm in ['P','E','H','EH','EH_permuted']:
        rel=arm+'_zero/'+subset+'_predictions.csv'
        path=out/rel
        assert sha(path)==source['artifact_sha256'][rel]
        used[rel]=sha(path)
        with path.open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
        assert [(x['image_id'],x['split_group_id']) for x in records]==base_ids
        q=np.array([[float(x['prob_'+str(j)]) for j in range(4)] for x in records])
        mass_b=base[:,1:].sum(1); mass_q=q[:,1:].sum(1)
        w=base[:,1:]/mass_b[:,None]; r=q[:,1:]/mass_q[:,None]
        u_r=np.log(q[:,0]/mass_q)-np.log(base[:,0]/mass_b)
        v_r=np.log(r/w)
        flat=softmax(np.log(base)+dual_to_flat(base,u_r,v_r))
        error=float(abs(flat-q).max())
        changes=int((flat.argmax(1)!=q.argmax(1)).sum())
        assert error<1e-12 and changes==0
        real.append({'subset':subset,'arm':arm,'rows':len(q),'max_probability_abs_error':error,'argmax_changes':changes})
report=root/'结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx'
assert sha(report)=='b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c'
for rel,digest in used.items():
    assert sha(out/rel)==digest
result={'status':'READ_ONLY_FUNCTION_FAMILY_EQUIVALENCE_AUDIT','seed':20261006,
'synthetic_probability_rows':5000,'dual_to_flat_max_abs_error':forward_error,
'flat_to_dual_max_abs_error':reverse_error,'inverse_u_max_abs_error':inverse_error,
'gauge_u_max_abs_error':gauge_u_error,'gauge_v_span_max_abs_error':gauge_v_span_error,
'autograd_rows':128,'gradient_max_abs_error':gradient_error,'hessian_vector_max_abs_error':hvp_error,
'strict_budget_inclusion_witness':{'base':[.5,1/6,1/6,1/6],'u':1.,'v':[-1.,0.,0.],
'alpha':1.,'beta':1.,'equivalent_flat_d':witness_d[0].tolist(),'flat_logit_span':witness_span},
'polytope_membership_rows':5000,'polytope_membership_mismatches':polytope_mismatches,'fixed_box_counterexample':{'w':[1/3,1/3,1/3],'beta':float(np.log(2)),'coordinate_lower':[.2,.2,.2],'coordinate_upper':[.5,.5,.5],'box_feasible_r':box_r.tolist(),'pairwise_ratio':float(box_r.max()/box_r.min()),'allowed_ratio':2.},'real_prediction_reconstructions':real,'source_summary_sha256':expected,'input_sha256':used,
'formal_report_sha256':sha(report),'raw_images_loaded':False,'visual_features_loaded':False,
'new_network_implemented':False,'optimizer_steps':0,'outer_manifest_read':False,
'true_labels_used_for_real_reconstruction':False,'independent_confirmation':False,
'fixed_architecture_parameter_families_proved_equal':False}
print(json.dumps(result,indent=2))
```

## 9. 核查限制

本轮无统计显著性检验、无数据删选、无预算选优，不把反复暴露的数据当独立测试；类角色为公开标本图像代理类别，不解释元素品位或工业回收。解析证明是作者自审；数值重放只证明所计算输出可重现，不代替独立理论审查或泛化实验。

[分析 JSON](../../outputs/theory/abmp_role_residual_equivalence_v1/audit_summary.json) 与 [验证记录](../../outputs/theory/abmp_role_residual_equivalence_v1/reproduction_verification.json) 保存输入哈希。正式报告哈希仍为 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。
