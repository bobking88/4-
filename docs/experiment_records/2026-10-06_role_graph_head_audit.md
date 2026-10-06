# 角色图头的可合并性与无效自适应权重核查

## Material Passport

- 日期：2026-10-06；类型：设计前只读代数与文献核查。
- 来源提交：097c9fe8c11c1e304be852389271ebee01b229da。
- 目的：检查“成对角色图”是否提供新的学习机制，避免以普通输出层的重新表达宣称网络创新。
- 范围：四类合成概率、合成中间表示与小矩阵；不读取真实图片、视觉特征、预测表或外层清单，不执行优化器步骤。
- 状态：VERIFIED，保存的完整代码与 JSON 精确重放；不是新网络规格或训练授权。
- 数学解释为作者自审，不是独立理论审稿；不提出首次主张。

## 1. 为什么这项检查改变下一步设计

上一轮证明双通道只是无约束普通残差的重新参数化。另一种容易想到的替代是“目标—难负样本图头”：预测类别对的纠错量，再聚合为四类概率。本轮进一步证明：

1. 固定图、固定正权重、共享表示上的线性边头可以精确合并为普通线性四类头。
2. 连接所有四类时，图头不扩大逐点中心化 logits 的自由度。
3. 对无约束最小二乘的树图，逐样本改变正边权也不改变输出；这种自适应模块在这里无效。
4. 只保留目标—含钛和目标—金属两条边会使脉石孤立，缺少一个分量间相对位移自由度。

这些结论只针对下文明确的算子，不禁止图网络、非线性关系学习或经过约束/正则的成对方法有效。但“画成角色图”本身仍不能证明网络创新。

## 2. 明确算子与线性合并

类序固定为 T（目标）、Ti（含钛干扰）、G（脉石）、M（金属光泽干扰）。令有向关联矩阵 D 的每一行在边的起点取 +1、终点取 -1；正定对角权重 W 为固定常量。

定义图 Laplacian 和最小范数势：

$$
L=D^\top W D,\qquad A=L^\dagger D^\top W,\qquad d=Ae.
\tag{1}
$$

这里是最小化 (Ds-e)^\top W(Ds-e) 的最小范数解；连通时用 1^\top d=0 固定共同平移。输出是 q=softmax(log p+d)，p 严格正。关联矩阵 D 与之前阶段映射矩阵 B^(z) 不同，本轮不涉及真实流程阶段。

若共享表示 f(x) 上的边头为 e=E f+b，则

$$
d=AE f+Ab,\qquad
q=\operatorname{softmax}(\log p+C f+c),\quad C=AE,\ c=Ab.
\tag{2}
$$

因此它只是普通输出头的固定线性组合。即使 f 是一个复杂非线性共享主干，最后这两层仍可合并。这里没有证明独立边分支、输入相关 W(x)、不同监督或正则后的优化路径也等价。

连通时，令 J=I-11^\top/K，有

$$
AD=L^\dagger L=J.
\tag{3}
$$

任意中心化普通头 C=JC、c=Jc 都可选 E=DC、b=Dc 精确实现。因此固定连通图头与共享表示上的不受限制中心化普通头，在输出函数族层面相同。图头参数可能冗余；参数数量不同不意味着独立输出能力更多。

## 3. 树图中的逐样本正权重不生效

连通树有 K-1 条边。D 在中心化子空间上可逆，所以对任意 e 都存在唯一中心化 d 使 Dd=e。此时最小二乘残差为零，

$$
\arg\min_{1^\top d=0}(Dd-e)^\top W(Dd-e)
=D^\dagger e,
\qquad W\succ0.
\tag{4}
$$

右侧不依赖 W。即使每张图片输出不同的正权重 W(x)，在这个无约束算子上也不能改变 d(x)，故不能用它解释预测收益。

同样，任意连通图上若 e 本来可积，即 e=Dd_0 且 d_0 中心化，则 A_W e=d_0，与正权重无关。只有存在不能同时满足的成对信号时，冗余图上的权重才可能改变这种聚合结果。

例外必须单独研究：约束、正则、非树拓扑、删除/置零边、非线性聚合或将权重用于前面的表示学习，均可能打破本节条件。本轮不验证这些不同结构。

## 4. 冗余边的不可见分量与等价正则

由正规方程，

$$
e=Dd+r,\qquad D^\top W r=0,\qquad Ar=0.
\tag{5}
$$

不可见分量 r 可以很大而概率完全不变；其大小不能直接解释为真实标签错误、校准误差或 OOD。四类完全图有六条边，但中心化输出仍只有三维。

对于正权重范数，展开得到标准能量关系

$$
e^\top We=d^\top Ld+r^\top Wr.
\tag{6}
$$

如果只固定中心化 d 并在所有产生该 d 的 e 中最小化边能量，则最小值恰为 d^\top Ld，在 e=Dd 达到。故此类边能量正则可由普通 logits 的图二次型正则复现；不能把它单独列为新图网络能力。这不是训练目标凸性、有限参数优化等价或泛化保证。

若额外监督边输出，可能改变共享表示和优化偏置；那是需要验证的学习作用，不能由式 (5) 推断不存在，也不能仅由概率代数断言成立。

## 5. 漏掉脉石的具体问题

若图只有 (T,Ti)、(T,M)，G 孤立，rank(L)=2。向量

$$
c=(1/4,1/4,-3/4,1/4)^\top
\tag{7}
$$

满足 Dc=0，但给 logits 加 c 不同于全体共同平移，会改变非孤立分量与脉石之间的概率。

因此这些边无法确定该分量间位移。最小范数逆只是用一个固定约定填补缺失自由度，不是从视觉证据估计它。均匀 p 下，加 c 与不加 c 的最大概率差为 0.140768。后续若使用该图，须显式补齐连接或分量质量头，并与普通等价头比较；不能把“只关注两个难负类”当作对所有四类完整建模。

## 6. 合成检查结果

- 三种图各 1024 行，共 3072 行；合并头概率最大差 6.66e-16，类别变化为零。
- 星形树与完全图输出秩都是 3；两条难负边的输出秩为 2，孤立类为 G。
- 加入一条完全图不可见循环信号后，概率最大差为 3.33e-16。
- 128 组正权重的树图重建最大势差为 5.77e-15。
- 128 组正权重的完全图可积信号重建最大势差为 6.66e-15。
- 能量分解与加权正交性在浮点容差内成立。

这些是固定代数性质的数值检查，不是 3072 张矿物测试图片，也不是网络训练实验。种子重复只用于代数核验，不当作独立统计重复。

## 7. 文献边界与发表时间

- [Wu、Lin、Weng：Pairwise Coupling](https://jmlr.csail.mit.edu/papers/v5/wu04a.html)，JMLR 2004-08。原文第 2 节已有类别对概率到整体概率的讨论，第 3–4 节给出线性系统方法。因此成对二分类聚合不作为首次创新。
- [Jiang 等：HodgeRank](https://link.springer.com/article/10.1007/s10107-010-0419-x)，[预印本](https://arxiv.org/abs/0811.1067) 首次 2008-11-07，在线 2010-11-28、卷期 2011-03。[作者全文](https://web.stanford.edu/~yyye/hodgeRank2011.pdf) 第 5 节定理 3 已给出加权边比较的 Laplacian 最小二乘解及不一致分量。本文式 (1)、(5)、(6) 是标准框架的重新推导/应用，不是本项目原创理论。
- [Khosla 等：Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)，首次提交 2020-04-23；[正文](https://arxiv.org/html/2004.11362v5) 第 3.2 节已有监督对比表示与难正/难负样本梯度分析。简单增加对比损失也不足以自动升级为理论创新。

本轮检索关键词为上述论文题名、pairwise coupling、weighted graph projection、supervised contrastive。只核验与候选机制直接有关的原文部分，不声称完成系统查新，也未运行它们的实际训练基线。

## 8. 对当前主线的设计影响，而非新规格

原目标仍是提高报告与第一篇论文的理论和网络创新，不能缩小为长期做校准审计。根据这些证据，设计优先级调整为：

1. 有界输出层保留作可核验约束与解释，但不是独立网络创新。
2. 固定图线性聚合只作为普通头的等价对照；无约束树图的正边权预测不进入新候选。
3. 真正的待研究作用应放在角色条件的非线性表示/特征适配，以及它是否以可控复杂度改善目标—干扰区分。
4. 同特征、同监督和可比参数能力的普通共享适配器、无角色结构版本是必要消融；只与旧未收敛头比较不足。
5. 任何表示更新、输出预算、损失或新数据范围均需单独设计与预登记；选择只在 fit 内分组验证，不能继续在已暴露 stop 上选赢者。

这不是已经批准的角色适配器网络规格；尚未实现或训练它。低秩适配、对比学习和图结构均有相关工作，具体组合仍须证明独立机制和效果，不提前赋予原创称号。C 路线的真实阶段、化验成本与工业过程仍后置。

## 9. 完整重放代码

在仓库根目录，用固定 Python 执行本文件唯一 Python 代码块并与保存 JSON 精确比较。重放不写文件。

~~~powershell
$replay = @'
import contextlib,io,json,re
from pathlib import Path
doc=Path('docs/experiment_records/2026-10-06_role_graph_head_audit.md').read_text(encoding='utf-8')
blocks=re.findall(r'\x60\x60\x60python\n(.*?)\n\x60\x60\x60',doc,re.S)
assert len(blocks)==1
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(blocks[0],'role_graph_algebra','exec'),{})
actual=json.loads(buf.getvalue())
expected=json.loads(Path('outputs/theory/abmp_role_graph_head_audit_v1/audit_summary.json').read_text(encoding='utf-8'))
assert actual==expected
print('EXACT_REPLAY_PASS')
'@
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -c $replay
~~~

```python
import json, hashlib
from pathlib import Path
import numpy as np

rng=np.random.default_rng(20261006)
K=4
J=np.eye(K)-np.ones((K,K))/K
def incidence(edges):
    B=np.zeros((len(edges),K))
    for n,(i,j) in enumerate(edges):
        B[n,i]=1.; B[n,j]=-1.
    return B
def softmax(x):
    z=x-x.max(1,keepdims=True)
    e=np.exp(z)
    return e/e.sum(1,keepdims=True)

graphs=[
    ('target_star',[(0,1),(0,2),(0,3)]),
    ('complete',[(i,j) for i in range(K) for j in range(i+1,K)]),
    ('two_hard_negative_edges',[(0,1),(0,3)]),
]
rows=1024
h=rng.normal(size=(rows,12))
base=softmax(rng.normal(size=(rows,K)))
graph_results=[]
for name,edges in graphs:
    B=incidence(edges)
    weights=rng.uniform(.2,2.,len(edges)); W=np.diag(weights)
    L=B.T@W@B
    A=np.linalg.pinv(L,hermitian=True,rcond=1e-12)@B.T@W
    E=rng.normal(size=(len(edges),12)); b=rng.normal(size=(len(edges),))
    edge=h@E.T+b
    projected=edge@A.T
    fused=h@(A@E).T+A@b
    q_graph=softmax(np.log(base)+projected)
    q_fused=softmax(np.log(base)+fused)
    max_error=float(np.abs(q_graph-q_fused).max())
    changes=int((q_graph.argmax(1)!=q_fused.argmax(1)).sum())
    assert max_error<1e-12 and changes==0
    rank=int(np.linalg.matrix_rank(L,tol=1e-10))
    ab=A@B
    if rank==K-1:
        assert np.abs(ab-J).max()<1e-12
        C=rng.normal(size=(K,12)); C=J@C
        c=J@rng.normal(size=K)
        representable= h@(A@(B@C)).T + A@(B@c)
        assert np.abs(representable-(h@C.T+c)).max()<1e-12
    cycle=edge-projected@B.T
    weighted_orthogonality=float(np.abs(cycle@W@B).max())
    assert weighted_orthogonality<1e-11
    edge_energy=np.einsum('ni,ij,nj->n',edge,W,edge)
    potential_energy=np.einsum('ni,ij,nj->n',projected,L,projected)
    cycle_energy=np.einsum('ni,ij,nj->n',cycle,W,cycle)
    energy_error=float(np.abs(edge_energy-potential_energy-cycle_energy).max())
    assert energy_error<1e-10
    graph_results.append({
        'graph':name,'edges':edges,'weights':weights.tolist(),
        'laplacian_rank':rank,'centered_output_rank':int(np.linalg.matrix_rank(A,tol=1e-10)),
        'rows':rows,'linear_head_fusion_max_probability_error':max_error,'argmax_changes':changes,
        'weighted_cycle_orthogonality_max_error':weighted_orthogonality,
        'weighted_energy_decomposition_max_error':energy_error,
        'cycle_max_abs':float(np.abs(cycle).max()),
        'connected_centered_output_surjectivity_checked':rank==K-1,
        'isolated_class_indices':[i for i in range(K) if not np.any(B[:,i])]
    })

B=incidence(graphs[0][1]); W=np.diag([.5,1.,2.])
L=B.T@W@B
A=np.linalg.pinv(L,hermitian=True)@B.T@W
assert np.linalg.matrix_rank(L)==3
tree_edge=rng.normal(size=(rows,3))
tree_potential=tree_edge@A.T
tree_residual=float(np.abs(tree_edge-tree_potential@B.T).max())
assert tree_residual<1e-12

B=incidence(graphs[1][1]); W=np.eye(len(B)); L=B.T@B
A=np.linalg.pinv(L,hermitian=True)@B.T
raw_cycle=np.array([1.,-1.,0.,1.,0.,0.])
assert np.array_equal(B.T@raw_cycle,np.zeros(4))
cycle_output=float(np.abs(A@raw_cycle).max())
assert cycle_output<1e-12
baseline_e=rng.normal(size=(rows,len(B)))
scale=rng.normal(size=(rows,1))
d1=baseline_e@A.T
d2=(baseline_e+scale*raw_cycle)@A.T
cycle_prob_error=float(np.abs(softmax(np.log(base)+d1)-softmax(np.log(base)+d2)).max())
assert cycle_prob_error<1e-12

B=incidence(graphs[2][1]); W=np.eye(2); L=B.T@B
A=np.linalg.pinv(L,hermitian=True)@B.T
component_offset=np.array([1.,1.,0.,1.])
component_offset-=component_offset.mean()
assert np.array_equal(B@component_offset,np.zeros(2))
assert np.abs((A@B)@component_offset).max()<1e-12
q0=softmax(np.log(np.array([[.25,.25,.25,.25]])))
q1=softmax(np.log(np.array([[.25,.25,.25,.25]]))+component_offset)
unidentified_prob_difference=float(np.abs(q0-q1).max())
assert unidentified_prob_difference>0.1
B=incidence(graphs[0][1])
tree_expected=tree_edge@np.linalg.pinv(B).T
tree_weight_errors=[]
for _ in range(128):
    Wv=np.diag(rng.uniform(.2,2.,len(B)))
    Av=np.linalg.pinv(B.T@Wv@B,hermitian=True)@B.T@Wv
    tree_weight_errors.append(float(np.abs(tree_edge@Av.T-tree_expected).max()))
assert max(tree_weight_errors)<1e-12
B=incidence(graphs[1][1])
centered=rng.normal(size=(rows,K))@J
integrable_edges=centered@B.T
integrable_weight_errors=[]
for _ in range(128):
    Wv=np.diag(rng.uniform(.2,2.,len(B)))
    Av=np.linalg.pinv(B.T@Wv@B,hermitian=True)@B.T@Wv
    integrable_weight_errors.append(float(np.abs(integrable_edges@Av.T-centered).max()))
assert max(integrable_weight_errors)<1e-12
formal=Path('结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx')
report_sha=hashlib.sha256(formal.read_bytes()).hexdigest()
assert report_sha=='b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c'
result={
    'status':'SYNTHETIC_FIXED_GRAPH_HEAD_ALGEBRA_AUDIT','seed':20261006,
    'source_commit':'097c9fe8c11c1e304be852389271ebee01b229da',
    'class_order':['target_mineral','ti_bearing_negative','gangue','metallic_hard_negative'],
    'graph_results':graph_results,'total_probability_rows':rows*len(graphs),
    'tree_projection_max_residual':tree_residual,
    'complete_graph_null_cycle':raw_cycle.tolist(),
    'null_cycle_projected_max_abs':cycle_output,
    'null_cycle_addition_max_probability_error':cycle_prob_error,
    'disconnected_graph_unidentified_centered_offset':component_offset.tolist(),
    'unidentified_offset_probability_difference':unidentified_prob_difference,
    'tree_positive_weight_repetitions':128,'tree_weight_changes_max_potential_error':max(tree_weight_errors),
    'integrable_complete_weight_repetitions':128,'integrable_weight_changes_max_potential_error':max(integrable_weight_errors),
    'raw_images_loaded':False,'visual_features_loaded':False,'real_prediction_tables_read':False,
    'optimizer_steps':0,'new_network_implemented':False,'architecture_approved':False,
    'independent_confirmation':False,'formal_report_sha256':report_sha,
    'scope':'fixed graph and fixed positive weights; algebra of output projection only, not neural training or generalization'
}
print(json.dumps(result,indent=2))
```

## 10. 交付与限制

[数值结果](../../outputs/theory/abmp_role_graph_head_audit_v1/audit_summary.json)、
[重放验证](../../outputs/theory/abmp_role_graph_head_audit_v1/reproduction_verification.json)。
正式报告 SHA256 保持 b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c；无正式报告、骨干、既有结果或外层数据修改。线性函数合并不推导出不同正则/初始化/参数冗余下训练曲线相同。验证属于作者自审，无独立理论确认。
