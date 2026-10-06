# 锚点通道审计与有界容量：完整只读重放

## Material Passport

- 类型：已批准强对照的事后诊断与合成数学性质验证，不是新网络实现。
- 日期：2026-10-05。
- 输入：既有锚点强对照预测 CSV 和登记汇总；不加载图片、权重、视觉特征或外层清单。
- Verification Status: VERIFIED。两个 JSON 均逐项精确重放；这不是独立数据确认，解析证明为作者自审。
- 来源提交：`58ec06a8f6b88e3c62e33c71009556cbdfbcb8e8`。
- 摘要及正式报告 SHA 在代码中固定；代码仅读取、计算并向 stdout 打印 JSON。
- 合成 `linprog` 只核验容量闭式解，不训练网络、不拟合真实数据，不用停止集选预算。
- 环境：沿用 `docs/experiment_protocols/abmp_anchor_linear_controls_v1_requirements.txt`；它是核心包版本记录，不是完整依赖锁。

## 如何重放

在当前仓库工作目录使用项目 Python。以下命令提取并执行本文两个完整代码块，逐项与已保存 JSON 比较；失败将返回非零退出状态。每段在独立命名空间执行，但同一进程内完成，并非第三方独立复现。

采用下面透明的多行 PowerShell here-string。此命令不写文件：

```powershell
$replay = @'
import contextlib, io, json, re
from pathlib import Path
doc = Path('docs/experiment_records/2026-10-05_bounded_role_reproduction.md').read_text(encoding='utf-8')
codes = re.findall(r'```python\n(.*?)\n```', doc, re.S)
paths = [
    'outputs/theory/abmp_anchor_channel_analysis_v1/diagnostic_summary.json',
    'outputs/theory/abmp_bounded_role_capacity_v1/geometry_properties.json',
]
assert len(codes) == len(paths) == 2
for code, path in zip(codes, paths):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, path, 'exec'), {})
    actual = json.loads(buf.getvalue())
    expected = json.loads(Path(path).read_text(encoding='utf-8'))
    assert actual == expected, path
    print('EXACT_REPLAY_PASS', path)
'@
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -c $replay
```

## 1. 通道损失诊断

每个已有组的 fit/stop 输出相对同一个 T0 做逐行分解。混合重建只用于代数诊断，不是训练过的通道消融，也不能据此选择模型。

```python
import csv,json,hashlib
from pathlib import Path
import numpy as np
root=Path.cwd()
out=root/'outputs/training/abmp_anchor_linear_controls_v1/development_fold_0'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
expected='2e018dbe2e223008b1adaf82680230e110dc98edad1fe557ff6138cff7331c32'
assert sha(out/'development_summary.json')==expected
source=json.loads((out/'development_summary.json').read_text(encoding='utf-8'))
used={}
def load_table(relative):
    path=out/relative
    digest=sha(path)
    assert digest==source['artifact_sha256'][relative]
    used[relative]=digest
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    p=np.array([[float(row['prob_'+str(j)]) for j in range(4)] for row in rows])
    y=np.array([int(row['true_class_id']) for row in rows])
    ids=[(row['image_id'],row['split_group_id']) for row in rows]
    assert p.shape==(340,4) and (p>0).all()
    assert np.allclose(p.sum(1),1,atol=1e-14,rtol=0)
    return p,y,ids
def losses(p,y):
    mass=p[:,1:].sum(1)
    r=p[:,1:]/mass[:,None]
    binary=np.where(y==0,-np.log(p[:,0]),-np.log(mass))
    conditional=np.zeros(len(y))
    mask=y!=0
    conditional[mask]=-np.log(r[mask,y[mask]-1])
    total=-np.log(p[np.arange(len(y)),y])
    error=float(np.max(abs(total-binary-conditional)))
    assert error<=1e-12
    return total,binary,conditional,r,error
def stats(p,y):
    pred=p.argmax(1)
    return {'nll':float(-np.log(p[np.arange(len(y)),y]).mean()),
            'correct_targets':int(((y==0)&(pred==0)).sum()),
            'ti_false_targets':int(((y==1)&(pred==0)).sum()),
            'metal_false_targets':int(((y==3)&(pred==0)).sum())}
results=[]
for split in ['fit','stop']:
    p,y,ids=load_table('T0_'+split+'_predictions.csv')
    bp,bb,bc,br,be=losses(p,y)
    for arm in ['P','E','H','EH','EH_permuted']:
        q,yq,idq=load_table(arm+'_zero/'+split+'_predictions.csv')
        assert np.array_equal(y,yq) and ids==idq
        qp,qb,qc,qr,qe=losses(q,y)
        delta=qp-bp
        target_only=np.column_stack([q[:,0],q[:,1:].sum(1)[:,None]*br])
        ratio_only=np.column_stack([p[:,0],p[:,1:].sum(1)[:,None]*qr])
        odds=np.log(q[:,0]/q[:,1:].sum(1))-np.log(p[:,0]/p[:,1:].sum(1))
        conditional_delta=np.log(qr/br)
        full_delta=np.log(q/p)
        full_span=np.ptp(full_delta,axis=1)
        assert (full_span<=abs(odds)+np.ptp(conditional_delta,axis=1)+1e-12).all()
        kl=(p*np.log(p/q)).sum(1)
        assert (kl<=full_span**2/8+1e-12).all()
        row={'subset':split,'arm':arm,'rows':len(y),'anchor':stats(p,y),'model':stats(q,y),
             'delta_nll':float(delta.mean()),'delta_binary_nll':float((qb-bb).mean()),
             'delta_conditional_nll':float((qc-bc).mean()),
             'decomposition_max_abs_error':max(be,qe),
             'target_only_hybrid':stats(target_only,y),'ratio_only_hybrid':stats(ratio_only,y),
             'absolute_target_odds_residual_max':float(abs(odds).max()),
             'conditional_log_ratio_span_max':float(np.ptp(conditional_delta,axis=1).max()),
             'full_log_residual_span_max':float(full_span.max()),'model_probability_min':float(q.min()),
             'delta_nll_by_class':{str(k):float(delta[y==k].mean()) for k in range(4)},
             'top_ten_positive_delta_contribution_to_full_mean':float(np.sort(delta[delta>0])[-10:].sum()/len(y))}
        if delta.mean()>0:
            row['conditional_share_of_net_delta']=float((qc-bc).mean()/delta.mean())
        results.append(row)
report=root/'结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx'
assert sha(report)=='b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c'
for rel,digest in used.items():
    assert sha(out/rel)==digest
assert sha(out/'development_summary.json')==expected
print(json.dumps({'status':'POSTHOC_READ_ONLY_CHANNEL_AUDIT','results':results,
                 'source_summary_sha256':expected,'input_sha256':used,
                 'formal_report_sha256':sha(report),'new_training':False,
                 'new_images_loaded':False,'outer_manifest_read':False,
                 'hybrids_are_trained_ablations':False,'selection_performed':False,
                 'independent_confirmation':False},indent=2))
```

## 2. 有限预算容量与性质验证

768 次线性规划核对闭式最大竞争概率；5000 次合成输入检查损失、KL、间隔与可达见证；两个反例说明有界变化仍可能损失正确目标或新增假目标。32 个真实表预算组合只作标签辅助的可达计数，不是真实模型表现。固定随机种子只是数值重放，不是训练重复或统计显著性证据。

```python
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
rng=np.random.default_rng(20261005)
def rho(w,beta):
    return float(w.max()/np.minimum(w.max(),np.exp(beta)*w).sum())
def lp_rho(w,beta):
    m=len(w); A=[]; b=[]
    for i in range(m):
        row=np.zeros(m+1); row[i]=1; row[-1]=-1; A.append(row); b.append(0.)
        for j in range(m):
            if i!=j:
                row=np.zeros(m+1); row[i]=w[j]; row[j]=-np.exp(beta)*w[i]
                A.append(row); b.append(0.)
    objective=np.zeros(m+1); objective[-1]=1
    ans=linprog(objective,A_ub=A,b_ub=b,A_eq=[np.r_[np.ones(m),0.]],b_eq=[1.],bounds=[(0.,None)]*(m+1),method='highs')
    assert ans.success
    return ans.fun
weights=np.r_[rng.dirichlet(np.ones(3),size=125),[[1/3]*3,[.98,.01,.01],[.7,.2,.1]]]
beta_grid=[0.,.25,.5,1.,2.,10.]
errors=[]
for w in weights:
    for beta in beta_grid:
        r=np.minimum(w.max(),np.exp(beta)*w); r/=r.sum()
        assert abs(r.max()-rho(w,beta))<1e-14
        assert np.ptp(np.log(r/w))<=beta+1e-12
        errors.append(abs(r.max()-lp_rho(w,beta)))
assert max(errors)<1e-8
z=rng.normal(0,3,size=(5000,4)); p=np.exp(z-z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
alpha=rng.uniform(0,2,size=5000); beta=rng.uniform(0,3,size=5000)
u=rng.uniform(-alpha,alpha); v=rng.uniform(-beta[:,None]/2,beta[:,None]/2,size=(5000,3))
pm=p[:,1:].sum(1); w=p[:,1:]/pm[:,None]
t=p[:,0]*np.exp(u)/(pm+p[:,0]*np.exp(u))
r=w*np.exp(v); r/=r.sum(1,keepdims=True)
q=np.column_stack([t,(1-t)[:,None]*r])
delta=np.log(p/q); nll_bound=alpha[:,None]+np.column_stack([np.zeros(len(p)),np.repeat(beta[:,None],3,1)])
assert (abs(delta)<=nll_bound+1e-10).all()
kl_pq=(p*delta).sum(1); kl_qp=-(q*delta).sum(1)
upper_pq=(alpha**2+pm*beta**2)/8
upper_qp=(alpha**2+(1-t)*beta**2)/8
assert (kl_pq<=upper_pq+1e-10).all() and (kl_qp<=upper_qp+1e-10).all()
order=np.argsort(p,axis=1); pred=order[:,-1]
margin=np.log(p[np.arange(len(p)),pred]/p[np.arange(len(p)),order[:,-2]])
protected=margin>alpha+beta
assert np.array_equal(q.argmax(1)[protected],pred[protected])
cap_r=np.minimum(w.max(1)[:,None],np.exp(beta)[:,None]*w)
cap_r/=cap_r.sum(1,keepdims=True)
cap_t=p[:,0]*np.exp(alpha)/(pm+p[:,0]*np.exp(alpha))
cap_q=np.column_stack([cap_t,(1-cap_t)[:,None]*cap_r])
actual_margin=np.log(cap_q[:,0]/cap_q[:,1:].max(1))
formula_margin=np.log(p[:,0]/pm)+alpha-np.log(cap_r.max(1))
margin_error=float(abs(actual_margin-formula_margin).max())
assert margin_error<1e-10
identity=np.column_stack([p[:,0],pm[:,None]*w])
identity_error=float(abs(identity-p).max())
assert identity_error<1e-14
counterexamples=[]
for name,base,label,shift in [
    ('bounded_update_can_lose_correct_target',[.26,.25,.245,.245],0,-.1),
    ('bounded_update_can_add_false_target',[.24,.25,.255,.255],1,.2)]:
    base=np.array(base); mass=base[1:].sum()
    t_new=base[0]*np.exp(shift)/(mass+base[0]*np.exp(shift))
    repaired=np.r_[t_new,(1-t_new)*base[1:]/mass]
    base_pred=int(base.argmax()); new_pred=int(repaired.argmax())
    if label==0:
        assert base_pred==0 and new_pred!=0
    else:
        assert base_pred!=0 and new_pred==0
    assert abs(np.log(base[label]/repaired[label]))<=abs(shift)+1e-12
    counterexamples.append({'name':name,'base_probability':base.tolist(),
        'repaired_probability':repaired.tolist(),'true_class_id':label,
        'alpha':abs(shift),'beta':0.,'u':shift,'v':[0.,0.,0.],
        'base_prediction':base_pred,'new_prediction':new_pred})
capacity=[]
root=Path.cwd(); output=root/'outputs/training/abmp_anchor_linear_controls_v1/development_fold_0'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
source=json.loads((output/'development_summary.json').read_text(encoding='utf-8'))
assert sha(output/'development_summary.json')=='2e018dbe2e223008b1adaf82680230e110dc98edad1fe557ff6138cff7331c32'
for split in ['fit','stop']:
    path=output/f'T0_{split}_predictions.csv'
    assert sha(path)==source['artifact_sha256'][path.name]
    with path.open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
    base=np.array([[float(row['prob_'+str(j)]) for j in range(4)] for row in records]); labels=np.array([int(row['true_class_id']) for row in records])
    mass=base[:,1:].sum(1); w=base[:,1:]/mass[:,None]; base_pred=base.argmax(1)
    for a in [0.,.25,.5,1.]:
        for b in [0.,.25,.5,1.]:
            minimum_rho=np.array([rho(row,b) for row in w])
            maximum_margin=np.log(base[:,0]/mass)+a-np.log(minimum_rho)
            attainable=maximum_margin>=0
            capacity.append({'subset':split,'alpha':a,'beta':b,'recoverable_missed_targets':int((attainable&(base_pred!=0)&(labels==0)).sum()),'new_false_target_capable_ti':int((attainable&(base_pred!=0)&(labels==1)).sum()),'new_false_target_capable_gangue':int((attainable&(base_pred!=0)&(labels==2)).sum()),'new_false_target_capable_metallic':int((attainable&(base_pred!=0)&(labels==3)).sum())})
report=root/'结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx'
assert sha(report)=='b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c'
result={'status':'READ_ONLY_SYNTHETIC_PROOF_AND_LABEL_ASSISTED_CAPACITY_DIAGNOSTIC','seed':20261005,'lp_comparisons':len(errors),'lp_max_abs_error':max(errors),'random_probability_checks':5000,'pointwise_loss_violations':int((abs(delta)>nll_bound+1e-10).sum()),'kl_bound_violations':int((kl_pq>upper_pq+1e-10).sum()+(kl_qp>upper_qp+1e-10).sum()),'protected_margin_rows':int(protected.sum()),'protected_margin_violations':0,'capacity':capacity,'maximum_margin_witness_checks':5000,'maximum_margin_max_abs_error':margin_error,'identity_max_abs_error':identity_error,'counterexamples':counterexamples,'source_summary_sha256':sha(output/'development_summary.json'),'formal_report_sha256':sha(report),'new_neural_training':False,'synthetic_lp_only':True,'new_images_loaded':False,'outer_manifest_read':False,'formal_report_unchanged':True,'budget_selection_performed':False,'independent_confirmation':False}
print(json.dumps(result,indent=2))
```
