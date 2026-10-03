# 端点与无约束重放的只读复算命令

日期：2026-10-03。此命令是研究诊断，不训练、不调参、不访问外层数据，不构成新方法的独立验证。

在 `D:\成信工科研\人工智能选矿\.worktrees\theory-aware-report` 工作目录执行。命令只向标准输出打印 JSON，不覆盖已有结果。依赖既有训练环境与本地冻结数据；原始图片和权重不随仓库分发。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$code = @'
import sys, json, csv, hashlib, statistics
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd()/'scripts'))
from run_verifier_trust_development import load_subset, calculate_probability_metrics
from verifier_risk import verification_family, conditional_risk_projection
root=Path.cwd(); torch.set_num_threads(1)
protocol=json.loads((root/'docs/experiment_protocols/abmp_verifier_trust_development_v1.json').read_text(encoding='utf-8'))
out=root/'outputs/training/abmp_verifier_trust_v1/development_fold_0'
files=[root/'docs/experiment_protocols/abmp_verifier_trust_development_v1.json',root/'scripts/verifier_risk.py']
files += sorted(out.glob('A4_seed*/stop_predictions.csv'))+sorted(out.glob('A4_seed*/fit_predictions.csv'))+sorted(out.glob('A4_seed*/best_head.pt'))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={p.relative_to(root).as_posix():digest(p) for p in files}
rows=[]
for name,short in [('gate_stop_projector_fit','fit'),('projector_stop','stop')]:
 data=load_subset(root,name,protocol['input_sha256']); p,L,y=data['p'],data['contradiction'],data['labels']
 lower=verification_family(p,L,torch.ones_like(L)); a,b=lower[:,:1],p[:,:1]
 safe=torch.where(L>0,L,torch.ones_like(L))
 crit=torch.where(L>0,-torch.log1p(p[:,:1]*torch.expm1(-L))/safe,p[:,:1])
 for seed in protocol['seeds']:
  with (out/f'A4_seed{seed}'/f'{short}_predictions.csv').open(encoding='utf-8-sig',newline='') as f: saved=list(csv.DictReader(f))
  assert [(r['image_id'],r['split_group_id'],int(r['true_class_id'])) for r in saved]==[(r['image_id'],r['split_group_id'],int(data['labels'][i])) for i,r in enumerate(data['records'])]
  raw=torch.tensor([[float(r['raw_head'])] for r in saved],dtype=torch.float64)
  q=conditional_risk_projection(p,L,raw)['probabilities']
  sq=torch.tensor([[float(r[f'prob_{k}']) for k in range(4)] for r in saved],dtype=torch.float64)
  assert float((q-sq).abs().max())<1e-12
  use_lower=(raw<crit)&(L>0); endpoint=torch.where(use_lower,lower,p)
  unprojected=torch.cat([raw,(1-raw)*p[:,1:]/(1-p[:,:1])],1)
  interior=((raw>a)&(raw<b)&(L>0)).squeeze(1)
  metrics={k:calculate_probability_metrics(v,y) for k,v in [('continuous',q),('endpoint',endpoint),('raw',unprojected)]}
  u=p[:,1:]/(1-p[:,:1]); uq=q[:,1:]/(1-q[:,:1])
  bce=lambda r,t:-r*torch.log(t)-(1-r)*torch.log1p(-t)
  gap=bce(raw,endpoint[:,:1])-bce(raw,q[:,:1])
  rows.append({'subset':short,'seed':seed,'n':len(y),'interior_count':int(interior.sum()),'lower_endpoint_count':int(use_lower.sum()),'continuous_endpoint_nll_delta':metrics['continuous']['nll']-metrics['endpoint']['nll'],'continuous_endpoint_class_difference':int((q.argmax(1)!=endpoint.argmax(1)).sum()),'raw_continuous_class_difference':int((q.argmax(1)!=unprojected.argmax(1)).sum()),'continuous_new_target_count':int(((q.argmax(1)==0)&(p.argmax(1)!=0)).sum()),'raw_new_target_count':int(((unprojected.argmax(1)==0)&(p.argmax(1)!=0)).sum()),'non_target_ratio_max_residual':float((uq-u).abs().max()),'estimated_endpoint_excess_mean':float(gap.mean()),'estimated_endpoint_excess_min':float(gap.min()),'estimated_endpoint_excess_violations':int((gap < -1e-12).sum()),'metrics':metrics})
aggregates={}
for subset in ['fit','stop']:
 rr=[r for r in rows if r['subset']==subset]
 aggregates[subset]={m:{k:{'mean':statistics.mean([r['metrics'][m][k] for r in rr]),'sample_std':statistics.stdev([r['metrics'][m][k] for r in rr])} for k in rr[0]['metrics'][m]} for m in ['continuous','endpoint','raw']}
g=torch.Generator().manual_seed(20261003); dtype=torch.float64
E,X=10000,4
pi=torch.rand(E,X,4,generator=g,dtype=dtype)+.01; pi/=pi.sum(2,keepdim=True)
pie=pi.mean(1); p=torch.rand(E,4,generator=g,dtype=dtype)+.01; p/=p.sum(1,keepdim=True)
L=2*torch.rand(E,1,generator=g,dtype=dtype); L[:3]=0
r=pie[:,:1]; u=p[:,1:]/(1-p[:,:1]); v=pie[:,1:]/(1-r)
a=torch.sigmoid(torch.logit(p[:,:1])-L); b=p[:,:1]; t=torch.maximum(a,torch.minimum(b,r))
q=torch.cat([t,(1-t)*u],1)
kl=lambda x,y:(x*(torch.log(x)-torch.log(y))).sum(-1)
ber=lambda x:torch.cat([x,1-x],-1)
full=kl(pi,q[:,None,:]).mean(); info=kl(pi,pie[:,None,:]).mean()
clamp=kl(ber(r),ber(t)).mean(); nontarget=((1-r).squeeze(1)*kl(v,u)).mean()
per_family=kl(pie,q); rhs=kl(ber(r),ber(t))+(1-r).squeeze(1)*kl(v,u)
safe=torch.where(L>0,L,torch.ones_like(L)); crit=torch.where(L>0,-torch.log1p(p[:,:1]*torch.expm1(-L))/safe,b)
ep=torch.where((r<crit)&(L>0),a,b); ek=kl(ber(r),ber(ep))-kl(ber(r),ber(t))
synthetic={'seed':20261003,'evidence_states':E,'image_states_per_evidence':X,'three_term_excess_risk':{'full':float(full),'evidence_information_loss':float(info),'target_interval_gap':float(clamp),'fixed_non_target_gap':float(nontarget),'max_per_evidence_chain_residual':float((per_family-rhs).abs().max()),'aggregate_identity_residual':float(abs(full-info-clamp-nontarget))},'true_conditional_endpoint_gap_min':float(ek.min()),'true_conditional_endpoint_gap_violations':int((ek < -1e-12).sum()),'true_conditional_interior_count':int(((r>a)&(r<b)).sum())}
hat_r=torch.rand(E,1,generator=g,dtype=dtype); hat_t=torch.maximum(a,torch.minimum(b,hat_r))
hat_q=torch.cat([hat_t,(1-hat_t)*u],1)
est=kl(ber(r),ber(hat_t))-kl(ber(r),ber(t))
actual_full=kl(pi,hat_q[:,None,:]).mean()
synthetic['four_term_actual_risk']={'full':float(actual_full),'estimation_gap':float(est.mean()),'estimation_gap_min':float(est.min()),'estimation_gap_violations':int((est < -1e-12).sum()),'aggregate_identity_residual':float(abs(actual_full-info-clamp-nontarget-est.mean()))}
assert synthetic['three_term_excess_risk']['max_per_evidence_chain_residual']<1e-12
assert synthetic['three_term_excess_risk']['aggregate_identity_residual']<1e-12
assert synthetic['true_conditional_endpoint_gap_violations']==0
assert synthetic['four_term_actual_risk']['estimation_gap_violations']==0
assert synthetic['four_term_actual_risk']['aggregate_identity_residual']<1e-12
assert before=={p.relative_to(root).as_posix():digest(p) for p in files}
print(json.dumps({'status':'POSTHOC_DEVELOPMENT_DIAGNOSTIC','date':'2026-10-03','selection':'analytic estimated-risk endpoint threshold; no labels used for endpoint/raw inference','new_training':False,'parameter_tuning':False,'outer_access':False,'source_unchanged':True,'input_sha256':before,'rows':rows,'aggregates':aggregates,'synthetic':synthetic},ensure_ascii=True))
'@
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -c $code
```
