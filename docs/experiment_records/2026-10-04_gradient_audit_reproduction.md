# 监督与梯度机制的只读复算命令

日期：2026-10-04。此命令不训练、不更新权重、不选新 checkpoint、不访问独立外层，只输出诊断 JSON。23 个源文件前后哈希、注册初始化与选中权重状态、逐图原始概率及全部解析式均有断言。

在 `D:\成信工科研\人工智能选矿\.worktrees\theory-aware-report` 工作目录运行。需要现有本地训练环境、冻结概率与权重；原始图像与权重不随仓库分发。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$code = @'
import sys,json,csv,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path.cwd()/'scripts'))
from run_verifier_trust_development import load_subset
from run_tc_oos_rsg_experiments import state_dict_sha256
from tc_oos_rsg import TargetRiskProjectionHead
from verifier_risk import conditional_risk_projection, verification_family
root=Path.cwd(); torch.set_num_threads(1)
protocol_path=root/'docs/experiment_protocols/abmp_verifier_trust_development_v1.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
out=root/'outputs/training/abmp_verifier_trust_v1/development_fold_0'
source_files=[protocol_path]+[root/'scripts'/name for name in ('run_verifier_trust_development.py','verifier_risk.py','tc_oos_rsg.py','run_tc_oos_rsg_experiments.py')]
for key in protocol['input_sha256']:
 subset,file=key.split('/')
 path=(root/'outputs/training/tc_oos_rsg_manifests_v1/fold_0'/f'{subset}.csv') if file=='manifest.csv' else (root/'outputs/theory/abmp_candidate_capacity_v1'/key)
 source_files.append(path)
for pattern in ('A4_seed*/fit_predictions.csv','A4_seed*/stop_predictions.csv','A4_seed*/best_head.pt','A4_seed*/result.json'):
 source_files += sorted(out.glob(pattern))
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={p.relative_to(root).as_posix():digest(p) for p in source_files}
data_by_subset={short:load_subset(root,name,protocol['input_sha256']) for short,name in [('fit','gate_stop_projector_fit'),('stop','projector_stop')]}
rows=[]
for seed in protocol['seeds']:
 result=json.loads((out/f'A4_seed{seed}'/'result.json').read_text(encoding='utf-8'))
 torch.manual_seed(seed)
 initial=TargetRiskProjectionHead(dropout=protocol['budget']['dropout']).double().eval()
 assert state_dict_sha256(initial.state_dict())==result['initial_sha256']
 selected=TargetRiskProjectionHead(dropout=protocol['budget']['dropout']).double().eval()
 selected.load_state_dict(torch.load(out/f'A4_seed{seed}'/'best_head.pt',map_location='cpu',weights_only=True))
 assert state_dict_sha256(selected.state_dict())==result['state_sha256']
 for subset,data in data_by_subset.items():
  with (out/f'A4_seed{seed}'/f'{subset}_predictions.csv').open(encoding='utf-8-sig',newline='') as h:
   saved=list(csv.DictReader(h))
  assert [(v['image_id'],v['split_group_id'],int(v['true_class_id'])) for v in saved]==[(v['image_id'],v['split_group_id'],int(data['labels'][i])) for i,v in enumerate(data['records'])]
  saved_raw=torch.tensor([[float(v['raw_head'])] for v in saved],dtype=torch.float64)
  p,L,y=data['p'],data['contradiction'],data['labels']; z=(y==0).double().reshape(-1,1)
  for stage,head in [('initial',initial),('selected',selected)]:
   with torch.no_grad(): raw=head(data['evidence'])
   if stage=='selected':
    assert float((raw-saved_raw).abs().max())<1e-12
   if stage=='initial' and subset=='fit':
    assert abs(float(F.binary_cross_entropy(raw,z))-result['initial_fit_objective'])<1e-12
   scores=torch.logit(raw).detach().requires_grad_(True)
   estimate=torch.sigmoid(scores)
   proj=conditional_risk_projection(p,L,estimate)
   q=proj['probabilities']; a,b=proj['lower'],proj['upper']
   assert bool((q>0).all())
   clipped_loss=-q[torch.arange(len(y)),y].log()
   clipped_grad=torch.autograd.grad(clipped_loss.sum(),scores,retain_graph=True)[0]
   smooth_q=verification_family(p,L,estimate)
   smooth_loss=-smooth_q[torch.arange(len(y)),y].log()
   smooth_grad=torch.autograd.grad(smooth_loss.sum(),scores,retain_graph=True)[0]
   smooth_analytic=L*(z-smooth_q[:,:1].detach())*estimate.detach()*(1-estimate.detach())
   smooth_residual=float((smooth_grad-smooth_analytic).abs().max())
   assert smooth_residual<1e-12
   raw_loss=F.binary_cross_entropy_with_logits(scores,z,reduction='none')
   raw_grad=torch.autograd.grad(raw_loss.sum(),scores)[0]
   assert float((estimate.detach()-raw).abs().max())<1e-15
   assert float((raw_grad-(estimate.detach()-z)).abs().max())<1e-12
   interior=(estimate.detach()>a)&(estimate.detach()<b)&(L>0)
   strict_low=(estimate.detach()<a)&(L>0)
   strict_high=(estimate.detach()>b)&(L>0)
   zero_width=(L==0)
   ties=(~interior)&(~strict_low)&(~strict_high)&(~zero_width)
   assert not bool(ties.any()), 'Nondegenerate boundary tie; handle separately'
   analytic=torch.where(interior,estimate.detach()-z,torch.zeros_like(z))
   residual=float((clipped_grad-analytic).abs().max())
   assert residual<1e-12
   conditional_u=p[:,1:]/(1-p[:,:1])
   conditional_const=torch.where(z.squeeze(1)==0,-torch.log(conditional_u[torch.arange(len(y)),(y-1).clamp_min(0)]),torch.zeros_like(y,dtype=torch.float64))
   reconstructed=F.binary_cross_entropy(proj['probabilities'][:,:1],z,reduction='none').squeeze(1)+conditional_const
   assert float((clipped_loss-reconstructed).detach().abs().max())<1e-12
   counts={}
   for label in range(4):
    mask=(y==label).reshape(-1,1)
    counts[str(label)]={'n':int(mask.sum()),'interior':int((mask&interior).sum()),'strict_lower':int((mask&strict_low).sum()),'strict_upper':int((mask&strict_high).sum()),'zero_width':int((mask&zero_width).sum()),'clipped_zero_score_gradient':int((mask&(clipped_grad==0)).sum()),'raw_zero_score_gradient':int((mask&(raw_grad==0)).sum())}
   rows.append({'seed':seed,'subset':subset,'stage':stage,'dropout_mode':'eval','n':len(y),'interior_count':int(interior.sum()),'strict_lower_count':int(strict_low.sum()),'strict_upper_count':int(strict_high.sum()),'zero_width_count':int(zero_width.sum()),'boundary_tie_count':int(ties.sum()),'clipped_zero_score_gradient_count':int((clipped_grad==0).sum()),'raw_zero_score_gradient_count':int((raw_grad==0).sum()),'smooth_strength_zero_score_gradient_count':int((smooth_grad==0).sum()),'smooth_strength_gradient_max_abs_residual':smooth_residual,'clipped_mean_abs_score_gradient':float(clipped_grad.abs().mean()),'smooth_strength_mean_abs_score_gradient':float(smooth_grad.abs().mean()),'raw_mean_abs_score_gradient':float(raw_grad.abs().mean()),'gradient_identity_max_abs_residual':residual,'loss_decomposition_max_abs_residual':float((clipped_loss-reconstructed).detach().abs().max()),'class_counts':counts})
dtype=torch.float64
a,b=.1,.4
r,hat=.2,.8
s=torch.tensor([[torch.logit(torch.tensor(hat,dtype=dtype))]],dtype=dtype,requires_grad=True)
t=torch.maximum(torch.tensor(a,dtype=dtype),torch.minimum(torch.tensor(b,dtype=dtype),torch.sigmoid(s)))
risk=-r*t.log()-(1-r)*torch.log1p(-t)
grad=float(torch.autograd.grad(risk,s)[0])
bce=lambda r,t:-r*torch.log(torch.tensor(t,dtype=dtype))-(1-r)*torch.log1p(-torch.tensor(t,dtype=dtype))
synthetic={'a':a,'b':b,'true_r':r,'estimated_r':hat,'clipped_t':float(t.detach()),'clipped_conditional_score_gradient':grad,'raw_bce_conditional_score_gradient':hat-r,'excess_above_interval_optimum':float(bce(r,b)-bce(r,r))}
assert grad==0 and synthetic['excess_above_interval_optimum']>0
g=torch.Generator().manual_seed(20261004); n=25000
pp=torch.rand(n,4,generator=g,dtype=dtype)+.01; pp/=pp.sum(1,keepdim=True)
LL=2*torch.rand(n,1,generator=g,dtype=dtype); LL[:3]=0
aa=torch.sigmoid(torch.logit(pp[:,:1])-LL); bb=pp[:,:1]
rr=.01+.98*torch.rand(n,1,generator=g,dtype=dtype)
hh=.01+.98*torch.rand(n,1,generator=g,dtype=dtype)
ss=torch.logit(hh).detach().requires_grad_(True); he=torch.sigmoid(ss)
tt=torch.maximum(aa,torch.minimum(bb,he)); star=torch.maximum(aa,torch.minimum(bb,rr))
lam=.25
ce=lambda r,t:-r*torch.log(t)-(1-r)*torch.log1p(-t)
combined=ce(rr,tt)+lam*ce(rr,he)
base=ce(rr,star)+lam*ce(rr,rr)
excess=combined-base; raw_kl=ce(rr,he)-ce(rr,rr)
delta=ce(rr,tt)-ce(rr,star)
jgrad=torch.autograd.grad(combined.sum(),ss)[0]
active=(he.detach()>aa)&(he.detach()<bb)&(LL>0)
analytic=(lam+active.double())*(he.detach()-rr)
mixed={'seed':20261004,'states':n,'lambda_illustrative_not_selected':lam,'max_exact_regret_identity_residual':float((excess-delta-lam*raw_kl).detach().abs().max()),'lower_bound_violations':int((excess<lam*raw_kl-1e-12).sum()),'upper_bound_violations':int((excess>(1+lam)*raw_kl+1e-12).sum()),'max_score_gradient_identity_residual':float((jgrad-analytic).abs().max()),'strictly_saturated_count':int(((LL>0)&(~active)).sum()),'clipped_only_score_gradient_zero_count':int((~active).sum()),'joint_score_gradient_zero_count':int((jgrad==0).sum()),'boundary_tie_count':int((((he.detach()==aa)|(he.detach()==bb))&(LL>0)).sum())}
assert mixed['max_exact_regret_identity_residual']<1e-12
assert mixed['lower_bound_violations']==mixed['upper_bound_violations']==0
assert mixed['max_score_gradient_identity_residual']<1e-12
assert mixed['boundary_tie_count']==0

after={p.relative_to(root).as_posix():digest(p) for p in source_files}
assert before==after
print(json.dumps({'status':'POSTHOC_READONLY_GRADIENT_DIAGNOSTIC','date':'2026-10-04','scope':'Fold 0 inner tables only; hypothetical loss score gradients at registered initial/selected A4 outputs, no training or label-based inference','new_training':False,'optimizer_steps':0,'outer_access':False,'source_unchanged':True,'input_sha256':before,'rows':rows,'synthetic_suboptimal_plateau':synthetic,'synthetic_joint_supervision':mixed},ensure_ascii=True))
'@
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -c $code
```
