# 信息收益与容量诊断的只读复现

日期：2026-10-04。在 `D:\成信工科研\人工智能选矿\.worktrees\theory-aware-report` 执行。以下为已实际运行的数学与概率表审计命令，不读取图片、不设置优化器、不更新模型。保存结果后再次执行会核对 JSON 值完全相同；若结果文件不存在，命令只打印，不写文件。

依赖既有脚本、强对照协议及登记六个输入文件。本地 Python 为 Torch 2.12.1+cu130；本命令只用 CPU float64、单线程、固定种子。结果是合成 oracle 与现有开发概率容量核验，不是新增矿物图像模型实验。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -c @'
import json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd() / "scripts"))
from run_verifier_trust_development import load_subset
from run_tc_oos_rsg_experiments import _file_sha256
torch.set_num_threads(1)
torch.manual_seed(20261004)
dtype = torch.float64
n = 10000
w = torch.tensor([.4, .6], dtype=dtype)[None, :]
a = .03 + .25 * torch.rand(n, 1, dtype=dtype)
b = a + .1 + .5 * torch.rand(n, 1, dtype=dtype)
rh = .01 + .98 * torch.rand(n, 2, dtype=dtype)
r = (rh * w).sum(1, keepdim=True)
th = torch.maximum(a, torch.minimum(b, rh))
t = torch.maximum(a, torch.minimum(b, r))
def ce(r, t):
    return -r*t.log() - (1-r)*torch.log1p(-t)
def kl(r, t):
    return ce(r,t) - ce(r,r)
coarse = ce(r,t).squeeze(1)
refined = (ce(rh,th)*w).sum(1)
gain = coarse-refined
info = ce(r,r).squeeze(1)-(ce(rh,rh)*w).sum(1)
jensen_penalty = (kl(rh,th)*w).sum(1)-kl(r,t).squeeze(1)
identity = float((gain-info+jensen_penalty).abs().max())
assert identity < 2e-15
assert bool((gain >= -2e-15).all()) and bool((info-gain >= -2e-15).all())
# Full categorical probabilities independently check cancellation of the frozen u term.
u = torch.rand(n,1,3,dtype=dtype)+.1
u = u/u.sum(2,keepdim=True)
v = torch.rand(n,2,3,dtype=dtype)+.1
v = v/v.sum(2,keepdim=True)
pi = torch.cat([rh[:,:,None], (1-rh[:,:,None])*v], 2)
q_coarse = torch.cat([t[:,:,None], (1-t[:,:,None])*u], 2).expand(-1,2,-1)
q_refined = torch.cat([th[:,:,None], (1-th[:,:,None])*u], 2)
categorical_gain = ((-(pi*q_coarse.log()).sum(2) + (pi*q_refined.log()).sum(2))*w).sum(1)
categorical_residual = float((categorical_gain-gain).abs().max())
assert categorical_residual < 3e-15
def example(a0,b0,rs,weights):
    rr=torch.tensor([rs],dtype=dtype)
    ww=torch.tensor([weights],dtype=dtype)
    mean=(rr*ww).sum(1,keepdim=True)
    aa=torch.tensor(a0,dtype=dtype); bb=torch.tensor(b0,dtype=dtype)
    tt=torch.maximum(aa,torch.minimum(bb,rr))
    mt=torch.maximum(aa,torch.minimum(bb,mean))
    return {"a":a0,"b":b0,"r_given_H":rs,"weights":weights,
            "target_information":float(ce(mean,mean)-(ce(rr,rr)*ww).sum()),
            "constrained_risk_gain":float(ce(mean,mt)-(ce(rr,tt)*ww).sum())}
examples = {"informative_but_upper_saturated":example(.1,.2,[.3,.5],[.5,.5]),
            "informative_inside_interval":example(.1,.6,[.2,.4],[.5,.5])}
protocol_path=Path("docs/experiment_protocols/abmp_verifier_strong_controls_v1.json")
protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
tables=[]
before={}
for name in ("gate_stop_projector_fit","projector_stop"):
    data=load_subset(Path.cwd(),name,protocol["input_sha256"])
    before.update(data["hashes"])
    p=data["p"]; labels=data["labels"]
    jmax=p[:,1:].max(1).values
    base_target=p.argmax(1)==0
    reachable_free_u=p[:,0]>=.25
    gap=(jmax/p[:,0]).log()
    per_class=[]
    for k in range(4):
        mask=(labels==k)&(~base_target)
        possible=mask&reachable_free_u
        per_class.append({"class_id":k,"baseline_non_target_winners":int(mask.sum()),
                          "target_feasible_by_free_non_target_ratios_only":int(possible.sum()),
                          "still_impossible_without_target_mass_increase":int((mask&(~reachable_free_u)).sum())})
    tables.append({"subset":name,"count":len(p),"base_target_winners":int(base_target.sum()),
                   "free_u_only_candidate_count":int((~base_target&reachable_free_u).sum()),
                   "per_true_class":per_class,
                   "minimum_target_odds_boost_log_margin_true_target_median":
                       float(gap[(labels==0)&(~base_target)].median())})
after={}
for name in ("gate_stop_projector_fit","projector_stop"):
    after.update(load_subset(Path.cwd(),name,protocol["input_sha256"])["hashes"])
assert before==after==protocol["input_sha256"]
result={"status":"MATHEMATICAL_AND_READONLY_CAPACITY_CHECK_ONLY","seed":20261004,"synthetic_E_states":n,
        "H_states_per_E":2,"risk_identity_max_abs_residual":identity,
        "categorical_identity_max_abs_residual":categorical_residual,
        "nonnegative_gain_violations":int((gain < -2e-15).sum()),
        "gain_above_target_information_violations":int((gain-info > 2e-15).sum()),
        "mean_synthetic_constrained_gain":float(gain.mean()),
        "mean_synthetic_target_information":float(info.mean()),
        "mean_synthetic_interval_jensen_penalty":float(jensen_penalty.mean()),
        "examples":examples,"frozen_probability_capacity":tables,
        "input_sha256":before,"registered_protocol_sha256":_file_sha256(protocol_path),
        "input_unchanged":True,"images_loaded":False,"optimizer_steps":0,"new_head_training":False}

pi2=torch.tensor([[.2,.64,.08,.08],[.2,.08,.64,.08]],dtype=dtype)
mean_pi=pi2.mean(0,keepdim=True)
full_info=float((pi2*(pi2.log()-mean_pi.log())).sum(1).mean())
assert float((pi2[:,:1]-mean_pi[:,:1]).abs().max())==0.
result["non_target_only_information_example"]={
    "posterior_given_H":pi2.tolist(),"target_information":0.,
    "four_class_information":full_info,
    "gain_with_fixed_non_target_ratios":0.,
    "gain_if_non_target_ratios_also_adapt":full_info}
p2=torch.tensor([[.26,.60,.10,.04]],dtype=dtype)
q2=torch.tensor([[.26,.74/3,.74/3,.74/3]],dtype=dtype)
eta=float((p2[:,1:].max(1).values/p2[:,0]).log())
delta=torch.tensor([[eta-1e-8],[eta+1e-8]],dtype=dtype)
odds_q=p2.repeat(2,1)*torch.cat([delta.exp(),torch.ones(2,3,dtype=dtype)],1)
odds_q=odds_q/odds_q.sum(1,keepdim=True)
assert int(p2.argmax())==1 and int(q2.argmax())==0 and q2[0,0]==p2[0,0]
assert odds_q.argmax(1).tolist()==[1,0]
result["capacity_counterexample"]={"base":p2.tolist()[0],"free_non_target_ratios":q2.tolist()[0],
    "target_mass_unchanged":True,"target_promotion":True,
    "minimum_log_odds_boost_if_non_target_ratios_fixed":eta,
    "classes_below_above_threshold":odds_q.argmax(1).tolist()}
saved=Path("outputs/theory/abmp_information_capacity_v1/diagnostic_summary.json")
if saved.exists():
    assert json.loads(saved.read_text(encoding="utf-8"))==result
print(json.dumps(result,indent=2))

'@
```

结果：`outputs/theory/abmp_information_capacity_v1/diagnostic_summary.json`。证明、假设及结论边界见 `2026-10-04_visual_information_and_role_capacity.md`。正式报告不修改，不进入其他真实折。
