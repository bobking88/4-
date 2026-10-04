# 静态偏置等价性证书的只读复现

日期：2026-10-04。在 `D:\成信工科研\人工智能选矿\.worktrees\theory-aware-report` 执行。下面的实际命令枚举有限差分约束、读取已登记概率表，不读取图片或设置优化器，不训练新校准臂。只有当数学系统可行时才构造一个证明用见证，不按真实标签选择偏置、不测新偏置的性能，也不部署。

依赖既有协议、六个登记输入、六张 P1 表和强对照摘要。CPU float64、单线程。保存结果后重放会核对 JSON 完全相同；结果不存在时仅打印，不写文件。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -c @'
import csv, itertools, json, math, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()/"scripts"))
from run_verifier_trust_development import load_subset
from run_tc_oos_rsg_experiments import _file_sha256
torch.set_num_threads(1)
root=Path.cwd()
protocol_path=root/"docs/experiment_protocols/abmp_verifier_strong_controls_v1.json"
assert _file_sha256(protocol_path)=="b8c1ec58746ca4d39961af3f93e8e7f4bc3b3e59e11fa47fd2f06fca4f435a4a"
protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
base=root/"outputs/training/abmp_verifier_strong_controls_v1/development_fold_0"
summary_path=base/"development_summary.json"
summary_hash=_file_sha256(summary_path)
summary=json.loads(summary_path.read_text(encoding="utf-8"))
tables_hash={}
pattern_lists={}
inputs_hash={}
data={name:load_subset(root,name,protocol["input_sha256"])
      for name in ("gate_stop_projector_fit","projector_stop")}
for d in data.values():
    inputs_hash.update(d["hashes"])
assert inputs_hash==protocol["input_sha256"]
def certify(theta,target,ids,classes):
    lo=float(theta[target].max()) if bool(target.any()) else -math.inf
    hi=float(theta[~target].min()) if bool((~target).any()) else math.inf
    feasible=lo<hi
    result={"count":len(theta),"target_winner_count":int(target.sum()),
            "lower_inclusive":lo,"upper_exclusive":hi,
            "interval_nonempty":feasible,"separation_width":hi-lo}
    if feasible:
        delta=(lo+hi)/2 if math.isfinite(lo) and math.isfinite(hi) else (
             lo+1 if math.isfinite(lo) else hi-1 if math.isfinite(hi) else 0.)
        matched=(theta<=delta)==target
        assert bool(matched.all())
        result["mathematical_midpoint_witness"]=delta
        result["target_pattern_exact_witness"]=True
    else:
        ilow=int(torch.where(target,theta,torch.full_like(theta,-math.inf)).argmax())
        ihigh=int(torch.where(~target,theta,torch.full_like(theta,math.inf)).argmin())
        assert target[ilow] and not target[ihigh] and theta[ilow]>=theta[ihigh]
        result["infeasibility_witness"]={
            "retain_target":{"image_id":ids[ilow],"required_delta_at_least":lo,
                             "existing_predicted_class":int(classes[ilow])},
            "exclude_target":{"image_id":ids[ihigh],"required_delta_strictly_below":hi,
                              "existing_predicted_class":int(classes[ihigh])}}
    return result
# Check both sides of the half-open feasibility condition without real data.
assert certify(torch.tensor([-1.,.5]),torch.tensor([True,False]),
               ["toy_a","toy_b"],[0,1])["interval_nonempty"]
assert not certify(torch.tensor([.5,-.5]),torch.tensor([True,False]),
                   ["toy_a","toy_b"],[0,1])["interval_nonempty"]

def class_bias_certificate(p,pred,ids):
    logs=p.log()
    edges={}
    for y in range(4):
        mask=pred==y
        if not bool(mask.any()):
            continue
        for j in range(4):
            if j==y:
                continue
            gaps=logs[:,y]-logs[:,j]
            vals=torch.where(mask,gaps,torch.full_like(gaps,math.inf))
            i=int(vals.argmin())
            edges[(y,j)]={"from":y,"to":j,"cost":float(vals[i]),
                          "strict":j<y,"image_id":ids[i]}
    cycles=[]
    for k in (2,3,4):
        for vertices in itertools.combinations(range(4),k):
            first=min(vertices)
            for rest in itertools.permutations([v for v in vertices if v!=first]):
                order=(first,)+rest
                es=[edges.get((order[t],order[(t+1)%k])) for t in range(k)]
                if all(e is not None for e in es):
                    cycles.append({"classes":list(order),"sum":sum(e["cost"] for e in es),
                                   "strict_edge_count":sum(e["strict"] for e in es),"edges":es})
    minimum=min(cycles,key=lambda c:c["sum"]) if cycles else None
    result={"scope":"All static four-class log-probability offsets, also any positive global temperature plus those offsets",
            "simple_cycles_checked":len(cycles),"tolerance":1e-12,
            "minimum_cycle_sum":minimum["sum"] if minimum else None}
    if minimum and minimum["sum"] < -1e-12:
        result.update({"status":"INFEASIBLE_NEGATIVE_CYCLE","cycle_witness":minimum})
    elif minimum and minimum["sum"]<=1e-12:
        result["status"]="NEAR_DEGENERATE_UNRESOLVED"
    else:
        eps=min([1e-4]+[c["sum"]/(4*c["strict_edge_count"])
                       for c in cycles if c["strict_edge_count"]])
        dist=[0.]*4
        for iteration in range(4):
            changed=False
            for (y,j),edge in edges.items():
                cost=edge["cost"]-eps*edge["strict"]
                if dist[j]>dist[y]+cost:
                    dist[j]=dist[y]+cost
                    changed=True
            if not changed:
                break
        offsets=torch.tensor(dist,dtype=torch.float64)
        offsets=offsets-offsets[0]
        witness=(logs+offsets).argmax(1)
        assert bool((witness==pred).all())
        result.update({"status":"FEASIBLE_EXACT_PATTERN_WITNESS","strict_epsilon":eps,
                       "offset_witness":offsets.tolist(),"four_class_pattern_exact_witness":True})
    return result


toy_p=torch.tensor([[.3,.4,.2,.1],[.2,.6,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7]],dtype=torch.float64)
toy_pred=(toy_p.log()+torch.tensor([.7,0.,0.,0.],dtype=torch.float64)).argmax(1)
assert toy_pred.tolist()==[0,1,2,3]
assert class_bias_certificate(toy_p,toy_pred,["toy_0","toy_1","toy_2","toy_3"])["status"]=="FEASIBLE_EXACT_PATTERN_WITNESS"
bad_p=torch.tensor([[.1,.7,.1,.1],[.7,.1,.1,.1]],dtype=torch.float64)
assert class_bias_certificate(bad_p,torch.tensor([0,1]),["bad_0","bad_1"])["status"]=="INFEASIBLE_NEGATIVE_CYCLE"
tie_p=torch.full((2,4),.25,dtype=torch.float64)
assert class_bias_certificate(tie_p,torch.tensor([0,1]),["tie_0","tie_1"])["status"]=="NEAR_DEGENERATE_UNRESOLVED"

runs=[]
for seed in (20261002,20261003,20261004):
    pieces=[]; by_subset=[]
    for name,short in (("gate_stop_projector_fit","fit"),("projector_stop","stop")):
        rel=f"P1_seed{seed}/{short}_predictions.csv"
        path=base/rel
        digest=_file_sha256(path)
        assert digest==summary["artifact_sha256"][rel]
        tables_hash[rel]=digest
        with path.open(encoding="utf-8-sig",newline="") as handle:
            rows=list(csv.DictReader(handle))
        d=data[name]; p=d["p"]; labels=d["labels"]; source=d["records"]
        assert len(rows)==len(source)==340
        for i,row in enumerate(rows):
            assert row["image_id"]==source[i]["image_id"]
            assert row["split_group_id"]==source[i]["split_group_id"]
            assert int(row["true_class_id"])==int(labels[i])
        q=torch.tensor([[float(row[f"prob_{k}"]) for k in range(4)] for row in rows],dtype=torch.float64)
        pred=torch.tensor([int(row["predicted_class_id"]) for row in rows])
        pattern_lists[(seed,short)]=pred.tolist()
        assert bool((pred==q.argmax(1)).all())
        assert float((q[:,0]-p[:,0]).max())<1e-12
        proportions=(q[:,1:]/(1-q[:,:1]))-(p[:,1:]/(1-p[:,:1]))
        assert float(proportions.abs().max())<1e-12
        non_winners=p[:,1:].argmax(1)+1
        assert bool((pred[pred!=0]==non_winners[pred!=0]).all())
        theta=(p[:,1:].max(1).values/p[:,0]).log()
        ids=[row["image_id"] for row in rows]
        result=certify(theta,pred==0,ids,pred)
        result["subset"]=name
        result["fixed_non_target_ratio_max_residual"]=float(proportions.abs().max())
        if result["interval_nonempty"]:
            delta=result["mathematical_midpoint_witness"]
            logq=p.log(); logq[:,0]+=delta
            witness=logq.argmax(1)
            assert bool((witness==pred).all())
            result["four_class_pattern_exact_witness"]=True
        result["static_class_bias"]=class_bias_certificate(p,pred,ids)
        by_subset.append(result)
        pieces.append((theta,pred,ids,p))
    joint=certify(torch.cat([x[0] for x in pieces]),torch.cat([x[1] for x in pieces])==0,
                  [i for x in pieces for i in x[2]],torch.cat([x[1] for x in pieces]))
    joint["static_class_bias"]=class_bias_certificate(torch.cat([x[3] for x in pieces]),torch.cat([x[1] for x in pieces]),[i for x in pieces for i in x[2]])
    runs.append({"seed":seed,"subsets":by_subset,"combined_fit_stop":joint})
for rel,digest in tables_hash.items():
    assert _file_sha256(base/rel)==digest
for name in data:
    assert load_subset(root,name,protocol["input_sha256"])["hashes"]==data[name]["hashes"]
assert _file_sha256(summary_path)==summary_hash
result={"status":"POSTHOC_READONLY_CLASSIFICATION_EQUIVALENCE_AUDIT",
        "scope":"Only registered exposed Fold 0 fit/stop probability tables; no new calibrated arm or performance selection.",
        "global_bias_family":"argmax(log(p) + delta*e_target); fixed non-target ratios",
        "target_index":0,"tie_policy":"target index 0 wins ties",
        "cross_seed_predictions_identical":all(pattern_lists[(20261002,short)]==pattern_lists[(seed,short)] for short in ("fit","stop") for seed in (20261003,20261004)),
        "criterion":"max(theta over existing target winners) < min(theta over existing non-target winners)",
        "true_labels_used_in_constraint_or_witness_selection":False,
        "images_loaded":False,"optimizer_steps":0,"new_calibration_training":False,
        "runs":runs,"input_sha256":inputs_hash,"prediction_table_sha256":tables_hash,
        "reference_summary_sha256":summary_hash,"registered_protocol_sha256":_file_sha256(protocol_path),
        "source_tables_unchanged":True,
        "interpretation":"Feasibility concerns finite classification patterns only; not risk equivalence, independent efficacy, or a deployed offset."}
saved=root/"outputs/theory/abmp_global_bias_equivalence_v1/diagnostic_summary.json"
if saved.exists():
    assert json.loads(saved.read_text(encoding="utf-8"))==result
print(json.dumps(result,indent=2))

'@
```

结果：`outputs/theory/abmp_global_bias_equivalence_v1/diagnostic_summary.json`。理论与解释：`docs/experiment_records/2026-10-04_global_bias_pattern_audit.md`。
