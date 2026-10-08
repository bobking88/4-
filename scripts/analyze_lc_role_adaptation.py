"""Locked LC-RFA evidence and inference replay; training replay is separately gated."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import torch

from lc_role_adapter import ARMS, RoleResidualModel, anchor_geometry
from lc_role_audit import audit_theory
from lc_role_data import SUBSETS, load_cache_subset, scoped_file, transform_subset
from lc_role_training import SEEDS, CHECKPOINTS, _linear_prediction, matched_budget_candidates, select_oof
import run_lc_role_adaptation as runner


def summarize_predictions(batch: dict, logq: torch.Tensor, anchor_logq: torch.Tensor) -> dict:
    anchor_geometry(logq)
    anchor_geometry(anchor_logq)
    labels = batch["labels"]
    n = len(labels)
    if (labels.shape != (n,) or labels.dtype != torch.long or labels.device.type != "cpu"
            or n != len(logq) or logq.shape != anchor_logq.shape
            or len(batch["records"]) != n or bool(((labels < 0) | (labels > 3)).any())):
        raise ValueError("Aligned labels, identities, and probabilities required.")
    q = logq.exp()
    prediction, old = logq.argmax(1), anchor_logq.argmax(1)
    confusion = torch.bincount(4*labels+prediction, minlength=16).reshape(4,4)
    per_class = []
    for k in range(4):
        tp, actual, predicted = int(confusion[k,k]), int(confusion[k].sum()), int(confusion[:,k].sum())
        precision, recall = tp/predicted if predicted else 0., tp/actual if actual else 0.
        per_class.append(dict(class_id=k, precision=precision, recall=recall,
                              f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                              true_count=actual, predicted_count=predicted,
                              precision_zero_denominator=predicted == 0, recall_zero_denominator=actual == 0))
    non_target = torch.logsumexp(logq[:,1:], 1)
    binary = -torch.where(labels == 0, logq[:,0], non_target)
    selected = logq[torch.arange(n),labels]
    conditional = torch.where(labels == 0, torch.zeros_like(selected), -(selected-non_target))
    confidence = q.max(1).values
    bins = (confidence*15).floor().long().clamp(max=14)
    correct = prediction == labels
    ece, rows = 0., []
    for k in range(15):
        mask = bins == k
        count = int(mask.sum())
        accuracy = float(correct[mask].double().mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count/n*abs(accuracy-mean_confidence)
        rows.append(dict(bin=k, lower=k/15, upper=(k+1)/15, count=count,
                         accuracy=accuracy, confidence=mean_confidence, right_closed=k == 14))
    new = int(((labels == 0)&(prediction == 0)&(old != 0)).sum())
    lost = int(((labels == 0)&(prediction != 0)&(old == 0)).sum())
    result = dict(row_count=n, accuracy=float(correct.double().mean()),
        macro_f1=sum(r["f1"] for r in per_class)/4, per_class=per_class,
        confusion_matrix=confusion.tolist(), confusion_direction="rows=true, columns=predicted",
        nll=float(-selected.mean()), binary_nll=float(binary.mean()),
        conditional_non_target_nll=float(conditional.mean()),
        brier=float((q-torch.nn.functional.one_hot(labels,4)).square().sum(1).mean()),
        ece=ece, ece_bins=rows, target_precision=per_class[0]["precision"],
        target_recall=per_class[0]["recall"], target_f1=per_class[0]["f1"],
        target_precision_zero_denominator=per_class[0]["precision_zero_denominator"],
        target_miss_rate=1-per_class[0]["recall"] if per_class[0]["true_count"] else None,
        correct_target_count=int(confusion[0,0]), new_correct_target_count=new,
        lost_correct_target_count=lost, net_correct_target_count=new-lost,
        anchor_high_confidence_error_count=int(((anchor_logq.exp().max(1).values >= .9)&(old != labels)).sum()))
    for k, name in ((1,"ti"),(2,"gangue"),(3,"metallic")):
        denominator = int(confusion[k].sum())
        result.update({name+"_false_target_count":int(confusion[k,0]),
                       name+"_false_target_denominator":denominator,
                       name+"_false_target_rate":int(confusion[k,0])/denominator if denominator else None})
    return result


def screen_development(results: list[dict], controls: list[dict]) -> dict:
    expected = {("R1",s) for s in SEEDS}
    expected |= {(a,s) for a in ("T0","H0","F0","S0","S1","R0","P","E","S1_matched","U1")
                 for s in ((0,) if a in ("T0","P","E") else SEEDS)}
    rows = results+controls
    keys = [(r["arm"],r["seed"]) for r in rows]
    reasons = []
    if len(keys) != len(set(keys)) or set(keys) != expected:
        reasons.append("MISSING_DUPLICATE_OR_UNREGISTERED_GROUP")
    lookup = dict(zip(keys,rows))
    for row in rows:
        if row.get("status") not in ("FIT_SELECTED","CONVERGED","ANCHOR"):
            reasons.append(f"INVALID_OR_TRUNCATED:{row['arm']}:{row['seed']}")
        if row.get("theory_violation_count",0) != 0:
            reasons.append(f"THEORY_VIOLATION:{row['arm']}:{row['seed']}")
        m = row.get("metrics",{})
        if any(not isinstance(m.get(k),(int,float)) or not math.isfinite(m[k]) for k in
               ("nll","correct_target_count","ti_false_target_count","metallic_false_target_count","net_correct_target_count")):
            reasons.append(f"INVALID_METRICS:{row['arm']}:{row['seed']}")
    per_seed, dominated = [], []
    if not reasons:
        for seed in SEEDS:
            r = lookup["R1",seed]["metrics"]
            anchor = lookup["T0",0]["metrics"]
            strong = [(a,lookup[a,0 if a in ("T0","P","E") else seed]["metrics"])
                      for a in ("T0","H0","F0","S0","S1","R0","P","E")]
            strongest_arm, strongest = min(strong,key=lambda pair:(pair[1]["nll"],pair[0]))
            checks = dict(nll_vs_anchor=r["nll"] <= anchor["nll"]-.005,
                          nll_vs_strongest=r["nll"] <= strongest["nll"]-.005,
                          net_correct_target=r["net_correct_target_count"] >= 1,
                          ti_invasion=r["ti_false_target_count"] <= anchor["ti_false_target_count"]+1,
                          metallic_invasion=r["metallic_false_target_count"] <= anchor["metallic_false_target_count"]+1,
                          matched_s1=r["nll"] < lookup["S1_matched",seed]["metrics"]["nll"])
            per_seed.append(dict(seed=seed, strongest_control=strongest_arm, checks=checks, passed=all(checks.values())))
            u = lookup["U1",seed]["metrics"]
            pairs = [(u["nll"],r["nll"]),( -u["correct_target_count"],-r["correct_target_count"]),
                     (u["ti_false_target_count"],r["ti_false_target_count"]),
                     (u["metallic_false_target_count"],r["metallic_false_target_count"])]
            if all(a <= b for a,b in pairs) and any(a < b for a,b in pairs):
                dominated.append(seed)
    passed = not reasons and bool(per_seed) and all(row["passed"] for row in per_seed)
    return dict(passed=passed, invalid_reasons=reasons, per_seed=per_seed, u1_dominating_seeds=dominated,
                bounded_net_benefit_supported=passed and not dominated,
                scope="repeatedly exposed development stop only; no independent confirmation or significance claim")


def _same(actual, expected, where):
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual,expected)):
            raise ValueError("REPLAY_MISMATCH: "+where)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual) != set(expected):
            raise ValueError("REPLAY_MISMATCH: "+where)
        for key in expected:
            _same(actual[key],expected[key],where+"/"+str(key))
    elif isinstance(expected,(list,tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError("REPLAY_MISMATCH: "+where)
        for i,value in enumerate(expected):
            _same(actual[i],value,where+"/"+str(i))
    elif actual != expected:
        raise ValueError("REPLAY_MISMATCH: "+where)


def _load(path):
    return torch.load(path,map_location="cpu",weights_only=True)


def _prepared(output, path, raw):
    prepared = _load(scoped_file(output,path))
    by_id = {r["image_id"]:i for i,r in enumerate(raw["records"])}
    train_ids = prepared["state"]["train_ids"]
    if train_ids != [r["image_id"] for r in prepared["train"]["records"]]:
        raise ValueError("Preprocessing train identity mismatch.")
    for part in ("train","validation"):
        if prepared.get(part) is None:
            continue
        ids = [r["image_id"] for r in prepared[part]["records"]]
        if any(i not in by_id for i in ids):
            raise ValueError("Preprocessing row outside locked fit.")
        replay = transform_subset(raw,prepared["state"],torch.tensor([by_id[i] for i in ids]))
        _same(replay,prepared[part],"preprocessing/"+part)
    return prepared


def _model_prediction(batch, artifact, arm, seed, config):
    if arm in ("P","E"):
        if (artifact.get("converged") is not True or not 0 <= artifact.get("gradient_l2",float("inf")) <= 1e-7
                or not 0 <= artifact.get("gap_upper_bound",float("inf")) <= 1e-8):
            raise ValueError("Unconverged main convex control.")
        return _linear_prediction(batch,arm,artifact["theta"]), None, None
    model = RoleResidualModel("S1" if arm == "S1_matched" else arm,seed=seed,
                              alpha_max=config["alpha_max"],beta_max=config["beta_max"])
    model.load_state_dict(artifact["state_dict"],strict=True)
    model.eval()
    with torch.no_grad():
        result = model(batch["h"],batch["e"],batch["log_anchor"])
    return result["logq"], model, result


def replay_oof(output, lock, raw, protocol):
    schedules = {j["job_id"]:j for j in runner.build_jobs(protocol) if j["phase"] == "cv"}
    if set(lock["cv_paths"]) != set(schedules):
        raise ValueError("Incomplete registered CV grid.")
    assignment = json.loads((output/"inner_assignment.json").read_text(encoding="utf-8"))["rows"]
    candidates, prepared_cache = [], {}
    for job_id,path in lock["cv_paths"].items():
        schedule = schedules[job_id]
        artifact_path = scoped_file(output,path)
        packet = _load(artifact_path.with_name("packet.pt"))
        arm,seed,config,fold = (schedule[k] for k in ("arm","seed","config","inner_fold"))
        if any(packet[k] != schedule[k] for k in ("arm","seed","config")) or packet["final"]:
            raise ValueError("CV packet differs from registered schedule.")
        prep_relative = packet["prepared_relative_path"]
        if prep_relative not in lock["artifacts"]:
            raise ValueError("Unpinned CV preprocessing.")
        if prep_relative not in prepared_cache:
            prepared_cache[prep_relative] = _prepared(output,prep_relative,raw)
        data = prepared_cache[prep_relative]
        expected_train = [r["image_id"] for r in assignment if r["inner_fold"] != fold]
        expected_val = [r["image_id"] for r in assignment if r["inner_fold"] == fold]
        if (data["state"]["train_ids"] != expected_train
                or [r["image_id"] for r in data["validation"]["records"]] != expected_val):
            raise ValueError("CV preprocessing leaks the held-out inner fold.")
        fitted = _load(artifact_path)
        convex = arm in ("P","E")
        if not convex and (set(fitted["checkpoints"]) != set(CHECKPOINTS) or fitted["optimizer_updates"] != 400):
            raise ValueError("Incomplete CV trajectory.")
        snapshots = {0:fitted} if convex else fitted["checkpoints"]
        for step,snapshot in snapshots.items():
            for subset in ("train","validation"):
                logq,_,_ = _model_prediction(data[subset],snapshot,arm,seed,config)
                _same(logq,snapshot[subset+"_logq"],job_id+"/"+str(step)+"/"+subset)
            candidates.append(dict(arm=arm,seed=seed,config=config,inner_fold=fold,updates=step,
                image_ids=expected_val,logq=logq,source="fit_inner_oof",**({"certificate":fitted} if convex else {})))
    saved = _load(output/"oof_predictions.pt")
    # Certificates include optimization evidence; compare predictions and selection keys separately.
    fields = ("arm","seed","config","inner_fold","updates","image_ids","logq","source")
    _same([{k:r[k] for k in fields} for r in candidates], [{k:r[k] for k in fields} for r in saved],"OOF predictions")
    selections = {arm:select_oof([r for r in candidates if r["arm"] == arm],assignment) for arm in ARMS+("P","E")}
    selections["S1_matched"] = select_oof(matched_budget_candidates(candidates,selections["R1"]),assignment)
    _same(selections,lock["selections"],"OOF selections")
    return len(candidates)


def _prediction_rows(batch, logq, arm, seed, diagnostics=None, result=None):
    rows = []
    anchor = batch["log_anchor"]
    for i,record in enumerate(batch["records"]):
        y = int(batch["labels"][i])
        row = dict(image_id=record["image_id"],split_group_id=record["split_group_id"],
            relative_path=record["relative_path"],arm=arm,seed=seed,class_id=y,predicted_id=int(logq[i].argmax()),
            anchor_predicted_id=int(anchor[i].argmax()),
            anchor_high_confidence_error=bool(anchor[i].exp().max() >= .90 and int(anchor[i].argmax()) != y))
        for k in range(4):
            row.update({f"logq_{k}":float(logq[i,k]),f"q_{k}":float(logq[i,k].exp()),
                        f"anchor_logq_{k}":float(anchor[i,k]),f"anchor_q_{k}":float(anchor[i,k].exp())})
        if diagnostics is not None:
            row.update({k:json.dumps(v,separators=(",",":")) if isinstance(v,list) else v
                        for k,v in diagnostics["rows"][i].items() if k not in ("image_id","class_id")})
            row["tanh_saturation_count"] = int((result["scores"][i].tanh().abs() >= .95).sum())
            row["adapter_tanh_saturation_counts"] = json.dumps(
                (result["delta"][i].abs()*math.sqrt(64) >= .95).sum(1).tolist(),separators=(",",":"))
        rows.append(row)
    return rows


def _final_evidence(output, lock, raw_fit, raw_stop):
    prepared = _prepared(output,lock["full_preprocessing_path"],raw_fit)
    if prepared["state"]["train_ids"] != [r["image_id"] for r in raw_fit["records"]]:
        raise ValueError("Final preprocessing did not use full fit exactly once.")
    fit = prepared["train"]
    stop = transform_subset(raw_stop,prepared["state"])
    for field in ("image_id","split_group_id"):
        if {r[field] for r in fit["records"]}&{r[field] for r in stop["records"]}:
            raise ValueError("Fit/stop identity leakage.")
    batches = dict(fit=fit,stop=stop)
    tensors, tables, summaries = {}, {}, []
    for final in lock["finals"]+ [dict(arm="T0",seed=0,state_path=None)]:
        arm,seed = final["arm"],final["seed"]
        selected = None if arm == "T0" else lock["selections"][arm]
        artifact = None if arm == "T0" else _load(scoped_file(output,final["state_path"]))
        if artifact is not None and arm not in ("P","E"):
            if (artifact.get("arm") != ("S1" if arm == "S1_matched" else arm)
                    or artifact.get("seed") != seed or artifact.get("config") != selected["config"]
                    or artifact.get("optimizer_updates") != selected["updates"]):
                raise ValueError("Final state differs from locked selection.")
        for subset,batch in batches.items():
            logq,model,result = (batch["log_anchor"],None,None) if arm == "T0" else _model_prediction(batch,artifact,arm,seed,selected["config"])
            if subset == "fit" and artifact is not None:
                _same(logq,artifact["train_logq"],"final "+arm+"/"+str(seed))
            audit = None if model is None else audit_theory(model,batch,result)
            name = f"{subset}-{arm}-{seed}"
            tensors[name] = logq
            tables[name] = _prediction_rows(batch,logq,arm,seed,audit,result)
            summaries.append(dict(subset=subset,arm=arm,seed=seed,
                status="ANCHOR" if arm == "T0" else "CONVERGED" if arm in ("P","E") else selected["status"],
                config=None if selected is None else selected["config"],
                metrics=summarize_predictions(batch,logq,batch["log_anchor"]),
                theory_violation_count=0 if audit is None else audit["violation_count"],
                theory_summary=None if audit is None else {k:v for k,v in audit.items() if k != "rows"}))
    stop_rows = [r for r in summaries if r["subset"] == "stop"]
    summary = dict(status="DEVELOPMENT_EVALUATED",final_group_count=len(lock["finals"]),groups=summaries,
        screen=screen_development([r for r in stop_rows if r["arm"] == "R1"],[r for r in stop_rows if r["arm"] != "R1"]),
        high_confidence_threshold=.90,tanh_saturation_threshold=.95,
        descriptive_thresholds_not_used_for_selection=True,
        scope="conditional on existing frozen expert; visual proxy; development not independent validation")
    return summary,tensors,tables


def _inputs(root,output,protocol):
    runner.validate_protocol(protocol,"evaluate")
    snapshot = runner.check_sources(root,protocol)
    lock = runner.read_selection_lock(output,protocol)
    if snapshot != lock["source_snapshot"]:
        raise ValueError("Sources differ from locked snapshot.")
    fit = load_cache_subset(root,protocol,SUBSETS[0])
    _same(fit,_load(output/"fit_raw.pt"),"original fit cache")
    return lock,fit


def evaluate_locked(root: Path, protocol: dict, output: Path) -> dict:
    lock,fit = _inputs(root,output,protocol)
    evaluation = output/"evaluation"
    if (evaluation/"evidence.lock.json").exists() or (evaluation/"predictions").exists():
        raise FileExistsError("Only one locked stop evidence export is permitted.")
    stop = load_cache_subset(root,protocol,SUBSETS[1])
    summary,tensors,tables = _final_evidence(output,lock,fit,stop)
    folder = evaluation/"predictions"
    folder.mkdir()
    for name,rows in tables.items():
        runner.write_csv(folder/(name+".csv"),rows)
    torch.save(tensors,evaluation/"final_predictions.pt")
    runner._json(evaluation/"analysis_summary.json",summary,exclusive=True)
    artifacts = {p.relative_to(evaluation).as_posix():runner.file_digest(p)
                 for p in [evaluation/"analysis_summary.json",evaluation/"final_predictions.pt",*folder.glob("*.csv")]}
    runner._json(evaluation/"evidence.lock.json",dict(selection_lock_sha256=runner.file_digest(output/"selection.lock.json"),
        artifacts=artifacts),exclusive=True)
    return summary


def _authorize_training_replay(root,output,path):
    if not path.is_file():
        raise ValueError("Independent training replay authorization file required.")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    approval = protocol.get("training_replay_authorization") or {}
    if (protocol.get("replay_status") != "APPROVED_TRAINING_REPLAY"
            or approval.get("scope") != "EXACT_LC_RFA_TRAINING_REPLAY"
            or approval.get("max_fits") != 422 or approval.get("max_seconds") != 21600
            or not approval.get("human_text") or approval["human_text"] == "认可计划，原生实施代码与合成验收"
            or protocol.get("parent_selection_lock_sha256") != runner.file_digest(output/"selection.lock.json")):
        raise ValueError("Independent bounded training replay authorization required.")
    runner._registered(root,path,protocol)
    return protocol


def _training_replay(root,output,registered,original,lock):
    comparable = {k:v for k,v in registered.items() if k not in
                  ("replay_status","training_replay_authorization","parent_selection_lock_sha256")}
    _same(comparable,original,"replay protocol unchanged search space")
    target = output/"training_replay"
    target.mkdir(exist_ok=False)
    runner._json(target/"registered_protocol.json",original,exclusive=True)
    started = time.monotonic()
    try:
        replay = runner._fit_stage(root,original,target,started,lock["source_snapshot"])
        _same(replay["selections"],lock["selections"],"training replay selections")
        def compare_artifact(new,old,label):
            ignored = {"worker_compute_seconds","elapsed_seconds","elapsed","seconds"}
            a,b = _load(new),_load(old)
            _same({k:v for k,v in a.items() if k not in ignored},{k:v for k,v in b.items() if k not in ignored},label)
        compare_artifact(scoped_file(target,replay["full_preprocessing_path"]),scoped_file(output,lock["full_preprocessing_path"]),"full preprocessing replay")
        for job_id,path in lock["cv_paths"].items():
            compare_artifact(scoped_file(target,replay["cv_paths"][job_id]),scoped_file(output,path),"CV training replay/"+job_id)
        final_lookup = {(r["arm"],r["seed"]):r for r in replay["finals"]}
        for row in lock["finals"]:
            other = final_lookup[row["arm"],row["seed"]]
            compare_artifact(scoped_file(target,other["state_path"]),scoped_file(output,row["state_path"]),"final training replay")
        result = dict(status="VERIFIED_TRAINING_REPLAY",training_replayed=True,stop_reselected=False,
                      actual_fit_count=replay["actual_fit_count"],consumed_seconds=replay["consumed_seconds"])
        runner._json(target/"verification.json",result,exclusive=True)
        return result
    except Exception as error:
        runner._json(target/"verification.json",dict(status="REPLAY_MISMATCH",error=str(error)),exclusive=True)
        raise


def verify_delivery(root: Path, output: Path, *, training_replay_protocol: Path | None = None) -> dict:
    authorization = None if training_replay_protocol is None else _authorize_training_replay(root,output,training_replay_protocol)
    protocol = json.loads((output/"registered_protocol.json").read_text(encoding="utf-8"))
    lock,fit = _inputs(root,output,protocol)
    evaluation = output/"evaluation"
    evidence = json.loads((evaluation/"evidence.lock.json").read_text(encoding="utf-8"))
    if evidence["selection_lock_sha256"] != runner.file_digest(output/"selection.lock.json"):
        raise ValueError("Evaluation selection identity changed.")
    expected_files = {"analysis_summary.json","final_predictions.pt"}|{
        f"predictions/{s}-{a}-{seed}.csv" for s in ("fit","stop")
        for a in runner.SELECTED_ARMS+("T0",) for seed in ((0,) if a in ("T0","P","E") else SEEDS)}
    if set(evidence["artifacts"]) != expected_files:
        raise ValueError("Incomplete evaluation evidence.")
    for relative,digest in evidence["artifacts"].items():
        if runner.file_digest(scoped_file(evaluation,relative)) != digest:
            raise ValueError("Evaluation prediction or summary changed/tampered: "+relative)
    candidate_count = replay_oof(output,lock,fit,protocol)
    stop = load_cache_subset(root,protocol,SUBSETS[1])
    summary,tensors,tables = _final_evidence(output,lock,fit,stop)
    _same(tensors,_load(evaluation/"final_predictions.pt"),"final predictions")
    _same(summary,json.loads((evaluation/"analysis_summary.json").read_text(encoding="utf-8")),"metrics/screen/theory")
    for name,rows in tables.items():
        with (evaluation/"predictions"/(name+".csv")).open(encoding="utf-8",newline="") as stream:
            saved = list(csv.DictReader(stream))
        expected = [{k:"" if v is None else str(v) for k,v in r.items()} for r in rows]
        _same(saved,expected,"prediction CSV/"+name)
    if authorization is not None:
        return _training_replay(root,output,authorization,protocol,lock)
    return dict(status="VERIFIED_INFERENCE_ONLY",training_replayed=False,oof_candidate_count=candidate_count,
                final_group_count=len(lock["finals"]),preprocessing="frozen fit-state application replay",
                tolerance="exact dtype/shape/tensor equality in the same environment; no 5% allowance",
                histories_and_gradients="hash-integrity checked; optimizer authenticity requires authorized training replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--replay-training",action="store_true")
    parser.add_argument("--replay-protocol",type=Path)
    args = parser.parse_args()
    if not args.verify or args.replay_training != (args.replay_protocol is not None):
        parser.error("--verify required; training replay needs both explicit replay flags.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(verify_delivery(runner.ROOT,args.output_dir,training_replay_protocol=args.replay_protocol),ensure_ascii=False))


if __name__ == "__main__":
    main()
