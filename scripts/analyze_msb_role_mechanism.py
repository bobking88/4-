"""Paired, descriptive MSB evidence and supervised saved-state verification."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import csv
import json
import math
from pathlib import Path
import torch

from msb_role_audit import audit_forward_contract
from msb_role_controls import GAP_TOLERANCE, MU, convex_objective, control_forward, make_convex_control
from msb_role_data import COUNTS, file_digest, identity_digest, load_prepared_fold, validate_fold_collection
from msb_role_model import CONVEX_UNITS, MSBRoleModel, reference_digest
from msb_role_training import CHECKPOINTS, PRIMARY, SEEDS, _input_digest, build_fit_matrix, validate_matrix_results

REFERENCES = ("SHARED-BIL", "CONCAT-MLP", "BIAS3-CVX", "LIN-UNB", "CLASSWISE4", "REPAIR4-COMP", "T0")
_CONTEXT = ContextVar("msb_analysis_context", default=None)


def replay_budget():
    jobs = build_fit_matrix()
    neural = sum(row["kind"]=="trajectory" for row in jobs)*len(CHECKPOINTS)
    convex = sum(row["kind"]=="convex" for row in jobs)
    masks = sum(row["kind"]=="trajectory" and row["unit"]=="MAIN" for row in jobs)*4
    return dict(neural_checkpoints=neural,convex_validation=convex,primary_role_masks=masks,
                total_validation=neural+convex+masks,train_certificates=convex)


def _equal(actual, expected, label):
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or actual.shape != expected.shape or actual.dtype != expected.dtype
                or not torch.isfinite(actual).all()
                or not torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)):
            raise ValueError(f"Saved evidence mismatch: {label}")
    elif isinstance(expected, dict):
        if not isinstance(actual,dict) or set(actual) != set(expected):
            raise ValueError(f"Evidence keys differ: {label}")
        for key in expected: _equal(actual[key],expected[key],label+"/"+key)
    elif isinstance(expected,(tuple,list)):
        if not isinstance(actual,(tuple,list)) or len(actual) != len(expected):
            raise ValueError(f"Evidence list differs: {label}")
        for i,(a,e) in enumerate(zip(actual,expected)): _equal(a,e,label+f"/{i}")
    elif actual != expected:
        raise ValueError(f"Evidence value differs: {label}")


def _rows(records):
    if not isinstance(records,list) or len(records) != 340:
        raise ValueError("OOF evaluation requires all340 fixed fit groups.")
    ids, groups, labels, probabilities = [], [], [], []
    for row in records:
        ids.append(row.get("image_id")); groups.append(row.get("split_group_id"))
        if (not isinstance(ids[-1],str) or not ids[-1] or not isinstance(groups[-1],str)
                or not groups[-1] or row.get("tc_subset") != "projector_fit"
                or str(row.get("four_class_id")) not in ("0","1","2","3")):
            raise ValueError("OOF row identity or scope invalid.")
        labels.append(int(row["four_class_id"]))
        probabilities.append(row.get("probabilities"))
    if len(set(ids)) != 340 or len(set(groups)) != 340:
        raise ValueError("OOF duplicate identities or groups.")
    y = torch.tensor(labels,dtype=torch.long)
    p = torch.tensor(probabilities,dtype=torch.float64)
    if (torch.bincount(y,minlength=4).tolist() != COUNTS or p.shape != (340,4)
            or not torch.isfinite(p).all() or bool((p<0).any()) or bool((p>1).any())
            or not torch.allclose(p.sum(1),torch.ones(340,dtype=torch.float64),atol=1e-10,rtol=0)):
        raise ValueError("OOF class totals or normalized probabilities invalid.")
    return y,p


def summarize_oof(records: list[dict]) -> dict:
    y,p = _rows(records)
    pred = p.argmax(1)
    confusion = torch.bincount(y*4+pred,minlength=16).reshape(4,4)
    counts = confusion.sum(1)
    tp = confusion.diag().to(torch.float64)
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / counts
    f1 = 2*tp / (counts+confusion.sum(0)).clamp_min(1)
    confidence = p.max(1).values
    correct = (pred == y).to(torch.float64)
    bins, ece = [], 0.
    for k in range(15):
        left,right = k/15,(k+1)/15
        selected = (confidence >= left) & ((confidence < right) if k<14 else (confidence<=1))
        n = int(selected.sum())
        accuracy = float(correct[selected].mean()) if n else None
        mean_confidence = float(confidence[selected].mean()) if n else None
        if n: ece += n/340*abs(accuracy-mean_confidence)
        bins.append(dict(left=left,right=right,count=n,accuracy=accuracy,confidence=mean_confidence))
    true_p = p[torch.arange(340),y]
    nll = float(-true_p.log().mean())
    return dict(n=340,counts=COUNTS.copy(),confusion=confusion.tolist(),accuracy=float(correct.mean()),
                macro_f1=float(f1.mean()),target_precision=float(precision[0]),target_recall=float(recall[0]),
                target_f1=float(f1[0]),target_correct=int(tp[0]),target_missed=66-int(tp[0]),
                false_target_counts=confusion[1:,0].tolist(),
                false_target_rates=(confusion[1:,0]/counts[1:]).tolist(),nll=nll,
                brier=float((p-torch.nn.functional.one_hot(y,4)).square().sum(1).mean()),
                ece15=ece,ece_bins=bins)


def paired_comparison(main: list[dict], other: list[dict]) -> dict:
    ym,pm = _rows(main); yo,po = _rows(other)
    for a,b in zip(main,other):
        if any(str(a[k]) != str(b[k]) for k in ("image_id","split_group_id","four_class_id")):
            raise ValueError("Paired row order/identity mismatch.")
    _equal(ym,yo,"paired labels")
    m,o = summarize_oof(main),summarize_oof(other)
    target = ym == 0
    corrected = target & (pm.argmax(1)==0) & (po.argmax(1)!=0)
    worsened = target & (pm.argmax(1)!=0) & (po.argmax(1)==0)
    return dict(target_net_corrections=int(corrected.sum()-worsened.sum()),
                target_corrected_ids=[main[i]["image_id"] for i in torch.where(corrected)[0].tolist()],
                target_worsened_ids=[main[i]["image_id"] for i in torch.where(worsened)[0].tolist()],
                false_target_delta=[a-b for a,b in zip(m["false_target_counts"],o["false_target_counts"])],
                macro_f1_delta=m["macro_f1"]-o["macro_f1"],nll_delta=m["nll"]-o["nll"],
                denominator_target=66,denominators_negative=COUNTS[1:].copy())


def _passes(comparison):
    if (comparison.get("denominator_target") != 66 or comparison.get("denominators_negative") != COUNTS[1:]
            or len(comparison.get("false_target_delta",[])) != 3):
        raise ValueError("Fixed paired denominators missing.")
    return (comparison["target_net_corrections"] >= 1
            and all(v<=0 for v in comparison["false_target_delta"])
            and math.isfinite(comparison["macro_f1_delta"]) and comparison["macro_f1_delta"] >= -1e-12)


def mechanism_gate(summaries: dict) -> dict:
    certificates = summaries.get("convex_certificates",[])
    expected = {(unit,fold) for unit in CONVEX_UNITS for fold in range(3)}
    if (len(certificates) != 12 or {(c.get("unit"),c.get("fold")) for c in certificates} != expected
            or any(c.get("converged") is not True or c.get("mu") != MU
                   or not isinstance(c.get("gap_upper"),(float,int))
                   or not 0<=c["gap_upper"]<=GAP_TOLERANCE for c in certificates)):
        raise ValueError("All12 converged convex certificates are required, not weak-winner comparisons.")
    comparisons = summaries.get("comparisons",{})
    seeds = {str(seed) for seed in SEEDS}
    if summaries.get("primary_checkpoint") != PRIMARY or set(comparisons) != set(REFERENCES):
        raise ValueError("Only the fixed100-update, seven-reference gate exists.")
    for by_seed in comparisons.values():
        if set(by_seed) != seeds: raise ValueError("All three seeds are required for every reference.")
    interaction = summaries.get("interaction_comparisons",{})
    if set(interaction) != seeds: raise ValueError("ADD comparison missing.")
    failures = [dict(reference=u,seed=s) for u,by_seed in comparisons.items()
                for s,comparison in by_seed.items() if not _passes(comparison)]
    passed = not failures
    return dict(decision="EXPLORATORY_ROLE_GATE_PASSED" if passed else "NO_ROLE_PROMOTION",
                primary_checkpoint=PRIMARY,failed_comparisons=failures,
                explicit_interaction_supported=passed and all(_passes(c) for c in interaction.values()),
                statistical_significance_claim=False,independent_confirmation=False,
                convex_reuse="One deterministic optimum per fold; not three independent seed fits.",
                causal_mechanism_claim=False)


def verify_checkpoint(unit, seed, point, validation, initial_reference, *, check_masks=False, deadline_check=lambda:None):
    deadline_check()
    model = MSBRoleModel(unit,seed=seed)
    expected_reference = reference_digest(model)
    if initial_reference != expected_reference or point.get("reference_digest") != expected_reference:
        raise ValueError("Reference identity differs from the fixed initialization.")
    if point["completed_updates"] == 0:
        _equal(point["model_state"],model.state_dict(),"initial checkpoint")
    model.load_state_dict(point["model_state"],strict=True)
    if reference_digest(model) != expected_reference:
        raise ValueError("Stored fixed reference changed.")
    with torch.no_grad():
        result = model(validation["h"],validation["e"],validation["log_anchor"])
        audit_forward_contract(validation,result)
        _equal(point["validation"],result,"validation replay")
        count = 1
        if check_masks:
            for off in point.get("role_off",[]):
                deadline_check()
                mask = tuple(off["mask"])
                forward = model(validation["h"],validation["e"],validation["log_anchor"],role_mask=mask)
                audit_forward_contract(validation,forward)
                _equal(off["forward"],forward,"same-state role-off replay")
                if not any(mask): _equal(forward["logq"],validation["log_anchor"],"all-off anchor")
                count += 1
    return count


def verify_convex_result(result, prepared, *, deadline_check=lambda:None):
    deadline_check()
    solution,train,val = result["solution"],prepared["train"],prepared["validation"]
    control = make_convex_control(result["unit"],train)
    control["coordinates"] = solution["coordinates"].detach().clone()
    _equal(solution["model_state"],control,"convex state and fold counts")
    coordinates = control["coordinates"].detach().clone().requires_grad_(True)
    loss = convex_objective(control,coordinates,train)
    gradient = torch.autograd.grad(loss,coordinates)[0]
    gap = float(gradient.square().sum()/(2*MU))
    if (solution.get("converged") is not True or solution.get("mu") != MU or not gap<=GAP_TOLERANCE
            or not math.isclose(solution["objective"],float(loss.detach()),abs_tol=1e-10,rel_tol=1e-10)
            or not math.isclose(solution["gap_upper"],gap,abs_tol=1e-15,rel_tol=1e-8)):
        raise ValueError("Recomputed convex objective/certificate invalid.")
    _equal(solution["gradient"],gradient.detach(),"convex gradient certificate")
    if (type(solution.get("iterations")) is not int or not 0<=solution["iterations"]<=200
            or type(solution.get("evaluations")) is not int or not 1<=solution["evaluations"]<=4201
            or len(solution.get("trace",[])) != solution["iterations"]+1):
        raise ValueError("Convex bounded-solver evidence incomplete.")
    deadline_check()
    with torch.no_grad():
        forward = control_forward(control,val["h"],val["e"],val["log_anchor"])
        audit_forward_contract(val,forward)
        _equal(result["validation"],forward,"convex validation replay")
    return dict(unit=result["unit"],converged=True,mu=MU,gap_upper=gap,
                train_certificate_forwards=1,validation_forwards=1)


@contextmanager
def supervised_analysis(root,output,protocol,guard,kind):
    """Bind read/replay permission to the already active worker guard, not a new run."""
    if _CONTEXT.get() is not None or kind not in ("analysis","final-verify"):
        raise RuntimeError("No nested or unknown analysis scope.")
    power = guard.evidence()
    if power.get("power_request_acquired") is not True or power.get("power_request_released"):
        raise RuntimeError("An active worker guard is required.")
    guard.check()
    token = _CONTEXT.set(dict(root=Path(root).resolve(),output=Path(output).resolve(),protocol=protocol,
                             guard=guard,kind=kind,consumed=False))
    try: yield
    finally: _CONTEXT.reset(token)


def _context(root,output,protocol,kind):
    context = _CONTEXT.get()
    if (context is None or context["root"] != Path(root).resolve() or context["output"] != Path(output).resolve()
            or context["protocol"] != protocol or context["kind"] != kind or context["consumed"]):
        raise RuntimeError("Saved evidence requires the single active supervised worker scope.")
    context["guard"].check()
    context["consumed"] = True
    return context


def _load_saved(root,output,protocol,context):
    from run_msb_role_mechanism import _path, protocol_digest, validate_protocol
    validate_protocol(root,protocol,"fit-mechanism")
    output = _path(root,output)
    if output != _path(root,protocol["output_relative"]): raise ValueError("Output identity changed.")
    state = json.loads((output/"stage_state.json").read_text(encoding="utf-8"))
    active = "analysis" if context["kind"] == "analysis" else "final_verify"
    if (state.get("status") != "RUNNING" or state.get("active_job") != active
            or state.get("stage") != "fit-mechanism" or state.get("protocol_digest") != protocol_digest(protocol)
            or state.get("actual_neural_fits") != 63 or state.get("actual_convex_fits") != 12):
        raise ValueError("All fixed jobs must complete in the published supervised attempt.")
    matrix = build_fit_matrix()
    expected_ids = [f"load_fold{fold}" for fold in range(3)]+[r["job_id"] for r in matrix]
    if context["kind"] == "final-verify": expected_ids += ["analysis"]
    jobs = state.get("jobs",[])
    if ([row.get("job_id") for row in jobs] != expected_ids
            or any(row.get("status") != "COMPLETE" or row.get("alive") or row.get("attempts") != 1
                   or row.get("supervision",{}).get("power_request_released") is not True for row in jobs)):
        raise ValueError("Incomplete, repeated, reordered or unsupervised jobs.")
    entries = []
    for schedule in matrix:
        context["guard"].check()
        relative = f"runs/{schedule['job_id']}/result.pt"
        path = _path(output,relative)
        if file_digest(path) != state["artifacts"].get(relative):
            raise ValueError("Saved fitted artifact hash mismatch before deserialization.")
        result = torch.load(path,map_location="cpu",weights_only=True)
        if (result.get("status") != "COMPLETE" or result.get("kind") != schedule["kind"]
                or result.get("worker_power",{}).get("power_request_released") is not True):
            raise ValueError("Fitted child completion evidence missing.")
        entries.append(dict(schedule=schedule,result=result))
    validate_matrix_results(entries)
    return entries


def _prediction_rows(result,forward,part="validation"):
    records,labels = result[part+"_records"],result[part+"_labels"]
    if len(records) != len(labels) or len(records) != len(forward["logq"]):
        raise ValueError("Prediction row count differs from stored labels.")
    p = forward["logq"].exp()
    rows = []
    for i,(row,y) in enumerate(zip(records,labels)):
        if str(row["four_class_id"]) != str(int(y)): raise ValueError("Saved row/label mismatch.")
        rows.append(dict(row,probabilities=p[i].tolist()))
    return rows


def _summarize_entries(entries,protocol):
    predictions,checkpoint_predictions,mask_predictions = {},{},{}
    fold_records, train_records, input_digests = {},{},{}
    certificates, optimization = [],[]
    main_train, bias_train = {},{}
    for entry in entries:
        schedule,result = entry["schedule"],entry["result"]
        unit,fold,seed = schedule["unit"],schedule["fold"],schedule["seed"]
        for part,cache in (("validation",fold_records),("train",train_records)):
            records = result[part+"_records"]
            ids = [r["image_id"] for r in records]
            if identity_digest(ids) != protocol["prepared_entries"][fold][f"registered_{part}_ids_digest"]:
                raise ValueError("Saved row order differs from registered prepared identities.")
            if fold in cache and cache[fold] != records: raise ValueError("Cross-unit fold identities changed.")
            cache[fold] = records
            key = (part,fold)
            digest = result[part+"_inputs_digest"]
            if key in input_digests and input_digests[key] != digest: raise ValueError("Cross-unit inputs changed.")
            input_digests[key] = digest
        if schedule["kind"] == "trajectory":
            if result["initial_reference_digest"] != result["final_reference_digest"]:
                raise ValueError("Trajectory reference changed.")
            for point in result["checkpoints"]:
                key = (unit,str(seed),point["completed_updates"])
                checkpoint_predictions.setdefault(key,[]).extend(_prediction_rows(result,point["validation"]))
                if point["completed_updates"] == PRIMARY:
                    predictions.setdefault(unit,{}).setdefault(str(seed),[]).extend(_prediction_rows(result,point["validation"]))
                    if unit == "MAIN":
                        masks = [tuple(row["mask"]) for row in point["role_off"]]
                        if masks != [(False,True,True),(True,False,True),(True,True,False),(False,False,False)]:
                            raise ValueError("Fixed primary same-state masks missing or changed.")
                        for off in point["role_off"]:
                            mask_key = (str(seed),str(tuple(off["mask"])))
                            mask_predictions.setdefault(mask_key,[]).extend(_prediction_rows(result,off["forward"]))
                        anchor = point["train"]["scores"]
                        if not torch.is_tensor(anchor): raise ValueError("Saved train forward missing.")
                        main_train[(fold,seed)] = point["gradients"]["losses"]["total"]
        else:
            predictions.setdefault(unit,{}).setdefault("deterministic",[]).extend(_prediction_rows(result,result["validation"]))
            sol = result["solution"]
            certificates.append(dict(unit=unit,fold=fold,converged=sol["converged"],mu=sol["mu"],gap_upper=sol["gap_upper"]))
            if unit == "BIAS3-CVX": bias_train[fold] = sol["objective"]
    all_ids = {r["image_id"] for rows in fold_records.values() for r in rows}
    if len(all_ids) != 340: raise ValueError("Three-fold validation identities overlap or omit groups.")
    for fold,rows in train_records.items():
        if {r["image_id"] for r in rows} != all_ids-{r["image_id"] for r in fold_records[fold]}:
            raise ValueError("Training fold is not the registered validation complement.")
    for by_seed in predictions.values():
        for rows in by_seed.values(): rows.sort(key=lambda r:r["image_id"])
    for rows in checkpoint_predictions.values(): rows.sort(key=lambda r:r["image_id"])
    for rows in mask_predictions.values(): rows.sort(key=lambda r:r["image_id"])
    predictions["T0"] = {str(seed):mask_predictions[(str(seed),str((False,False,False)))] for seed in SEEDS}
    summaries = {u:{s:summarize_oof(rows) for s,rows in by_seed.items()} for u,by_seed in predictions.items()}
    comparisons = {u:{str(seed):paired_comparison(predictions["MAIN"][str(seed)],
                     predictions[u]["deterministic" if u in CONVEX_UNITS else str(seed)]) for seed in SEEDS} for u in REFERENCES}
    interaction = {str(seed):paired_comparison(predictions["MAIN"][str(seed)],predictions["ADD"][str(seed)]) for seed in SEEDS}
    for (fold,seed),value in main_train.items():
        gap = float(value)-bias_train[fold]
        optimization.append(dict(fold=fold,seed=seed,main_training_objective=float(value),
                                 certified_bias_objective=bias_train[fold],difference=gap,
                                 main_above_feasible_bias=gap>1e-10))
    gate_input = dict(primary_checkpoint=PRIMARY,comparisons=comparisons,interaction_comparisons=interaction,
                      convex_certificates=certificates)
    return dict(scope="EXPOSED_FIT_MECHANISM_NOT_INDEPENDENT",delivery_verified=False,
                primary_checkpoint=PRIMARY,predictions=predictions,summaries=summaries,
                checkpoint_summaries=[dict(unit=u,seed=s,checkpoint=c,metrics=summarize_oof(rows))
                                      for (u,s,c),rows in checkpoint_predictions.items()],
                mask_summaries=[dict(seed=s,mask=m,metrics=summarize_oof(rows)) for (s,m),rows in mask_predictions.items()],
                optimization_diagnostics=optimization,gate_input=gate_input,gate=mechanism_gate(gate_input))


def collect_saved_predictions(root: Path,output: Path,protocol: dict) -> dict:
    context = _context(root,output,protocol,"analysis")
    entries = _load_saved(root,output,protocol,context)
    return _summarize_entries(entries,protocol)


def verify_saved_mechanism(root: Path,output: Path,protocol: dict) -> dict:
    context = _context(root,output,protocol,"final-verify")
    entries = _load_saved(root,output,protocol,context)
    folds = [load_prepared_fold(Path(root),entry,protocol["authorization"]) for entry in protocol["prepared_entries"]]
    validate_fold_collection(folds)
    forwards,train_checks = 0,0
    for entry in entries:
        context["guard"].check()
        schedule,result = entry["schedule"],entry["result"]
        prepared = folds[schedule["fold"]]
        for part in ("train","validation"):
            if result[part+"_inputs_digest"] != _input_digest(prepared[part]):
                raise ValueError("Saved input identity differs from the frozen prepared tensors.")
            _equal(result[part+"_records"],prepared[part]["records"],part+" records")
            _equal(result[part+"_labels"],prepared[part]["labels"],part+" labels")
        if schedule["kind"] == "trajectory":
            for point in result["checkpoints"]:
                forwards += verify_checkpoint(schedule["unit"],schedule["seed"],point,prepared["validation"],
                              result["initial_reference_digest"],
                              check_masks=schedule["unit"]=="MAIN" and point["completed_updates"]==PRIMARY,
                              deadline_check=context["guard"].check)
        else:
            verified = verify_convex_result(result,prepared,deadline_check=context["guard"].check)
            forwards += verified["validation_forwards"]
            train_checks += verified["train_certificate_forwards"]
        budget = replay_budget()
        if forwards>budget["total_validation"] or train_checks>budget["train_certificates"]:
            raise ValueError("Fixed final replay budget exceeded.")
    if (forwards,train_checks) != (budget["total_validation"],budget["train_certificates"]):
        raise ValueError("Incomplete final replay/certificate matrix.")
    evidence = _summarize_entries(entries,protocol)
    evidence.update(delivery_verified=True,verification=dict(validation_forwards=forwards,
                    train_certificate_forwards=train_checks,optimizer_replay=False,temperature_fits=0))
    return evidence


def write_mechanism_report(evidence: dict,output: Path) -> dict:
    output = Path(output)
    if output.exists(): raise FileExistsError("Mechanism report is exclusive; no overwrite or resume.")
    output.mkdir(parents=True,exist_ok=False)
    summary = {k:v for k,v in evidence.items() if k != "predictions"}
    with (output/"summary.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(summary,stream,ensure_ascii=False,allow_nan=False,indent=2)
    with (output/"predictions.csv").open("x",encoding="utf-8",newline="") as stream:
        fields = ["unit","seed","image_id","split_group_id","four_class_id","p0","p1","p2","p3"]
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for unit,by_seed in evidence.get("predictions",{}).items():
            for seed,rows in by_seed.items():
                for row in rows:
                    writer.writerow(dict(unit=unit,seed=seed,**{k:row[k] for k in fields[2:5]},
                                         **{f"p{k}":v for k,v in enumerate(row["probabilities"])}))
    with (output/"summary.md").open("x",encoding="utf-8",newline="\n") as stream:
        stream.write("# MSB mechanism evidence\n\nScope: "+evidence["scope"]+".\n\n")
        stream.write("This is descriptive development evidence, not independent confirmation or a significance test.\n\n")
        stream.write("Gate: "+evidence["gate"]["decision"]+". Delivery replay verified: "+str(evidence["delivery_verified"])+".\n")
        stream.write("Training objectives and convex optimization deficits are separate from OOF validation metrics.\n")
        stream.write("A passed role gate does not by itself establish interaction necessity, novelty, or industrial sorting performance.\n")
    artifacts = {name:file_digest(output/name) for name in ("summary.json","predictions.csv","summary.md")}
    return dict(scope=evidence["scope"],delivery_verified=evidence["delivery_verified"],gate=evidence["gate"],artifacts=artifacts)
