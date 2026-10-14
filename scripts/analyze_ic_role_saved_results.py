"""Offline algebra on saved IC-RFA validation outputs; never execute a model."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/training/ic_role_mechanism_v2/development_fold_0"
OUTPUT = ROOT / "outputs/ic_role_offline_theory_audit_v1"
UNITS = ("C-S-R", "C-S-U", "HC-S")
SEEDS = (20261002, 20261003, 20261004)
STEPS = (0, 1, 2, 5, 20, 50, 100, 200, 400)
TOL = 1e-10


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def margin(logq, labels):
    rivals = logq.clone()
    rivals[torch.arange(len(labels)), labels] = -torch.inf
    return logq[torch.arange(len(labels)), labels] - rivals.max(1).values


def envelope(anchor):
    log_non = torch.logsumexp(anchor[:, 1:], 1)
    ell = anchor[:, 0] - log_non
    logw = anchor[:, 1:] - log_non[:, None]
    b = 4 * anchor[:, 0].exp() * log_non.exp()
    alpha = beta = .5 * b
    maximum = logw.max(1).values
    log_rho = maximum - torch.logsumexp(
        torch.minimum(maximum[:, None], logw + beta[:, None]), 1)
    target = ell + alpha - log_rho
    outputs = [target]
    # Independent bounded coordinates give a closed-box oracle, not a network.
    for j in range(3):
        others = [k for k in range(3) if k != j]
        boosted = logw[:, j] + beta
        log_ry = boosted - torch.logaddexp(
            boosted, torch.logsumexp(logw[:, others], 1))
        against_target = log_ry - ell + alpha
        against_non = logw[:, j] - logw[:, others].max(1).values + beta
        outputs.append(torch.minimum(against_target, against_non))
    return dict(all_margins=torch.stack(outputs, 1), b=b, alpha=alpha, beta=beta)


def write_csv(name, rows):
    with (OUTPUT / name).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT.exists():
        raise ValueError("Exclusive offline output already exists; no overwrite/retry.")
    evidence_path = INPUT / "final_acceptance/evidence.json"
    if digest(evidence_path) != "28104aae46f7f1d7e011c8d6ff8d4ca357ceb9d03a557264984a35b80187206b":
        raise ValueError("Saved final evidence identity mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["delivery_verified"] and evidence["theory_violation_count"] == 0
    torch.set_num_threads(1)
    provenance = []

    def load_saved(path):
        provenance.append(dict(path=path.relative_to(ROOT).as_posix(),
                               size_bytes=path.stat().st_size, sha256=digest(path)))
        return torch.load(path, map_location="cpu", weights_only=True)

    prepared, row_ids, groups, reachability = {}, set(), set(), []
    for fold in range(3):
        saved = load_saved(INPUT / f"runs/prepare_fold{fold}/result.pt")
        assert saved["inner_fold"] == fold
        val = saved["validation"]
        anchor, labels, records = val["log_anchor"], val["labels"], val["records"]
        assert torch.isfinite(anchor).all() and not anchor.requires_grad
        assert torch.allclose(torch.logsumexp(anchor, 1), torch.zeros(len(labels), dtype=torch.float64), atol=TOL, rtol=0)
        geometry = envelope(anchor)
        true_envelope = geometry["all_margins"][torch.arange(len(labels)), labels]
        original_margin = margin(anchor, labels)
        for i, record in enumerate(records):
            assert record["tc_subset"] == "gate_stop_projector_fit"
            assert record["image_id"] not in row_ids and record["split_group_id"] not in groups
            assert int(record["four_class_id"]) == int(labels[i])
            row_ids.add(record["image_id"])
            groups.add(record["split_group_id"])
            reachability.append(dict(image_id=record["image_id"], fold=fold,
                class_id=int(labels[i]), anchor_prediction=int(anchor[i].argmax()),
                anchor_true_margin=float(original_margin[i]), b=float(geometry["b"][i]),
                alpha=float(geometry["alpha"][i]), beta=float(geometry["beta"][i]),
                true_class_oracle_margin=float(true_envelope[i]),
                true_class_strictly_reachable=bool(true_envelope[i] > TOL),
                true_class_boundary=bool(abs(float(true_envelope[i])) <= TOL),
                target_oracle_margin=float(geometry["all_margins"][i, 0]),
                target_strictly_reachable=bool(geometry["all_margins"][i, 0] > TOL)))
        prepared[fold] = dict(anchor=anchor, labels=labels, records=records,
                              envelope=true_envelope)
        del saved, val
    assert len(row_ids) == len(groups) == 340
    assert [sum(r["class_id"] == j for r in reachability) for j in range(4)] == [66, 130, 97, 47]

    primary_rows, aggregates, arrays, integrity = [], [], {}, []
    for unit in UNITS:
        for seed in SEEDS:
            for fold in range(3):
                saved = load_saved(INPUT / f"runs/{unit}_fold{fold}_seed{seed}/result.pt")
                batch = prepared[fold]
                labels, anchor, records = batch["labels"], batch["anchor"], batch["records"]
                assert saved["unit"] == unit and saved["seed"] == seed and saved["inner_fold"] == fold
                assert saved["validation_records"] == records and tuple(saved["checkpoints"]) == STEPS
                for step in STEPS:
                    point = saved["checkpoints"][step]
                    on = point["validation_logq"]
                    diagnostic = point["branch_diagnostics"]["validation"]
                    off = diagnostic["off_logq"]
                    removed = diagnostic["removed_branch_logq"]
                    on_margin, off_margin, anchor_margin = [margin(x, labels) for x in (on, off, anchor)]
                    assert torch.isfinite(on).all() and torch.isfinite(off).all()
                    assert torch.all(on_margin <= batch["envelope"] + TOL)
                    if step == 0:
                        assert torch.allclose(on, anchor, atol=TOL, rtol=0)
                    rows = point["bounds"]["validation"]["rows"]
                    assert [r["image_id"] for r in rows] == [r["image_id"] for r in records]
                    eta = torch.tensor([r["eta_main"] for r in rows], dtype=torch.float64)
                    assert torch.all((on_margin - off_margin).abs() <= eta + TOL)
                    key = (unit, seed, step)
                    arrays.setdefault(key, []).append(dict(fold=fold, on=on.clone(), off=off.clone(),
                        anchor=anchor, labels=labels, eta=eta, on_margin=on_margin,
                        off_margin=off_margin, anchor_margin=anchor_margin,
                        removed=removed.clone(), records=records,
                        delta_norms=diagnostic["delta_norms"].clone(),
                        branch_cosines=diagnostic["branch_cosines"].clone()))
                    if step == 100 and unit != "HC-S":
                        for i, record in enumerate(records):
                            row = dict(unit=unit, seed=seed, fold=fold, image_id=record["image_id"],
                                class_id=int(labels[i]), anchor_prediction=int(anchor[i].argmax()),
                                off_prediction=int(off[i].argmax()), on_prediction=int(on[i].argmax()),
                                anchor_true_margin=float(anchor_margin[i]), off_true_margin=float(off_margin[i]),
                                on_true_margin=float(on_margin[i]), true_class_oracle_margin=float(batch["envelope"][i]),
                                shared_true_margin_change=float(off_margin[i]-anchor_margin[i]),
                                adapter_true_margin_change=float(on_margin[i]-off_margin[i]),
                                eta_main=float(eta[i]), off_wrong_adapter_certificate=bool(off_margin[i] < -eta[i]-TOL),
                                adapter_max_log_drift=float((on[i]-off[i]).abs().max()))
                            for j in range(3):
                                row[f"removed_role{j+1}_prediction"] = int(removed[i, j].argmax())
                                row[f"removed_role{j+1}_max_log_drift"] = float((removed[i, j]-on[i]).abs().max())
                            primary_rows.append(row)
                del saved
                gc.collect()

    for (unit, seed, step), folds in arrays.items():
        combined = {key: torch.cat([f[key] for f in folds]) for key in
            ("on", "off", "anchor", "labels", "eta", "on_margin", "off_margin", "anchor_margin", "removed", "delta_norms", "branch_cosines")}
        on, off, anchor, labels = [combined[k] for k in ("on", "off", "anchor", "labels")]
        i = torch.arange(340)
        prediction, off_prediction, original = [x.argmax(1) for x in (on, off, anchor)]
        correct, off_correct, original_correct = [x == labels for x in (prediction, off_prediction, original)]
        shared_nll = float((anchor[i, labels]-off[i, labels]).mean())
        adapter_nll = float((off[i, labels]-on[i, labels]).mean())
        total_nll = float((anchor[i, labels]-on[i, labels]).mean())
        assert abs(shared_nll + adapter_nll - total_nll) <= TOL
        row = dict(unit=unit, seed=seed, step=step, row_count=340,
            nll=float(-on[i, labels].mean()), off_nll=float(-off[i, labels].mean()),
            anchor_nll=float(-anchor[i, labels].mean()), correct_count=int(correct.sum()),
            target_correct_count=int((correct & (labels == 0)).sum()),
            shared_changed_count=int((off_prediction != original).sum()),
            shared_new_correct_count=int((off_correct & ~original_correct).sum()),
            shared_lost_correct_count=int((~off_correct & original_correct).sum()),
            adapter_changed_count=int((prediction != off_prediction).sum()),
            adapter_new_correct_count=int((correct & ~off_correct).sum()),
            adapter_lost_correct_count=int((~correct & off_correct).sum()),
            off_wrong_count=int((~off_correct).sum()),
            off_wrong_adapter_certificate_count=int((combined["off_margin"] < -combined["eta"]-TOL).sum()),
            shared_nll_change=shared_nll, adapter_nll_change=adapter_nll, total_nll_change=total_nll,
            shared_mean_abs_true_margin_change=float((combined["off_margin"]-combined["anchor_margin"]).abs().mean()),
            adapter_mean_abs_true_margin_change=float((combined["on_margin"]-combined["off_margin"]).abs().mean()),
            adapter_max_log_drift=float((on-off).abs().max()))
        for j in range(3):
            row[f"removed_role{j+1}_changed_count"] = int((combined["removed"][:, j].argmax(1) != prediction).sum())
            row[f"role{j+1}_mean_delta_norm"] = float(combined["delta_norms"][:, j].mean())
            row[f"branch_cosine_pair{j+1}_mean"] = float(combined["branch_cosines"][:, j].mean())
        aggregates.append(row)
        if step == 100:
            registered = next(m for m in evidence["metrics"] if m["unit"] == unit and m["seed"] == seed)
            assert abs(row["nll"] - registered["nll"]) <= TOL
            assert row["target_correct_count"] == registered["correct_target_count"]
            integrity.append(dict(unit=unit, seed=seed, primary_saved_metric_match=True))
        arrays[(unit, seed, step)] = combined

    paired = []
    for seed in SEEDS:
        for step in STEPS:
            r, u, hc = [arrays[(unit, seed, step)] for unit in UNITS]
            labels = r["labels"]
            assert torch.equal(labels, u["labels"]) and torch.equal(labels, hc["labels"])
            i = torch.arange(340)
            rp, up = r["on"].argmax(1), u["on"].argmax(1)
            on_delta = float((u["on"][i, labels]-r["on"][i, labels]).mean())
            off_delta = float((u["off"][i, labels]-r["off"][i, labels]).mean())
            adaptation_delta = float(((r["off"]-r["on"])-(u["off"]-u["on"]))[i, labels].mean())
            assert abs(on_delta-off_delta-adaptation_delta) <= TOL
            paired.append(dict(seed=seed, step=step, row_count=340,
                r_u_changed_count=int((rp != up).sum()),
                r_u_new_correct_count=int(((rp == labels) & (up != labels)).sum()),
                r_u_lost_correct_count=int(((rp != labels) & (up == labels)).sum()),
                r_u_on_nll_difference=on_delta, r_u_off_nll_difference=off_delta,
                r_u_adapter_nll_contribution_difference=adaptation_delta,
                r_hc_changed_count=int((rp != hc["on"].argmax(1)).sum())))

    classes = []
    for j, name in enumerate(("target", "ti_negative", "gangue", "metallic_negative")):
        rows = [r for r in reachability if r["class_id"] == j]
        errors = [r for r in rows if r["anchor_prediction"] != j]
        classes.append(dict(class_id=j, class_name=name, count=len(rows), anchor_wrong_count=len(errors),
            anchor_wrong_reachable_count=sum(r["true_class_strictly_reachable"] for r in errors),
            anchor_wrong_unreachable_count=sum(not r["true_class_strictly_reachable"] for r in errors),
            strict_oracle_correct_count=sum(r["true_class_strictly_reachable"] for r in rows),
            negative_target_reachable_count=sum(r["target_strictly_reachable"] for r in rows) if j else 0))
    summary = dict(status="ANALYZED_SAVED_OUTPUTS_ONLY", baseline_commit="7f694a7f4d175365642f0c1a95da03fa0a961d9a",
        approval_question="call_ec886148d74c4ae595b130823340a25d", unique_groups=340,
        saved_prepare_files=3, saved_trajectory_files=27, checkpoints_per_trajectory=9,
        new_model_forwards=0, new_gradients=0, new_fits=0, stop_reads=0, raw_image_reads=0,
        primary_checkpoint=100, oracle_is_label_conditioned_not_a_predictor=True,
        boundaries=sum(r["true_class_boundary"] for r in reachability),
        classes=classes, primary_attribution=[r for r in aggregates if r["step"] == 100],
        primary_pairs=[r for r in paired if r["step"] == 100], arithmetic_metric_checks=integrity,
        scope="Conditional frozen-expert fit-only diagnosis; no independent confirmation or novelty verification.")
    OUTPUT.mkdir()
    write_csv("anchor_reachability.csv", reachability)
    write_csv("class_reachability.csv", classes)
    write_csv("attribution_primary.csv", primary_rows)
    write_csv("trajectory_attribution_summary.csv", aggregates)
    write_csv("paired_role_summary.csv", paired)
    for name, data in (("diagnosis.json", summary), ("input_manifest.json", provenance)):
        with (OUTPUT / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
