"""Synthetic metrics and delivery replay, with no real stop access."""
import copy
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from lc_role_adapter import RoleResidualModel
from lc_role_data import fit_preprocessing, make_inner_assignment, transform_subset
from lc_role_fixtures import raw
from lc_role_training import SEEDS, select_oof, matched_budget_candidates
import run_lc_role_adaptation as runner


def metric_batch(labels):
    return dict(labels=torch.tensor(labels), records=[dict(image_id=str(i)) for i in range(len(labels))])


def screen_rows():
    metrics = dict(nll=1., correct_target_count=10, ti_false_target_count=5,
                   metallic_false_target_count=3, net_correct_target_count=0)
    controls = [dict(arm=arm, seed=seed, status="FIT_SELECTED", metrics=copy.deepcopy(metrics),
                     theory_violation_count=0)
                for arm in ("T0", "H0", "F0", "S0", "S1", "R0", "P", "E", "S1_matched", "U1")
                for seed in ((0,) if arm in ("T0", "P", "E") else SEEDS)]
    results = [dict(arm="R1", seed=seed, status="FIT_SELECTED", theory_violation_count=0,
                    metrics=dict(metrics, nll=.98, correct_target_count=11, net_correct_target_count=1))
               for seed in SEEDS]
    return results, controls


def make_delivery(root):
    """Frozen synthetic zero-init states test inference, not training authenticity."""
    protocol = dict(runner.draft_settings(), status="APPROVED_FOR_DEVELOPMENT", source_sha256={},
        training_authorization=dict(scope="REAL_LC_RFA_FIT_AND_ONCE_STOP", max_fits=422,
            max_seconds=21600, stop_evaluations=1, human_text="synthetic test fixture only"))
    output = root/"delivery"
    output.mkdir()
    (output/"registered_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    source = raw()
    stop = raw(seed=99)
    for row in stop["records"]:
        row["image_id"] = "stop-"+row["image_id"]
        row["split_group_id"] = "stop-"+row["split_group_id"]
    torch.save(source, output/"fit_raw.pt")
    assignment = make_inner_assignment(source)
    (output/"inner_assignment.json").write_text(json.dumps(dict(rows=assignment)), encoding="utf-8")
    prepared = {}
    for fold in (0, 1, 2, "full"):
        indices = torch.tensor([i for i, row in enumerate(assignment) if fold == "full" or row["inner_fold"] != fold])
        val = None if fold == "full" else torch.tensor([i for i, row in enumerate(assignment) if row["inner_fold"] == fold])
        state = fit_preprocessing(source, indices)
        path = output/f"prepared-{fold}.pt"
        torch.save(dict(state=state, train=transform_subset(source, state, indices),
                        validation=None if val is None else transform_subset(source, state, val)), path)
        prepared[fold] = path
    candidates, cv_paths = [], {}
    for schedule in runner.build_jobs(protocol):
        if schedule["phase"] != "cv":
            continue
        folder = output/"runs"/schedule["job_id"]
        folder.mkdir(parents=True)
        arm, seed, config, fold = (schedule[k] for k in ("arm", "seed", "config", "inner_fold"))
        data = torch.load(prepared[fold], weights_only=True)
        packet = dict(kind="convex" if arm in ("P", "E") else "neural", arm=arm,
                      seed=seed, config=config, prepared_path=str(prepared[fold]),
                      prepared_relative_path=prepared[fold].relative_to(output).as_posix(), final=False)
        torch.save(packet, folder/"packet.pt")
        if arm in ("P", "E"):
            result = dict(arm=arm, theta=torch.zeros(5 if arm == "P" else 24, 4, dtype=torch.float64),
                          converged=True, gradient_l2=0., gap_upper_bound=0.,
                          train_logq=torch.log_softmax(data["train"]["log_anchor"],1),
                          validation_logq=torch.log_softmax(data["validation"]["log_anchor"],1))
            snapshots = {0: result}
        else:
            model = RoleResidualModel(arm, seed=seed, alpha_max=config["alpha_max"], beta_max=config["beta_max"])
            with torch.no_grad():
                predictions = {part:model(data[part]["h"],data[part]["e"],data[part]["log_anchor"])["logq"] for part in ("train","validation")}
            snap = dict(state_dict=model.state_dict(), train_logq=predictions["train"],
                        validation_logq=predictions["validation"])
            snapshots = {step: snap for step in runner.CHECKPOINTS}
            result = dict(arm=arm, seed=seed, config=config, optimizer_updates=400, checkpoints=snapshots)
        torch.save(result, folder/"result.pt")
        cv_paths[schedule["job_id"]] = (folder/"result.pt").relative_to(output).as_posix()
        for step, snap in snapshots.items():
            candidates.append(dict(arm=arm, seed=seed, config=config, inner_fold=fold, updates=step,
                image_ids=[r["image_id"] for r in data["validation"]["records"]],
                logq=snap["validation_logq"], source="fit_inner_oof", certificate=result if arm in ("P", "E") else {}))
    selections = {arm: select_oof([r for r in candidates if r["arm"] == arm], assignment) for arm in runner.ARMS+("P", "E")}
    selections["S1_matched"] = select_oof(matched_budget_candidates(candidates, selections["R1"]), assignment)
    torch.save(candidates, output/"oof_predictions.pt")
    finals = []
    full = torch.load(prepared["full"], weights_only=True)["train"]
    for label in runner.SELECTED_ARMS:
        arm = "S1" if label == "S1_matched" else label
        for seed in ((0,) if arm in ("P", "E") else SEEDS):
            config = selections[label]["config"]
            if arm in ("P", "E"):
                result = dict(arm=arm, theta=torch.zeros(5 if arm == "P" else 24, 4, dtype=torch.float64),
                              converged=True, gradient_l2=0., gap_upper_bound=0., train_logq=torch.log_softmax(full["log_anchor"],1))
            else:
                model = RoleResidualModel(arm, seed=seed, alpha_max=config["alpha_max"], beta_max=config["beta_max"])
                with torch.no_grad():
                    logq = model(full["h"],full["e"],full["log_anchor"])["logq"]
                result = dict(arm=arm, seed=seed, config=config, optimizer_updates=selections[label]["updates"],
                              state_dict=model.state_dict(), train_logq=logq)
            path = f"final-{label}-{seed}.pt"
            torch.save(result, output/path)
            finals.append(dict(arm=label, seed=seed, state_path=path))
    lock = dict(status="FIT_LOCKED", protocol_sha256=runner.protocol_digest(protocol), source_snapshot={},
        selections=selections, finals=finals, full_preprocessing_path=prepared["full"].relative_to(output).as_posix(),
        cv_paths=cv_paths, artifacts={p.relative_to(output).as_posix(): runner.file_digest(p) for p in output.rglob("*") if p.is_file()},
        consumed_seconds=1., actual_fit_count=395, temperature_fit_count=4)
    runner.write_selection_lock(output, lock)
    (output/"evaluation").mkdir()
    return protocol, output, source, stop


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        try:
            self.analysis = importlib.import_module("analyze_lc_role_adaptation")
        except ModuleNotFoundError:
            self.fail("Missing analysis implementation")
        torch.set_num_threads(1)

    def test_metrics_confusion_channels_and_brier(self):
        q = torch.tensor([[.6,.2,.1,.1],[.7,.1,.1,.1],[.1,.2,.6,.1],[.1,.6,.2,.1]], dtype=torch.float64)
        batch = metric_batch([0,1,2,3])
        m = self.analysis.summarize_predictions(batch, q.log(), q.log())
        self.assertEqual(m["confusion_matrix"], [[1,0,0,0],[1,0,0,0],[0,0,1,0],[0,1,0,0]])
        self.assertAlmostEqual(m["nll"], m["binary_nll"]+m["conditional_non_target_nll"], places=14)
        expected = (q-torch.eye(4, dtype=torch.float64)).square().sum(1).mean().item()
        self.assertAlmostEqual(m["brier"], expected, places=14)
        self.assertEqual(m["ti_false_target_rate"], 1.)
        self.assertEqual(m["metallic_false_target_rate"], 0.)
        self.assertEqual(m["ti_false_target_denominator"], 1)

    def test_zero_target_predictions_and_net_target_changes(self):
        batch = metric_batch([0,0,0,1,2,3])
        q = torch.tensor([[.6,.2,.1,.1],[.6,.2,.1,.1],[.1,.7,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7]], dtype=torch.float64)
        anchor = q.clone()
        anchor[:3] = q[torch.tensor([2,0,0])]
        m = self.analysis.summarize_predictions(batch, q.log(), anchor.log())
        self.assertEqual((m["new_correct_target_count"],m["lost_correct_target_count"],m["net_correct_target_count"]), (1,1,0))
        q[:,0] = .01
        q /= q.sum(1,keepdim=True)
        m = self.analysis.summarize_predictions(batch, q.log(), anchor.log())
        self.assertEqual(m["target_precision"], 0.)
        self.assertTrue(m["target_precision_zero_denominator"])

    def test_ece_fixed_bins_edges_empty_and_one(self):
        # exp of a tiny finite log probability rounds the winner to 1 in float64.
        q = torch.tensor([[.8,.1,.05,.05],[1.,1e-30,1e-30,1e-30]], dtype=torch.float64)
        m = self.analysis.summarize_predictions(metric_batch([0,1]), q.log(), q.log())
        self.assertEqual(len(m["ece_bins"]),15)
        self.assertEqual(m["ece_bins"][12]["count"],1)
        self.assertEqual(m["ece_bins"][14]["count"],1)
        self.assertIsNone(m["ece_bins"][0]["accuracy"])
        self.assertAlmostEqual(m["ece"],.6)

    def test_screen_requires_every_seed_and_complete_controls(self):
        results, controls = screen_rows()
        self.assertTrue(self.analysis.screen_development(results,controls)["passed"])
        results[-1]["metrics"]["nll"] = .999
        self.assertFalse(self.analysis.screen_development(results,controls)["passed"])
        results, controls = screen_rows()
        self.assertFalse(self.analysis.screen_development(results,controls[:-1])["passed"])

    def test_screen_rejects_truncation_violations_and_false_target_regression(self):
        for field, value in (("status","OPTIMIZATION_TRUNCATED"),("theory_violation_count",1)):
            results, controls = screen_rows()
            results[0][field] = value
            self.assertFalse(self.analysis.screen_development(results,controls)["passed"])
        results, controls = screen_rows()
        results[0]["metrics"]["ti_false_target_count"] = 7
        self.assertFalse(self.analysis.screen_development(results,controls)["passed"])

    def test_u1_dominance_denies_bounded_net_benefit(self):
        results, controls = screen_rows()
        for row in controls:
            if row["arm"] == "U1":
                row["metrics"].update(nll=.97,correct_target_count=12,ti_false_target_count=4,metallic_false_target_count=2)
        screen = self.analysis.screen_development(results,controls)
        self.assertFalse(screen["bounded_net_benefit_supported"])
        self.assertEqual(screen["u1_dominating_seeds"], list(SEEDS))

    def test_replay_training_needs_independent_authorization_before_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"unauthorized.json"
            path.write_text(json.dumps(dict(status="DRAFT_NOT_AUTHORIZED")), encoding="utf-8")
            with patch.object(self.analysis, "load_cache_subset") as loader:
                with self.assertRaises(ValueError):
                    self.analysis.verify_delivery(root,root,training_replay_protocol=path)
                loader.assert_not_called()

    def test_training_replay_compares_states_exactly_and_records_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save(dict(state_dict={"weight":torch.ones(1)},worker_compute_seconds=1.),root/"state.pt")
            original = runner.draft_settings()
            lock = dict(source_snapshot={},selections={},full_preprocessing_path="state.pt",
                        cv_paths={"cv":"state.pt"},finals=[dict(arm="R1",seed=SEEDS[0],state_path="state.pt")])
            def fit(_root,_protocol,target,_started,_snapshot):
                torch.save(dict(state_dict={"weight":torch.zeros(1)},worker_compute_seconds=2.),target/"state.pt")
                return dict(lock,actual_fit_count=1,consumed_seconds=.1)
            with patch.object(runner,"_fit_stage",side_effect=fit):
                with self.assertRaisesRegex(ValueError,"REPLAY_MISMATCH"):
                    self.analysis._training_replay(root,root,original,original,lock)
            report = json.loads((root/"training_replay"/"verification.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"],"REPLAY_MISMATCH")

    def test_cv_packet_keeps_a_portable_pinned_preparation_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            torch.save(dict(synthetic=True),output/"prepared.pt")
            job = runner._packet_job(output,dict(kind="neural",prepared_path=str(output/"prepared.pt")),"portable")
            packet = torch.load(Path(job["result_path"]).with_name("packet.pt"),weights_only=True)
            self.assertEqual(packet.get("prepared_relative_path"),"prepared.pt")

    def test_delivery_exports_all_groups_and_default_verify_is_inference_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, output, source, stop = make_delivery(root)
            def load(_root,_protocol,subset):
                return source if subset == runner.SUBSETS[0] else stop
            with patch.object(self.analysis,"load_cache_subset",side_effect=load), patch.object(torch.optim.Adam,"step",side_effect=AssertionError("optimizer forbidden")):
                summary = self.analysis.evaluate_locked(root,protocol,output)
                verified = self.analysis.verify_delivery(root,output)
            self.assertEqual(summary["final_group_count"],26)
            self.assertEqual(len(list((output/"evaluation"/"predictions").glob("*.csv"))),54)
            self.assertEqual(verified["status"],"VERIFIED_INFERENCE_ONLY")
            self.assertEqual(verified["oof_candidate_count"],2124)
            self.assertFalse(verified["training_replayed"])
            self.assertFalse(summary["screen"]["passed"])

    def test_altered_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol, output, source, stop = make_delivery(root)
            with patch.object(self.analysis,"load_cache_subset",side_effect=lambda r,p,s: source if s == runner.SUBSETS[0] else stop):
                self.analysis.evaluate_locked(root,protocol,output)
                path = output/"evaluation"/"predictions"/f"stop-R1-{SEEDS[0]}.csv"
                path.write_text(path.read_text(encoding="utf-8").replace("synthetic/0.jpg","altered.jpg"),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"changed|tampered"):
                    self.analysis.verify_delivery(root,output)


if __name__ == "__main__":
    unittest.main()
