"""Synthetic-only evidence, replay and falsification checks for MSB."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from msb_synthetic_helpers import make_prepared_fixture
from msb_role_model import MSBRoleModel, reference_digest
from msb_role_training import fit_trajectory, fit_convex_fold, SEEDS, CHECKPOINTS, build_fit_matrix, _input_digest
from msb_role_data import identity_digest
from msb_role_audit import gradient_components
try:
    import analyze_msb_role_mechanism as analysis
except ModuleNotFoundError:
    analysis = None


def prediction_rows(*, perfect=False, target_error=True):
    data = make_prepared_fixture()["batch"]
    rows = []
    for i, row in enumerate(data["records"]):
        y = int(data["labels"][i])
        predicted = y if perfect else (1 if y == 0 and target_error and i == 0 else y)
        p = [.02]*4
        p[predicted] = .94
        rows.append(dict(row, probabilities=p))
    return rows


def gate_fixture():
    good, base = prediction_rows(perfect=True), prediction_rows()
    comparison = analysis.paired_comparison(good, base)
    return dict(primary_checkpoint=100,
                comparisons={unit:{str(seed):copy.deepcopy(comparison) for seed in SEEDS}
                             for unit in analysis.REFERENCES},
                interaction_comparisons={str(seed):copy.deepcopy(comparison) for seed in SEEDS},
                convex_certificates=[dict(unit=u, fold=f, converged=True, mu=.001, gap_upper=0.)
                                     for u in analysis.CONVEX_UNITS for f in range(3)])


def metadata_matrix_fixture():
    """75 initial-state metadata fixtures, NOT75 trained or converged models."""
    fixture = make_prepared_fixture()
    entries = []
    protocol = dict(prepared_entries=[])
    for data in fixture["folds"]:
        protocol["prepared_entries"].append({f"registered_{part}_ids_digest":identity_digest(
            [r["image_id"] for r in data[part]["records"]]) for part in ("train","validation")})
    for schedule in build_fit_matrix():
        data = fixture["folds"][schedule["fold"]]
        result = dict(unit=schedule["unit"],seed=schedule["seed"],complete=True)
        for part in ("train","validation"):
            result.update({part+"_records":data[part]["records"],part+"_labels":data[part]["labels"],
                           part+"_inputs_digest":_input_digest(data[part])})
        if schedule["kind"] == "trajectory":
            model = MSBRoleModel(schedule["unit"],seed=schedule["seed"])
            output = {part:model(data[part]["h"],data[part]["e"],data[part]["log_anchor"])
                      for part in ("train","validation")}
            gradients = gradient_components(model,output["train"],data["train"]["labels"])
            points = []
            for step in CHECKPOINTS:
                masks = [(False,True,True),(True,False,True),(True,True,False),(False,False,False)]
                points.append(dict(completed_updates=step,train=output["train"],validation=output["validation"],
                                   reference_digest=reference_digest(model),gradients=gradients,
                                   role_off=[dict(mask=m,forward=output["validation"]) for m in masks]
                                   if schedule["unit"]=="MAIN" and step==100 else []))
            result.update(checkpoints=points,initial_reference_digest=reference_digest(model),
                          final_reference_digest=reference_digest(model))
        else:
            result.update(validation=dict(logq=data["validation"]["log_anchor"]),
                          solution=dict(converged=True,mu=.001,gap_upper=0.,objective=1.))
        entries.append(dict(schedule=schedule,result=result))
    return entries,protocol


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(analysis, "MSB evidence analyzer missing")
        self.data = make_prepared_fixture(24)["folds"][0]

    def test_oof_count_row_order_and_duplicate_fail(self):
        rows = prediction_rows(perfect=True)
        summary = analysis.summarize_oof(rows)
        self.assertEqual(summary["counts"], [66,130,97,47])
        self.assertEqual(summary["macro_f1"], 1.)
        self.assertEqual(summary["false_target_counts"], [0,0,0])
        with self.assertRaises(ValueError): analysis.summarize_oof(rows[:-1])
        with self.assertRaises(ValueError): analysis.summarize_oof(rows[:-1]+[rows[0]])
        with self.assertRaises(ValueError): analysis.paired_comparison(rows, rows[::-1])

    def test_metrics_probabilities_and_ece_endpoints(self):
        rows = prediction_rows(perfect=True)
        for row in rows:
            row["probabilities"] = [float(k == int(row["four_class_id"])) for k in range(4)]
        summary = analysis.summarize_oof(rows)
        self.assertEqual(summary["ece15"], 0.)
        self.assertEqual(summary["brier"], 0.)
        self.assertEqual(summary["nll"], 0.)
        self.assertEqual(summary["ece_bins"][-1]["count"], 340)
        bad = copy.deepcopy(rows); bad[0]["probabilities"] = [.5]*4
        with self.assertRaises(ValueError): analysis.summarize_oof(bad)

    def test_nll_change_without_argmax_gain_fails_gate(self):
        rows = prediction_rows()
        sharper = copy.deepcopy(rows)
        for row in sharper:
            winner = max(range(4), key=lambda k:row["probabilities"][k])
            row["probabilities"] = [.01 if k != winner else .97 for k in range(4)]
        fixture = gate_fixture()
        fixture["comparisons"]["T0"][str(SEEDS[0])] = analysis.paired_comparison(sharper,rows)
        result = analysis.mechanism_gate(fixture)
        self.assertEqual(result["decision"], "NO_ROLE_PROMOTION")
        self.assertFalse(result["statistical_significance_claim"])

    def test_bias_or_shared_equal_not_role_success(self):
        for unit in ("BIAS3-CVX", "SHARED-BIL"):
            fixture = gate_fixture()
            fixture["comparisons"][unit][str(SEEDS[1])] = analysis.paired_comparison(
                prediction_rows(perfect=True),prediction_rows(perfect=True))
            self.assertEqual(analysis.mechanism_gate(fixture)["decision"], "NO_ROLE_PROMOTION")

    def test_unsolved_convex_cannot_be_called_weak_winner(self):
        fixture = gate_fixture(); fixture["convex_certificates"][0]["gap_upper"] = 1e-4
        with self.assertRaises(ValueError): analysis.mechanism_gate(fixture)
        fixture = gate_fixture(); fixture["convex_certificates"].pop()
        with self.assertRaises(ValueError): analysis.mechanism_gate(fixture)

    def test_mixed_seed_or_false_target_tradeoff_not_promoted(self):
        fixture = gate_fixture()
        fixture["comparisons"]["T0"][str(SEEDS[2])]["false_target_delta"][0] = 1
        self.assertEqual(analysis.mechanism_gate(fixture)["decision"], "NO_ROLE_PROMOTION")
        fixture = gate_fixture(); fixture["comparisons"]["T0"].pop(str(SEEDS[2]))
        with self.assertRaises(ValueError): analysis.mechanism_gate(fixture)
        fixture = gate_fixture()
        fixture["interaction_comparisons"][str(SEEDS[0])]["target_net_corrections"] = 0
        result = analysis.mechanism_gate(fixture)
        self.assertEqual(result["decision"], "EXPLORATORY_ROLE_GATE_PASSED")
        self.assertFalse(result["explicit_interaction_supported"])

    def trajectory(self):
        return fit_trajectory(self.data["train"],self.data["validation"],"MAIN",SEEDS[0],
                              max_updates=2,checkpoints=(0,1,2))

    def test_same_state_off_returns_anchor_not_old_shared_head(self):
        result = self.trajectory(); point = result["checkpoints"][-1]
        model = MSBRoleModel("MAIN",seed=SEEDS[0]); model.load_state_dict(point["model_state"])
        with torch.no_grad():
            forward = model(self.data["validation"]["h"],self.data["validation"]["e"],
                            self.data["validation"]["log_anchor"],role_mask=(False,False,False))
        point["role_off"] = [dict(mask=(False,False,False),forward=forward)]
        count = analysis.verify_checkpoint("MAIN",SEEDS[0],point,self.data["validation"],
                                           result["initial_reference_digest"],check_masks=True)
        self.assertEqual(count,2)
        torch.testing.assert_close(forward["logq"],self.data["validation"]["log_anchor"])

    def test_tampered_state_or_prediction_fails(self):
        result = self.trajectory(); point = result["checkpoints"][-1]
        self.assertEqual(analysis.verify_checkpoint("MAIN",SEEDS[0],point,self.data["validation"],
                          result["initial_reference_digest"]),1)
        bad = copy.deepcopy(point); bad["validation"]["logq"][0,0] += .01
        with self.assertRaises(ValueError):
            analysis.verify_checkpoint("MAIN",SEEDS[0],bad,self.data["validation"],result["initial_reference_digest"])
        bad = copy.deepcopy(point)
        key = next(k for k in bad["model_state"] if k.endswith("U0"))
        bad["model_state"][key][0,0] += .01
        with self.assertRaises(ValueError):
            analysis.verify_checkpoint("MAIN",SEEDS[0],bad,self.data["validation"],result["initial_reference_digest"])

    def test_convex_certificate_independently_recomputed(self):
        for unit in analysis.CONVEX_UNITS:
            result = fit_convex_fold(self.data["train"],self.data["validation"],unit,deadline_check=lambda:None)
            evidence = analysis.verify_convex_result(result,self.data)
            self.assertLessEqual(evidence["gap_upper"],1e-8)
            self.assertEqual(evidence["train_certificate_forwards"],1)
            self.assertEqual(evidence["validation_forwards"],1)
            bad = copy.deepcopy(result); bad["solution"]["gradient"][0] += .01
            with self.assertRaises(ValueError): analysis.verify_convex_result(bad,self.data)

    def test_unsupervised_replay_opens_nothing(self):
        with patch("torch.load",side_effect=AssertionError("unmanaged load")):
            for fn in (analysis.collect_saved_predictions,analysis.verify_saved_mechanism):
                with self.assertRaises(RuntimeError): fn(Path("unused"),Path("unused"),{})

    def test_initial_state_metadata_matrix_is_not_a_positive_mechanism(self):
        entries,protocol = metadata_matrix_fixture()
        evidence = analysis._summarize_entries(entries,protocol)
        self.assertFalse(evidence["delivery_verified"])
        self.assertEqual(evidence["gate"]["decision"],"NO_ROLE_PROMOTION")
        self.assertEqual(len(evidence["checkpoint_summaries"]),189)
        self.assertEqual(len(evidence["mask_summaries"]),12)
        self.assertEqual(len(evidence["optimization_diagnostics"]),9)
        self.assertEqual(evidence["summaries"]["MAIN"][str(SEEDS[0])]["counts"],[66,130,97,47])

    def test_guard_scope_is_single_use_and_does_not_read_data(self):
        class FakeGuard:
            def evidence(self): return dict(power_request_acquired=True,power_request_released=False)
            def check(self): return 0.
        entries,protocol = metadata_matrix_fixture()
        root = Path("synthetic-only").resolve(); output = root/"fixture"
        with analysis.supervised_analysis(root,output,protocol,FakeGuard(),"analysis"):
            with patch.object(analysis,"_load_saved",return_value=entries):
                result = analysis.collect_saved_predictions(root,output,protocol)
                self.assertFalse(result["delivery_verified"])
                with self.assertRaises(RuntimeError): analysis.collect_saved_predictions(root,output,protocol)

    def test_final_replay_requires_exact615_validation_and12_certificates(self):
        self.assertEqual(analysis.replay_budget(),dict(neural_checkpoints=567,convex_validation=12,
                         primary_role_masks=36,total_validation=615,train_certificates=12))

    def test_artifact_hash_checked_before_deserialization(self):
        import run_msb_role_mechanism as runner
        class FakeGuard:
            def check(self): return 0.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); output=root/"fixture"; output.mkdir()
            matrix=build_fit_matrix()
            ids=[f"load_fold{fold}" for fold in range(3)]+[j["job_id"] for j in matrix]
            protocol=dict(output_relative="fixture")
            relative=f"runs/{matrix[0]['job_id']}/result.pt"
            path=output/relative; path.parent.mkdir(parents=True); path.write_bytes(b"not a model")
            state=dict(status="RUNNING",active_job="analysis",stage="fit-mechanism",
                       protocol_digest=runner.protocol_digest(protocol),actual_neural_fits=63,actual_convex_fits=12,
                       jobs=[dict(job_id=i,status="COMPLETE",alive=False,attempts=1,
                                  supervision=dict(power_request_released=True)) for i in ids],
                       artifacts={relative:"0"*64})
            (output/"stage_state.json").write_text(json.dumps(state),encoding="utf-8")
            with patch.object(runner,"validate_protocol"),patch("torch.load",side_effect=AssertionError("hash not checked")):
                with self.assertRaisesRegex(ValueError,"hash"):
                    analysis._load_saved(root,output,protocol,dict(kind="analysis",guard=FakeGuard()))

    def test_final_orchestration_counts_are_not_training_evidence(self):
        # Mock only the orchestration boundary; actual tensor replay is covered above.
        fixture=make_prepared_fixture()
        entries,protocol=metadata_matrix_fixture()
        context=dict(guard=type("Guard",(),{"check":lambda self:0.})())
        prepared_entries=[dict(fold=i) for i in range(3)]
        protocol.update(prepared_entries=prepared_entries,authorization={})
        calls=[]
        def checkpoint(unit,seed,point,batch,ref,**kwargs):
            calls.append((unit,point["completed_updates"],kwargs["check_masks"]))
            return 5 if kwargs["check_masks"] else 1
        with patch.object(analysis,"_context",return_value=context),patch.object(analysis,"_load_saved",return_value=entries),patch.object(analysis,"load_prepared_fold",side_effect=lambda r,e,a:fixture["folds"][e["fold"]]),patch.object(analysis,"verify_checkpoint",side_effect=checkpoint),patch.object(analysis,"verify_convex_result",return_value=dict(validation_forwards=1,train_certificate_forwards=1)),patch.object(analysis,"_summarize_entries",return_value=dict(scope="SYNTHETIC_ORCHESTRATION_ONLY")):
            evidence=analysis.verify_saved_mechanism(Path("unused"),Path("unused"),protocol)
        self.assertEqual(len(calls),567)
        self.assertEqual(sum(c[2] for c in calls),9)
        self.assertEqual(evidence["verification"]["validation_forwards"],615)
        self.assertEqual(evidence["verification"]["train_certificate_forwards"],12)

    def test_report_is_exclusive_and_cannot_promote_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"report"
            evidence = dict(scope="SYNTHETIC_FIXTURE",delivery_verified=False,
                            gate=analysis.mechanism_gate(gate_fixture()),predictions={},summaries={})
            result = analysis.write_mechanism_report(evidence,path)
            self.assertFalse(result["delivery_verified"])
            report = json.loads((path/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["scope"],"SYNTHETIC_FIXTURE")
            self.assertIn("not independent",(path/"summary.md").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError): analysis.write_mechanism_report(evidence,path)


if __name__ == "__main__": unittest.main()
