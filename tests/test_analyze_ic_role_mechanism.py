"""Hand-checkable pooled predictions and synthetic saved-state replay."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from ic_role_model import ICRoleModel, UNITS
from ic_role_training import SEEDS, fit_trajectory
from test_ic_role_training import processed
try:
    import analyze_ic_role_mechanism as analysis
except ModuleNotFoundError:
    analysis = None


def logprob(prediction):
    return torch.nn.functional.one_hot(torch.tensor(prediction), 4).double().mul(.8).add(.05).log()


def oof_fixture():
    assignment = [dict(image_id=f"r{i}", split_group_id=f"g{i}", class_id=i % 4,
                       inner_fold=0 if i < 4 else 1 if i < 12 else 2) for i in range(24)]
    artifacts = []
    for unit in UNITS:
        for fold in range(3):
            rows = [r for r in assignment if r["inner_fold"] == fold]
            records = [dict(image_id=r["image_id"], split_group_id=r["split_group_id"],
                            four_class_id=str(r["class_id"])) for r in rows]
            labels = torch.tensor([r["class_id"] for r in rows])
            anchor = logprob(labels.tolist())
            prediction = labels if fold == 0 else (labels+1) % 4
            for seed in SEEDS:
                artifacts.append(dict(unit=unit, seed=seed, inner_fold=fold, validation_records=records,
                    validation_labels=labels, validation_anchor=anchor,
                    checkpoints={100: dict(validation_logq=logprob(prediction.tolist()), completed_updates=100),
                                 200: dict(validation_logq=anchor, completed_updates=200)}))
    return artifacts, assignment


class MechanismAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(analysis, "Inference-only mechanism analysis missing")
        torch.set_num_threads(1)

    def test_oof_missing_duplicate_or_reordered_rows_rejected(self):
        artifacts, assignment = oof_fixture()
        for bad in (artifacts[:-1], artifacts+[artifacts[0]]):
            with self.assertRaises(ValueError): analysis.collect_oof(bad, assignment)
        changed = copy.deepcopy(artifacts)
        changed[0]["validation_records"].reverse()
        with self.assertRaises(ValueError): analysis.collect_oof(changed, assignment)
        changed = copy.deepcopy(assignment)
        changed[1]["image_id"] = changed[0]["image_id"]
        with self.assertRaises(ValueError): analysis.collect_oof(artifacts, changed)

    def test_pooled_metrics_not_unweighted_fold_average(self):
        artifacts, assignment = oof_fixture()
        pooled = analysis.collect_oof(artifacts, assignment)
        row = pooled["groups"]["C-S-R|20261002"]
        self.assertEqual(row["metrics"]["row_count"], 24)
        self.assertAlmostEqual(row["metrics"]["accuracy"], 4/24)
        self.assertNotAlmostEqual(row["metrics"]["accuracy"], 1/3)
        self.assertEqual(len(row["metrics"]["ece_bins"]), 15)

    def test_100_primary_no_best_epoch_selection(self):
        artifacts, assignment = oof_fixture()
        result = analysis.collect_oof(artifacts, assignment)
        self.assertEqual(result["primary_checkpoint"], 100)
        self.assertAlmostEqual(result["groups"]["C-S-R|20261002"]["metrics"]["accuracy"], 1/6)
        self.assertNotIn("winner", result)
        with self.assertRaises(ValueError): analysis.collect_oof(artifacts, assignment, checkpoint=99)

    def test_paired_new_lost_counts_and_false_target_denominators(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3])
        records = [dict(image_id=f"r{i}", split_group_id=f"g{i}") for i in range(8)]
        right = dict(labels=labels, records=records, logq=logprob([1, 0, 1, 1, 0, 2, 0, 3]))
        left = dict(labels=labels, records=records, logq=logprob([0, 1, 0, 0, 1, 2, 3, 0]))
        comparison = analysis.compare_units(left, right)
        self.assertEqual(comparison["new_correct_target_count"], 2)
        self.assertEqual(comparison["lost_correct_target_count"], 1)
        self.assertEqual(comparison["net_correct_target_count"], 1)
        self.assertEqual(comparison["left_metrics"]["ti_false_target_denominator"], 2)
        self.assertEqual(comparison["left_metrics"]["metallic_false_target_denominator"], 2)
        self.assertEqual(comparison["left_metrics"]["ti_false_target_count"], 1)
        self.assertEqual(len(comparison["row_changes"]), 8)

    def test_saved_kernel_corruption_is_rejected(self):
        train, val = processed(8, "train", 7), processed(8, "val", 8)
        trajectory = fit_trajectory(train, val, "C-S-R", SEEDS[0], max_updates=1)
        trajectory["initial_kernels"][0]["Kmain"][0, 0] += .1
        with self.assertRaises(ValueError): analysis._verify_trajectory(trajectory, train, val)

    def test_failed_terminal_delivery_never_opens_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"out"
            output.mkdir()
            manifest = dict(stage="synthetic", scope="synthetic_saved_processed_inputs", artifacts={}, trajectories=[])
            (output/"artifact_manifest.json").write_text(json.dumps(manifest))
            for status in ("RUNNING", "FAILED", "POWER_RELEASE_FAILED", "TIMED_OUT"):
                (output/"stage_state.json").write_text(json.dumps(dict(status=status, stage="synthetic", elapsed_seconds=7201., alive=False,
                    power=dict(power_request_acquired=True, power_request_released=False), jobs=[])))
                with patch.object(analysis.torch, "load", side_effect=AssertionError("Never open artifact")):
                    with self.assertRaisesRegex(ValueError, "terminal|budget|power|supervision"):
                        analysis.verify_mechanism_delivery(root, output)

    def test_diagnostic_schema_and_ablation_tampering_rejected(self):
        train, val = processed(8, "train", 7), processed(8, "val", 8)
        original = fit_trajectory(train, val, "C-S-R", SEEDS[0], max_updates=1)
        for kind in ("empty_gradient", "missing_jacobian", "nonfinite_jacobian", "false_ablation", "row_order"):
            bad = copy.deepcopy(original)
            if kind == "empty_gradient":
                for key in ("data", "regularization", "total"): bad["gradient_vectors"][0][key] = torch.empty(0, dtype=torch.float64)
                bad["gradient_vectors"][0]["layout"] = []
            elif kind == "missing_jacobian": del bad["checkpoints"][1]["train_jacobians"]
            elif kind == "nonfinite_jacobian": bad["checkpoints"][1]["train_jacobians"]["1"]["main"]["jacobian"][0, 0] = float("nan")
            elif kind == "false_ablation": bad["checkpoints"][1]["branch_diagnostics"]["validation"]["removed_branch_max_log_drift"].fill_(123.)
            elif kind == "row_order": bad["checkpoints"][1]["train_jacobians"]["1"]["aux"]["row_ids"].reverse()
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError): analysis._verify_trajectory(bad, train, val)

    def test_provisional_analysis_requires_live_supervision(self):
        helper = getattr(analysis, "_verify_active_mechanism", None)
        self.assertIsNotNone(helper, "Provisional analysis must be separated from terminal acceptance")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "supervision"):
                helper(Path(directory), Path(directory)/"out")

    def test_inference_verifier_cannot_optimize_or_open_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"synthetic"
            output.mkdir()
            train, val = processed(8, "train", 7), processed(8, "val", 8)
            trajectory = fit_trajectory(train, val, "C-S-R", SEEDS[0], max_updates=1)
            trajectory["inner_fold"] = 0
            trajectory_path = output/"trajectory.pt"
            torch.save(dict(trajectory=trajectory, train=train, validation=val), trajectory_path)
            manifest = dict(stage="synthetic", artifacts={"trajectory.pt": analysis.file_digest(trajectory_path)},
                            trajectories=["trajectory.pt"], scope="synthetic_saved_processed_inputs")
            (output/"artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (output/"stage_state.json").write_text(json.dumps(dict(status="MECHANISM_COMPLETE", stage="synthetic",
                elapsed_seconds=.1, alive=False, power=dict(power_request_acquired=True, power_request_released=True), jobs=[])))
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimization")), \
                    patch.object(analysis, "load_fit_bundle", side_effect=AssertionError("No real file")):
                evidence = analysis.verify_mechanism_delivery(root, output)
            self.assertEqual(evidence["trajectory_count"], 1)
            self.assertEqual(evidence["checkpoint_count"], 2)
            self.assertEqual(evidence["theory_violation_count"], 0)
            self.assertFalse(evidence["independent_confirmation"])
            report = analysis.write_mechanism_report(evidence, output/"report")
            prose = Path(report["markdown"]).read_text(encoding="utf-8")
            for phrase in ("仅fit探索", "固定400预算不是收敛", "无独立确认", "已有前例"):
                self.assertIn(phrase, prose)
            packet = torch.load(trajectory_path, weights_only=True)
            packet["trajectory"]["checkpoints"][1]["validation_logq"][0, 0] += .01
            torch.save(packet, trajectory_path)
            manifest["artifacts"]["trajectory.pt"] = analysis.file_digest(trajectory_path)
            (output/"artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError): analysis.verify_mechanism_delivery(root, output)


if __name__ == "__main__": unittest.main()
