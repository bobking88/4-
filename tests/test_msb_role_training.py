"""Fixed schedules and synthetic-only trajectories."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from msb_synthetic_helpers import make_prepared_fixture
try:
    import msb_role_training as training
except ModuleNotFoundError:
    training=None


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(training,"MSB fixed training missing")
        self.data=make_prepared_fixture(24)["folds"][0]
        timer=patch.object(training,"stage_elapsed_seconds",return_value=0.)
        timer.start(); self.addCleanup(timer.stop)

    def short(self, validation=None, unit="MAIN"):
        return training.fit_trajectory(self.data["train"],validation or self.data["validation"],unit,20261002,max_updates=2,checkpoints=(0,1,2))

    def test_63_neural_plus_12_convex_jobs(self):
        jobs=training.build_fit_matrix()
        self.assertEqual(len(jobs),75)
        self.assertEqual(sum(j["kind"]=="trajectory" for j in jobs),63)
        self.assertEqual(sum(j["kind"]=="convex" for j in jobs),12)
        self.assertEqual(len({j["job_id"] for j in jobs}),75)
        for j in jobs:
            if j["kind"]=="trajectory":
                self.assertEqual(j["max_updates"],400)
                self.assertEqual(j["primary_checkpoint"],100)

    def test_no_validation_grad_or_selection(self):
        before=self.short()
        validation=copy.deepcopy(self.data["validation"])
        validation["labels"]=validation["labels"].roll(1)
        for r,y in zip(validation["records"],validation["labels"]): r["four_class_id"]=str(int(y))
        after=self.short(validation)
        for key in before["checkpoints"][-1]["model_state"]:
            self.assertTrue(torch.equal(before["checkpoints"][-1]["model_state"][key],after["checkpoints"][-1]["model_state"][key]))
        self.assertNotIn("best_checkpoint",before)

    def test_checkpoint_is_completed_updates(self):
        result=self.short()
        self.assertEqual([p["completed_updates"] for p in result["checkpoints"]],[0,1,2])
        self.assertEqual(result["completed_updates"],2)
        self.assertTrue(result["complete"])
        self.assertFalse(result["primary_present"])

    def test_reference_unchanged_all_steps(self):
        result=self.short()
        self.assertEqual(result["initial_reference_digest"],result["final_reference_digest"])
        for point in result["checkpoints"]:
            self.assertEqual(point["reference_digest"],result["initial_reference_digest"])
            self.assertIn("total",point["gradients"]["vectors"])

    def test_convex_not_repeated_as_three_seed_evidence(self):
        jobs=[j for j in training.build_fit_matrix() if j["kind"]=="convex"]
        self.assertTrue(all(j["seed"] is None for j in jobs))
        self.assertEqual(len({(j["unit"],j["fold"]) for j in jobs}),12)
        result=training.fit_convex_fold(self.data["train"],self.data["validation"],"CLASSWISE4",deadline_check=lambda:None)
        self.assertTrue(result["complete"])
        self.assertTrue(result["solution"]["converged"])
        self.assertIsNone(result["seed"])

    def test_missing_or_unsolved_job_fails_complete(self):
        entries=[]
        for job in training.build_fit_matrix():
            r=dict(complete=True,unit=job["unit"],seed=job["seed"],
                   completed_updates=400,primary_present=True,primary_checkpoint=100,
                   checkpoints=[dict(completed_updates=i) for i in (0,1,2,5,20,50,100,200,400)],
                   solution=dict(converged=True,gap_upper=0.,mu=.001))
            entries.append(dict(schedule=job,result=r))
        training.validate_matrix_results(entries)
        with self.assertRaises(ValueError): training.validate_matrix_results(entries[:-1])
        bad=copy.deepcopy(entries)
        next(e for e in bad if e["schedule"]["kind"]=="convex")["result"]["solution"]["converged"]=False
        with self.assertRaises(ValueError): training.validate_matrix_results(bad)

    def test_primary_masks_require_actual_hundred_updates(self):
        result=training.fit_trajectory(self.data["train"],self.data["validation"],"MAIN",20261002,max_updates=100,checkpoints=(0,100))
        self.assertEqual(result["completed_updates"],100)
        self.assertTrue(result["primary_present"])
        point=result["checkpoints"][-1]
        self.assertEqual(len(point["role_off"]),4)
        torch.testing.assert_close(point["role_off"][-1]["forward"]["logq"],self.data["validation"]["log_anchor"],atol=1e-12,rtol=1e-12)

    def test_invalid_real_short_budget_or_overlap_rejected(self):
        real=copy.deepcopy(self.data["train"])
        for r in real["records"]: r["image_id"]="pretend-real-"+r["image_id"]
        with self.assertRaises(ValueError):
            training.fit_trajectory(real,self.data["validation"],"MAIN",20261002,max_updates=2,checkpoints=(0,1,2))
        with self.assertRaises(ValueError):
            training.fit_trajectory(self.data["train"],self.data["train"],"MAIN",20261002,max_updates=2,checkpoints=(0,1,2))

    def test_empty_checkpoint_schedule_rejected(self):
        with self.assertRaises(ValueError):
            training.fit_trajectory(self.data["train"],self.data["validation"],"MAIN",20261002,max_updates=2,checkpoints=())
