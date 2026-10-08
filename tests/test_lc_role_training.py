import copy
import inspect
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lc_role_data import fit_preprocessing, transform_subset
from lc_role_adapter import RoleResidualModel
from lc_role_fixtures import raw
try:
    import lc_role_training as training
except ModuleNotFoundError:
    training = None


class InnerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(training, "LC-RFA fit-only training module missing")
        torch.set_num_threads(1)
        data = raw()
        state = fit_preprocessing(data, torch.arange(80))
        self.train = transform_subset(data, state, torch.arange(80))
        self.validation = transform_subset(data, state, torch.arange(80, 120))
        self.config = dict(regularization=.01, alpha_max=.25, beta_max=.25, checkpoints=[0, 20])
        self.assignment = [dict(image_id=f"row-{i}", split_group_id=f"group-{i}",
                                inner_fold=0 if i < 80 else 1 if i < 104 else 2,
                                class_id=i % 4) for i in range(120)]

    def candidates(self, config, nlls, updates=20, arm="R1"):
        rows = []
        for seed_index, seed in enumerate(training.SEEDS):
            for fold in range(3):
                assignment = [r for r in self.assignment if r["inner_fold"] == fold]
                nll = nlls[seed_index][fold]
                true_prob = math.exp(-nll)
                p = torch.full((len(assignment), 4), (1-true_prob)/3, dtype=torch.float64)
                p[range(len(p)), [r["class_id"] for r in assignment]] = true_prob
                rows.append(dict(arm=arm, config=config.copy(), updates=updates, seed=seed,
                                 inner_fold=fold, image_ids=[r["image_id"] for r in assignment],
                                 logq=p.log(), source="fit_inner_oof"))
        return rows

    def test_pooled_nll_not_fold_average_or_best_seed(self):
        a = dict(regularization=.01, alpha_max=.25, beta_max=.25)
        b = dict(regularization=.01, alpha_max=.5, beta_max=.5)
        c = dict(regularization=.1, alpha_max=.5, beta_max=.5)
        candidates = self.candidates(a, [[2, .1, .1]]*3)
        candidates += self.candidates(b, [[.8]*3, [.9]*3, [1.]*3])
        candidates += self.candidates(c, [[.1]*3, [2.]*3, [2.]*3])
        winner = training.select_oof(candidates, self.assignment)
        self.assertEqual(winner["pooled_row_count"], 120)
        self.assertEqual(winner["config"], b)
        self.assertAlmostEqual(winner["mean_pooled_nll"], .9, places=13)
        self.assertEqual(winner["source"], "fit_inner_oof")
        self.assertEqual(len(winner["scores"]), 3)
        self.assertEqual(set(winner["seed_pooled_nll"]), {str(s) for s in training.SEEDS})

    def test_exact_ties_and_matched_budget_filter(self):
        a = dict(regularization=.01, alpha_max=.5, beta_max=.5)
        b = dict(regularization=.1, alpha_max=.25, beta_max=.25)
        tied = self.candidates(a, [[1.]*3]*3, updates=20)
        tied += self.candidates(b, [[1.]*3]*3, updates=20)
        tied += self.candidates(b, [[1.]*3]*3, updates=0)
        chosen = training.select_oof(tied, self.assignment)
        self.assertEqual(chosen["config"], b)
        self.assertEqual(chosen["updates"], 0)
        s1 = [dict(row, arm="S1") for row in tied]
        subset = training.matched_budget_candidates(s1, chosen)
        self.assertEqual(len(subset), 18)
        self.assertTrue(all(row["config"]["alpha_max"] == .25 for row in subset))
        self.assertEqual(training.select_oof(subset, self.assignment)["updates"], 0)

    def test_duplicate_missing_wrong_fold_or_untrusted_oof_rejected(self):
        config = {key: self.config[key] for key in ("regularization", "alpha_max", "beta_max")}
        valid = self.candidates(config, [[1]*3]*3)
        variants = [valid + [copy.deepcopy(valid[0])], valid[:-1]]
        wrong = copy.deepcopy(valid)
        wrong[0]["image_ids"][0] = "row-100"
        variants.append(wrong)
        wrong = copy.deepcopy(valid)
        wrong[0]["source"] = "projector_stop"
        variants.append(wrong)
        wrong = copy.deepcopy(valid)
        wrong[0]["logq"][0, 0] = float("nan")
        variants.append(wrong)
        wrong = copy.deepcopy(valid)
        wrong[0]["seed"] = 99
        variants.append(wrong)
        for rows in variants:
            with self.assertRaises(ValueError):
                training.select_oof(rows, self.assignment)

    def test_all_neural_arms_single_trajectory_and_exact_replay(self):
        for arm in ("H0", "F0", "S0", "S1", "R0", "R1", "U1"):
            fit = training.fit_neural(self.train, self.validation, arm, self.config, 20261002)
            self.assertEqual(set(fit["checkpoints"]), {0, 20})
            self.assertEqual(fit["optimizer_updates"], 20)
            self.assertEqual(len(fit["history"]), 21)
            self.assertEqual({r["step"] for r in fit["gradients"]}, set(range(1, 6)))
            model = RoleResidualModel(arm, seed=20261002, alpha_max=.25, beta_max=.25)
            model.load_state_dict(fit["checkpoints"][20]["state_dict"])
            with torch.no_grad():
                prediction = model(self.validation["h"], self.validation["e"], self.validation["log_anchor"])["logq"]
            torch.testing.assert_close(prediction, fit["checkpoints"][20]["validation_logq"], atol=0, rtol=0)
            final = training.refit_neural(self.train, arm, self.config, 20261002, 20)
            for key, value in final["state_dict"].items():
                torch.testing.assert_close(value, fit["checkpoints"][20]["state_dict"][key], atol=0, rtol=0)
            zero = training.refit_neural(self.train, arm, self.config, 20261002, 0)
            torch.testing.assert_close(zero["train_logq"], self.train["log_anchor"], atol=1e-12, rtol=1e-12)
            self.assertEqual(zero["optimizer_updates"], 0)

    def test_gradients_are_separated_l2_includes_bias_and_missing_class_fails(self):
        fit = training.fit_neural(self.train, self.validation, "R1", self.config, 20261002)
        first = [r for r in fit["gradients"] if r["step"] == 1 and r["parameter"] == "adapters.0.down.weight"][0]
        self.assertEqual(first["data_gradient_l2"], 0)
        self.assertGreater(first["total_gradient_l2"], 0)
        initial = RoleResidualModel("R1", seed=20261002, alpha_max=.25, beta_max=.25)
        expected = .005*sum(p.detach().square().sum() for p in initial.parameters())
        self.assertAlmostEqual(fit["history"][0]["l2"], float(expected), places=13)
        missing = copy.deepcopy(self.validation)
        missing["labels"][:] = 0
        with self.assertRaises(ValueError):
            training.fit_neural(self.train, missing, "R1", self.config, 20261002)

    def test_checkpoint_budget_nan_scope_and_stop_interface(self):
        for updates in (-1, 401, 1):
            with self.assertRaises(ValueError):
                training.refit_neural(self.train, "R1", self.config, 20261002, updates)
        bad = copy.deepcopy(self.train)
        bad["h"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            training.fit_neural(bad, self.validation, "R1", self.config, 20261002)
        exposed = copy.deepcopy(self.validation)
        exposed["records"][0]["tc_subset"] = "projector_stop"
        with self.assertRaises(ValueError):
            training.fit_neural(self.train, exposed, "R1", self.config, 20261002)
        self.assertNotIn("stop", inspect.signature(training.refit_neural).parameters)
        config = {key: self.config[key] for key in ("regularization", "alpha_max", "beta_max")}
        selected = training.select_oof(self.candidates(config, [[1]*3]*3, updates=400), self.assignment)
        self.assertEqual(selected["status"], "OPTIMIZATION_TRUNCATED")
        final = training.refit_neural(self.train, "H0", self.config, 20261002, 400)
        self.assertEqual(final["optimizer_updates"], 400)
        self.assertEqual(final["status"], "OPTIMIZATION_TRUNCATED")

    def test_convex_controls_have_stationarity_certificates_and_replay(self):
        for arm in ("P", "E"):
            fit = training.fit_convex(self.train, self.validation, arm, .1)
            self.assertTrue(fit["converged"])
            self.assertLessEqual(fit["gradient_l2"], 1e-7)
            self.assertLessEqual(fit["gap_upper_bound"], 1e-8)
            self.assertEqual(len(fit["solver_stages"]), 2)
            x = self.validation["P" if arm == "P" else "e"]
            augmented = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
            expected = torch.log_softmax(self.validation["log_anchor"] + augmented@fit["theta"], 1)
            torch.testing.assert_close(expected, fit["validation_logq"], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            training.fit_convex(self.train, self.validation, "EH", .1)

    def test_uncertified_convex_oof_cannot_enter_selection(self):
        config = dict(regularization=.1, alpha_max=0, beta_max=0)
        candidates = [dict(row, seed=0) for row in self.candidates(config, [[1]*3]*3, updates=0, arm="P")[:3]]
        with self.assertRaises(ValueError):
            training.select_oof(candidates, self.assignment)
        for row in candidates:
            row["certificate"] = dict(converged=True, gradient_l2=1e-9, gap_upper_bound=1e-12)
        self.assertEqual(training.select_oof(candidates, self.assignment)["arm"], "P")
        for field, value in (("converged", False), ("gradient_l2", 1), ("gap_upper_bound", float("nan"))):
            broken = copy.deepcopy(candidates)
            broken[0]["certificate"][field] = value
            with self.assertRaises(ValueError):
                training.select_oof(broken, self.assignment)

    def test_all_arms_grouped_synthetic_selection_refit_and_replay(self):
        source = raw()
        from lc_role_data import make_inner_assignment
        assignment = make_inner_assignment(source)
        folds = []
        for fold in range(3):
            train_indices = torch.tensor([i for i, row in enumerate(assignment) if row["inner_fold"] != fold])
            val_indices = torch.tensor([i for i, row in enumerate(assignment) if row["inner_fold"] == fold])
            state = fit_preprocessing(source, train_indices)
            folds.append((transform_subset(source, state, train_indices), transform_subset(source, state, val_indices)))
        all_fit = transform_subset(source, fit_preprocessing(source, torch.arange(120)))
        for arm in ("H0", "F0", "S0", "S1", "R0", "R1", "U1", "P", "E"):
            oof = []
            config = dict(regularization=.1, alpha_max=.25, beta_max=.25, checkpoints=[0, 20])
            seeds = (0,) if arm in ("P", "E") else training.SEEDS
            for seed in seeds:
                for fold, (train, val) in enumerate(folds):
                    common = dict(arm=arm, seed=seed, inner_fold=fold, source="fit_inner_oof",
                                  image_ids=[r["image_id"] for r in val["records"]])
                    if arm in ("P", "E"):
                        fitted = training.fit_convex(train, val, arm, .1)
                        oof.append(dict(common, config=dict(regularization=.1, alpha_max=0, beta_max=0),
                                        updates=0, logq=fitted["validation_logq"], certificate=fitted))
                    else:
                        fitted = training.fit_neural(train, val, arm, config, seed)
                        oof.extend(dict(common, config={k: config[k] for k in ("regularization", "alpha_max", "beta_max")},
                                        updates=step, logq=snapshot["validation_logq"])
                                   for step, snapshot in fitted["checkpoints"].items())
            choice = training.select_oof(oof, assignment)
            self.assertEqual(choice["pooled_row_count"], 120)
            self.assertEqual(choice["arm"], arm)
            if arm in ("P", "E"):
                final = training.fit_convex(all_fit, None, arm, choice["config"]["regularization"])
                self.assertTrue(final["converged"])
            else:
                final = training.refit_neural(all_fit, arm, choice["config"], training.SEEDS[0], choice["updates"])
                model = RoleResidualModel(arm, seed=training.SEEDS[0], alpha_max=.25, beta_max=.25)
                model.load_state_dict(final["state_dict"])
                with torch.no_grad():
                    replay = model(all_fit["h"], all_fit["e"], all_fit["log_anchor"])["logq"]
                torch.testing.assert_close(replay, final["train_logq"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
