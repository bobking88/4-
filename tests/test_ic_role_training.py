"""Synthetic trajectory tests; never open the registered mineral feature bundle."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ic_role_model import ICRoleModel, UNITS, reference_identity
from test_ic_role_model import make_fixture

try:
    import ic_role_training as training
except ModuleNotFoundError:
    training = None


def processed(n, prefix, seed):
    result = make_fixture(n, seed)
    result["records"] = [dict(image_id=f"{prefix}-{i}", split_group_id=f"{prefix}-group-{i}",
                              four_class_id=str(int(result["labels"][i])),
                              tc_subset="gate_stop_projector_fit") for i in range(n)]
    return result


class ICTrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(training, "IC fixed trajectory module is not implemented")
        torch.set_num_threads(1)
        self.train, self.validation = processed(12, "train", 7), processed(8, "val", 8)

    def test_gradients_are_vectors_and_sum_before_update(self):
        for unit in ("C-S-R", "Z-O-U"):
            model = ICRoleModel(unit, seed=20261002)
            result = model(self.train["h"], self.train["e"], self.train["log_anchor"], auxiliary=True)
            grads = training.gradient_components(model, result, self.train["labels"])
            self.assertEqual(grads["data"].shape, (3024,))
            self.assertEqual(grads["regularization"].shape, (3024,))
            self.assertEqual(grads["total"].shape, (3024,))
            torch.testing.assert_close(grads["total"], grads["data"]+grads["regularization"], atol=1e-10, rtol=1e-10)
            self.assertEqual(sum(row["numel"] for row in grads["layout"]), 3024)
            if unit == "C-S-R":
                self.assertEqual(float(grads["regularization"].norm()), 0.)
            else:
                self.assertGreater(float(grads["regularization"].norm()), 0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_checkpoint_phase_and_reference_hashes(self):
        result = training.fit_trajectory(self.train, self.validation, "C-S-R", 20261002, max_updates=5)
        self.assertEqual(sorted(result["checkpoints"]), [0, 1, 2, 5])
        self.assertEqual([g["pre_update_index"] for g in result["gradient_vectors"]], [0, 1, 2, 3, 4])
        self.assertEqual(result["optimizer_updates"], 5)
        zero = result["checkpoints"][0]
        torch.testing.assert_close(zero["train_logq"], self.train["log_anchor"], atol=1e-12, rtol=0)
        self.assertEqual(zero["completed_updates"], 0)
        initial = zero["identity"]["reference_current_sha256"]
        for step, packet in result["checkpoints"].items():
            self.assertEqual(packet["completed_updates"], step)
            self.assertEqual(packet["identity"]["reference_current_sha256"], initial)
            self.assertTrue(packet["identity"]["references_match_initial"])
            self.assertIn("gradient_norms", packet)
            self.assertIn("branch_diagnostics", packet)
        rebuilt = ICRoleModel("C-S-R", seed=20261002)
        rebuilt.load_state_dict(result["checkpoints"][5]["state_dict"])
        output = rebuilt(self.validation["h"], self.validation["e"], self.validation["log_anchor"])
        torch.testing.assert_close(output["logq"], result["checkpoints"][5]["validation_logq"], atol=0, rtol=0)
        self.assertTrue(reference_identity(rebuilt)["references_match_initial"])

    def test_validation_does_not_choose_epoch_or_change_trajectory(self):
        first = training.fit_trajectory(self.train, self.validation, "C-S-U", 20261002, max_updates=5)
        changed = copy.deepcopy(self.validation)
        changed["labels"] = (changed["labels"]+1) % 4
        changed["h"] += 100
        for i, record in enumerate(changed["records"]):
            record["four_class_id"] = str(int(changed["labels"][i]))
        second = training.fit_trajectory(self.train, changed, "C-S-U", 20261002, max_updates=5)
        for step in first["checkpoints"]:
            for key, value in first["checkpoints"][step]["state_dict"].items():
                torch.testing.assert_close(value, second["checkpoints"][step]["state_dict"][key], atol=0, rtol=0)
        self.assertNotIn("winner", first)
        self.assertNotIn("best_epoch", first)
        self.assertEqual(first["primary_checkpoint"], 100)

    def test_fixed_matrix_has_108_without_refits(self):
        matrix = training.build_fit_matrix()
        self.assertEqual(len(matrix), 108)
        self.assertEqual(len({(r["unit"], r["fold"], r["seed"]) for r in matrix}), 108)
        self.assertEqual([r["unit"] for r in matrix[::9]], list(UNITS))
        self.assertTrue(all(r["max_updates"] == 400 and r["primary_checkpoint"] == 100 for r in matrix))
        self.assertEqual(matrix[0]["seed"], 20261002)
        self.assertEqual(matrix[1]["seed"], 20261003)
        self.assertTrue(all(r["stage"] == "inner-fit-trajectory" for r in matrix))

    def test_failure_returns_no_promoted_partial_result(self):
        original = training.gradient_components
        count = 0

        def fail_third(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise ValueError("Synthetic gradient interruption")
            return original(*args, **kwargs)

        with patch.object(training, "gradient_components", side_effect=fail_third):
            with self.assertRaisesRegex(ValueError, "interruption"):
                training.fit_trajectory(self.train, self.validation, "C-S-R", 20261002, max_updates=5)
        self.assertEqual(count, 3)

    def test_all_twelve_units_short_trajectory_and_bad_scope(self):
        for unit in UNITS:
            result = training.fit_trajectory(self.train, self.validation, unit, 20261002, max_updates=1)
            self.assertEqual(result["optimizer_updates"], 1)
            self.assertEqual(len(result["initial_kernels"]), 3)
        for updates in (-1, 401, True):
            with self.assertRaises(ValueError):
                training.fit_trajectory(self.train, self.validation, "C-S-R", 20261002, max_updates=updates)
        with self.assertRaises(ValueError):
            training.fit_trajectory(self.train, None, "C-S-R", 20261002, max_updates=1)
        bad = copy.deepcopy(self.validation)
        bad["records"][0]["split_group_id"] = self.train["records"][0]["split_group_id"]
        with self.assertRaises(ValueError):
            training.fit_trajectory(self.train, bad, "C-S-R", 20261002, max_updates=1)
        bad = copy.deepcopy(self.validation)
        bad["records"][0]["tc_subset"] = "projector_stop"
        with self.assertRaises(ValueError):
            training.fit_trajectory(self.train, bad, "C-S-R", 20261002, max_updates=1)


if __name__ == "__main__":
    unittest.main()
