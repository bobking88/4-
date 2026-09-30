import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_abmp_rsg_development import (
    development_grid, evaluate_policy, select_development_candidate,
    train_policy, prepare_output_directory,
)


def candidate(nll=.8, recall=.6, activation=.1, eps=.04):
    return {
        "config": {"epsilon_max": eps, "tau_p": .7, "tau_m": .1, "delta": .01, "lambda_cal": 0.0},
        "metrics": {"nll": nll, "target_recall": recall},
        "audit": {"posterior_violations": 0, "margin_violations": 0, "anchor_count": 12, "anchor_retention_rate": 1., "posterior_conditional_activation_rate": activation, "margin_conditional_activation_rate": 0., "max_difference_from_q0": .02, "max_difference_from_unbounded": .01, "epsilon_mean": .02, "anchor_coverage": .2, "epsilon_std": .003},
    }


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_registered_grid_has_27_unique_candidates(self):
        grid = development_grid()
        self.assertEqual(len(grid), 27)
        self.assertEqual(len({tuple(sorted(c.items())) for c in grid}), 27)
        self.assertEqual({c["epsilon_max"] for c in grid}, {.02, .04, .08})

    def test_selection_rejects_nonfinite_or_inactive_or_unsafe(self):
        for change in ({"nll": float("nan")}, {"activation": 0}, {"recall": .58}):
            self.assertIsNone(select_development_candidate([candidate(**change)], {"target_recall": .6}))
        broken = candidate()
        broken["audit"]["posterior_violations"] = 1
        self.assertIsNone(select_development_candidate([broken], {"target_recall": .6}))

    def test_selection_minimizes_nll_then_budget(self):
        larger = candidate(nll=.8, eps=.08)
        smaller = candidate(nll=.80005, eps=.02)
        best = select_development_candidate([larger, smaller], {"target_recall": .6})
        self.assertIs(best, smaller)
        strict_best = candidate(nll=.79, eps=.08)
        self.assertIs(select_development_candidate([strict_best, smaller], {"target_recall": .6}), strict_best)

    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                prepare_output_directory(Path(folder))

    def test_training_changes_heads_and_returns_finite_audit(self):
        torch.manual_seed(23)
        q0 = torch.rand(24, 4)
        q0 /= q0.sum(1, keepdim=True)
        qp = torch.rand(24, 4)
        qp /= qp.sum(1, keepdim=True)
        from tc_oos_rsg import build_projection_evidence
        evidence = build_projection_evidence(q0, qp, torch.full((24, 1), .5), torch.full((24, 1), .6), torch.full((24, 1), .4))
        cache = {"q0": q0, "q_candidate": qp, "evidence": evidence, "labels": torch.arange(24) % 4}
        trained = train_policy(cache, cache, development_grid()[0], epochs=3, seed=1)
        self.assertEqual(len(trained["history"]), 3)
        self.assertNotEqual(trained["initial_state_sha256"], trained["state_sha256"])
        result = evaluate_policy(cache, trained)
        self.assertTrue(torch.isfinite(result["probabilities"]).all())
        for name, value in result["audit"].items():
            if name.endswith("violations"):
                self.assertEqual(value, 0, name)


if __name__ == "__main__":
    unittest.main()
