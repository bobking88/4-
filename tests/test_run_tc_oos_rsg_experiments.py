from __future__ import annotations

import csv
import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _row(image_id: str, group_id: str) -> dict[str, str]:
    return {"image_id": image_id, "split_group_id": group_id}


def _candidate(epsilon: float, recall: float, nll: float, epoch: int = 1):
    from run_tc_oos_rsg_experiments import ProjectionCandidate

    return ProjectionCandidate(
        state_dict={},
        epoch=epoch,
        epsilon_target=epsilon,
        stop_metrics={"target_recall": recall, "nll": nll},
        state_sha256=f"state-{epsilon}-{epoch}",
    )


class TCOOSRSGPipelineTests(unittest.TestCase):
    @staticmethod
    def _training_cache() -> dict[str, torch.Tensor]:
        return {
            "features": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]],
                dtype=torch.float32,
            ),
            "direct": torch.tensor(
                [
                    [0.70, 0.10, 0.10, 0.10],
                    [0.10, 0.70, 0.10, 0.10],
                    [0.60, 0.20, 0.10, 0.10],
                    [0.15, 0.65, 0.10, 0.10],
                ],
                dtype=torch.float32,
            ),
            "mapped": torch.tensor(
                [
                    [0.40, 0.30, 0.20, 0.10],
                    [0.30, 0.40, 0.20, 0.10],
                    [0.45, 0.35, 0.10, 0.10],
                    [0.25, 0.45, 0.20, 0.10],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
            "ti": torch.full((4, 1), 0.9, dtype=torch.float32),
            "metal": torch.full((4, 1), 0.9, dtype=torch.float32),
        }

    def test_gate_and_projector_trainers_return_hashed_locked_states(self) -> None:
        from run_tc_oos_rsg_experiments import (
            SelectedGate,
            build_projection_cache,
            train_oos_gate,
            train_projection_head,
        )

        fit = self._training_cache()
        stop = {key: value.clone() for key, value in fit.items()}
        gate_network = torch.nn.Linear(2, 1)
        selected_gate = train_oos_gate(
            gate_network,
            fit,
            stop,
            epochs=2,
            seed=7,
            batch_size=2,
        )
        self.assertIsInstance(selected_gate, SelectedGate)
        self.assertEqual(len(selected_gate.state_sha256), 64)
        self.assertIn("nll", selected_gate.stop_metrics)

        gate_network.load_state_dict(selected_gate.state_dict)
        gate_network.eval()
        with torch.no_grad():
            fit_gate = torch.sigmoid(gate_network(fit["features"]))
            stop_gate = torch.sigmoid(gate_network(stop["features"]))
        projection = train_projection_head(
            build_projection_cache(fit_gate, fit),
            build_projection_cache(stop_gate, stop),
            epsilon_target=0.01,
            epochs=2,
            seed=8,
            batch_size=2,
        )
        self.assertEqual(projection.epsilon_target, 0.01)
        self.assertEqual(len(projection.state_sha256), 64)
        self.assertLessEqual(projection.stop_metrics["max_target_harm"], 0.010001)

    def test_gate_and_projector_use_disjoint_update_sets(self) -> None:
        from run_tc_oos_rsg_experiments import audit_pipeline_update_sets

        subsets = {
            "expert_fit": [_row("expert", "g-expert")],
            "gate_fit": [_row("gate", "g-gate")],
            "gate_stop_projector_fit": [_row("projector", "g-projector")],
            "projector_stop": [_row("stop", "g-stop")],
            "outer_eval": [_row("outer", "g-outer")],
        }
        audit = audit_pipeline_update_sets(subsets)
        self.assertEqual(audit["cross_subset_group_overlap_count"], 0)
        self.assertEqual(
            audit["parameter_update_subsets"],
            {"expert": "expert_fit", "gate": "gate_fit", "projector": "gate_stop_projector_fit"},
        )

        subsets["gate_stop_projector_fit"][0]["split_group_id"] = "g-gate"
        with self.assertRaisesRegex(ValueError, "group overlap"):
            audit_pipeline_update_sets(subsets)

    def test_candidate_posteriors_apply_verifiers_before_projection(self) -> None:
        from hrgv_network import apply_residual_target_verifiers
        from run_tc_oos_rsg_experiments import build_verifier_complete_branches

        direct = torch.tensor(
            [[0.70, 0.10, 0.10, 0.10], [0.20, 0.50, 0.20, 0.10]],
            dtype=torch.float64,
        )
        mapped = torch.tensor(
            [[0.30, 0.40, 0.20, 0.10], [0.50, 0.10, 0.20, 0.20]],
            dtype=torch.float64,
        )
        gate = torch.tensor([[0.8], [0.2]], dtype=torch.float64)
        ti = torch.tensor([[0.2], [0.9]], dtype=torch.float64)
        metallic = torch.tensor([[0.9], [0.3]], dtype=torch.float64)
        cache = {"direct": direct, "mapped": mapped, "ti": ti, "metal": metallic}

        branches = build_verifier_complete_branches(gate, cache)

        expected_q0 = apply_residual_target_verifiers(
            0.5 * direct + 0.5 * mapped, ti, metallic
        )
        expected_q_phi = apply_residual_target_verifiers(
            gate * direct + (1.0 - gate) * mapped, ti, metallic
        )
        self.assertTrue(torch.allclose(branches["q0"], expected_q0))
        self.assertTrue(torch.allclose(branches["q_candidate"], expected_q_phi))
        self.assertFalse(
            torch.allclose(branches["q0"], 0.5 * direct + 0.5 * mapped)
        )

    def test_selector_uses_recall_then_nll_then_smaller_epsilon(self) -> None:
        from run_tc_oos_rsg_experiments import select_projection_candidate

        selected = select_projection_candidate(
            [
                _candidate(0.0, recall=0.78, nll=0.20),
                _candidate(0.02, recall=0.795, nll=0.80),
                _candidate(0.01, recall=0.795, nll=0.80),
                _candidate(0.04, recall=0.81, nll=0.90),
            ],
            baseline_metrics={"target_recall": 0.80, "nll": 1.00},
        )

        self.assertEqual(selected.epsilon_target, 0.01)
        self.assertEqual(selected.stop_metrics["nll"], 0.80)

    def test_selector_returns_explicit_fallback_when_no_candidate_is_safe(self) -> None:
        from run_tc_oos_rsg_experiments import FallbackSelection, select_projection_candidate

        selected = select_projection_candidate(
            [_candidate(0.08, recall=0.70, nll=0.10)],
            baseline_metrics={"target_recall": 0.80, "nll": 1.00},
        )

        self.assertIsInstance(selected, FallbackSelection)
        self.assertEqual(selected.reason, "no_safe_candidate")
        self.assertEqual(selected.baseline_metrics["nll"], 1.00)

    def test_outer_tensor_loader_runs_only_after_selection_lock(self) -> None:
        from run_tc_oos_rsg_experiments import lock_before_outer_evaluation

        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "selection_lock.json"
            events: list[str] = []

            def load_outer():
                self.assertTrue(lock_path.is_file())
                self.assertEqual(
                    json.loads(lock_path.read_text(encoding="utf-8"))["status"],
                    "locked_before_outer_tensor_loading",
                )
                return "outer-cache"

            result = lock_before_outer_evaluation(
                lock_path,
                {"method": "TC-OOS-RSG"},
                load_outer,
                event_trace=events,
            )

            self.assertEqual(result, "outer-cache")
            self.assertEqual(events, ["lock_written", "outer_tensor_loaded"])

    def test_all_six_registered_epsilon_values_are_evaluated(self) -> None:
        from run_tc_oos_rsg_experiments import (
            EPSILON_TARGET_GRID,
            train_registered_projection_candidates,
        )

        seen: list[float] = []

        def trainer(epsilon_target: float):
            seen.append(epsilon_target)
            return _candidate(epsilon_target, recall=0.80, nll=1.0 + epsilon_target)

        candidates = train_registered_projection_candidates(trainer)

        self.assertEqual(tuple(seen), EPSILON_TARGET_GRID)
        self.assertEqual(
            tuple(candidate.epsilon_target for candidate in candidates),
            EPSILON_TARGET_GRID,
        )

    def test_method_matrix_exports_m0_through_m6_with_safe_main_routes(self) -> None:
        from run_tc_oos_rsg_experiments import (
            ProjectionCandidate,
            build_projection_cache,
            evaluate_method_matrix,
            state_dict_sha256,
        )
        from tc_oos_rsg import TargetRiskProjectionHead

        cache = self._training_cache()
        gate = torch.tensor([[0.9], [0.1], [0.8], [0.2]], dtype=torch.float32)
        projection_cache = build_projection_cache(gate, cache)
        head = TargetRiskProjectionHead()
        state = {name: value.detach().clone() for name, value in head.state_dict().items()}
        candidate = ProjectionCandidate(
            state_dict=state,
            epoch=1,
            epsilon_target=0.01,
            stop_metrics={"target_recall": 1.0, "nll": 1.0},
            state_sha256=state_dict_sha256(state),
        )

        methods = evaluate_method_matrix(
            projection_cache,
            selected=candidate,
            hard_negative_selected=candidate,
            m0_probabilities=projection_cache["q0"],
        )

        self.assertEqual(set(methods), {f"M{index}" for index in range(7)})
        for name, result in methods.items():
            self.assertEqual(result["probabilities"].shape, (4, 4), name)
        self.assertLessEqual(float(methods["M5"]["max_target_harm"]), 0.010001)
        self.assertLessEqual(float(methods["M6"]["max_target_harm"]), 0.010001)

    def test_evidence_ablations_mask_only_registered_dimensions(self) -> None:
        from run_tc_oos_rsg_experiments import build_evidence_ablations

        evidence = torch.arange(36, dtype=torch.float32).reshape(2, 18)
        ablations = build_evidence_ablations(evidence)

        self.assertEqual(
            set(ablations),
            {"remove_verifier_probabilities", "remove_abs_difference", "remove_uncertainty"},
        )
        self.assertTrue(torch.equal(ablations["remove_verifier_probabilities"][:, 13:15], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(ablations["remove_abs_difference"][:, 8:12], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(ablations["remove_uncertainty"][:, 15:18], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(ablations["remove_verifier_probabilities"][:, :13], evidence[:, :13]))

    def test_selection_lock_hashes_every_manifest_and_locked_model(self) -> None:
        from run_tc_oos_rsg_experiments import build_selection_lock_payload

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = {"gate_fit": root / "gate.csv", "outer_eval": root / "outer.csv"}
            models = {"expert": root / "expert.pt", "gate": root / "gate.pt"}
            for path in [*manifests.values(), *models.values()]:
                path.write_bytes(path.name.encode("utf-8"))

            payload = build_selection_lock_payload(
                manifest_paths=manifests,
                model_paths=models,
                selected_epsilon=0.01,
                selection_reason="minimum_safe_nll",
                gate_state_sha256="g" * 64,
                projector_state_sha256="p" * 64,
            )

            self.assertEqual(payload["selected_epsilon_target"], 0.01)
            self.assertFalse(payload["outer_tensors_loaded"])
            for name, path in manifests.items():
                self.assertEqual(
                    payload["manifest_sha256"][name],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            for name, path in models.items():
                self.assertEqual(
                    payload["model_file_sha256"][name],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )

    def test_unbounded_route_is_a_true_no_safety_layer_ablation(self) -> None:
        from run_tc_oos_rsg_experiments import project_with_optional_safety

        q0 = torch.tensor([[0.8, 0.1, 0.05, 0.05]], dtype=torch.float64)
        q_candidate = torch.tensor([[0.2, 0.6, 0.1, 0.1]], dtype=torch.float64)
        raw_route = torch.ones((1, 1), dtype=torch.float64)

        bounded = project_with_optional_safety(
            q0, q_candidate, raw_route, epsilon_target=0.0, apply_safety=True
        )
        unbounded = project_with_optional_safety(
            q0, q_candidate, raw_route, epsilon_target=0.0, apply_safety=False
        )

        self.assertAlmostEqual(float(bounded["final_probabilities"][0, 0]), 0.8)
        self.assertAlmostEqual(float(unbounded["final_probabilities"][0, 0]), 0.2)
        self.assertAlmostEqual(float(unbounded["projected_route"][0, 0]), 1.0)

    def test_prediction_writer_accepts_manifest_record_class_id(self) -> None:
        from run_tc_oos_rsg_experiments import _write_prediction_csv

        record = SimpleNamespace(
            image_id="sample-1",
            split_group_id="group-1",
            mineral_label="ilmenite",
            class_id=2,
        )
        probabilities = torch.tensor([[0.1, 0.2, 0.6, 0.1]])

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "predictions.csv"
            _write_prediction_csv(
                output_path,
                records=[record],
                probabilities=probabilities,
                route=None,
                fold=0,
            )
            with output_path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["true_class_id"], "2")


if __name__ == "__main__":
    unittest.main()
