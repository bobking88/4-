from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class HRGVProbabilityPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def test_entropy_and_js_divergence_have_batch_column_shape(self) -> None:
        from hrgv_network import jensen_shannon_divergence, normalized_entropy

        first = self.torch.tensor(
            [[0.70, 0.10, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]],
            dtype=self.torch.float32,
        )
        second = self.torch.tensor(
            [[0.60, 0.20, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]],
            dtype=self.torch.float32,
        )

        entropy = normalized_entropy(first, self.torch)
        divergence = jensen_shannon_divergence(first, second, self.torch)

        self.assertEqual(tuple(entropy.shape), (2, 1))
        self.assertEqual(tuple(divergence.shape), (2, 1))
        self.assertTrue(self.torch.isfinite(entropy).all())
        self.assertTrue(self.torch.isfinite(divergence).all())
        self.assertAlmostEqual(float(divergence[1]), 0.0, places=6)

    def test_js_divergence_rejects_mismatched_shapes(self) -> None:
        from hrgv_network import jensen_shannon_divergence

        first = self.torch.full((2, 4), 0.25)
        second = self.torch.full((2, 3), 1.0 / 3.0)

        with self.assertRaisesRegex(ValueError, "same shape"):
            jensen_shannon_divergence(first, second, self.torch)

    def test_mixture_obeys_gate_endpoints_and_normalization(self) -> None:
        from hrgv_network import mix_role_experts

        direct = self.torch.tensor([[0.70, 0.10, 0.10, 0.10]], dtype=self.torch.float32)
        mapped = self.torch.tensor([[0.20, 0.40, 0.30, 0.10]], dtype=self.torch.float32)

        direct_only = mix_role_experts(direct, mapped, self.torch.ones(1, 1))
        mapped_only = mix_role_experts(direct, mapped, self.torch.zeros(1, 1))
        halfway = mix_role_experts(direct, mapped, self.torch.full((1, 1), 0.5))

        self.assertTrue(self.torch.allclose(direct_only, direct))
        self.assertTrue(self.torch.allclose(mapped_only, mapped))
        self.assertTrue(self.torch.allclose(halfway.sum(dim=1), self.torch.ones(1)))
        self.assertTrue(self.torch.allclose(halfway, (direct + mapped) / 2.0))

    def test_residual_verifier_has_neutral_zone_and_target_monotonicity(self) -> None:
        from hrgv_network import apply_residual_target_verifiers

        fused = self.torch.tensor([[0.60, 0.20, 0.10, 0.10]], dtype=self.torch.float32)
        supportive = apply_residual_target_verifiers(
            fused,
            self.torch.full((1, 1), 0.80),
            self.torch.full((1, 1), 0.90),
        )
        suppress_ti = apply_residual_target_verifiers(
            fused,
            self.torch.full((1, 1), 0.25),
            self.torch.ones(1, 1),
        )
        suppress_both = apply_residual_target_verifiers(
            fused,
            self.torch.full((1, 1), 0.25),
            self.torch.full((1, 1), 0.25),
        )

        self.assertTrue(self.torch.allclose(supportive, fused, atol=1e-6))
        self.assertLess(float(suppress_ti[0, 0]), float(supportive[0, 0]))
        self.assertLess(float(suppress_both[0, 0]), float(suppress_ti[0, 0]))
        self.assertTrue(self.torch.allclose(suppress_both.sum(dim=1), self.torch.ones(1)))

    def test_multiplicative_pilot_penalizes_supportive_verifiers(self) -> None:
        from hrgv_network import apply_multiplicative_target_verifiers

        fused = self.torch.tensor([[0.60, 0.20, 0.10, 0.10]], dtype=self.torch.float32)
        corrected = apply_multiplicative_target_verifiers(
            fused,
            self.torch.full((1, 1), 0.80),
            self.torch.full((1, 1), 0.90),
        )

        self.assertLess(float(corrected[0, 0]), float(fused[0, 0]))


class RPGProbabilityPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def test_partitioned_entropy_obeys_chain_rule(self) -> None:
        from hrgv_network import role_partitioned_uncertainty

        species = self.torch.tensor([[0.20, 0.30, 0.10, 0.40]])
        role_matrix = self.torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

        values = role_partitioned_uncertainty(species, role_matrix, self.torch)

        self.assertTrue(self.torch.allclose(
            values["total_species_entropy"],
            values["between_role_entropy"] + values["within_role_entropy"],
            atol=1e-6,
        ))

    def test_within_role_redistribution_preserves_mapped_role_posterior(self) -> None:
        from hrgv_network import role_partitioned_uncertainty

        role_matrix = self.torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        first = role_partitioned_uncertainty(
            self.torch.tensor([[0.20, 0.30, 0.10, 0.40]]), role_matrix, self.torch
        )
        second = role_partitioned_uncertainty(
            self.torch.tensor([[0.45, 0.05, 0.25, 0.25]]), role_matrix, self.torch
        )

        self.assertTrue(self.torch.allclose(
            first["mapped_role_probabilities"], second["mapped_role_probabilities"]
        ))

    def test_singleton_role_has_zero_conditional_entropy(self) -> None:
        from hrgv_network import role_partitioned_uncertainty

        values = role_partitioned_uncertainty(
            self.torch.tensor([[0.25, 0.75]]), self.torch.eye(2), self.torch
        )

        self.assertTrue(self.torch.allclose(
            values["within_role_entropy"], self.torch.zeros((1, 1))
        ))

    def test_gate_feature_modes_do_not_conflate_partitioned_uncertainties(self) -> None:
        from hrgv_network import role_partitioned_gate_features

        partition = {
            "between_role_entropy": self.torch.tensor([[0.4]]),
            "within_role_entropy": self.torch.tensor([[0.3]]),
            "total_species_entropy": self.torch.tensor([[0.7]]),
        }

        features = role_partitioned_gate_features(
            self.torch.tensor([[0.6]]), partition, "partitioned", self.torch
        )
        without_between = role_partitioned_gate_features(
            self.torch.tensor([[0.6]]), partition, "without_between", self.torch
        )

        self.assertTrue(self.torch.allclose(
            features, self.torch.tensor([[0.4, 0.3, 0.2]])
        ))
        self.assertTrue(self.torch.allclose(
            without_between, self.torch.tensor([[0.0, 0.3, 0.0]])
        ))


class CGDCProbabilityPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def test_disagreement_calibration_is_identity_when_experts_agree(self) -> None:
        from hrgv_network import disagreement_calibrated_role_probabilities

        posterior = self.torch.tensor(
            [[0.70, 0.10, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]],
            dtype=self.torch.float32,
        )
        residual = self.torch.tensor(
            [[3.0, -2.0, 1.0, -4.0], [-2.0, 3.0, -1.0, 4.0]],
            dtype=self.torch.float32,
        )

        calibrated, gain = disagreement_calibrated_role_probabilities(
            posterior, posterior, posterior, residual, self.torch
        )

        self.assertTrue(self.torch.allclose(gain, self.torch.zeros((2, 1))))
        self.assertTrue(self.torch.allclose(calibrated, posterior, atol=1e-6))

    def test_disagreement_calibration_keeps_simplex_and_bounded_log_odds_shift(self) -> None:
        from hrgv_network import disagreement_calibrated_role_probabilities

        direct = self.torch.tensor([[0.70, 0.10, 0.10, 0.10]], dtype=self.torch.float32)
        mapped = self.torch.tensor([[0.10, 0.70, 0.10, 0.10]], dtype=self.torch.float32)
        fused = 0.5 * (direct + mapped)
        residual = self.torch.tensor([[7.0, -7.0, 1.0, -1.0]], dtype=self.torch.float32)

        calibrated, gain = disagreement_calibrated_role_probabilities(
            fused, direct, mapped, residual, self.torch
        )
        odds_shift = ((calibrated[:, 0] / calibrated[:, 1]).log() - (fused[:, 0] / fused[:, 1]).log()).abs()

        self.assertTrue(self.torch.allclose(calibrated.sum(dim=1), self.torch.ones(1), atol=1e-6))
        self.assertTrue((calibrated >= 0).all())
        self.assertLessEqual(float(odds_shift[0]), float(2.0 * gain[0, 0]) + 1e-6)

    def test_disagreement_gain_has_global_half_budget(self) -> None:
        from hrgv_network import disagreement_calibrated_role_probabilities

        direct = self.torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=self.torch.float32)
        mapped = self.torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=self.torch.float32)
        fused = 0.5 * (direct + mapped)
        residual = self.torch.tensor([[5.0, -5.0, 0.0, 0.0]], dtype=self.torch.float32)

        _, gain = disagreement_calibrated_role_probabilities(
            fused, direct, mapped, residual, self.torch
        )

        self.assertGreater(float(gain[0, 0]), 0.49)
        self.assertLessEqual(float(gain[0, 0]), 0.5 + 1e-6)

    def test_adapter_decomposition_penalizes_aligned_residuals_more_than_orthogonal(self) -> None:
        from hrgv_network import adapter_decomposition_loss

        direct_delta = self.torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=self.torch.float32)
        orthogonal_delta = self.torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=self.torch.float32)

        orthogonal_loss = adapter_decomposition_loss(
            direct_delta, orthogonal_delta, self.torch
        )
        aligned_loss = adapter_decomposition_loss(direct_delta, direct_delta, self.torch)

        self.assertLess(float(orthogonal_loss), 1e-6)
        self.assertGreater(float(aligned_loss), 0.99)

    def test_masked_verifier_loss_uses_only_target_and_selected_negative(self) -> None:
        from hrgv_network import masked_verifier_loss

        logits = self.torch.tensor(
            [[0.0, 2.0], [2.0, 0.0], [8.0, -8.0], [-8.0, 8.0]],
            dtype=self.torch.float32,
            requires_grad=True,
        )
        roles = self.torch.tensor([0, 1, 2, 3], dtype=self.torch.long)
        criterion = self.torch.nn.CrossEntropyLoss()

        loss, eligible_count = masked_verifier_loss(
            logits,
            roles,
            negative_role_id=1,
            criterion=criterion,
        )

        expected = criterion(logits[:2], self.torch.tensor([1, 0]))
        self.assertEqual(eligible_count, 2)
        self.assertTrue(self.torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(self.torch.allclose(logits.grad[2:], self.torch.zeros_like(logits.grad[2:])))

    def test_masked_verifier_loss_returns_differentiable_zero_without_eligible_samples(self) -> None:
        from hrgv_network import masked_verifier_loss

        logits = self.torch.randn(3, 2, requires_grad=True)
        roles = self.torch.tensor([2, 2, 2], dtype=self.torch.long)

        loss, eligible_count = masked_verifier_loss(
            logits,
            roles,
            negative_role_id=1,
            criterion=self.torch.nn.CrossEntropyLoss(),
        )
        loss.backward()

        self.assertEqual(eligible_count, 0)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(self.torch.allclose(logits.grad, self.torch.zeros_like(logits.grad)))


class RegretGatePrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def test_regret_target_prefers_expert_with_lower_true_class_loss(self) -> None:
        from hrgv_network import regret_gate_targets

        direct = self.torch.tensor(
            [[0.70, 0.10, 0.10, 0.10], [0.10, 0.20, 0.60, 0.10]],
            dtype=self.torch.float32,
        )
        mapped = self.torch.tensor(
            [[0.30, 0.30, 0.20, 0.20], [0.10, 0.20, 0.20, 0.50]],
            dtype=self.torch.float32,
        )
        labels = self.torch.tensor([0, 2], dtype=self.torch.long)

        targets = regret_gate_targets(
            direct,
            mapped,
            labels,
            target_temperature=0.20,
            gap_temperature=0.50,
            torch=self.torch,
        )

        self.assertGreater(float(targets["soft_oracle_gate"][0]), 0.5)
        self.assertGreater(float(targets["soft_oracle_gate"][1]), 0.5)
        self.assertTrue(
            self.torch.equal(
                targets["hard_oracle_gate"],
                self.torch.ones((2, 1), dtype=self.torch.float32),
            )
        )

    def test_regret_gap_weight_increases_with_expert_loss_gap(self) -> None:
        from hrgv_network import regret_gate_targets

        direct = self.torch.tensor(
            [[0.55, 0.15, 0.15, 0.15], [0.90, 0.04, 0.03, 0.03]],
            dtype=self.torch.float32,
        )
        mapped = self.torch.tensor(
            [[0.45, 0.20, 0.20, 0.15], [0.10, 0.30, 0.30, 0.30]],
            dtype=self.torch.float32,
        )
        labels = self.torch.tensor([0, 0], dtype=self.torch.long)

        targets = regret_gate_targets(
            direct,
            mapped,
            labels,
            target_temperature=0.20,
            gap_temperature=0.50,
            torch=self.torch,
        )

        self.assertGreater(
            float(targets["gate_gap_weight"][1]),
            float(targets["gate_gap_weight"][0]),
        )

    def test_weighted_soft_gate_loss_rewards_matching_gate_targets(self) -> None:
        from hrgv_network import weighted_soft_gate_loss

        targets = self.torch.tensor([[0.90], [0.10]], dtype=self.torch.float32)
        weights = self.torch.tensor([[1.00], [0.50]], dtype=self.torch.float32)
        matching = self.torch.tensor([[0.85], [0.15]], dtype=self.torch.float32)
        opposing = self.torch.tensor([[0.15], [0.85]], dtype=self.torch.float32)

        matching_loss = weighted_soft_gate_loss(matching, targets, weights, self.torch)
        opposing_loss = weighted_soft_gate_loss(opposing, targets, weights, self.torch)

        self.assertLess(float(matching_loss), float(opposing_loss))

    def test_weighted_soft_gate_loss_returns_differentiable_zero_for_zero_weights(self) -> None:
        from hrgv_network import weighted_soft_gate_loss

        gate = self.torch.tensor([[0.30], [0.70]], requires_grad=True)
        targets = self.torch.tensor([[1.00], [0.00]])
        weights = self.torch.zeros((2, 1))

        loss = weighted_soft_gate_loss(gate, targets, weights, self.torch)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(gate.grad)
        self.assertTrue(self.torch.allclose(gate.grad, self.torch.zeros_like(gate.grad)))

    def test_routing_regret_bound_holds_for_deterministic_probability_pairs(self) -> None:
        from hrgv_network import gate_routing_diagnostics

        count = 200
        direct_true = self.torch.linspace(0.10, 0.90, count)
        mapped_true = self.torch.linspace(0.85, 0.15, count)
        gate = self.torch.linspace(0.01, 0.99, count).view(-1, 1)
        direct = self.torch.stack(
            [direct_true, (1.0 - direct_true) / 3.0, (1.0 - direct_true) / 3.0,
             (1.0 - direct_true) / 3.0],
            dim=1,
        )
        mapped = self.torch.stack(
            [mapped_true, (1.0 - mapped_true) / 3.0, (1.0 - mapped_true) / 3.0,
             (1.0 - mapped_true) / 3.0],
            dim=1,
        )
        labels = self.torch.zeros(count, dtype=self.torch.long)

        diagnostics = gate_routing_diagnostics(gate, direct, mapped, labels, self.torch)
        epsilon = 0.05
        upper_bound = (
            diagnostics["absolute_hard_gate_error"]
            * diagnostics["true_probability_gap"]
            / epsilon
        )

        self.assertTrue((diagnostics["routing_regret_nll"] >= -1e-7).all())
        self.assertTrue((diagnostics["routing_regret_nll"] <= upper_bound + 1e-6).all())

    def test_soft_oracle_approximation_bound_holds(self) -> None:
        from hrgv_network import regret_gate_targets

        direct_true = self.torch.linspace(0.10, 0.90, 200)
        mapped_true = self.torch.linspace(0.85, 0.15, 200)
        direct = self.torch.stack([direct_true, 1.0 - direct_true], dim=1)
        mapped = self.torch.stack([mapped_true, 1.0 - mapped_true], dim=1)
        labels = self.torch.zeros(200, dtype=self.torch.long)
        temperature = 0.20

        targets = regret_gate_targets(
            direct,
            mapped,
            labels,
            target_temperature=temperature,
            gap_temperature=0.50,
            torch=self.torch,
        )
        approximation_error = (
            targets["soft_oracle_gate"] - targets["hard_oracle_gate"]
        ).abs()
        upper_bound = self.torch.exp(-targets["expert_loss_gap"].abs() / temperature)

        self.assertTrue((approximation_error <= upper_bound + 1e-7).all())

    def test_gate_routing_diagnostics_identify_one_right_one_wrong_rows(self) -> None:
        from hrgv_network import gate_routing_diagnostics

        direct = self.torch.tensor(
            [[0.70, 0.10, 0.10, 0.10], [0.60, 0.10, 0.20, 0.10]],
            dtype=self.torch.float32,
        )
        mapped = self.torch.tensor(
            [[0.20, 0.60, 0.10, 0.10], [0.10, 0.60, 0.20, 0.10]],
            dtype=self.torch.float32,
        )
        gate = self.torch.tensor([[0.80], [0.30]], dtype=self.torch.float32)
        labels = self.torch.tensor([0, 1], dtype=self.torch.long)

        diagnostics = gate_routing_diagnostics(gate, direct, mapped, labels, self.torch)

        self.assertTrue(diagnostics["one_right_one_wrong"].all())
        self.assertTrue(diagnostics["hard_gate_selection_correct"].all())
        self.assertTrue((diagnostics["routing_regret_nll"] >= 0).all())
        self.assertEqual(tuple(diagnostics["fused_true_probability"].shape), (2, 1))


class HRGVModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.dependencies = require_training_dependencies()
        cls.torch = cls.dependencies["torch"]

    def build_mapping(self):
        from mineral_hierarchy import SpeciesRoleMapping

        labels = tuple(f"species_{index:02d}" for index in range(17))
        role_ids = (0, 0, 1, 1, 3, 3, 2, 2, 0, 1, 2, 3, 0, 1, 2, 3, 2)
        return SpeciesRoleMapping(
            species_labels=labels,
            species_role_ids=role_ids,
            species_to_index={label: index for index, label in enumerate(labels)},
        )

    def build_role_matrix(self, mapping):
        matrix = self.torch.zeros(4, len(mapping.species_labels), dtype=self.torch.float32)
        matrix[
            self.torch.tensor(mapping.species_role_ids, dtype=self.torch.long),
            self.torch.arange(len(mapping.species_labels)),
        ] = 1.0
        return matrix

    def test_model_output_contract_and_probability_invariants(self) -> None:
        from hrgv_network import HierarchicalRiskGatedVerificationNet

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
        )
        outputs = model(self.torch.randn(2, 3, 64, 64))

        expected_shapes = {
            "role_logits": (2, 4),
            "species_logits": (2, 17),
            "direct_role_probabilities": (2, 4),
            "mapped_role_probabilities": (2, 4),
            "gate": (2, 1),
            "fused_role_probabilities": (2, 4),
            "ti_verifier_logits": (2, 2),
            "metallic_verifier_logits": (2, 2),
            "ti_target_probability": (2, 1),
            "metallic_target_probability": (2, 1),
            "final_role_probabilities": (2, 4),
            "embeddings": (2, 8),
            "expert_js_divergence": (2, 1),
        }
        self.assertEqual(set(outputs), set(expected_shapes))
        for key, shape in expected_shapes.items():
            self.assertEqual(tuple(outputs[key].shape), shape, key)
            self.assertTrue(self.torch.isfinite(outputs[key]).all(), key)
        for key in (
            "direct_role_probabilities",
            "mapped_role_probabilities",
            "fused_role_probabilities",
            "final_role_probabilities",
        ):
            self.assertTrue(
                self.torch.allclose(outputs[key].sum(dim=1), self.torch.ones(2), atol=1e-6),
                key,
            )
        self.assertTrue(((outputs["gate"] >= 0) & (outputs["gate"] <= 1)).all())
        self.assertIn("role_matrix", dict(model.named_buffers()))
        self.assertEqual(model.verifier_mode, "residual")
        self.assertTrue(model.detach_verifier_features)

    def test_model_rejects_unknown_verifier_mode(self) -> None:
        from hrgv_network import HierarchicalRiskGatedVerificationNet

        mapping = self.build_mapping()

        with self.assertRaisesRegex(ValueError, "verifier_mode"):
            HierarchicalRiskGatedVerificationNet(
                self.dependencies["models"],
                self.build_role_matrix(mapping),
                pretrained=False,
                verifier_mode="unknown",
            )

    def test_cgdc_model_exposes_decomposed_experts_and_calibrated_posterior(self) -> None:
        from hrgv_network import HierarchicalRiskGatedVerificationNet

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
            enable_cgdc=True,
            calibration_hidden_dim=16,
        )
        outputs = model(self.torch.randn(2, 3, 64, 64))

        for key in (
            "direct_adapter_delta",
            "species_adapter_delta",
            "calibration_residual",
            "disagreement_gain",
            "calibrated_role_probabilities",
        ):
            self.assertIn(key, outputs)
            self.assertTrue(self.torch.isfinite(outputs[key]).all(), key)
        self.assertEqual(tuple(outputs["direct_adapter_delta"].shape), (2, 1280))
        self.assertEqual(tuple(outputs["species_adapter_delta"].shape), (2, 1280))
        self.assertEqual(tuple(outputs["calibration_residual"].shape), (2, 4))
        self.assertEqual(tuple(outputs["disagreement_gain"].shape), (2, 1))
        self.assertTrue(
            self.torch.allclose(
                outputs["calibrated_role_probabilities"].sum(dim=1),
                self.torch.ones(2),
                atol=1e-6,
            )
        )

    def test_rpg_model_exposes_role_partitioned_uncertainty(self) -> None:
        from hrgv_network import HierarchicalRiskGatedVerificationNet

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
            enable_rpg=True,
            rpg_entropy_mode="partitioned",
        )
        outputs = model(self.torch.randn(2, 3, 64, 64))

        for key in (
            "total_species_entropy",
            "between_role_entropy",
            "within_role_entropy",
        ):
            self.assertIn(key, outputs)
            self.assertEqual(tuple(outputs[key].shape), (2, 1))
            self.assertTrue(self.torch.isfinite(outputs[key]).all(), key)
        self.assertTrue(
            self.torch.allclose(
                outputs["total_species_entropy"],
                outputs["between_role_entropy"] + outputs["within_role_entropy"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            self.torch.allclose(
                outputs["mapped_role_probabilities"].sum(dim=1),
                self.torch.ones(2),
                atol=1e-6,
            )
        )

    def test_cgdc_loss_trains_adapters_and_calibration_head(self) -> None:
        from hrgv_network import (
            HRGVLossWeights,
            HierarchicalRiskGatedVerificationNet,
            compute_hrgv_losses,
        )

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
            enable_cgdc=True,
            calibration_hidden_dim=16,
        )
        outputs = model(self.torch.randn(6, 3, 64, 64))
        total_loss, terms = compute_hrgv_losses(
            outputs=outputs,
            role_labels=self.torch.tensor([0, 0, 1, 1, 3, 3], dtype=self.torch.long),
            species_labels=self.torch.tensor([0, 1, 2, 3, 4, 5], dtype=self.torch.long),
            mapping=mapping,
            final_role_criterion=self.torch.nn.NLLLoss(),
            direct_role_criterion=self.torch.nn.CrossEntropyLoss(),
            species_criterion=self.torch.nn.CrossEntropyLoss(),
            verifier_criterion=self.torch.nn.CrossEntropyLoss(),
            weights=HRGVLossWeights(decomposition=0.02, calibration=0.25),
            temperature=0.10,
            torch=self.torch,
        )
        total_loss.backward()

        self.assertIn("decomposition_loss", terms)
        self.assertIn("calibration_loss", terms)
        self.assertGreaterEqual(float(terms["decomposition_loss"].detach()), 0.0)
        self.assertGreater(float(terms["calibration_loss"].detach()), 0.0)
        self.assertIsNotNone(model.direct_adapter[0].weight.grad)
        self.assertIsNotNone(model.species_adapter[0].weight.grad)
        self.assertIsNotNone(model.calibration_network[0].weight.grad)

    def test_regret_gate_loss_is_isolated_from_backbone_and_experts_when_detached(self) -> None:
        from hrgv_network import (
            HierarchicalRiskGatedVerificationNet,
            regret_gate_targets,
            weighted_soft_gate_loss,
        )

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
            detach_gate_features=True,
        )
        outputs = model(self.torch.randn(4, 3, 64, 64))
        role_labels = self.torch.tensor([0, 1, 2, 3], dtype=self.torch.long)
        targets = regret_gate_targets(
            outputs["direct_role_probabilities"],
            outputs["mapped_role_probabilities"],
            role_labels,
            target_temperature=0.20,
            gap_temperature=0.50,
            torch=self.torch,
        )
        gate_loss = weighted_soft_gate_loss(
            outputs["gate"],
            targets["soft_oracle_gate"],
            targets["gate_gap_weight"],
            self.torch,
        )

        gate_loss.backward()

        self.assertIsNotNone(model.gate_network[0].weight.grad)
        self.assertGreater(float(model.gate_network[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(next(model.features.parameters()).grad)
        self.assertIsNone(model.role_head.weight.grad)
        self.assertIsNone(model.species_head.weight.grad)

    def test_zero_regret_weight_preserves_original_hrgv_total_loss(self) -> None:
        from hrgv_network import (
            HRGVLossWeights,
            HierarchicalRiskGatedVerificationNet,
            compute_hrgv_losses,
        )

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
        )
        role_labels = self.torch.tensor([0, 0, 1, 1, 3, 3], dtype=self.torch.long)
        species_labels = self.torch.tensor([0, 1, 2, 3, 4, 5], dtype=self.torch.long)
        outputs = model(self.torch.randn(6, 3, 64, 64))
        weights = HRGVLossWeights(gate_regret=0.0)

        total_loss, terms = compute_hrgv_losses(
            outputs=outputs,
            role_labels=role_labels,
            species_labels=species_labels,
            mapping=mapping,
            final_role_criterion=self.torch.nn.NLLLoss(),
            direct_role_criterion=self.torch.nn.CrossEntropyLoss(),
            species_criterion=self.torch.nn.CrossEntropyLoss(),
            verifier_criterion=self.torch.nn.CrossEntropyLoss(),
            weights=weights,
            temperature=0.10,
            torch=self.torch,
        )
        original_formula = (
            terms["final_role_loss"]
            + weights.direct * terms["direct_role_loss"]
            + weights.species * terms["species_loss"]
            + weights.consistency * terms["consistency_loss"]
            + weights.verifier * terms["verifier_loss"]
            + weights.contrast * terms["contrast_loss"]
        )

        self.assertTrue(self.torch.allclose(total_loss, original_formula, atol=1e-7))
        self.assertIn("gate_regret_loss", terms)

    def test_complete_loss_backpropagates_through_every_hrgv_module(self) -> None:
        from hrgv_network import (
            HRGVLossWeights,
            HierarchicalRiskGatedVerificationNet,
            compute_hrgv_losses,
        )

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
        )
        images = self.torch.randn(6, 3, 64, 64)
        role_labels = self.torch.tensor([0, 0, 1, 1, 3, 3], dtype=self.torch.long)
        species_labels = self.torch.tensor([0, 1, 2, 3, 4, 5], dtype=self.torch.long)
        outputs = model(images)

        total_loss, terms = compute_hrgv_losses(
            outputs=outputs,
            role_labels=role_labels,
            species_labels=species_labels,
            mapping=mapping,
            final_role_criterion=self.torch.nn.NLLLoss(),
            direct_role_criterion=self.torch.nn.CrossEntropyLoss(),
            species_criterion=self.torch.nn.CrossEntropyLoss(),
            verifier_criterion=self.torch.nn.CrossEntropyLoss(),
            weights=HRGVLossWeights(),
            temperature=0.10,
            torch=self.torch,
        )
        total_loss.backward()

        self.assertTrue(self.torch.isfinite(total_loss))
        self.assertEqual(
            set(terms),
            {
                "final_role_loss",
                "direct_role_loss",
                "species_loss",
                "consistency_loss",
                "ti_verifier_loss",
                "metallic_verifier_loss",
                "verifier_loss",
                "contrast_loss",
                "gate_regret_loss",
                "decomposition_loss",
                "calibration_loss",
            },
        )
        gradient_parameters = {
            "backbone": next(model.features.parameters()),
            "role_head": model.role_head.weight,
            "species_head": model.species_head.weight,
            "gate": model.gate_network[0].weight,
            "ti_verifier": model.ti_verifier_head.weight,
            "metallic_verifier": model.metallic_verifier_head.weight,
            "projection": model.projection_head[-1].weight,
        }
        for name, parameter in gradient_parameters.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(self.torch.isfinite(parameter.grad).all(), name)

    def test_verifier_only_loss_is_isolated_from_backbone_by_default(self) -> None:
        from hrgv_network import (
            HierarchicalRiskGatedVerificationNet,
            masked_verifier_loss,
        )

        mapping = self.build_mapping()
        model = HierarchicalRiskGatedVerificationNet(
            self.dependencies["models"],
            self.build_role_matrix(mapping),
            pretrained=False,
            embedding_dim=8,
            gate_hidden_dim=16,
        )
        outputs = model(self.torch.randn(6, 3, 64, 64))
        roles = self.torch.tensor([0, 0, 1, 1, 3, 3], dtype=self.torch.long)
        criterion = self.torch.nn.CrossEntropyLoss()
        ti_loss, _ = masked_verifier_loss(
            outputs["ti_verifier_logits"], roles, negative_role_id=1, criterion=criterion
        )
        metallic_loss, _ = masked_verifier_loss(
            outputs["metallic_verifier_logits"],
            roles,
            negative_role_id=3,
            criterion=criterion,
        )

        (ti_loss + metallic_loss).backward()

        self.assertIsNone(next(model.features.parameters()).grad)
        self.assertIsNotNone(model.ti_verifier_head.weight.grad)
        self.assertIsNotNone(model.metallic_verifier_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
