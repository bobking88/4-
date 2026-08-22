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
