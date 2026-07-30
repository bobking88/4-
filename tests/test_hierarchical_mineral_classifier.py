from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@dataclass(frozen=True)
class FakeRecord:
    mineral_label: str
    four_class_label: str


class HierarchicalMineralClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def test_species_target_tensor_follows_the_frozen_mapping(self) -> None:
        from mineral_hierarchy import aggregate_role_probabilities, build_species_mapping
        from train_hierarchical_mineral_classifier import species_target_tensor

        records = [
            FakeRecord("rutile", "ti_bearing_negative"),
            FakeRecord("magnetite", "target_mineral"),
            FakeRecord("quartz", "gangue_negative"),
        ]
        mapping = build_species_mapping(records)

        targets = species_target_tensor(records, mapping, self.torch)

        self.assertEqual(targets.tolist(), [2, 0, 1])

    def test_consistency_loss_is_lower_when_role_logits_match_species_aggregation(self) -> None:
        from mineral_hierarchy import aggregate_role_probabilities, build_species_mapping
        from train_hierarchical_mineral_classifier import compute_hierarchy_consistency_loss

        mapping = build_species_mapping(
            [
                FakeRecord("magnetite", "target_mineral"),
                FakeRecord("ilmenite", "target_mineral"),
                FakeRecord("rutile", "ti_bearing_negative"),
                FakeRecord("quartz", "gangue_negative"),
            ]
        )
        species_logits = self.torch.tensor([[1.0, 1.0, 3.0, -2.0]], dtype=self.torch.float32)
        mapped_probabilities = aggregate_role_probabilities(
            self.torch.softmax(species_logits, dim=1), mapping, self.torch
        )
        mapped_probabilities = mapped_probabilities.clamp_min(self.torch.finfo(self.torch.float32).eps)
        mapped_probabilities = mapped_probabilities / mapped_probabilities.sum(dim=1, keepdim=True)
        matching_role_logits = mapped_probabilities.log()
        mismatched_role_logits = self.torch.tensor([[4.0, -2.0, 3.0, -2.0]], dtype=self.torch.float32)

        matching_loss = compute_hierarchy_consistency_loss(
            matching_role_logits, species_logits, mapping, self.torch
        )
        mismatched_loss = compute_hierarchy_consistency_loss(
            mismatched_role_logits, species_logits, mapping, self.torch
        )

        self.assertLess(float(matching_loss), float(mismatched_loss))

    def test_model_supports_a_backward_pass_with_all_heads(self) -> None:
        from train_hierarchical_mineral_classifier import HierarchicalRoleAwareEfficientNet
        from train_mineral_classifier import require_training_dependencies

        dependencies = require_training_dependencies()
        model = HierarchicalRoleAwareEfficientNet(
            dependencies["models"], num_roles=4, num_species=17, pretrained=False, embedding_dim=8
        )
        images = self.torch.randn(2, 3, 64, 64)
        role_logits, species_logits, binary_logits, embeddings = model(images)
        loss = role_logits.sum() + species_logits.sum() + binary_logits.sum() + embeddings.sum()

        loss.backward()

        self.assertIsNotNone(model.role_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
