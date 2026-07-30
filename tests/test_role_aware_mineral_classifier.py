from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RoleAwareLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def test_target_binary_labels_map_only_target_class_to_one(self) -> None:
        from train_role_aware_mineral_classifier import compute_binary_inverse_frequency_weights, target_binary_labels

        labels = self.torch.tensor([0, 1, 2, 3])

        self.assertEqual(target_binary_labels(labels).tolist(), [1, 0, 0, 0])
        weights = compute_binary_inverse_frequency_weights([80, 20])
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(sum(weights) / 2, 1.0)

    def test_select_prediction_records_truncates_only_for_smoke_run(self) -> None:
        from train_role_aware_mineral_classifier import select_prediction_records

        records = ["a", "b", "c"]

        self.assertEqual(select_prediction_records(records, 3), records)
        self.assertEqual(select_prediction_records(records, 2), ["a", "b"])

    def test_role_aware_contrastive_loss_is_lower_when_target_and_hard_negatives_are_separated(self) -> None:
        from train_role_aware_mineral_classifier import compute_role_aware_contrastive_loss

        labels = self.torch.tensor([0, 0, 1, 1, 3, 3])
        separated = self.torch.tensor(
            [[1.0, 0.0], [0.98, 0.02], [-1.0, 0.0], [-0.98, 0.02], [0.0, -1.0], [0.02, -0.98]]
        )
        mixed = self.torch.tensor(
            [[1.0, 0.0], [0.98, 0.02], [0.96, 0.04], [0.94, 0.06], [0.92, 0.08], [0.90, 0.10]]
        )

        separated_loss = compute_role_aware_contrastive_loss(separated, labels, temperature=0.1, torch=self.torch)
        mixed_loss = compute_role_aware_contrastive_loss(mixed, labels, temperature=0.1, torch=self.torch)

        self.assertLess(float(separated_loss), float(mixed_loss))

    def test_role_aware_contrastive_loss_is_zero_without_a_same_class_positive(self) -> None:
        from train_role_aware_mineral_classifier import compute_role_aware_contrastive_loss

        labels = self.torch.tensor([0, 1, 3])
        embeddings = self.torch.eye(3)

        loss = compute_role_aware_contrastive_loss(embeddings, labels, temperature=0.1, torch=self.torch)

        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
