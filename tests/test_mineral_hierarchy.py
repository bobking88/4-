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


class MineralHierarchyTests(unittest.TestCase):
    def test_mapping_uses_sorted_species_and_the_fixed_role_order(self) -> None:
        from mineral_hierarchy import build_species_mapping

        mapping = build_species_mapping(
            [
                FakeRecord("rutile", "ti_bearing_negative"),
                FakeRecord("magnetite", "target_mineral"),
                FakeRecord("quartz", "gangue_negative"),
            ]
        )

        self.assertEqual(mapping.species_labels, ("magnetite", "quartz", "rutile"))
        self.assertEqual(mapping.species_role_ids, (0, 2, 1))
        self.assertEqual(mapping.species_to_index["rutile"], 2)

    def test_mapping_rejects_a_species_assigned_to_more_than_one_role(self) -> None:
        from mineral_hierarchy import build_species_mapping

        with self.assertRaisesRegex(ValueError, "multiple roles"):
            build_species_mapping(
                [
                    FakeRecord("magnetite", "target_mineral"),
                    FakeRecord("magnetite", "metallic_hard_negative"),
                ]
            )

    def test_aggregate_role_probabilities_sums_species_in_each_role(self) -> None:
        from mineral_hierarchy import aggregate_role_probabilities, build_species_mapping
        from train_mineral_classifier import require_training_dependencies

        torch = require_training_dependencies()["torch"]
        mapping = build_species_mapping(
            [
                FakeRecord("magnetite", "target_mineral"),
                FakeRecord("ilmenite", "target_mineral"),
                FakeRecord("rutile", "ti_bearing_negative"),
                FakeRecord("quartz", "gangue_negative"),
            ]
        )
        probabilities = torch.tensor([[0.10, 0.20, 0.30, 0.40]], dtype=torch.float32)

        role_probabilities = aggregate_role_probabilities(probabilities, mapping, torch)

        self.assertTrue(torch.allclose(role_probabilities, torch.tensor([[0.30, 0.40, 0.30, 0.0]])))
        self.assertTrue(torch.allclose(role_probabilities.sum(dim=1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
