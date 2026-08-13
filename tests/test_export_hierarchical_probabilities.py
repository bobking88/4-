from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@dataclass(frozen=True)
class FakeRecord:
    image_id: str
    mindat_photo_id: str
    mineral_label: str
    four_class_label: str
    class_id: int
    split_group_id: str
    image_path: Path


class ExportHierarchicalProbabilitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from train_mineral_classifier import require_training_dependencies

        cls.torch = require_training_dependencies()["torch"]

    def test_probability_rows_include_both_role_paths_and_metadata(self) -> None:
        from export_hierarchical_probabilities import build_probability_rows
        from mineral_hierarchy import build_species_mapping

        records = [
            FakeRecord(
                image_id="VTM-1",
                mindat_photo_id="101",
                mineral_label="magnetite",
                four_class_label="target_mineral",
                class_id=0,
                split_group_id="DG-1",
                image_path=Path("image.jpg"),
            )
        ]
        mapping = build_species_mapping(records)
        role_logits = self.torch.tensor([[2.0, 0.0, -1.0, -2.0]])
        species_logits = self.torch.tensor([[1.5]])
        metadata = {
            "VTM-1": {
                "locality": "Test locality",
                "photographer_or_credit": "Test photographer",
            }
        }

        rows = build_probability_rows(
            records, role_logits, species_logits, mapping, metadata, self.torch
        )

        row = rows[0]
        role_columns = [key for key in row if key.startswith("role_probability_")]
        mapped_columns = [key for key in row if key.startswith("mapped_role_probability_")]
        self.assertEqual(len(role_columns), 4)
        self.assertEqual(len(mapped_columns), 4)
        self.assertAlmostEqual(sum(float(row[key]) for key in role_columns), 1.0, places=5)
        self.assertAlmostEqual(sum(float(row[key]) for key in mapped_columns), 1.0, places=5)
        self.assertEqual(row["locality"], "Test locality")
        self.assertEqual(row["photographer_or_credit"], "Test photographer")

    def test_probability_row_lengths_must_match(self) -> None:
        from export_hierarchical_probabilities import build_probability_rows
        from mineral_hierarchy import build_species_mapping

        records = [
            FakeRecord("VTM-1", "101", "magnetite", "target_mineral", 0, "DG-1", Path("a.jpg"))
        ]
        mapping = build_species_mapping(records)

        with self.assertRaisesRegex(ValueError, "length"):
            build_probability_rows(
                records,
                self.torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
                self.torch.tensor([[1.0]]),
                mapping,
                {},
                self.torch,
            )


if __name__ == "__main__":
    unittest.main()
