from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_oos_confirmation_manifests import (  # noqa: E402
    build_confirmation_manifests,
    partition_training_rows,
)


class OOSConfirmationManifestTests(unittest.TestCase):
    @staticmethod
    def rows(group_count: int = 40) -> list[dict[str, str]]:
        return [
            {
                "image_id": f"i{index:03d}",
                "relative_path": f"m/i{index:03d}.jpg",
                "mineral_label": "mineral_a",
                "four_class_label": "target_mineral",
                "four_class_id": "0",
                "mindat_photo_id": f"p{index:03d}",
                "split_group_id": f"g{index:03d}",
                "split": "train",
            }
            for index in range(group_count)
        ]

    def test_partition_is_deterministic_and_keeps_groups_intact(self) -> None:
        rows = self.rows()
        rows.append({**rows[0], "image_id": "second-view"})

        first = partition_training_rows(rows, seed=17)
        second = partition_training_rows(list(reversed(rows)), seed=17)

        first_assignment = {
            row["image_id"]: row["confirmation_subset"] for row in first
        }
        second_assignment = {
            row["image_id"]: row["confirmation_subset"] for row in second
        }
        self.assertEqual(first_assignment, second_assignment)
        self.assertEqual(
            {row["confirmation_subset"] for row in first if row["split_group_id"] == "g000"},
            {first_assignment["i000"]},
        )
        self.assertEqual(
            {row["confirmation_subset"] for row in first},
            {"expert_fit", "expert_stop", "gate_fit", "gate_stop", "final_eval"},
        )
        self.assertTrue(all(row["split"] == "train" for row in first))
        self.assertTrue(all("confirmation_subset" not in row for row in rows))

    def test_manifests_isolate_selection_and_match_seen_counts(self) -> None:
        partition = partition_training_rows(self.rows(), seed=23)

        manifests, audit = build_confirmation_manifests(partition, seen_seed=29)

        expert = manifests["expert"]
        gate_seen = manifests["gate_seen"]
        gate_unseen = manifests["gate_unseen"]
        gate_stop = manifests["gate_stop"]
        final_eval = manifests["final_eval"]
        self.assertEqual({row["split"] for row in expert}, {"train", "val"})
        self.assertTrue(all(row["split"] == "train" for row in gate_seen))
        self.assertTrue(all(row["split"] == "train" for row in gate_unseen))
        self.assertTrue(all(row["split"] == "val" for row in gate_stop))
        self.assertTrue(all(row["split"] == "test" for row in final_eval))

        stratum = lambda row: (row["mineral_label"], row["four_class_label"])
        self.assertEqual(Counter(map(stratum, gate_seen)), Counter(map(stratum, gate_unseen)))

        group_sets = {
            name: {row["split_group_id"] for row in records}
            for name, records in manifests.items()
        }
        held_names = ("gate_unseen", "gate_stop", "final_eval")
        for index, left in enumerate(held_names):
            for right in held_names[index + 1 :]:
                self.assertFalse(group_sets[left] & group_sets[right])
        self.assertFalse(group_sets["expert"] & group_sets["gate_unseen"])
        self.assertFalse(group_sets["expert"] & group_sets["gate_stop"])
        self.assertFalse(group_sets["expert"] & group_sets["final_eval"])
        self.assertTrue(group_sets["gate_seen"].issubset(group_sets["expert"]))
        self.assertTrue(audit["final_eval_locked"])
        self.assertTrue(audit["exact_seen_unseen_strata_match"])
        self.assertEqual(audit["cross_function_group_overlap"], {})

    def test_rejects_nontraining_or_too_few_groups(self) -> None:
        nontraining = self.rows()
        nontraining[0]["split"] = "test"
        with self.assertRaises(ValueError):
            partition_training_rows(nontraining)
        with self.assertRaises(ValueError):
            partition_training_rows(self.rows(group_count=4))


if __name__ == "__main__":
    unittest.main()
