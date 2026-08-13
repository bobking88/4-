from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class SourceHoldoutManifestTests(unittest.TestCase):
    def test_photographer_and_duplicate_groups_never_cross_splits(self) -> None:
        from build_source_holdout_manifest import allocate_source_groups, validate_source_holdout

        roles = ("target_mineral", "ti_bearing_negative", "gangue_negative", "metallic_hard_negative")
        rows = []
        for role_index, role in enumerate(roles):
            for index in range(12):
                rows.append({
                    "image_id": f"{role_index}-{index}", "four_class_label": role,
                    "photographer_or_credit": f"Photographer {role_index}-{index // 2}",
                    "split_group_id": f"DG-{role_index}-{index}",
                })
        rows[1]["split_group_id"] = rows[0]["split_group_id"]

        first = allocate_source_groups(rows, 0.2, 7)
        second = allocate_source_groups(rows, 0.2, 7)
        audit = validate_source_holdout(first, minimum_test_per_role=1)

        self.assertEqual(first, second)
        self.assertEqual(audit["photographer_cross_split_count"], 0)
        self.assertEqual(audit["split_group_cross_split_count"], 0)

    def test_missing_photographers_are_rejected(self) -> None:
        from build_source_holdout_manifest import allocate_source_groups

        with self.assertRaisesRegex(ValueError, "photographer"):
            allocate_source_groups([
                {"image_id": "1", "four_class_label": "target_mineral",
                 "photographer_or_credit": "", "split_group_id": "DG-1"}
            ], 0.2, 7)


if __name__ == "__main__":
    unittest.main()
