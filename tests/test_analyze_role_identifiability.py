from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class RoleIdentifiabilityTests(unittest.TestCase):
    def make_mapping(self):
        from mineral_hierarchy import SpeciesRoleMapping

        return SpeciesRoleMapping(
            species_labels=("a", "b", "c", "d"),
            species_role_ids=(0, 0, 0, 1),
            species_to_index={"a": 0, "b": 1, "c": 2, "d": 3},
        )

    def test_role_consistent_candidates_preserve_role_identifiability(self):
        from analyze_role_identifiability import build_candidate_set_rows, summarize_candidate_sets

        mapping = self.make_mapping()
        rows = build_candidate_set_rows(mapping, candidate_sizes=(2, 3), seed=7)
        summary = summarize_candidate_sets(rows)

        self.assertEqual(summary["role_consistent"]["role_unique_rate"], 1.0)
        self.assertLess(summary["role_consistent"]["species_unique_rate"], 1.0)

    def test_candidate_set_construction_is_deterministic_and_marks_conflicts(self):
        from analyze_role_identifiability import build_candidate_set_rows

        mapping = self.make_mapping()
        first = build_candidate_set_rows(mapping, candidate_sizes=(2,), seed=7)
        second = build_candidate_set_rows(mapping, candidate_sizes=(2,), seed=7)

        self.assertEqual(first, second)
        conflict_rows = [row for row in first if row["scenario"] == "role_conflict"]
        self.assertTrue(conflict_rows)
        self.assertTrue(all(row["role_unique"] is False for row in conflict_rows))

    def test_summary_contains_counts_and_candidate_size_breakdown(self):
        from analyze_role_identifiability import build_candidate_set_rows, summarize_candidate_sets

        summary = summarize_candidate_sets(
            build_candidate_set_rows(self.make_mapping(), candidate_sizes=(2,), seed=7)
        )

        self.assertEqual(summary["row_count"], 7)
        self.assertEqual(summary["by_candidate_size"]["2"]["row_count"], 7)
        self.assertEqual(summary["role_conflict"]["role_unique_count"], 0)


if __name__ == "__main__":
    unittest.main()
