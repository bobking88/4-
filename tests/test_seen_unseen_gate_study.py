import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_seen_unseen_gate_study import audit_supervision_records, select_earliest_best


def record(image_id, group_id, mineral, role):
    return SimpleNamespace(
        image_id=image_id,
        split_group_id=group_id,
        mineral_label=mineral,
        four_class_label=role,
    )


class GateSupervisionAuditTests(unittest.TestCase):
    def test_accepts_exact_strata_match_and_group_isolation(self):
        seen = [record("s1", "fit1", "ilmenite", "target_mineral")]
        unseen = [record("u1", "gate1", "ilmenite", "target_mineral")]
        expert = seen + [record("v1", "stop1", "rutile", "ti_bearing_negative")]

        audit = audit_supervision_records(expert, seen, unseen)

        self.assertEqual(audit["seen_count"], 1)
        self.assertEqual(audit["unseen_count"], 1)
        self.assertTrue(audit["exact_species_role_count_match"])
        self.assertTrue(audit["unseen_absent_from_expert"])

    def test_rejects_mismatched_species_role_counts(self):
        seen = [record("s1", "fit1", "ilmenite", "target_mineral")]
        unseen = [record("u1", "gate1", "rutile", "ti_bearing_negative")]

        with self.assertRaisesRegex(ValueError, "species-role counts"):
            audit_supervision_records(seen, seen, unseen)

    def test_rejects_unseen_group_present_in_expert(self):
        seen = [record("s1", "fit1", "ilmenite", "target_mineral")]
        unseen = [record("u1", "fit1", "ilmenite", "target_mineral")]

        with self.assertRaisesRegex(ValueError, "unseen groups"):
            audit_supervision_records(seen, seen, unseen)


class SelectionTests(unittest.TestCase):
    def test_selects_earliest_minimum(self):
        history = [
            {"epoch": 1, "final_nll_nats": 0.8},
            {"epoch": 2, "final_nll_nats": 0.7},
            {"epoch": 3, "final_nll_nats": 0.7},
        ]

        self.assertEqual(select_earliest_best(history, "final_nll_nats")["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
