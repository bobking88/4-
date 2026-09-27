from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_oos_rsg_confirmation import (  # noqa: E402
    EXPERT_GATE_SEEDS,
    audit_confirmation_records,
)
from train_mineral_classifier import ManifestRecord  # noqa: E402


class OOSRSGConfirmationTests(unittest.TestCase):
    @staticmethod
    def record(
        image_id: str,
        group: str,
        split: str,
        mineral: str = "mineral_a",
        role: str = "target_mineral",
        class_id: int = 0,
    ) -> ManifestRecord:
        return ManifestRecord(
            image_id=image_id,
            image_path=Path(f"{image_id}.jpg"),
            mineral_label=mineral,
            four_class_label=role,
            class_id=class_id,
            mindat_photo_id=image_id,
            split_group_id=group,
            split=split,
        )

    def valid_records(self):
        expert = [
            self.record("fit-a", "fit-a", "train"),
            self.record("fit-b", "fit-b", "train"),
            self.record("expert-stop", "expert-stop", "val"),
        ]
        seen = [replace(expert[0], split="train")]
        unseen = [self.record("unseen", "unseen", "train")]
        gate_stop = [self.record("gate-stop", "gate-stop", "val")]
        final_eval = [self.record("final", "final", "test")]
        return expert, seen, unseen, gate_stop, final_eval

    def test_registered_experts_have_one_paired_gate_seed_each(self) -> None:
        self.assertEqual(
            EXPERT_GATE_SEEDS,
            {20260924: 20260934, 20260925: 20260935, 20260926: 20260936},
        )

    def test_audit_accepts_exact_strata_and_all_required_isolation(self) -> None:
        audit = audit_confirmation_records(*self.valid_records())

        self.assertTrue(audit["exact_seen_unseen_strata_match"])
        self.assertEqual(audit["unexpected_group_overlap"], {})
        self.assertEqual(audit["counts"]["final_eval"], 1)
        self.assertTrue(audit["final_eval_locked"])

    def test_audit_rejects_final_eval_leakage(self) -> None:
        expert, seen, unseen, gate_stop, final_eval = self.valid_records()
        final_eval[0] = replace(final_eval[0], split_group_id=unseen[0].split_group_id)

        with self.assertRaises(ValueError):
            audit_confirmation_records(expert, seen, unseen, gate_stop, final_eval)

    def test_audit_rejects_changed_seen_unseen_strata(self) -> None:
        expert, seen, unseen, gate_stop, final_eval = self.valid_records()
        unseen[0] = replace(
            unseen[0],
            mineral_label="other",
            four_class_label="gangue_negative",
            class_id=2,
        )

        with self.assertRaises(ValueError):
            audit_confirmation_records(expert, seen, unseen, gate_stop, final_eval)


if __name__ == "__main__":
    unittest.main()
