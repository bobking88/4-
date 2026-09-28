from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _row(
    index: int,
    group_id: str,
    mineral: str = "ilmenite",
    role: str = "target_mineral",
    confirmation_subset: str = "expert_fit",
) -> dict[str, str]:
    return {
        "image_id": f"VTM-{index:06d}",
        "relative_path": f"images/{index:06d}.jpg",
        "mineral_label": mineral,
        "four_class_label": role,
        "four_class_id": "0",
        "mindat_photo_id": str(index),
        "split_group_id": group_id,
        "split": "train",
        "confirmation_subset": confirmation_subset,
    }


class TCOOSRSGManifestTests(unittest.TestCase):
    def test_outer_partition_is_deterministic_and_group_intact(self) -> None:
        from build_tc_oos_rsg_manifests import build_outer_folds

        rows = [_row(index, f"g-{index:02d}") for index in range(1, 13)]
        rows.extend([
            _row(101, "g-shared", mineral="rutile", role="ti_bearing_negative"),
            _row(102, "g-shared", mineral="rutile", role="ti_bearing_negative"),
        ])
        shuffled = list(rows)
        random.Random(17).shuffle(shuffled)

        first = build_outer_folds(rows, seed=20260927)
        second = build_outer_folds(shuffled, seed=20260927)

        first_assignment = {row["image_id"]: row["outer_fold"] for row in first}
        second_assignment = {row["image_id"]: row["outer_fold"] for row in second}
        self.assertEqual(first_assignment, second_assignment)
        shared_folds = {
            row["outer_fold"] for row in first if row["split_group_id"] == "g-shared"
        }
        self.assertEqual(len(shared_folds), 1)

    def test_outer_folds_jointly_cover_every_development_row_once(self) -> None:
        from build_tc_oos_rsg_manifests import audit_tc_manifests, build_outer_folds

        rows = []
        for index in range(18):
            rows.append(_row(index + 1, f"target-{index:02d}"))
            rows.append(_row(
                index + 101,
                f"gangue-{index:02d}",
                mineral="quartz",
                role="gangue_waste",
            ))

        partitioned = build_outer_folds(rows)
        audit = audit_tc_manifests(partitioned)

        self.assertEqual(len(partitioned), len(rows))
        self.assertEqual(len({row["image_id"] for row in partitioned}), len(rows))
        self.assertEqual({row["outer_fold"] for row in partitioned}, {"0", "1", "2"})
        self.assertEqual({row["tc_subset"] for row in partitioned}, {"outer_eval"})
        self.assertEqual(audit["row_count"], len(rows))
        self.assertEqual(audit["cross_subset_group_overlap_count"], 0)

    def test_inner_subsets_follow_registered_names_and_group_isolation(self) -> None:
        from build_tc_oos_rsg_manifests import build_inner_subsets

        rows = [
            {**_row(index + 1, f"g-{index:02d}"), "outer_fold": "1"}
            for index in range(20)
        ]
        partitioned = build_inner_subsets(rows, seed=20260928)

        counts = Counter(row["tc_subset"] for row in partitioned)
        self.assertEqual(
            counts,
            Counter({
                "expert_fit": 12,
                "expert_stop": 2,
                "gate_fit": 2,
                "gate_stop_projector_fit": 2,
                "projector_stop": 2,
            }),
        )
        by_group = defaultdict(set)
        for row in partitioned:
            by_group[row["split_group_id"]].add(row["tc_subset"])
        self.assertTrue(all(len(subsets) == 1 for subsets in by_group.values()))

    def test_spent_final_eval_rows_are_rejected(self) -> None:
        from build_tc_oos_rsg_manifests import build_outer_folds

        rows = [_row(index + 1, f"g-{index:02d}") for index in range(6)]
        rows.append(_row(99, "spent", confirmation_subset="final_eval"))

        with self.assertRaisesRegex(ValueError, "final_eval"):
            build_outer_folds(rows)

    def test_titanomagnetite_is_balanced_four_four_five_when_thirteen_groups_exist(self) -> None:
        from build_tc_oos_rsg_manifests import build_outer_folds

        rows = [
            _row(index + 1, f"tm-{index:02d}", mineral="titanomagnetite")
            for index in range(13)
        ]

        partitioned = build_outer_folds(rows)
        counts = Counter(row["outer_fold"] for row in partitioned)

        self.assertEqual(sorted(counts.values()), [4, 4, 5])

    def test_cross_subset_group_leakage_is_rejected(self) -> None:
        from build_tc_oos_rsg_manifests import audit_tc_manifests

        rows = [
            {
                **_row(1, "shared"),
                "outer_fold": "0",
                "tc_subset": "expert_fit",
                "tc_protocol_version": "tc_oos_rsg_v1",
            },
            {
                **_row(2, "shared"),
                "outer_fold": "0",
                "tc_subset": "projector_stop",
                "tc_protocol_version": "tc_oos_rsg_v1",
            },
        ]

        with self.assertRaisesRegex(ValueError, "multiple tc_subset"):
            audit_tc_manifests(rows)

    def test_cli_writes_all_fold_manifests_and_hashes(self) -> None:
        from build_tc_oos_rsg_manifests import main

        rows = []
        for index in range(30):
            rows.append(_row(index + 1, f"target-{index:02d}"))
            rows.append(_row(
                index + 101,
                f"gangue-{index:02d}",
                mineral="quartz",
                role="gangue_negative",
            ))
        spent = _row(999, "spent", confirmation_subset="final_eval")
        rows.append(spent)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            output = root / "registered"
            protocol = root / "protocol.md"
            protocol.write_text("# Locked synthetic protocol\n", encoding="utf-8")
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            main([
                "--partition",
                str(source),
                "--output-dir",
                str(output),
                "--protocol-document",
                str(protocol),
            ])

            expected_root = {"partition.csv", "audit.json", "registered_protocol.json"}
            self.assertTrue(expected_root.issubset({path.name for path in output.iterdir()}))
            expected_fold_files = {
                "outer_eval.csv",
                "expert_fit.csv",
                "expert_stop.csv",
                "gate_fit.csv",
                "gate_stop_projector_fit.csv",
                "projector_stop.csv",
            }
            all_rows = []
            for fold in range(3):
                fold_dir = output / f"fold_{fold}"
                self.assertEqual({path.name for path in fold_dir.iterdir()}, expected_fold_files)
                for path in fold_dir.glob("*.csv"):
                    with path.open(encoding="utf-8", newline="") as handle:
                        file_rows = list(csv.DictReader(handle))
                    self.assertTrue(file_rows)
                    self.assertTrue(all(row["outer_fold"] == str(fold) for row in file_rows))
                    all_rows.extend(file_rows)

            registration = json.loads(
                (output / "registered_protocol.json").read_text(encoding="utf-8")
            )
            hashes = registration["manifest_sha256"]
            self.assertEqual(len(hashes), 19)
            self.assertEqual(set(hashes), {
                "partition.csv",
                *{
                    f"fold_{fold}/{filename}"
                    for fold in range(3)
                    for filename in expected_fold_files
                },
            })
            self.assertNotIn(spent["image_id"], {row["image_id"] for row in all_rows})
            self.assertEqual(registration["development_row_count"], 60)
            self.assertEqual(registration["spent_final_eval_row_count"], 1)
            for relative_path, expected_hash in hashes.items():
                import hashlib

                actual_hash = hashlib.sha256((output / relative_path).read_bytes()).hexdigest()
                self.assertEqual(actual_hash, expected_hash)


if __name__ == "__main__":
    unittest.main()
