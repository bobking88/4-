from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _record(
    image_id: str,
    group_id: str,
    mineral: str,
    role: str,
) -> dict[str, str]:
    role_ids = {
        "target_mineral": "0",
        "ti_bearing_negative": "1",
        "gangue_negative": "2",
        "metallic_hard_negative": "3",
    }
    return {
        "image_id": image_id,
        "relative_path": f"images/{image_id}.png",
        "mineral_label": mineral,
        "four_class_label": role,
        "four_class_id": role_ids[role],
        "mindat_photo_id": image_id,
        "split_group_id": group_id,
        "split": "train",
        "confirmation_subset": "expert_fit",
        "outer_fold": "0",
        "tc_subset": "expert_fit",
        "tc_protocol_version": "tc_oos_rsg_v1",
    }


class TCExpertWrapperTests(unittest.TestCase):
    def _valid_records(self):
        fit = [
            _record("fit-t", "fit-g-t", "ilmenite", "target_mineral"),
            _record("fit-ti", "fit-g-ti", "rutile", "ti_bearing_negative"),
            _record("fit-g", "fit-g-g", "quartz", "gangue_negative"),
            _record("fit-m", "fit-g-m", "pyrite", "metallic_hard_negative"),
        ]
        stop = [
            _record("stop-t", "stop-g-t", "ilmenite", "target_mineral"),
            _record("stop-ti", "stop-g-ti", "rutile", "ti_bearing_negative"),
            _record("stop-g", "stop-g-g", "quartz", "gangue_negative"),
            _record("stop-m", "stop-g-m", "pyrite", "metallic_hard_negative"),
        ]
        outer = [
            _record("outer-t", "outer-g-t", "ilmenite", "target_mineral"),
            _record("outer-ti", "outer-g-ti", "rutile", "ti_bearing_negative"),
            _record("outer-g", "outer-g-g", "quartz", "gangue_negative"),
            _record("outer-m", "outer-g-m", "pyrite", "metallic_hard_negative"),
        ]
        return fit, stop, outer

    def test_audit_rejects_group_overlap(self) -> None:
        from run_tc_oos_rsg_experts import audit_expert_manifests

        fit, stop, outer = self._valid_records()
        stop[0]["split_group_id"] = fit[0]["split_group_id"]

        with self.assertRaisesRegex(ValueError, "group overlap"):
            audit_expert_manifests(fit, stop, outer)

    def test_audit_allows_multiple_images_in_one_group_within_a_subset(self) -> None:
        from run_tc_oos_rsg_experts import audit_expert_manifests

        fit, stop, outer = self._valid_records()
        second_view = _record(
            "fit-t-second-view",
            fit[0]["split_group_id"],
            "ilmenite",
            "target_mineral",
        )
        fit.append(second_view)

        audit = audit_expert_manifests(fit, stop, outer)

        self.assertEqual(audit["row_counts"]["expert_fit"], 5)
        self.assertEqual(audit["group_counts"]["expert_fit"], 4)

    def test_audit_rejects_nonmatching_role_species_maps(self) -> None:
        from run_tc_oos_rsg_experts import audit_expert_manifests

        fit, stop, outer = self._valid_records()
        stop[0]["four_class_label"] = "metallic_hard_negative"
        stop[0]["four_class_id"] = "3"

        with self.assertRaisesRegex(ValueError, "species-role mapping"):
            audit_expert_manifests(fit, stop, outer)

    def test_default_commands_use_matched_efficientnet_b0_configuration(self) -> None:
        from run_tc_oos_rsg_experts import build_training_commands

        commands = build_training_commands(
            project_root=Path("project"),
            expert_manifest=Path("expert.csv"),
            baseline_manifest=Path("baseline.csv"),
            dataset_root=Path("dataset"),
            output_root=Path("output"),
            python_executable=Path("python"),
            torch_home=Path("torch-cache"),
            fold=2,
        )

        expert = list(commands["expert"].arguments)
        baseline = list(commands["baseline"].arguments)
        self.assertEqual(expert[expert.index("--backbone") + 1], "efficientnet_b0")
        self.assertEqual(baseline[baseline.index("--model") + 1], "efficientnet_b0")
        self.assertEqual(expert[expert.index("--seed") + 1], "20260929")
        self.assertEqual(baseline[baseline.index("--seed") + 1], "20260929")
        self.assertIn("--validation-only", expert)
        self.assertNotIn("--no-pretrained", expert)
        self.assertNotIn("--no-pretrained", baseline)
        self.assertEqual(commands["expert"].configuration["pretrained"], True)
        self.assertEqual(commands["baseline"].configuration["pretrained"], True)

    def test_selection_locks_contain_manifest_config_and_checkpoint_hashes(self) -> None:
        from run_tc_oos_rsg_experts import write_selection_lock

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_paths = {
                "expert_fit": root / "fit.csv",
                "expert_stop": root / "stop.csv",
                "training_manifest": root / "training.csv",
            }
            for name, path in source_paths.items():
                path.write_text(f"{name}\n", encoding="utf-8")
            config = root / "run_config.json"
            checkpoint = root / "best_model.pt"
            metrics = root / "best_validation_metrics.json"
            config.write_text('{"backbone":"efficientnet_b0"}\n', encoding="utf-8")
            checkpoint.write_bytes(b"synthetic checkpoint")
            metrics.write_text('{"epoch":1,"macro_f1":0.5}\n', encoding="utf-8")

            lock = write_selection_lock(
                output_dir=root,
                source_manifests=source_paths,
                config_path=config,
                checkpoint_path=checkpoint,
                metrics_path=metrics,
            )

            stored = json.loads((root / "selection_lock.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, lock)
            self.assertEqual(
                lock["checkpoint_sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                lock["configuration_sha256"],
                hashlib.sha256(config.read_bytes()).hexdigest(),
            )
            self.assertEqual(set(lock["manifest_sha256"]), set(source_paths))
            for name, path in source_paths.items():
                self.assertEqual(
                    lock["manifest_sha256"][name],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
