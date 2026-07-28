from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_mineral_classifier import (
    CLASS_LABELS,
    compute_focal_loss,
    compute_class_weights,
    configure_torch_home,
    load_manifest_records,
    resolve_device_name,
)


class ManifestLoadingTests(unittest.TestCase):
    def test_loads_existing_records_without_changing_fixed_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            image_path = dataset_root / "raw_positive" / "magnetite" / "one.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"not-read-by-loader")
            manifest = root / "split.csv"
            rows = [
                {
                    "image_id": "VTM-000001",
                    "relative_path": "raw_positive/magnetite/one.jpg",
                    "mineral_label": "magnetite",
                    "four_class_label": "target_mineral",
                    "four_class_id": "0",
                    "mindat_photo_id": "123",
                    "split_group_id": "DG-1",
                    "split": "test",
                }
            ]
            with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            records = load_manifest_records(manifest, dataset_root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].split, "test")
            self.assertEqual(records[0].class_id, 0)
            self.assertEqual(records[0].image_path, image_path)

    def test_rejects_unexpected_class_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            image_path = dataset_root / "raw" / "one.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"x")
            manifest = root / "split.csv"
            manifest.write_text(
                "image_id,relative_path,mineral_label,four_class_label,four_class_id,mindat_photo_id,split_group_id,split\n"
                "VTM-000001,raw/one.jpg,magnetite,gangue_negative,0,123,DG-1,train\n",
                encoding="utf-8-sig",
            )

            with self.assertRaisesRegex(ValueError, "mapping"):
                load_manifest_records(manifest, dataset_root)


class ClassWeightTests(unittest.TestCase):
    def test_inverse_frequency_weights_upweight_small_classes(self) -> None:
        weights = compute_class_weights([10, 20, 5, 5])

        self.assertEqual(len(weights), len(CLASS_LABELS))
        self.assertGreater(weights[2], weights[0])
        self.assertGreater(weights[3], weights[1])
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)


class FocalLossTests(unittest.TestCase):
    def test_gamma_zero_matches_weighted_cross_entropy(self) -> None:
        dependencies = __import__("train_mineral_classifier").require_training_dependencies()
        torch = dependencies["torch"]
        logits = torch.tensor([[2.0, 0.5], [0.1, 1.5]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        class_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

        focal = compute_focal_loss(logits, targets, class_weights, gamma=0.0, torch=torch)
        cross_entropy = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)

        self.assertTrue(torch.allclose(focal, cross_entropy))


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_uses_cpu_when_cuda_is_unavailable(self) -> None:
        self.assertEqual(resolve_device_name("auto", cuda_available=False), "cpu")

    def test_auto_uses_cuda_when_available(self) -> None:
        self.assertEqual(resolve_device_name("auto", cuda_available=True), "cuda")

    def test_explicit_device_is_preserved(self) -> None:
        self.assertEqual(resolve_device_name("cpu", cuda_available=True), "cpu")


class TorchHomeTests(unittest.TestCase):
    def test_configures_project_local_torch_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "torch-cache"
            previous = os.environ.get("TORCH_HOME")
            try:
                configured = configure_torch_home(cache_path)
                self.assertEqual(configured, cache_path)
                self.assertEqual(Path(os.environ["TORCH_HOME"]), cache_path)
                self.assertTrue(cache_path.is_dir())
            finally:
                if previous is None:
                    os.environ.pop("TORCH_HOME", None)
                else:
                    os.environ["TORCH_HOME"] = previous


if __name__ == "__main__":
    unittest.main()
