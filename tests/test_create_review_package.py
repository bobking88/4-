from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from create_review_package import (
    create_review_package,
    resolve_image_path,
    stratified_sample,
)


class StratifiedSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = []
        for mineral in ("magnetite", "rutile"):
            for split, count in (("train", 8), ("val", 3), ("test", 3)):
                for index in range(count):
                    self.rows.append(
                        {
                            "image_id": f"{mineral}-{split}-{index}",
                            "relative_path": f"raw/{mineral}/{index}.jpg",
                            "mineral_label": mineral,
                            "four_class_label": "target_mineral",
                            "split": split,
                        }
                    )
        for index in range(2):
            self.rows.append(
                {
                    "image_id": f"titanomagnetite-train-{index}",
                    "relative_path": f"raw/titanomagnetite/{index}.jpg",
                    "mineral_label": "titanomagnetite",
                    "four_class_label": "target_mineral",
                    "split": "train",
                }
            )

    def test_sample_is_deterministic_and_capped_per_mineral(self) -> None:
        first = stratified_sample(self.rows, per_mineral=5, seed=20260727)
        second = stratified_sample(self.rows, per_mineral=5, seed=20260727)

        self.assertEqual(first, second)
        counts = Counter(row["mineral_label"] for row in first)
        self.assertEqual(counts["magnetite"], 5)
        self.assertEqual(counts["rutile"], 5)
        self.assertEqual(counts["titanomagnetite"], 2)

    def test_sample_covers_each_available_split_when_quota_allows(self) -> None:
        sample = stratified_sample(self.rows, per_mineral=5, seed=20260727)
        magnetite_splits = {
            row["split"] for row in sample if row["mineral_label"] == "magnetite"
        }
        self.assertEqual(magnetite_splits, {"train", "val", "test"})

    def test_resolve_image_path_uses_dataset_root(self) -> None:
        root = Path(r"D:\example\dataset")
        result = resolve_image_path(root, "raw_positive/magnetite/example.jpg")
        self.assertEqual(
            result,
            root / "raw_positive" / "magnetite" / "example.jpg",
        )


class ReviewPackageTests(unittest.TestCase):
    def test_package_writes_review_csv_and_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            output_dir = root / "review"
            image_dir = dataset_root / "raw_positive" / "magnetite"
            image_dir.mkdir(parents=True)

            manifest_path = root / "manifest.csv"
            rows = []
            for index, color in enumerate(("red", "green", "blue")):
                image_path = image_dir / f"sample_{index}.jpg"
                Image.new("RGB", (320, 320), color=color).save(image_path)
                rows.append(
                    {
                        "image_id": f"VTM-{index:06d}",
                        "relative_path": str(
                            image_path.relative_to(dataset_root)
                        ).replace("\\", "/"),
                        "mineral_label": "magnetite",
                        "four_class_label": "target_mineral",
                        "four_class_id": "0",
                        "mindat_photo_id": str(100 + index),
                        "split_group_id": f"DG-{index}",
                        "split": "train",
                    }
                )
            with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            summary = create_review_package(
                manifest_path=manifest_path,
                dataset_root=dataset_root,
                output_dir=output_dir,
                per_mineral=2,
                seed=20260727,
                columns=2,
                tile_size=120,
            )

            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["missing_images"], 0)
            review_path = output_dir / "review_queue.csv"
            self.assertTrue(review_path.exists())
            self.assertTrue((output_dir / "contact_sheets" / "magnetite_001.jpg").exists())
            with review_path.open(encoding="utf-8-sig", newline="") as handle:
                review_rows = list(csv.DictReader(handle))
            self.assertEqual(len(review_rows), 2)
            self.assertIn("review_decision", review_rows[0])
            self.assertEqual(review_rows[0]["review_decision"], "")


if __name__ == "__main__":
    unittest.main()
