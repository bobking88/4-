from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class ProxyAblationManifestTests(unittest.TestCase):
    def test_magnetite_is_removed_without_removing_true_target_species(self) -> None:
        from build_proxy_ablation_manifest import build_proxy_ablation_rows

        rows = [
            {"image_id": "1", "mineral_label": "magnetite"},
            {"image_id": "2", "mineral_label": "ilmenite"},
            {"image_id": "3", "mineral_label": "titanomagnetite"},
            {"image_id": "4", "mineral_label": "quartz"},
        ]

        result = build_proxy_ablation_rows(rows)

        self.assertEqual([row["image_id"] for row in result], ["2", "3", "4"])

    def test_titanomagnetite_counts_are_traced(self) -> None:
        from build_proxy_ablation_manifest import summarize_titanomagnetite_provenance

        audit = []
        for index in range(35):
            audit.append({
                "image_id": str(index), "mineral_label": "titanomagnetite",
                "exclusion_reason": "" if index < 23 else (
                    "exclude_exact_label_conflict" if index < 33 else "exclude_near_label_conflict"
                ),
            })
        final = [{"image_id": str(index), "mineral_label": "titanomagnetite"} for index in range(23)]

        summary = summarize_titanomagnetite_provenance(audit, final)

        self.assertEqual(summary["downloaded"], 35)
        self.assertEqual(summary["excluded_cross_label_conflict"], 12)
        self.assertEqual(summary["final"], 23)


if __name__ == "__main__":
    unittest.main()
