from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def make_row(
    true_species: str,
    predicted_species: str,
    true_role: str,
    direct: tuple[float, float, float, float],
    mapped: tuple[float, float, float, float],
) -> dict[str, str]:
    from train_mineral_classifier import CLASS_LABELS

    row = {
        "mineral_label": true_species,
        "predicted_species": predicted_species,
        "four_class_label": true_role,
    }
    for label, probability in zip(CLASS_LABELS, direct):
        row[f"role_probability_{label}"] = str(probability)
    for label, probability in zip(CLASS_LABELS, mapped):
        row[f"mapped_role_probability_{label}"] = str(probability)
    return row


class HierarchyConsistencyTests(unittest.TestCase):
    def test_species_error_can_preserve_role_correctness(self) -> None:
        from analyze_hierarchy_consistency import calculate_hierarchy_metrics

        rows = [
            make_row(
                "magnetite",
                "ilmenite",
                "target_mineral",
                (0.8, 0.1, 0.05, 0.05),
                (0.7, 0.2, 0.05, 0.05),
            ),
            make_row(
                "quartz",
                "quartz",
                "gangue_negative",
                (0.05, 0.05, 0.85, 0.05),
                (0.05, 0.05, 0.85, 0.05),
            ),
        ]

        summary = calculate_hierarchy_metrics(rows)

        self.assertEqual(summary["species_wrong_role_correct_count"], 1)
        self.assertEqual(summary["species_error_count"], 1)
        self.assertAlmostEqual(summary["species_accuracy"], 0.5)
        self.assertAlmostEqual(summary["mapped_role_accuracy"], 1.0)

    def test_pinsker_bound_holds(self) -> None:
        from analyze_hierarchy_consistency import kl_divergence, total_variation

        mapped = (0.7, 0.2, 0.05, 0.05)
        direct = (0.6, 0.25, 0.1, 0.05)
        tv = total_variation(mapped, direct)
        kl = kl_divergence(mapped, direct)

        self.assertLessEqual(tv, math.sqrt(0.5 * kl) + 1e-12)

    def test_invalid_probabilities_are_rejected(self) -> None:
        from analyze_hierarchy_consistency import total_variation

        with self.assertRaisesRegex(ValueError, "sum"):
            total_variation((0.7, 0.2), (0.1, 0.2))

    def test_kl_is_stable_for_rounded_nearly_identical_probabilities(self) -> None:
        from analyze_hierarchy_consistency import kl_divergence

        mapped = (0.50000001, 0.29999999, 0.1, 0.1)
        direct = (0.50000002, 0.29999998, 0.1, 0.1)

        self.assertGreaterEqual(kl_divergence(mapped, direct), 0.0)


if __name__ == "__main__":
    unittest.main()
