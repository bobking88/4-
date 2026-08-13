from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class CalibratedSelectiveRecognitionTests(unittest.TestCase):
    def test_validation_subsets_do_not_share_groups(self) -> None:
        from calibrate_selective_recognition import split_validation_groups

        rows = [
            {
                "image_id": f"VTM-{index}",
                "split_group_id": f"G-{index // 2}",
                "four_class_label": "target_mineral" if index < 6 else "gangue_negative",
            }
            for index in range(12)
        ]

        fit, certify = split_validation_groups(rows, seed=7)
        fit_groups = {row["split_group_id"] for row in fit}
        certify_groups = {row["split_group_id"] for row in certify}

        self.assertTrue(fit_groups.isdisjoint(certify_groups))
        self.assertEqual(len(fit) + len(certify), len(rows))

    def test_temperature_is_positive_and_reduces_fit_nll(self) -> None:
        import numpy as np

        from calibrate_selective_recognition import fit_temperature, multiclass_nll

        logits = np.asarray([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
        labels = np.asarray([0, 1, 1, 0])
        temperature = fit_temperature(logits, labels)

        self.assertGreater(temperature, 0.0)
        self.assertLessEqual(
            multiclass_nll(logits / temperature, labels),
            multiclass_nll(logits, labels) + 1e-8,
        )

    def test_no_certificate_is_reported_when_bound_exceeds_target(self) -> None:
        from calibrate_selective_recognition import select_certified_threshold

        rows = [
            {"calibrated_confidence": confidence, "correct": False}
            for confidence in (0.99, 0.9, 0.8, 0.7)
        ]
        result = select_certified_threshold(
            rows, thresholds=(0.0, 0.5), delta=0.01, alpha=0.05
        )

        self.assertEqual(result["status"], "no_certified_threshold")

    def test_clopper_pearson_upper_is_valid_for_zero_errors(self) -> None:
        from calibrate_selective_recognition import clopper_pearson_upper

        bound = clopper_pearson_upper(0, 100, confidence=0.95)

        self.assertGreater(bound, 0.0)
        self.assertLess(bound, 0.1)
        self.assertTrue(math.isfinite(bound))


if __name__ == "__main__":
    unittest.main()
