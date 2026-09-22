import sys
import unittest
import warnings
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_oos_gate_degeneracy import compute_examples


class OOSGateDegeneracyTests(unittest.TestCase):
    def test_computation_emits_no_tensor_conversion_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_examples()
        self.assertEqual(caught, [])

    def test_equal_experts_have_zero_gate_gradient_and_weight(self):
        result = compute_examples()
        self.assertLessEqual(abs(result["equal_experts"]["final_nll_gradient"]), 1e-12)
        self.assertEqual(result["equal_experts"]["gap_weight"], 0.0)

    def test_distinct_experts_restore_nonzero_signal(self):
        result = compute_examples()
        self.assertGreater(abs(result["distinct_experts"]["final_nll_gradient"]), 1e-6)
        self.assertGreater(result["distinct_experts"]["gap_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
