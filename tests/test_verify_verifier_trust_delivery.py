import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import verify_verifier_trust_delivery as verifier
except ModuleNotFoundError:
    verifier = None


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(verifier, "delivery verifier implementation is missing")

    def test_table_replay_checks_values_and_identity_not_only_row_count(self):
        data = {"records": [{"image_id": "A", "split_group_id": "G"}], "labels": torch.tensor([0]),
                "contradiction": torch.tensor([[.5]], dtype=torch.float64)}
        predictions = {"raw": torch.tensor([[.25]], dtype=torch.float64), "gamma": torch.tensor([[.25]], dtype=torch.float64),
                       "probabilities": torch.tensor([[.6, .2, .1, .1]], dtype=torch.float64)}
        row = dict(image_id="A", split_group_id="G", true_class_id="0", raw_head=".25", gamma=".25", contradiction=".5", predicted_class_id="0", prob_0=".6", prob_1=".2", prob_2=".1", prob_3=".1")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            for update, valid in (({}, True), ({"image_id": "B"}, False), ({"prob_0": ".5"}, False), ({"raw_head": "nan"}, False)):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(dict(row, **update))
                if valid:
                    self.assertEqual(verifier.verify_table(path, data, predictions), 0.)
                else:
                    with self.assertRaises(ValueError):
                        verifier.verify_table(path, data, predictions)


if __name__ == "__main__":
    unittest.main()
