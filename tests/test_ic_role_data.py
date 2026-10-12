"""Synthetic-only tests for the isolated IC-RFA fit data contract."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lc_role_fixtures import raw

try:
    import ic_role_data as data
except ModuleNotFoundError:
    data = None


def assignment(source):
    return [dict(image_id=r["image_id"], split_group_id=r["split_group_id"],
                 class_id=int(source["labels"][i]), inner_fold=(i // 4) % 3)
            for i, r in enumerate(source["records"])]


class ICDataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "IC fit-only data module is not implemented")
        torch.set_num_threads(1)
        self.raw = raw()
        self.rows = assignment(self.raw)

    def assert_state_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            if torch.is_tensor(left[key]):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            elif isinstance(left[key], dict):
                self.assert_state_equal(left[key], right[key])
            else:
                self.assertEqual(left[key], right[key])

    def package(self, root):
        source = raw(340)
        labels = torch.repeat_interleave(torch.arange(4), torch.tensor([66, 130, 97, 47]))
        source["labels"] = labels
        for i, record in enumerate(source["records"]):
            record["four_class_id"] = str(int(labels[i]))
        rows, seen = [], [0] * 4
        for i, record in enumerate(source["records"]):
            k = int(labels[i])
            rows.append(dict(image_id=record["image_id"], split_group_id=record["split_group_id"],
                             class_id=k, inner_fold=seen[k] % 3))
            seen[k] += 1
        fit = root / data.FIT_PATH
        assigned = root / data.ASSIGNMENT_PATH
        fit.parent.mkdir(parents=True)
        torch.save(source, fit)
        assigned.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        return source, rows, fit, assigned

    def synthetic_digest(self, path):
        return data.FIT_SHA if path.name == "fit_raw.pt" else data.ASSIGNMENT_SHA

    def test_loader_never_opens_stop_or_combined_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, rows, fit, _ = self.package(root)
            original_load = torch.load
            opened = []

            def checked_load(path, **kwargs):
                self.assertEqual(Path(path).resolve(), fit.resolve())
                self.assertEqual(kwargs, dict(map_location="cpu", weights_only=True))
                opened.append(Path(path).name)
                return original_load(path, **kwargs)

            # Only the digest is replaced: a generated fixture cannot have the
            # published real blob SHA. Parsing and all identity checks stay real.
            with patch.object(data, "file_digest", side_effect=self.synthetic_digest):
                with patch.object(torch, "load", side_effect=checked_load):
                    restored, accepted = data.load_fit_bundle(root)
            self.assertEqual(opened, ["fit_raw.pt"])
            self.assertEqual(accepted, rows)
            torch.testing.assert_close(restored["H"], source["H"], rtol=0, atol=0)

    def test_hash_failure_precedes_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.package(root)
            with patch.object(torch, "load", side_effect=AssertionError("Must not deserialize")):
                with self.assertRaisesRegex(ValueError, "hash"):
                    data.load_fit_bundle(root)

    def test_assignment_reorder_duplicate_missing_class_rejected(self):
        bad_rows = copy.deepcopy(self.rows)
        bad_rows[0], bad_rows[1] = bad_rows[1], bad_rows[0]
        with self.assertRaises(ValueError):
            data.validate_assignment(self.raw, bad_rows)
        for key in ("image_id", "split_group_id"):
            changed = copy.deepcopy(self.raw)
            changed["records"][1][key] = changed["records"][0][key]
            with self.assertRaises(ValueError):
                data.validate_assignment(changed, assignment(changed))
        changed = copy.deepcopy(self.raw)
        rows = copy.deepcopy(self.rows)
        for i, r in enumerate(rows):
            if r["inner_fold"] == 0:
                changed["labels"][i] = 0
                changed["records"][i]["four_class_id"] = "0"
                r["class_id"] = 0
        with self.assertRaises(ValueError):
            data.validate_assignment(changed, rows)

    def test_validation_changes_do_not_change_preprocessing(self):
        prepared = data.prepare_inner_fold(self.raw, self.rows, 0)
        changed, rows = copy.deepcopy(self.raw), copy.deepcopy(self.rows)
        validation = prepared["validation_indices"]
        changed["p"][validation] = .25
        for key in ("H", "evidence", "contradiction"):
            changed[key][validation] += 10000
        for i in validation.tolist():
            k = (int(changed["labels"][i]) + 1) % 4
            changed["labels"][i] = k
            changed["records"][i]["four_class_id"] = str(k)
            rows[i]["class_id"] = k
        again = data.prepare_inner_fold(changed, rows, 0)
        self.assert_state_equal(prepared["state"], again["state"])
        self.assertEqual(prepared["state"]["basis"].shape, (1280, 64))
        self.assertEqual(len(prepared["state"]["train_ids"]), 80)
        self.assertFalse(set(prepared["state"]["train_ids"]) &
                         {r["image_id"] for r in prepared["validation"]["records"]})

    def test_fullfit_state_cannot_be_reused(self):
        bad_state = data.fit_preprocessing(self.raw, torch.arange(120))
        with patch.object(data, "fit_preprocessing", return_value=bad_state):
            with self.assertRaisesRegex(ValueError, "training.*identity"):
                data.prepare_inner_fold(self.raw, self.rows, 0)

    def test_repeat_serialization_and_batch_transform(self):
        prepared = data.prepare_inner_fold(self.raw, self.rows, 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic-state.pt"
            torch.save(prepared["state"], path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
        self.assert_state_equal(prepared["state"], restored)
        indices = prepared["validation_indices"]
        full = data.transform_subset(self.raw, restored, indices)
        a = data.transform_subset(self.raw, restored, indices[:20])
        b = data.transform_subset(self.raw, restored, indices[20:])
        for key in ("h", "e", "log_anchor"):
            torch.testing.assert_close(full[key], torch.cat([a[key], b[key]]), rtol=1e-12, atol=1e-12)

    def test_loader_class_count_and_assignment_container_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, rows, fit, assigned = self.package(root)
            source["labels"][0] = 1
            source["records"][0]["four_class_id"] = "1"
            rows[0]["class_id"] = 1
            torch.save(source, fit)
            assigned.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            with patch.object(data, "file_digest", side_effect=self.synthetic_digest):
                with self.assertRaises(ValueError):
                    data.load_fit_bundle(root)
            assigned.write_text(json.dumps(rows), encoding="utf-8")
            with patch.object(data, "file_digest", side_effect=self.synthetic_digest):
                with self.assertRaises(ValueError):
                    data.load_fit_bundle(root)

    def test_invalid_fold_and_nonfinite_input_fail(self):
        for fold in (-1, 3, True):
            with self.assertRaises(ValueError):
                data.prepare_inner_fold(self.raw, self.rows, fold)
        changed = copy.deepcopy(self.raw)
        changed["H"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            data.validate_assignment(changed, self.rows)


if __name__ == "__main__":
    unittest.main()
