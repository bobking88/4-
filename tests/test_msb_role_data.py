"""Data-contract tests use only synthetic temporary tensors."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from msb_synthetic_helpers import make_prepared_fixture
try:
    import msb_role_data as data
except ModuleNotFoundError:
    data = None


class DataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "MSB prepared-only contracts missing")
        self.fixture = make_prepared_fixture()

    def test_missing_or_reordered_rows_fail(self):
        for mutate in (lambda x: x["train"]["records"].reverse(),
                       lambda x: x["train"]["records"].pop(),
                       lambda x: x["state"]["train_ids"].reverse()):
            item = copy.deepcopy(self.fixture["folds"][0]); mutate(item)
            with self.assertRaises(ValueError):
                data.validate_prepared_fold(item, 0)

    def test_train_validation_and_oof_groups_disjoint(self):
        result = data.validate_fold_collection(self.fixture["folds"])
        self.assertEqual(result["counts"], [66, 130, 97, 47])
        self.assertEqual(result["groups"], 340)
        bad = copy.deepcopy(self.fixture["folds"])
        bad[1]["validation"]["records"][0] = copy.deepcopy(bad[0]["validation"]["records"][0])
        with self.assertRaises(ValueError): data.validate_fold_collection(bad)

    def test_wrong_fold_or_stop_record_fail(self):
        with self.assertRaises(ValueError): data.validate_prepared_fold(self.fixture["folds"][0], 1)
        bad = copy.deepcopy(self.fixture["folds"][0])
        bad["train"]["records"][0]["tc_subset"] = "projector_stop"
        with self.assertRaises(ValueError): data.validate_prepared_fold(bad, 0)

    def test_hash_checked_before_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "outputs/training/ic_role_mechanism_v2/development_fold_0/runs/prepare_fold0/result.pt"
            path = root / relative; path.parent.mkdir(parents=True)
            torch.save(self.fixture["folds"][0], path)
            entry = dict(fold=0, relative_path=relative, sha256="0" * 64)
            authorization = dict(human_approved=True, scope="msb_rre_v1_fit_only_once")
            with patch.object(torch, "load", side_effect=AssertionError("deserialized too early")):
                with self.assertRaisesRegex(ValueError, "hash"):
                    data.load_prepared_fold(root, entry, authorization)

    def test_no_raw_or_refit_fallback(self):
        for relative in ("fit_raw.pt", "projector_stop.pt", "../escape.pt"):
            with self.assertRaises(ValueError):
                data.load_prepared_fold(Path.cwd(), dict(fold=0,relative_path=relative),
                                        dict(human_approved=True,scope="msb_rre_v1_fit_only_once"))
        with self.assertRaises(ValueError):
            data.load_prepared_fold(Path.cwd(), {}, {})

    def test_validation_label_changes_do_not_change_input_state(self):
        item = copy.deepcopy(self.fixture["folds"][0])
        before = copy.deepcopy(item["state"])
        data.validate_prepared_fold(item, 0)
        self.assertEqual(item["state"], before)
        item["validation"]["labels"] = item["validation"]["labels"].roll(1)
        for r, y in zip(item["validation"]["records"], item["validation"]["labels"]):
            r["four_class_id"] = str(int(y))
        data.validate_prepared_fold(item, 0)
        self.assertEqual(item["state"], before)

    def test_registered_loader_roundtrip_and_digest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "outputs/training/ic_role_mechanism_v2/development_fold_0/runs/prepare_fold0/result.pt"
            path = root / relative; path.parent.mkdir(parents=True)
            item = self.fixture["folds"][0]; torch.save(item, path)
            digest = lambda x: hashlib.sha256(json.dumps(x,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
            entry = dict(fold=0,relative_path=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         registered_train_ids_digest=digest([r["image_id"] for r in item["train"]["records"]]),
                         registered_validation_ids_digest=digest([r["image_id"] for r in item["validation"]["records"]]))
            auth = dict(human_approved=True,scope="msb_rre_v1_fit_only_once")
            loaded = data.load_prepared_fold(root,entry,auth)
            self.assertTrue(torch.equal(loaded["train"]["h"],item["train"]["h"]))
            entry["registered_train_ids_digest"] = "0" * 64
            with self.assertRaises(ValueError): data.load_prepared_fold(root,entry,auth)
