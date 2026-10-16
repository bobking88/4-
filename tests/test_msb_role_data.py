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

    def summaries(self, folds=None):
        self.assertTrue(callable(getattr(data, "prepared_identity_summary", None)),
                        "Prepared identity summaries missing")
        result = []
        for fold, item in enumerate(folds or self.fixture["folds"]):
            entry = dict(fold=fold, relative_path=f"{data.PREPARED_BASE}/prepare_fold{fold}/result.pt",
                         sha256="1" * 64)
            for part in ("train", "validation"):
                entry[f"registered_{part}_ids_digest"] = data.identity_digest(
                    [r["image_id"] for r in item[part]["records"]])
            result.append(data.prepared_identity_summary(item, fold, entry))
        return result

    def test_every_train_and_validation_row_requires_exact_fit_scope(self):
        for part in ("train", "validation"):
            for scope in (None, "", "projector_stop", "other", "Projector_fit", "projector_fit "):
                with self.subTest(part=part, scope=scope):
                    item = copy.deepcopy(self.fixture["folds"][0])
                    item[part]["records"][0]["tc_subset"] = scope
                    with self.assertRaises(ValueError): data.validate_prepared_fold(item, 0)
            item = copy.deepcopy(self.fixture["folds"][0])
            del item[part]["records"][0]["tc_subset"]
            with self.assertRaises(ValueError): data.validate_prepared_fold(item, 0)

    def test_joint_summary_rejects_locally_valid_oof_duplicate(self):
        folds = copy.deepcopy(self.fixture["folds"])
        folds[1] = copy.deepcopy(folds[0]); folds[1]["inner_fold"] = 1
        for fold, item in enumerate(folds): data.validate_prepared_fold(item, fold)
        summaries = self.summaries(folds)
        with self.assertRaises(ValueError): data.validate_identity_collection(summaries)

    def test_cross_fold_training_identity_and_scope_conflict(self):
        summaries = self.summaries()
        for field, value in (("image_id", "different"), ("four_class_id", "3"), ("tc_subset", "other")):
            bad = copy.deepcopy(summaries)
            bad[0]["train_records"][0][field] = value
            with self.assertRaises(ValueError): data.validate_identity_collection(bad)

    def test_summary_schema_rejects_bool_indices_and_unknown_fields(self):
        summaries = self.summaries()
        self.assertEqual(data.validate_identity_collection(summaries)["groups"], 340)
        for mutate in (lambda s: s[0]["train_indices"].__setitem__(0, True),
                       lambda s: s[0].__setitem__("h", []),
                       lambda s: s[0]["train_records"][0].__setitem__("extra", "x")):
            bad = copy.deepcopy(summaries); mutate(bad)
            with self.assertRaises(ValueError): data.validate_identity_collection(bad)
        def no_tensor(value):
            self.assertFalse(torch.is_tensor(value))
            if isinstance(value, dict):
                for child in value.values(): no_tensor(child)
            elif isinstance(value, list):
                for child in value: no_tensor(child)
        no_tensor(summaries)

    def test_fold_specific_features_are_not_required_equal(self):
        folds = copy.deepcopy(self.fixture["folds"])
        folds[1]["train"]["h"] += 100
        result = data.validate_fold_collection(folds)
        self.assertEqual(result["counts"], [66, 130, 97, 47])
        summaries = self.summaries(folds)
        self.assertEqual(data.validate_identity_collection(summaries), result)

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
