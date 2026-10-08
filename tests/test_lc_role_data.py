import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lc_role_fixtures import raw, digest
try:
    import lc_role_data as data
except ModuleNotFoundError:
    data = None


class RoleDataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "LC-RFA data module missing")
        torch.set_num_threads(1)
        self.raw = raw()
        self.train = torch.arange(80)

    def assert_state_equal(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            if torch.is_tensor(a[key]):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            elif isinstance(a[key], dict):
                self.assert_state_equal(a[key], b[key])
            else:
                self.assertEqual(a[key], b[key])

    def test_preprocessing_uses_only_training_values_and_labels(self):
        state = data.fit_preprocessing(self.raw, self.train)
        changed = copy.deepcopy(self.raw)
        for key in ("p", "H", "evidence", "contradiction"):
            if key == "p":
                changed[key][80:] = .25
            else:
                changed[key][80:] += 10000
        changed["labels"][80:] = 0
        self.assert_state_equal(state, data.fit_preprocessing(changed, self.train))
        self.assertEqual(state["basis"].shape, (1280, 64))
        self.assertEqual(state["train_ids"], [r["image_id"] for r in self.raw["records"][:80]])

    def test_repeat_serialization_and_constant_columns(self):
        self.raw["evidence"][:, 0] = 3
        state = data.fit_preprocessing(self.raw, self.train)
        self.assertEqual(float(state["e_scale"][0]), 1e-8)
        out = data.transform_subset(self.raw, state)
        self.assertEqual(out["e"].shape, (120, 23))
        self.assertEqual(out["P"].shape, (120, 4))
        self.assertTrue(torch.isfinite(out["h"]).all())
        self.assertTrue(torch.equal(out["e"][:, 0], torch.zeros(120, dtype=torch.float64)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            torch.save(state, path)
            restored = torch.load(path, weights_only=True)
        self.assert_state_equal(state, restored)
        self.assert_state_equal(state, data.fit_preprocessing(self.raw, self.train))
        replay = data.transform_subset(self.raw, restored)
        for key in ("h", "e", "P", "log_anchor"):
            torch.testing.assert_close(out[key], replay[key], rtol=0, atol=0)

    def test_pca_and_training_class_failures(self):
        with self.assertRaises(ValueError):
            data.fit_preprocessing(self.raw, torch.arange(64))
        changed = copy.deepcopy(self.raw)
        changed["labels"][:80] = 0
        with self.assertRaises(ValueError):
            data.fit_preprocessing(changed, self.train)
        for indices in (torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([120])):
            with self.assertRaises(ValueError):
                data.fit_preprocessing(self.raw, indices)

    def test_group_assignment_and_missing_validation_class(self):
        assignment = data.make_inner_assignment(self.raw)
        self.assertEqual(assignment, data.make_inner_assignment(self.raw))
        self.assertEqual(len(assignment), 120)
        for fold in range(3):
            val = [r for r in assignment if r["inner_fold"] == fold]
            train = [r for r in assignment if r["inner_fold"] != fold]
            self.assertEqual({r["class_id"] for r in val}, set(range(4)))
            self.assertFalse({r["split_group_id"] for r in val} & {r["split_group_id"] for r in train})
        changed = copy.deepcopy(self.raw)
        changed["labels"][:] = 0
        with self.assertRaises(ValueError):
            data.make_inner_assignment(changed)
        changed = copy.deepcopy(self.raw)
        for row in changed["records"]:
            row["split_group_id"] = "one-group"
        with self.assertRaises(ValueError):
            data.make_inner_assignment(changed)

    def cache_fixture(self, folder):
        source = raw(340)
        H = source.pop("H").float()
        path = folder / "cache.pt"
        torch.save({"fit": H, "stop": H.clone()}, path)
        audit = [dict(row, subset="gate_stop_projector_fit", feature_sha256=digest(feature))
                 for row, feature in zip(source["records"], H)]
        audit_path = folder / "audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        protocol = {"outer_fold": 0, "input_hashes": {}, "cache_path": "cache.pt",
                    "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "audit_path": "audit.json",
                    "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()}
        return source, H, audit, protocol

    def test_cache_checks_original_dtype_before_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, H, _, protocol = self.cache_fixture(root)
            with patch.object(data, "load_subset", return_value=source) as loader:
                out = data.load_cache_subset(root, protocol, "gate_stop_projector_fit")
            loader.assert_called_once_with(root, "gate_stop_projector_fit", {})
            self.assertEqual(out["H"].dtype, torch.float64)
            torch.testing.assert_close(out["H"], H.double(), rtol=0, atol=0)

    def test_cache_identity_hash_scope_and_row_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, H, audit, protocol = self.cache_fixture(root)
            mutations = [lambda s: s["records"][0].update(image_id=s["records"][1]["image_id"]),
                         lambda s: s["records"][0].update(relative_path="wrong.jpg"),
                         lambda s: s["records"][0].update(split_group_id="wrong"),
                         lambda s: s["labels"].__setitem__(0, 3)]
            for mutate in mutations:
                changed = copy.deepcopy(source)
                mutate(changed)
                with patch.object(data, "load_subset", return_value=changed):
                    with self.assertRaises(ValueError):
                        data.load_cache_subset(root, protocol, "gate_stop_projector_fit")
            for bad in (dict(protocol, outer_fold=1), dict(protocol, cache_sha256="bad"),
                        dict(protocol, audit_sha256="bad"), dict(protocol, cache_path="../outside.pt")):
                with self.assertRaises(ValueError):
                    data.load_cache_subset(root, bad, "gate_stop_projector_fit")
            with self.assertRaises(ValueError):
                data.load_cache_subset(root, protocol, "fold_1")
            torch.save({"fit": H.flip(0), "stop": H}, root / "cache.pt")
            protocol["cache_sha256"] = hashlib.sha256((root / "cache.pt").read_bytes()).hexdigest()
            with patch.object(data, "load_subset", return_value=source):
                with self.assertRaises(ValueError):
                    data.load_cache_subset(root, protocol, "gate_stop_projector_fit")


if __name__ == "__main__":
    unittest.main()
