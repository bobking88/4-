import json
import hashlib
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_abmp_candidate_capacity import (
    analyze_component_cache, build_component_branches,
    target_segment_interval, validate_sources,
)


def fixture_cache(direct, mapped):
    return {"direct": torch.tensor(direct, dtype=torch.float64),
            "mapped": torch.tensor(mapped, dtype=torch.float64),
            "ti": torch.full((len(direct), 1), .9, dtype=torch.float64),
            "metal": torch.full((len(direct), 1), .9, dtype=torch.float64),
            "original_gate": torch.full((len(direct), 1), .5, dtype=torch.float64),
            "labels": torch.zeros(len(direct), dtype=torch.long)}


def source_fixture(root):
    protocol, source = root / "protocol", root / "source"
    protocol.mkdir()
    for name in ("expert", "manifests", "tc_projection/gate"):
        (source / name).mkdir(parents=True, exist_ok=True)
    names = ("expert_fit", "expert_stop", "gate_fit", "gate_stop_projector_fit", "projector_stop")
    texts = {}
    for index, name in enumerate(names):
        texts[name] = f"image_id,split_group_id,mindat_photo_id\ni{index},g{index},{index+1}\n"
        (protocol / f"{name}.csv").write_text(texts[name], encoding="utf-8")
    training = "image_id,split_group_id,mindat_photo_id\ni0,g0,1\ni1,g1,2\n"
    (source / "manifests/expert_training.csv").write_text(training, encoding="utf-8")
    for name in ("expert/best_model.pt", "tc_projection/gate/best_model.pt"):
        (source / name).write_bytes(b"fixture")
    (source / "expert/run_config.json").write_text('{"verifier_mode":"residual"}', encoding="utf-8")
    (source / "expert/environment.json").write_text('{"verifier_mode":"residual"}', encoding="utf-8")
    (source / "tc_projection/run_config.json").write_text('{}', encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    expert_lock = {"manifest_sha256": {"training_manifest": digest(source / "manifests/expert_training.csv")},
                   "configuration_sha256": digest(source / "expert/run_config.json"),
                   "checkpoint_sha256": digest(source / "expert/best_model.pt")}
    (source / "expert/selection_lock.json").write_text(json.dumps(expert_lock), encoding="utf-8")
    lock = {"protocol_version": "tc_oos_rsg_v1", "status": "locked_before_outer_tensor_loading",
            "manifest_sha256": {name: digest(protocol / f"{name}.csv") for name in names},
            "model_file_sha256": {name: digest(source / path) for name, path in {
                "expert": "expert/best_model.pt", "expert_lock": "expert/selection_lock.json",
                "gate": "tc_projection/gate/best_model.pt", "configuration": "tc_projection/run_config.json"}.items()}}
    (source / "tc_projection/selection_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return protocol, source


class CapacityTests(unittest.TestCase):
    def test_two_nontarget_endpoints_can_have_target_interior(self):
        direct = torch.tensor([[.34, .50, .08, .08]], dtype=torch.float64)
        mapped = torch.tensor([[.34, .08, .50, .08]], dtype=torch.float64)
        interval = target_segment_interval(direct, mapped)
        self.assertTrue(bool(interval["feasible"].item()))
        self.assertAlmostEqual(float(interval["lower"]), .16 / .42, places=7)
        self.assertAlmostEqual(float(interval["upper"]), .26 / .42, places=7)
        self.assertNotEqual(int(direct.argmax()), 0)
        self.assertNotEqual(int(mapped.argmax()), 0)

    def test_constant_nontarget_direction_has_no_capacity(self):
        posterior = torch.tensor([[.1, .7, .1, .1]], dtype=torch.float64)
        interval = target_segment_interval(posterior, posterior)
        self.assertFalse(bool(interval["feasible"].item()))
        self.assertTrue(bool(torch.isfinite(interval["lower"]).all()))

    def test_subnormal_direction_keeps_empty_interval_certificate_finite(self):
        first = torch.tensor([[0., 1., 0., 0.]], dtype=torch.float64)
        second = torch.tensor([[5e-324, 1., 0., 0.]], dtype=torch.float64)
        interval = target_segment_interval(first, second)
        self.assertFalse(bool(interval["feasible"].item()))
        self.assertTrue(bool(torch.isfinite(interval["lower"]).all()))
        self.assertTrue(bool(torch.isfinite(interval["upper"]).all()))

    def test_verification_commutes_with_renormalised_mixture(self):
        cache = fixture_cache([[.7, .1, .1, .1]], [[.2, .6, .1, .1]])
        cache["ti"].fill_(.3)
        gate = torch.tensor([[.7]], dtype=torch.float64)
        branches = build_component_branches(cache, gate, {})
        effective = branches["effective_candidate_gate"]
        expected = effective * branches["direct_verified"] + (1-effective) * branches["mapped_verified"]
        torch.testing.assert_close(expected, branches["oos_verified"], atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(float(branches["target_scale"]), math.exp(-.4))
        self.assertNotAlmostEqual(float(effective), float(gate), places=5)

    def test_verifier_removal_discloses_true_and_false_targets(self):
        cache = fixture_cache([[.45, .40, .1, .05]]*2, [[.45, .40, .1, .05]]*2)
        cache["ti"].zero_()
        cache["metal"].zero_()
        cache["labels"] = torch.tensor([0, 1])
        result = analyze_component_cache(cache, torch.full((2,1),.5,dtype=torch.float64), {})
        self.assertEqual(result["verification_effects"]["fixed"]["true_target_removed"], 1)
        self.assertEqual(result["verification_effects"]["fixed"]["false_target_removed"], 1)
        self.assertEqual(result["verification_effects"]["fixed"]["target_promotions"], 0)
        json.dumps(result, allow_nan=False)

    def test_full_segment_can_recover_target_when_restricted_segment_cannot(self):
        cache = fixture_cache([[.7,.2,.05,.05]], [[.1,.8,.05,.05]])
        result = analyze_component_cache(cache, torch.tensor([[.55]],dtype=torch.float64), {})
        self.assertEqual(result["segment_capacity"]["full_verified"]["missed_fixed_target_feasible"], 1)
        self.assertEqual(result["segment_capacity"]["restricted_verified"]["missed_fixed_target_feasible"], 0)
        self.assertEqual(result["nested_segment_violation_count"], 0)

    def test_rejects_invalid_posterior_and_interval(self):
        q = torch.tensor([[.7,.1,.1,.1]],dtype=torch.float64)
        with self.assertRaises(ValueError):
            target_segment_interval(q, q, lower=.8, upper=.2)
        with self.assertRaises(ValueError):
            target_segment_interval(q*2, q)

    def test_missing_source_lock_rejected_without_outer_access(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                validate_sources(Path(directory), Path(directory))

    def test_source_validation_reads_inner_only_and_binds_training_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, source = source_fixture(Path(directory))
            result = validate_sources(protocol, source)
            self.assertFalse(result["outer_manifest_read"])
            self.assertEqual(result["group_audit"]["cross_subset_group_overlap_count"], 0)
            self.assertNotIn("outer_eval", result["manifest_sha256"])
            (source / "manifests/expert_training.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training manifest"):
                validate_sources(protocol, source)

    def test_source_validation_rejects_changed_inner_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, source = source_fixture(Path(directory))
            (protocol / "projector_stop.csv").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                validate_sources(protocol, source)

    def test_source_validation_rejects_cross_subset_group_overlap_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol, source = source_fixture(Path(directory))
            path = protocol / "projector_stop.csv"
            path.write_text("image_id,split_group_id,mindat_photo_id\ni4,g0,5\n", encoding="utf-8")
            lock_path = source / "tc_projection/selection_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["manifest_sha256"]["projector_stop"] = hashlib.sha256(path.read_bytes()).hexdigest()
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_sources(protocol, source)

    def test_nondefault_verifier_configuration_is_used(self):
        cache = fixture_cache([[.7, .1, .1, .1]], [[.2, .6, .1, .1]])
        cache["ti"].fill_(.2)
        cache["metal"].fill_(.1)
        config = {"ti_threshold": .4, "metallic_threshold": .2, "ti_strength": .3, "metallic_strength": .7}
        result = build_component_branches(cache, cache["original_gate"], config)
        self.assertAlmostEqual(float(result["target_scale"]), math.exp(-.5))
        with self.assertRaises(ValueError):
            build_component_branches(cache, cache["original_gate"], {"ti_strength": math.nan})

    def test_existing_output_is_rejected_before_source_or_image_reads(self):
        from audit_abmp_candidate_capacity import main
        with tempfile.TemporaryDirectory() as directory:
            argv = ["audit", "--dataset-root", directory, "--output-dir", directory]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    main()

    def test_every_feasible_interval_witness_has_the_required_target_margin(self):
        generator = torch.Generator().manual_seed(20261002)
        first = torch.softmax(torch.randn(1000, 4, generator=generator, dtype=torch.float64), 1)
        second = torch.softmax(torch.randn(1000, 4, generator=generator, dtype=torch.float64), 1)
        interval = target_segment_interval(first, second, minimum_margin=.01)
        mask = interval["feasible"].flatten()
        g = ((interval["lower"] + interval["upper"])/2)[mask]
        witness = g*first[mask] + (1-g)*second[mask]
        self.assertTrue(bool(((witness[:, :1]-witness[:, 1:]) >= .01-1e-12).all()))


if __name__ == "__main__":
    unittest.main()
