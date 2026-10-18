"""Contracts use scratch metadata and simulated interfaces only."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import check_msb_native_supervision as native
import run_msb_native_supervision_synthetic_acceptance as harness


def fixture_protocol():
    root = Path(__file__).resolve().parents[1]
    authority = json.loads((root/"outputs/msb_completion_repair_synthetic_v1/final/source_identity.json").read_text(encoding="utf-8"))
    paths = tuple(authority["source_sha256"]) + (
        "scripts/check_msb_native_supervision.py", "tests/test_check_msb_native_supervision.py",
        "tests/run_msb_native_supervision_synthetic_acceptance.py")
    return dict(protocol="msb_native_supervision_v1", status="APPROVED_NATIVE_ONLY",
                scope="NATIVE_SUPERVISION_NO_DATA", execution_contract="msb_parent_bound_v2",
                approval_call_id="call_"+"f"*32, approval_answer="APPROVED_SINGLE_NATIVE_NO_DATA",
                output_relative="outputs/msb_native_supervision_v1/acceptance_once",
                max_seconds=480, case_order=["N1","N2","N3","N4"],
                case_limits=[90,45,75,45], require_ac=True, max_attempts=1,
                real_training_authorized=False, source_sha256={p:"1"*64 for p in paths})


def fixture_approval(protocol):
    return dict(call_id=protocol["approval_call_id"], answer=protocol["approval_answer"],
                scope="NATIVE_SUPERVISION_NO_DATA", confirmed_from_human=True,
                registration_commit="a"*40)


class ContractTests(unittest.TestCase):
    def function(self, name):
        value = getattr(native, name, None)
        self.assertTrue(callable(value), "Missing contract implementation: "+name)
        return value

    def test_remaining_uses_larger_elapsed_including_preparation(self):
        remaining = self.function("remaining_acceptance_seconds")
        self.assertEqual(remaining(10.,1000,now_monotonic=45.,now_tick_ms=36000),445.)
        self.assertEqual(remaining(10.,1000,now_monotonic=70.,now_tick_ms=46000),420.)

    def test_deadline_equality_backwards_and_nonfinite_rejected(self):
        remaining = self.function("remaining_acceptance_seconds")
        for mono,tick in ((490.,481000),(491.,2000),(9.,2000),(11.,999),(math.nan,2000),(math.inf,2000)):
            with self.subTest(mono=mono,tick=tick), self.assertRaises((ValueError,RuntimeError)):
                remaining(10.,1000,now_monotonic=mono,now_tick_ms=tick)

    def test_fixed_protocol_validates_without_reading_assets(self):
        validate = self.function("validate_native_protocol")
        with tempfile.TemporaryDirectory() as tmp:
            protocol=fixture_protocol()
            with patch.object(Path,"read_bytes",side_effect=AssertionError("No asset reads in pure validation")):
                result=validate(Path(tmp),protocol,approval=fixture_approval(protocol))
            self.assertEqual(result["output_relative"],"outputs/msb_native_supervision_v1/acceptance_once")
            self.assertEqual(result["case_limits"],[90,45,75,45])

    def test_protocol_mutations_and_true_data_fields_rejected(self):
        validate = self.function("validate_native_protocol")
        changes=[("max_seconds",481),("case_order",["N2","N1","N3","N4"]),
                 ("case_limits",[90,45,76,45]),("max_attempts",2),("require_ac",False),
                 ("real_training_authorized",True),("execution_contract","msb_parent_bound_v1"),
                 ("prepared_entries",["forbidden.pt"]),("output_relative","../outside")]
        with tempfile.TemporaryDirectory() as tmp:
            for key,value in changes:
                p=fixture_protocol();p[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):
                    validate(Path(tmp),p,approval=fixture_approval(p))

    def test_nonreal_and_unconfirmed_approval_rejected(self):
        validate = self.function("validate_native_protocol")
        with tempfile.TemporaryDirectory() as tmp:
            for token in ("call_a42aed1b7bab474bb17246c3df69dc4e","call_6961d4dac6b64d7aba8e538f01c5e097"):
                p=fixture_protocol();p["approval_call_id"]=token
                with self.subTest(token=token),self.assertRaises(ValueError):
                    validate(Path(tmp),p,approval=fixture_approval(p))
            p=fixture_protocol();a=fixture_approval(p);a["confirmed_from_human"]=False
            with self.assertRaises(ValueError): validate(Path(tmp),p,approval=a)

    def test_old_output_missing_source_and_extra_source_rejected(self):
        validate = self.function("validate_native_protocol")
        with tempfile.TemporaryDirectory() as tmp:
            p=fixture_protocol();output=Path(tmp)/p["output_relative"];output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):validate(Path(tmp),p,approval=fixture_approval(p))
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("missing","extra"):
                p=fixture_protocol()
                if mode=="missing":p["source_sha256"].pop("scripts/msb_execution_safety.py")
                else:p["source_sha256"]["unregistered.py"]="1"*64
                with self.subTest(mode=mode),self.assertRaises(ValueError):
                    validate(Path(tmp),p,approval=fixture_approval(p))

    def test_no_execute_flag_denies_before_service_or_file_read(self):
        run=self.function("run_acceptance")
        services=Mock()
        with patch.object(Path,"read_text",side_effect=AssertionError("Unauthorized read")):
            with self.assertRaises(ValueError):
                run(Path("."),Path("never-read.json"),execute_native=False,approval={},services=services)
        self.assertEqual(services.mock_calls,[])


class HarnessGateTests(unittest.TestCase):
    def test_gate_rejects_nonpassing_result_classes(self):
        for outcome in ("empty","skip","expected","unexpected","failure","error","pass"):
            r=unittest.TestResult()
            if outcome!="empty":r.testsRun=1
            if outcome=="skip":r.skipped=[("test","reason")]
            if outcome=="expected":r.expectedFailures=[("test","trace")]
            if outcome=="unexpected":r.unexpectedSuccesses=["test"]
            if outcome=="failure":r.failures=[("test","trace")]
            if outcome=="error":r.errors=[("test","trace")]
            with self.subTest(outcome=outcome):
                self.assertEqual(harness._acceptance_passed(r,forbidden_calls=[],harness_error=None),outcome=="pass")
        self.assertFalse(harness._acceptance_passed(r,forbidden_calls=["native"],harness_error=None))
        self.assertFalse(harness._acceptance_passed(r,forbidden_calls=[],harness_error="trace"))

    def test_actual_main_never_passes_a_skipped_input(self):
        r=unittest.TestResult();r.testsRun=1;r.skipped=[("injected","fixture")]
        previous=(Path.cwd(),tempfile.tempdir,list(sys.path))
        with tempfile.TemporaryDirectory() as tmp:
            try:
                with patch.object(harness,"ROOT",Path(tmp)),patch.object(harness,"_snapshot_sources",return_value={}),patch.object(harness,"_run_modules",return_value=r):
                    code=harness.main(["--phase","contract-green"])
                summary=json.loads((Path(tmp)/"outputs/msb_native_supervision_synthetic_v1/contract-green/attempt-001/acceptance_summary.json").read_text())
                self.assertEqual((code,summary["status"],summary["skipped"]),(1,"FAILED",1))
            finally:
                os.chdir(previous[0]);tempfile.tempdir=previous[1];sys.path[:]=previous[2]
