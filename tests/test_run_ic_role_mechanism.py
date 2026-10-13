"""Only temporary synthetic packets and mocked real protocol checks."""
import json
import sys
import tempfile
import unittest
import platform
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
try:
    import run_ic_role_mechanism as runner
except ModuleNotFoundError:
    runner = None


class FakeGuard:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def check(self): return 0.
    def evidence(self): return dict(power_request_acquired=True, power_request_released=True)


class MechanismRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "New gated mechanism runner missing")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_draft_or_consumed_protocol_cannot_run_real(self):
        for status in ("DRAFT", "MECHANISM_COMPLETE", "FAILED", "APPROVED"):
            with self.assertRaises(ValueError), patch.object(runner, "load_fit_bundle") as load:
                runner.validate_protocol(self.root, {"status": status}, "fit-mechanism")
            load.assert_not_called()
        with self.assertRaises(ValueError):
            runner.run_stage(self.root, None, self.root/"new", "fit-mechanism")
        self.assertFalse((self.root/"new").exists())

    def test_existing_output_and_path_escape_rejected(self):
        existing = self.root/"old"
        existing.mkdir()
        for path in (existing, self.root/".."/"escape"):
            with self.assertRaises((ValueError, FileExistsError)):
                runner.run_stage(self.root, None, path, "synthetic")

    def test_stop_or_combined_cache_path_rejected(self):
        protocol = runner.synthetic_protocol()
        for key, value in (("cache_path", "cache.pt"), ("fit_path", "stop.pt"),
                           ("assignment_path", "../assignment.json")):
            with self.assertRaises(ValueError):
                runner.validate_protocol(self.root, dict(protocol, **{key: value}), "synthetic")
        for kind in ("evaluate-stop", "refit", "replay-training"):
            packet = self.root/f"{kind}.json"
            packet.write_text(json.dumps(dict(kind=kind)), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.worker(packet)

    def simulate(self, outcome="COMPLETE", guard=FakeGuard, elapsed=None):
        jobs = []
        def job(job, protocol, remaining):
            jobs.append(job)
            return dict(job_id=job["job_id"], status=outcome, alive=False, elapsed_seconds=.01,
                        supervision=dict(power_request_released=True), attempts=1)
        with patch.object(runner, "WindowsAwakeGuard", guard), patch.object(runner, "require_ac_power"), \
                patch.object(runner, "NativeEnvironmentAPI"), \
                patch.object(runner, "run_supervised_job", side_effect=job), \
                patch.object(runner, "stage_elapsed_seconds", side_effect=elapsed or (lambda **kw: 0.)):
            result = runner.run_stage(self.root, None, self.root/"new", "synthetic")
        return result, jobs

    def test_timeout_interrupt_release_fail_closed(self):
        for status in ("TIMED_OUT", "SUPERVISION_INTERRUPTED", "POWER_RELEASE_FAILED", "POWER_REQUEST_FAILED"):
            with self.subTest(status=status):
                result, jobs = self.simulate(status)
                self.assertEqual(result["status"], status)
                self.assertEqual(len(jobs), 1)
                self.assertFalse(result["alive"])
                self.assertEqual(json.loads((self.root/"new"/"stage_state.json").read_text())["status"], status)
                self.temporary.cleanup()
                self.temporary = tempfile.TemporaryDirectory()
                self.addCleanup(self.temporary.cleanup)
                self.root = Path(self.temporary.name)
        class ReleaseFail(FakeGuard):
            def __exit__(self, *args): raise RuntimeError("POWER_RELEASE_FAILED: synthetic")
        result, jobs = self.simulate(guard=ReleaseFail)
        self.assertEqual(result["status"], "POWER_RELEASE_FAILED")

    def test_108_count_three_temperatures_and_whole_stage_budget(self):
        result, jobs = self.simulate()
        self.assertEqual(result["status"], "MECHANISM_COMPLETE")
        self.assertEqual(result["trajectory_count"], 108)
        self.assertEqual(result["temperature_count"], 3)
        self.assertEqual(len(jobs), 112)
        self.assertEqual(result["state_history"], ["PREPARING", "RUNNING", "MECHANISM_COMPLETE"])
        self.assertEqual(result["stage"], "synthetic")
        self.assertFalse(result["real_data_opened"])
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        result, jobs = self.simulate(elapsed=lambda **kw: 7201.)
        self.assertEqual(result["status"], "TIMED_OUT")
        self.assertEqual(len(jobs), 0)

    def test_completed_packet_is_rejected_before_fitting(self):
        output = self.root/"out"
        folder = output/"runs"/"trajectory"
        folder.mkdir(parents=True)
        protocol = self.root/"protocol.json"
        protocol.write_text("{}")
        (output/"stage_state.json").write_text(json.dumps(dict(status="RUNNING", stage="fit-mechanism")))
        (output/"prepared.pt").write_bytes(b"synthetic placeholder")
        (folder/"result.pt").write_bytes(b"already completed")
        packet = dict(kind="trajectory", root=str(self.root), output=str(output), stage="fit-mechanism",
                      protocol_path=str(protocol), prepared_relative="prepared.pt", prepared_sha256="0",
                      max_updates=400, primary_checkpoint=100, unit="C-S-R", seed=20261002, fold=0)
        path = folder/"packet.json"
        path.write_text(json.dumps(packet))
        with patch.object(runner, "validate_protocol"), patch.object(runner, "WindowsAwakeGuard", FakeGuard), \
                patch.object(runner, "NativeEnvironmentAPI"), patch.object(runner, "file_digest", return_value="0"), \
                patch.object(runner.torch, "load", return_value=dict(train={}, validation={})), \
                patch.object(runner, "fit_trajectory", return_value={}) as fit:
            with self.assertRaises(FileExistsError): runner.worker(path)
            fit.assert_not_called()

    def test_interrupted_packet_claim_remains_consumed(self):
        output = self.root/"out"
        folder = output/"runs"/"prepare"
        folder.mkdir(parents=True)
        (output/"stage_state.json").write_text(json.dumps(dict(status="RUNNING", stage="synthetic")))
        path = folder/"packet.json"
        path.write_text(json.dumps(dict(kind="prepare", root=str(self.root), output=str(output), stage="synthetic", fold=0)))
        with patch.object(runner, "WindowsAwakeGuard", side_effect=RuntimeError("POWER_REQUEST_FAILED")), \
                patch.object(runner, "NativeEnvironmentAPI"):
            with self.assertRaises(RuntimeError): runner.worker(path)
        with self.assertRaises(FileExistsError): runner.worker(path)

    def test_executed_dependency_hash_changes_fail_closed(self):
        for dependency in ("scripts/verifier_strong_controls.py", "scripts/analyze_lc_role_adaptation.py"):
            self.assertIn(dependency, runner.SOURCE_FILES)
            protocol = dict(runner.synthetic_protocol(), status="APPROVED_REAL_ONCE",
                authorization=dict(scope="ic_rfa_b_v1_fit_only_mechanism_once", human_approved=True,
                                   user_message_id="synthetic-fixture-not-real-approval", approval_utc="fixture"),
                fit_path=runner.FIT_PATH, fit_sha256=runner.FIT_SHA,
                assignment_path=runner.ASSIGNMENT_PATH, assignment_sha256=runner.ASSIGNMENT_SHA,
                source_sha256={name: "0" for name in runner.SOURCE_FILES}, protocol_path="protocol.json",
                output_relative="out", environment=dict(python=sys.version.split()[0], torch=str(torch.__version__), dtype="float64", device="cpu"))
            def digest(path):
                return "1" if Path(path).relative_to(self.root).as_posix() == dependency else "0"
            with patch.object(runner, "file_digest", side_effect=digest):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    runner.validate_protocol(self.root, protocol, "fit-mechanism")


if __name__ == "__main__": unittest.main()
