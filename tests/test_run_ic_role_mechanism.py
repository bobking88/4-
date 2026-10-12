"""Only temporary synthetic packets and mocked real protocol checks."""
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__": unittest.main()
