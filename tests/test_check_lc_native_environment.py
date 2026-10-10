import base64
import copy
import ctypes
import hashlib
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lc_execution_safety as safety
try:
    import check_lc_native_environment as native
except ModuleNotFoundError:
    native = None


class FakeHost:
    def __init__(self):
        self.seconds = 0.
        self.requests = []
        self.ac = True
        self.last_input = 80000
        self.release_ok = True
        self.settings_changed = False

    def sample(self):
        if not self.ac:
            raise RuntimeError("SUPERVISION_INTERRUPTED: AC unavailable")
        return 100.+self.seconds, 80.+self.seconds

    def set_execution_state(self, flags):
        self.requests.append(flags)
        return 0 if len(self.requests) == 2 and not self.release_ok else 0x80000000

    def snapshot(self):
        return dict(ac_connected=self.ac, battery_percent=99,
                    session_idle_seconds=(100000+int(self.seconds*1000)-self.last_input)/1000.,
                    last_input_tick_ms=self.last_input)

    def power_settings(self):
        return dict(scheme="381b4222-f694-41f0-9685-ff5bb260df2e",
                    ac_standby_seconds=301 if self.settings_changed else 300,
                    dc_standby_seconds=300)

    def advance(self, seconds):
        while seconds > 0:
            step = min(10., seconds)
            self.seconds += step
            seconds -= step
            safety.stage_elapsed_seconds()


class NativeAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(native, "Independent native acceptance module missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = {
            "protocol": "lc_native_environment_v1", "status": "APPROVED_NATIVE_ONLY",
            "scope": "NATIVE_IDLE_SUPERVISION_NO_TRAINING",
            "approval_call_id": "call_d317f0f1f2fd4fceab4110e48d049e53",
            "approval_answer": "批准环境验收代码与一次原生运行（推荐）",
            "idle_seconds": 360, "timeout_seconds": .25, "max_seconds": 480,
            "long_worker_seconds": 420, "heartbeat_seconds": 10,
            "standby_seconds": 300, "require_ac": True, "max_attempts": 1,
            "output_dir": "outputs/lc_native_environment_v1/acceptance_once",
            "source_sha256": {},
        }
        for name in native.SOURCE_FILES:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"registered source\n")
            self.protocol["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.protocol_path = self.root/"docs/experiment_protocols/lc_native_environment_v1.json"
        self.protocol_path.parent.mkdir(parents=True)
        self.protocol_path.write_text(json.dumps(self.protocol, ensure_ascii=False), encoding="utf-8")
        self.host = FakeHost()
        self.jobs = []
        self.overrides = {}
        worker_patch = patch.object(native, "run_supervised_job", side_effect=self.worker)
        worker_patch.start()
        self.addCleanup(worker_patch.stop)
        dll_patch = patch.object(safety.ctypes, "WinDLL", side_effect=AssertionError("Unmocked native API in synthetic controller test"), create=True)
        dll_patch.start()
        self.addCleanup(dll_patch.stop)

    def worker(self, job, protocol, remaining):
        self.jobs.append(job)
        long = len(self.jobs) == 1
        self.assertEqual(job["command"], [sys.executable, "-c", "import time; time.sleep(360)" if long else "import time; time.sleep(30)"])
        self.assertEqual(protocol, {"per_fit_seconds": 420 if long else .25, "heartbeat_seconds": 10})
        self.assertGreater(remaining, 0)
        self.host.advance(360 if long else .25)
        row = dict(job_id=job["job_id"], status="COMPLETE" if long else "TIMED_OUT",
                   attempts=1, pid=1234, exit_code=0 if long else 1, alive=False,
                   elapsed_seconds=360 if long else .25, error="", log_tail="",
                   supervision={"power_request_acquired": True, "power_request_released": True}, heartbeats=[])
        row.update(self.overrides.get(len(self.jobs), {}))
        return row

    def run_probe(self):
        with patch.object(native, "NativeEnvironmentAPI", return_value=self.host), \
             patch.object(native, "verify_remote_registration", return_value="a"*40), \
             patch.object(native, "run_supervised_job", side_effect=self.worker):
            return native.run_acceptance(self.root, self.protocol_path, execute_native=True)

    def test_no_execute_flag_cannot_create_output_or_power_request(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            native.run_acceptance(self.root, self.protocol_path, execute_native=False)
        self.assertFalse((self.root/self.protocol["output_dir"]).exists())
        self.assertEqual(self.host.requests, [])

    def test_draft_changed_budget_or_unapproved_scope_rejected(self):
        for field, value in (("status", "DRAFT"), ("idle_seconds", 359), ("max_seconds", 481),
                             ("require_ac", False), ("max_attempts", 2), ("scope", "REAL_TRAINING"),
                             ("output_dir", "outputs/another_run"), ("approval_answer", "code only")):
            with self.subTest(field=field):
                protocol = copy.deepcopy(self.protocol)
                protocol[field] = value
                with self.assertRaises(ValueError):
                    native.validate_protocol(self.root, protocol)

    def test_changed_registered_source_rejected_before_native_api(self):
        (self.root/native.SOURCE_FILES[0]).write_bytes(b"changed")
        with patch.object(native, "NativeEnvironmentAPI") as api:
            with self.assertRaisesRegex(ValueError, "source"):
                self.run_probe()
            api.assert_not_called()
        self.assertEqual(self.host.requests, [])

    def test_remote_failure_does_not_request_power_or_start_worker(self):
        with patch.object(native, "verify_remote_registration", side_effect=ValueError("registration")), \
             patch.object(native, "NativeEnvironmentAPI", return_value=self.host):
            with self.assertRaisesRegex(ValueError, "registration"):
                native.run_acceptance(self.root, self.protocol_path, execute_native=True)
        self.assertEqual(self.host.requests, [])
        self.assertEqual(self.jobs, [])

    def test_source_changed_during_remote_preflight_cannot_start_native(self):
        def change(root, path):
            (root/native.SOURCE_FILES[0]).write_bytes(b"changed after preflight")
            return "a"*40
        with patch.object(native, "verify_remote_registration", side_effect=change), \
             patch.object(native, "NativeEnvironmentAPI", return_value=self.host):
            with self.assertRaisesRegex(ValueError, "source"):
                native.run_acceptance(self.root, self.protocol_path, execute_native=True)
        self.assertEqual(self.host.requests, [])

    def test_complete_idle_and_timeout_cleanup_release_request(self):
        result = self.run_probe()
        self.assertEqual(result["status"], "PASS_NATIVE_ACCEPTANCE")
        self.assertEqual(result["real_training_authorized"], False)
        self.assertEqual(result["power_request"]["power_request_released"], True)
        self.assertEqual(self.host.requests, [0x80000001, 0x80000000])
        self.assertEqual(len(self.jobs), 2)
        output = self.root/self.protocol["output_dir"]
        self.assertEqual(json.loads((output/"stage_state.json").read_text())["status"], "PASS_NATIVE_ACCEPTANCE")
        self.assertEqual(len((output/"worker_ledger.jsonl").read_text().splitlines()), 2)

    def test_existing_output_never_resumes_or_retries(self):
        self.run_probe()
        with self.assertRaises(FileExistsError):
            self.run_probe()
        self.assertEqual(len(self.jobs), 2)

    def test_ac_disconnected_initially_starts_no_worker(self):
        self.host.ac = False
        result = self.run_probe()
        self.assertEqual(result["status"], "FAILED_NATIVE_ACCEPTANCE")
        self.assertEqual(self.jobs, [])
        self.assertEqual(self.host.requests, [])

    def test_ac_loss_mid_worker_aborts_and_releases(self):
        def loss(job, protocol, remaining):
            self.jobs.append(job)
            self.host.ac = False
            self.host.sample()
        with patch.object(native, "run_supervised_job", side_effect=loss):
            # run_probe normally replaces worker; exercise the controller with this boundary directly.
            with patch.object(native, "NativeEnvironmentAPI", return_value=self.host), \
                 patch.object(native, "verify_remote_registration", return_value="a"*40):
                result = native.run_acceptance(self.root, self.protocol_path, execute_native=True)
        self.assertEqual(result["status"], "FAILED_NATIVE_ACCEPTANCE")
        self.assertEqual(len(self.jobs), 1)
        self.assertEqual(self.host.requests[-1], 0x80000000)

    def assert_long_failure_stops(self, failure):
        self.overrides = {1: failure}
        self.assertEqual(self.run_probe()["status"], "FAILED_NATIVE_ACCEPTANCE")
        self.assertEqual(len(self.jobs), 1)

    def test_long_worker_failure_stops_before_short_job(self):
        self.assert_long_failure_stops({"status": "FAILED"})

    def test_long_unreaped_child_stops_before_short_job(self):
        self.assert_long_failure_stops({"alive": True})

    def test_long_missing_exit_stops_before_short_job(self):
        self.assert_long_failure_stops({"exit_code": None})

    def test_long_missing_attempt_stops_before_short_job(self):
        self.assert_long_failure_stops({"attempts": 0})

    def test_long_incomplete_duration_stops_before_short_job(self):
        self.assert_long_failure_stops({"elapsed_seconds": 359})

    def test_long_unreleased_request_stops_before_short_job(self):
        self.assert_long_failure_stops({"supervision": {"power_request_acquired": True, "power_request_released": False}})

    def test_short_child_must_time_out_and_be_reaped(self):
        self.overrides = {2: {"status": "COMPLETE", "exit_code": 0}}
        self.assertEqual(self.run_probe()["status"], "FAILED_NATIVE_ACCEPTANCE")
        self.assertEqual(len(self.jobs), 2)

    def test_exhausted_cumulative_budget_starts_no_second_worker(self):
        ordinary = self.worker
        def spend(job, protocol, remaining):
            row = ordinary(job, protocol, remaining)
            self.host.advance(120)
            return row
        with patch.object(native, "NativeEnvironmentAPI", return_value=self.host), \
             patch.object(native, "verify_remote_registration", return_value="a"*40), \
             patch.object(native, "run_supervised_job", side_effect=spend):
            result = native.run_acceptance(self.root, self.protocol_path, execute_native=True)
        self.assertEqual(result["status"], "FAILED_NATIVE_ACCEPTANCE")
        self.assertEqual(len(self.jobs), 1)

    def test_recent_input_is_inconclusive_not_pass_or_retry(self):
        ordinary = self.worker
        def interaction(job, protocol, remaining):
            row = ordinary(job, protocol, remaining)
            if len(self.jobs) == 1:
                self.host.last_input = 450000
            return row
        with patch.object(native, "NativeEnvironmentAPI", return_value=self.host), \
             patch.object(native, "verify_remote_registration", return_value="a"*40), \
             patch.object(native, "run_supervised_job", side_effect=interaction):
            result = native.run_acceptance(self.root, self.protocol_path, execute_native=True)
        self.assertEqual(result["status"], "INCONCLUSIVE_IDLE_WINDOW")
        self.assertEqual(len(self.jobs), 2)

    def test_settings_change_or_release_failure_cannot_pass(self):
        self.host.release_ok = False
        self.assertEqual(self.run_probe()["status"], "FAILED_NATIVE_ACCEPTANCE")
        self.assertEqual(len(self.jobs), 2)

    def test_supervision_gap_during_release_cannot_pass(self):
        previous = self.host.set_execution_state
        def slow_release(flags):
            returned = previous(flags)
            if flags == 0x80000000:
                self.host.seconds += 120
            return returned
        self.host.set_execution_state = slow_release
        result = self.run_probe()
        self.assertEqual(result["status"], "FAILED_NATIVE_ACCEPTANCE")
        self.assertGreaterEqual(result["elapsed_seconds"], 480)

    def test_power_setting_change_cannot_pass(self):
        ordinary = self.worker
        def change(job, protocol, remaining):
            row = ordinary(job, protocol, remaining)
            self.host.settings_changed = True
            return row
        with patch.object(native, "NativeEnvironmentAPI", return_value=self.host), \
             patch.object(native, "verify_remote_registration", return_value="a"*40), \
             patch.object(native, "run_supervised_job", side_effect=change):
            self.assertEqual(native.run_acceptance(self.root, self.protocol_path, execute_native=True)["status"], "FAILED_NATIVE_ACCEPTANCE")

    def test_remote_protocol_and_source_bytes_must_match(self):
        entries = []
        for name in native.SOURCE_FILES:
            payload = (self.root/name).read_bytes()
            digest = hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest()
            entries.append(dict(path=name, sha=digest, type="blob"))
        tree = {"truncated": False, "tree": entries}
        content = {"content": base64.b64encode(self.protocol_path.read_bytes()).decode()}
        responses = [{"object": {"sha": "a"*40}}, content,
                     {"tree": {"sha": "b"*40}}, tree]
        def response(command, **kwargs):
            return types.SimpleNamespace(stdout=json.dumps(responses.pop(0)).encode())
        with patch.object(native.subprocess, "run", side_effect=response):
            self.assertEqual(native.verify_remote_registration(self.root, self.protocol_path), "a"*40)
        tree["tree"][0]["sha"] = "c"*40
        responses.extend([{"object": {"sha": "a"*40}}, content, {"tree": {"sha": "b"*40}}, tree])
        with patch.object(native.subprocess, "run", side_effect=response):
            with self.assertRaisesRegex(ValueError, "source"):
                native.verify_remote_registration(self.root, self.protocol_path)


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(native, "Independent native acceptance module missing")
        self.kernel, self.user, self.power = Mock(), Mock(), Mock()
        self.kernel.GetTickCount64.return_value = (1 << 32)+5000
        def active(pointer):
            pointer._obj.value = 300000000
            return 1
        self.kernel.QueryUnbiasedInterruptTime.side_effect = active
        def status(pointer):
            pointer._obj.ACLineStatus = 1
            pointer._obj.BatteryLifePercent = 255
            return 1
        self.kernel.GetSystemPowerStatus.side_effect = status
        def last_input(pointer):
            pointer._obj.dwTime = (1 << 32)-5000
            return 1
        self.user.GetLastInputInfo.side_effect = last_input
        self.guid = (ctypes.c_ubyte*16).from_buffer_copy(uuid.UUID("381b4222-f694-41f0-9685-ff5bb260df2e").bytes_le)
        def scheme(root, pointer):
            pointer._obj.value = ctypes.addressof(self.guid)
            return 0
        self.power.PowerGetActiveScheme.side_effect = scheme
        def index(root, scheme, subgroup, setting, pointer):
            pointer._obj.value = 300
            return 0
        self.power.PowerReadACValueIndex.side_effect = index
        self.power.PowerReadDCValueIndex.side_effect = index
        self.kernel.LocalFree.return_value = None

    def api(self):
        def dll(name, **kwargs):
            return {"kernel32": self.kernel, "user32": self.user, "PowrProf": self.power}[name]
        with patch.object(safety.os, "name", "nt"), patch.object(safety.ctypes, "WinDLL", side_effect=dll, create=True):
            return native.NativeEnvironmentAPI()

    def test_bridge_reads_ac_session_idle_wrap_and_settings_without_writes(self):
        api = self.api()
        snapshot = api.snapshot()
        self.assertEqual(snapshot["session_idle_seconds"], 10.)
        self.assertIsNone(snapshot["battery_percent"])
        self.assertTrue(snapshot["ac_connected"])
        self.assertEqual(api.power_settings(), dict(scheme="381b4222-f694-41f0-9685-ff5bb260df2e", ac_standby_seconds=300, dc_standby_seconds=300))
        self.assertEqual(ctypes.sizeof(native.SYSTEM_POWER_STATUS), 12)
        self.assertEqual(ctypes.sizeof(native.LASTINPUTINFO), 8)
        self.kernel.LocalFree.assert_called_once()
        self.kernel.SetThreadExecutionState.assert_not_called()

    def test_native_read_errors_fail_closed_and_free_scheme(self):
        api = self.api()
        self.user.GetLastInputInfo.return_value = 0
        self.user.GetLastInputInfo.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "idle"):
            api.snapshot()
        self.power.PowerReadACValueIndex.side_effect = None
        self.power.PowerReadACValueIndex.return_value = 5
        with self.assertRaisesRegex(RuntimeError, "power setting"):
            api.power_settings()
        self.kernel.LocalFree.assert_called_once()

    def test_power_monitor_fails_on_unknown_or_battery_only_ac(self):
        api = self.api()
        for line in (0, 255):
            def status(pointer):
                pointer._obj.ACLineStatus = line
                return 1
            self.kernel.GetSystemPowerStatus.side_effect = status
            with self.assertRaisesRegex(RuntimeError, "AC"):
                api.sample()


if __name__ == "__main__":
    unittest.main()
