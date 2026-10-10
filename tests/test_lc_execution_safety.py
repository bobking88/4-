import sys
import threading
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import lc_execution_safety as safety
except ModuleNotFoundError:
    safety = None


class FakeWindowsAPI:
    def __init__(self, samples=((100., 80.), (100.1, 80.1)), states=(0x80000000, 0x80000001)):
        self.samples = iter(samples)
        self.states = iter(states)
        self.requests = []

    def set_execution_state(self, flags):
        self.requests.append(flags)
        return next(self.states)

    def sample(self):
        return next(self.samples)


class ExecutionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(safety, "Execution safety module missing")

    def test_native_bridge_uses_sleep_inclusive_and_active_clock_units(self):
        kernel = Mock()
        kernel.GetTickCount64.return_value = 123450
        def active_clock(pointer):
            pointer._obj.value = 1_200_000_000
            return 1
        kernel.QueryUnbiasedInterruptTime.side_effect = active_clock
        kernel.SetThreadExecutionState.return_value = 0x80000000
        with patch.object(safety.os, "name", "nt"), patch.object(safety.ctypes, "WinDLL", return_value=kernel, create=True):
            api = safety.WindowsExecutionAPI()
            self.assertEqual(api.sample(), (123.45, 120.))
            self.assertEqual(api.set_execution_state(0x80000001), 0x80000000)
        self.assertEqual(kernel.SetThreadExecutionState.argtypes, [safety.ctypes.c_uint32])
        self.assertEqual(kernel.GetTickCount64.restype, safety.ctypes.c_uint64)

    def test_failed_native_clock_and_unsupported_platform_fail_closed(self):
        kernel = Mock()
        kernel.GetTickCount64.return_value = 1000
        kernel.QueryUnbiasedInterruptTime.return_value = 0
        with patch.object(safety.os, "name", "nt"), patch.object(safety.ctypes, "WinDLL", return_value=kernel, create=True):
            api = safety.WindowsExecutionAPI()
            with self.assertRaisesRegex(RuntimeError, "SUPERVISION_INTERRUPTED"):
                api.sample()
        with patch.object(safety.os, "name", "posix"):
            with self.assertRaisesRegex(RuntimeError, "POWER_REQUEST_FAILED"):
                safety.WindowsExecutionAPI()

    def test_success_restores_previous_flags_without_forcing_display(self):
        api = FakeWindowsAPI(states=(0x80000002, 0x80000001))
        with safety.WindowsAwakeGuard(api=api) as guard:
            self.assertAlmostEqual(guard.check(), .1)
        self.assertEqual(api.requests, [0x80000001, 0x80000002])

    def test_failed_request_never_enters_protected_body(self):
        api = FakeWindowsAPI(states=(0,))
        entered = False
        with self.assertRaisesRegex(RuntimeError, "POWER_REQUEST_FAILED"):
            with safety.WindowsAwakeGuard(api=api):
                entered = True
        self.assertFalse(entered)
        self.assertEqual(api.requests, [0x80000001])

    def test_clock_initialization_failure_releases_request(self):
        api = FakeWindowsAPI(samples=())
        with self.assertRaises(StopIteration):
            with safety.WindowsAwakeGuard(api=api):
                self.fail("Clock failure must not start work")
        self.assertEqual(api.requests, [0x80000001, 0x80000000])

    def test_exception_and_keyboard_interrupt_release_request(self):
        for error in (ValueError("synthetic"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                api = FakeWindowsAPI()
                with self.assertRaises(type(error)):
                    with safety.WindowsAwakeGuard(api=api):
                        raise error
                self.assertEqual(api.requests, [0x80000001, 0x80000000])

    def test_failed_release_is_not_a_successful_exit(self):
        api = FakeWindowsAPI(states=(0x80000000, 0))
        with self.assertRaisesRegex(RuntimeError, "POWER_RELEASE_FAILED"):
            with safety.WindowsAwakeGuard(api=api):
                pass

    def test_poll_gap_fails_even_when_active_clock_also_advances(self):
        api = FakeWindowsAPI(samples=((100., 80.), (131., 111.)))
        with self.assertRaisesRegex(RuntimeError, "SUPERVISION_INTERRUPTED"):
            with safety.WindowsAwakeGuard(api=api) as guard:
                guard.check()
        self.assertEqual(api.requests[-1], 0x80000000)

    def test_sleep_delta_fails_without_waiting_for_long_poll_gap(self):
        api = FakeWindowsAPI(samples=((100., 80.), (105., 80.1)))
        with self.assertRaisesRegex(RuntimeError, "SUPERVISION_INTERRUPTED"):
            with safety.WindowsAwakeGuard(api=api) as guard:
                guard.check()

    def test_backward_nonfinite_or_active_ahead_clock_fails_closed(self):
        for sample in ((99., 80.), (101., 79.), (float("nan"), 81.), (float("inf"), 81.), (101., 85.)):
            with self.subTest(sample=sample):
                api = FakeWindowsAPI(samples=((100., 80.), sample))
                with self.assertRaisesRegex(RuntimeError, "SUPERVISION_INTERRUPTED"):
                    with safety.WindowsAwakeGuard(api=api) as guard:
                        guard.check()

    def test_nested_worker_checks_parent_gap_before_starting(self):
        parent = FakeWindowsAPI(samples=((100., 80.), (131., 111.)))
        child = FakeWindowsAPI()
        with self.assertRaisesRegex(RuntimeError, "SUPERVISION_INTERRUPTED"):
            with safety.WindowsAwakeGuard(api=parent):
                with safety.WindowsAwakeGuard(api=child):
                    self.fail("No new worker after a stage-level gap")
        self.assertEqual(child.requests, [])

    def test_thread_affinity_failure_does_not_release_another_thread_state(self):
        api = FakeWindowsAPI()
        guard = safety.WindowsAwakeGuard(api=api)
        guard.__enter__()
        errors = []
        def other_thread():
            try:
                guard.__exit__(None, None, None)
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("POWER_RELEASE_FAILED", errors[0])
        self.assertEqual(api.requests, [0x80000001])
        guard.__exit__(None, None, None)
        self.assertEqual(api.requests[-1], 0x80000000)


if __name__ == "__main__":
    unittest.main()
