"""Temporary Windows power requests and fail-closed elapsed-time supervision."""
from __future__ import annotations

import contextvars
import ctypes
import math
import os
import threading

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
MAX_POLL_GAP_SECONDS = 30.
SUSPEND_TOLERANCE_SECONDS = 2.
_CURRENT_GUARD = contextvars.ContextVar("lc_execution_guard", default=None)


class WindowsExecutionAPI:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("POWER_REQUEST_FAILED: Windows execution API required")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        self.kernel.SetThreadExecutionState.restype = ctypes.c_uint32
        self.kernel.GetTickCount64.argtypes = []
        self.kernel.GetTickCount64.restype = ctypes.c_uint64
        self.kernel.QueryUnbiasedInterruptTime.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.kernel.QueryUnbiasedInterruptTime.restype = ctypes.c_int

    def set_execution_state(self, flags):
        return self.kernel.SetThreadExecutionState(flags)

    def sample(self):
        active = ctypes.c_uint64()
        wall = self.kernel.GetTickCount64()/1000.
        if not self.kernel.QueryUnbiasedInterruptTime(ctypes.byref(active)):
            raise RuntimeError("SUPERVISION_INTERRUPTED: active clock unavailable")
        return wall, active.value/10_000_000.


class WindowsAwakeGuard:
    """Thread-affine request; neither user sleep nor OS suspension is prevented."""
    def __init__(self, *, api=None):
        self.api = WindowsExecutionAPI() if api is None else api
        self.owner = None
        self.previous = None
        self.token = None
        self.parent = None
        self.elapsed_seconds = 0.
        self.max_gap = 0.
        self.max_suspend = 0.
        self.request_acquired = False
        self.request_released = False

    def __enter__(self):
        self.parent = _CURRENT_GUARD.get()
        if self.parent is not None:
            self.parent.check()
        previous = self.api.set_execution_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if not previous:
            raise RuntimeError("POWER_REQUEST_FAILED: SetThreadExecutionState returned zero")
        self.previous, self.owner = int(previous), threading.get_ident()
        self.request_acquired = True
        try:
            self.start = self.last = self.api.sample()
            if any(not math.isfinite(v) or v < 0 for v in self.start):
                raise RuntimeError("SUPERVISION_INTERRUPTED: invalid initial clock")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.token = _CURRENT_GUARD.set(self)
        return self

    def check(self):
        if self.owner != threading.get_ident() or self.previous is None:
            raise RuntimeError("SUPERVISION_INTERRUPTED: guard is inactive or on another thread")
        if self.parent is not None:
            self.parent.check()
        sample = self.api.sample()
        if any(not math.isfinite(v) or v < 0 for v in sample):
            raise RuntimeError("SUPERVISION_INTERRUPTED: invalid clock sample")
        wall_delta, active_delta = (sample[i]-self.last[i] for i in (0, 1))
        self.elapsed_seconds = max(self.elapsed_seconds, sample[0]-self.start[0])
        self.max_gap = max(self.max_gap, wall_delta)
        self.max_suspend = max(self.max_suspend, wall_delta-active_delta)
        if (wall_delta < 0 or active_delta < 0 or wall_delta > MAX_POLL_GAP_SECONDS
                or abs(wall_delta-active_delta) > SUSPEND_TOLERANCE_SECONDS):
            raise RuntimeError("SUPERVISION_INTERRUPTED: "
                               f"wall_gap={wall_delta:.6f}s active_gap={active_delta:.6f}s")
        self.last = sample
        return self.elapsed_seconds

    def evidence(self):
        return dict(wall_elapsed_seconds=self.elapsed_seconds, max_poll_gap_seconds=self.max_gap,
                    max_suspend_delta_seconds=self.max_suspend,
                    poll_gap_limit_seconds=MAX_POLL_GAP_SECONDS,
                    suspend_tolerance_seconds=SUSPEND_TOLERANCE_SECONDS,
                    power_request_acquired=self.request_acquired,
                    power_request_released=self.request_released)

    def __exit__(self, *args):
        if self.previous is None:
            return
        if self.owner != threading.get_ident():
            raise RuntimeError("POWER_RELEASE_FAILED: request belongs to another thread")
        try:
            if not self.api.set_execution_state(self.previous | ES_CONTINUOUS):
                raise RuntimeError("POWER_RELEASE_FAILED: SetThreadExecutionState returned zero")
            self.request_released = True
        finally:
            if self.token is not None:
                _CURRENT_GUARD.reset(self.token)
                self.token = None
            self.previous = None


def stage_elapsed_seconds(*, check=True):
    guard = _CURRENT_GUARD.get()
    if guard is None:
        return 0.
    while guard.parent is not None:
        guard = guard.parent
    return guard.check() if check else guard.elapsed_seconds
