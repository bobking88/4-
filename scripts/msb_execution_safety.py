"""MSB-only parent capabilities and cumulative Windows execution contracts."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import re
import struct
import threading

MAX_MS = 7_200_000
LEASE_MS = 30_000
RENEW_MS = 15_000
MAX_MESSAGE = 8192
_CURRENT = ContextVar("msb_worker_scope", default=None)
_FIELDS = {"kind", "run_id", "job_id", "protocol_digest", "packet_digest", "nonce", "worker_pid",
           "parent_pid", "parent_created", "start_tick_ms", "deadline_tick_ms", "sequence",
           "lease_until_tick_ms", "job_assigned", "power_released"}


def _interrupted(reason):
    return RuntimeError("SUPERVISION_INTERRUPTED: "+reason)


@dataclass(frozen=True)
class RunBudget:
    start_tick_ms: int
    deadline_tick_ms: int

    def __post_init__(self):
        if (type(self.start_tick_ms) is not int or type(self.deadline_tick_ms) is not int
                or self.start_tick_ms < 0 or self.deadline_tick_ms-self.start_tick_ms != MAX_MS):
            raise ValueError("Fixed integer cumulative budget required.")

    def elapsed_seconds(self, now_tick_ms: int) -> float:
        if type(now_tick_ms) is not int or now_tick_ms < self.start_tick_ms:
            raise _interrupted("invalid or backwards cumulative clock")
        if now_tick_ms >= self.deadline_tick_ms:
            raise RuntimeError("TIMED_OUT: inherited cumulative deadline")
        return (now_tick_ms-self.start_tick_ms)/1000.

    def remaining_seconds(self, now_tick_ms: int) -> float:
        self.elapsed_seconds(now_tick_ms)
        return (self.deadline_tick_ms-now_tick_ms)/1000.


@dataclass(frozen=True)
class _VerifiedBinding:
    run_id: str
    job_id: str
    protocol_digest: str
    packet_digest: str
    nonce: str
    worker_pid: int
    parent_pid: int
    parent_created: int
    parent_handle: int
    read_handle: int
    write_handle: int
    budget: RunBudget
    sequence: int
    lease_until_tick_ms: int


def control_message(binding, kind, *, sequence=None, lease_until_tick_ms=None, power_released=False):
    return dict(kind=kind, run_id=binding.run_id, job_id=binding.job_id,
                protocol_digest=binding.protocol_digest, packet_digest=binding.packet_digest,
                nonce=binding.nonce, worker_pid=binding.worker_pid, parent_pid=binding.parent_pid,
                parent_created=binding.parent_created, start_tick_ms=binding.budget.start_tick_ms,
                deadline_tick_ms=binding.budget.deadline_tick_ms,
                sequence=binding.sequence if sequence is None else sequence,
                lease_until_tick_ms=binding.lease_until_tick_ms if lease_until_tick_ms is None else lease_until_tick_ms,
                job_assigned=kind != "HELLO", power_released=power_released)


def _validate_message(message):
    if (not isinstance(message, dict) or set(message) != _FIELDS
            or message["kind"] not in ("HELLO", "GRANT", "RENEW", "ACK", "REVOKE", "DONE")
            or any(type(message[k]) is not str or not message[k] for k in ("run_id", "job_id"))
            or any(type(message[k]) is not str or not re.fullmatch(r"[a-f0-9]{64}",message[k])
                   for k in ("protocol_digest", "packet_digest", "nonce"))
            or any(type(message[k]) is not int or message[k] < 0 for k in
                   ("worker_pid", "parent_pid", "parent_created", "sequence", "lease_until_tick_ms"))
            or min(message["worker_pid"], message["parent_pid"]) < 1
            or type(message["job_assigned"]) is not bool or type(message["power_released"]) is not bool):
        raise ValueError("Invalid bounded MSB control message.")
    RunBudget(message["start_tick_ms"],message["deadline_tick_ms"])


def encode_message(message: dict) -> bytes:
    _validate_message(message)
    payload = json.dumps(message, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(payload) <= MAX_MESSAGE: raise ValueError("Control payload exceeds bound.")
    return struct.pack(">I", len(payload))+payload


def decode_message(frame: bytes) -> dict:
    if not isinstance(frame, bytes) or len(frame) < 4: raise ValueError("Truncated control prefix.")
    size = struct.unpack(">I",frame[:4])[0]
    if not 0 < size <= MAX_MESSAGE or len(frame) != size+4: raise ValueError("Invalid control frame size.")
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError("Duplicate control key.")
            result[key] = value
        return result
    def invalid(value): raise ValueError("Nonfinite control value.")
    result = json.loads(frame[4:].decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    _validate_message(result)
    return result


class MSBWorkerScope:
    def __init__(self, binding: dict, *, api, awake_guard):
        if not isinstance(binding, _VerifiedBinding):
            raise ValueError("Live inherited capability required, not disk metadata.")
        self.binding, self.api, self.awake = binding, api, awake_guard
        self.sequence, self.lease_until = binding.sequence, binding.lease_until_tick_ms
        self.phase = "BINDING"
        self.last_tick = binding.budget.start_tick_ms
        self.failure, self.token, self.watch = None, None, None
        self.lock = threading.RLock()
        self.owner = threading.get_ident()
        self.bound = False

    def _alive(self):
        if not self.api.process_alive(self.binding.parent_handle): raise _interrupted("original parent exited")
        now = self.api.tick_ms()
        if now < self.last_tick: raise _interrupted("cumulative tick moved backwards")
        self.binding.budget.elapsed_seconds(now)
        if now >= self.lease_until: raise _interrupted("parent lease expired")
        self.last_tick = now
        return now

    def __enter__(self):
        try:
            if _CURRENT.get() is not None or self.phase != "BINDING" or self.bound:
                raise _interrupted("nested or reused worker scope")
            if self.api.process_identity(self.binding.parent_handle) != (self.binding.parent_pid,self.binding.parent_created):
                raise _interrupted("parent handle creation identity differs")
            if self.sequence != 1: raise _interrupted("initial GRANT sequence differs")
            self._alive()
            if not self.last_tick < self.lease_until <= min(self.last_tick+LEASE_MS,self.binding.budget.deadline_tick_ms):
                raise _interrupted("invalid GRANT lease")
            self.bound = True
            self.token = _CURRENT.set(self)
            self.watch = self.api.start_watchdog(self._pump)
            return self
        except BaseException:
            self._close()
            raise

    def _pump(self):
        try:
            with self.lock:
                if self.phase == "CLOSED": return
                now = self._alive()
                message = self.api.read_message(self.binding.read_handle)
                if message is None: return
                _validate_message(message)
                expected = control_message(self.binding,"RENEW")
                mutable = {"sequence", "lease_until_tick_ms"}
                if (any(message[k] != expected[k] for k in _FIELDS-mutable)
                        or message["sequence"] != self.sequence+1
                        or not now < message["lease_until_tick_ms"] <= min(now+LEASE_MS,self.binding.budget.deadline_tick_ms)):
                    raise _interrupted("renewal replay or immutable binding changed")
                self.sequence, self.lease_until = message["sequence"],message["lease_until_tick_ms"]
                self.api.write_message(self.binding.write_handle,control_message(
                    self.binding,"ACK",sequence=self.sequence,lease_until_tick_ms=self.lease_until))
        except BaseException as error:
            with self.lock:
                self.failure = str(error)
            self.api.abort_worker()

    def check(self) -> float:
        with self.lock:
            if threading.get_ident() != self.owner or not self.bound or self.phase == "CLOSED" or self.failure:
                raise _interrupted(self.failure or "inactive worker scope")
            now = self._alive()
            if self.phase == "WORKING": self.awake.check()
            return self.binding.budget.elapsed_seconds(now)

    def begin_work(self):
        if self.phase != "BINDING": raise _interrupted("invalid working transition")
        self.check()
        power = self.awake.evidence()
        if power.get("power_request_acquired") is not True or power.get("power_request_released") is not False:
            raise _interrupted("working requires acquired awake guard")
        self.awake.check()
        self.phase = "WORKING"

    def begin_publish(self, worker_power: dict):
        if self.phase != "WORKING": raise _interrupted("invalid publish transition")
        actual = self.awake.evidence()
        if (worker_power != actual or actual.get("power_request_acquired") is not True
                or actual.get("power_request_released") is not True):
            raise RuntimeError("POWER_RELEASE_FAILED: worker request not released")
        self.phase = "PUBLISHING"
        self.check()

    def evidence(self):
        return dict(parent_binding_verified=self.bound, job_assigned=self.bound, lease_sequence=self.sequence,
                    start_tick_ms=self.binding.budget.start_tick_ms, deadline_tick_ms=self.binding.budget.deadline_tick_ms,
                    last_tick_ms=self.last_tick, phase=self.phase, revoked=self.failure is not None,
                    nonce_sha256=hashlib.sha256(self.binding.nonce.encode()).hexdigest())

    def _close(self):
        self.phase = "CLOSED"
        try:
            if self.watch is not None and not self.api.stop_watchdog(self.watch):
                raise _interrupted("watchdog did not exit")
        finally:
            if self.token is not None: _CURRENT.reset(self.token); self.token = None
            for handle in (self.binding.read_handle,self.binding.write_handle,self.binding.parent_handle):
                self.api.close_handle(handle)

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                if self.phase != "PUBLISHING": raise _interrupted("worker did not finish publish phase")
                self.check()
                self.api.write_message(self.binding.write_handle,control_message(
                    self.binding,"DONE",sequence=self.sequence,lease_until_tick_ms=self.lease_until,power_released=True))
        finally:
            self._close()


def msb_stage_elapsed_seconds() -> float:
    scope = _CURRENT.get()
    if scope is None: raise _interrupted("no inherited MSB worker scope")
    return scope.check()
