"""MSB-only parent capabilities and cumulative Windows execution contracts."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import struct
import subprocess
import sys
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
            or message["kind"] not in ("HELLO", "GRANT", "RENEW", "ACK", "REVOKE", "CLOSE_REQUEST", "CLOSE_GRANT", "DONE")
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


class MSBParentControl:
    def __init__(self,binding,*,lease_until_tick_ms: int,granted_tick_ms: int):
        binding.budget.remaining_seconds(granted_tick_ms)
        if (type(lease_until_tick_ms) is not int or not granted_tick_ms < lease_until_tick_ms
                <= min(granted_tick_ms+LEASE_MS,binding.budget.deadline_tick_ms)):
            raise _interrupted("invalid parent initial lease")
        self.binding = binding
        self.issued = (1,lease_until_tick_ms)
        self.last_acked = None
        self.outstanding = True
        self.phase = "RUNNING"
        self.done = self.close_granted = False
        self.last_renew = self.last_tick = granted_tick_ms

    def _check(self,now):
        self.binding.budget.remaining_seconds(now)
        if now < self.last_tick or now >= self.issued[1]:
            raise _interrupted("parent control clock or lease expired")
        self.last_tick = now

    def _message(self,kind,*,power_released=False):
        return control_message(self.binding,kind,sequence=self.issued[0],
                               lease_until_tick_ms=self.issued[1],power_released=power_released)

    def _grant_close(self):
        if self.outstanding or self.last_acked != self.issued:
            raise _interrupted("close requires exact acknowledged lease")
        self.phase,self.close_granted = "CLOSE_GRANTED",True
        return [self._message("CLOSE_GRANT",power_released=True)]

    def accept(self,message: dict,now_tick_ms: int) -> list[dict]:
        self._check(now_tick_ms)
        _validate_message(message)
        kind = message["kind"]
        pair = (message["sequence"],message["lease_until_tick_ms"])
        expected = control_message(self.binding,kind,sequence=pair[0],lease_until_tick_ms=pair[1],
                                   power_released=kind in ("CLOSE_REQUEST","DONE"))
        if message != expected: raise _interrupted("control immutable binding differs")
        if kind == "ACK" and self.outstanding and pair == self.issued and self.phase in ("RUNNING","CLOSE_WAIT_ACK"):
            self.last_acked,self.outstanding = pair,False
            return self._grant_close() if self.phase == "CLOSE_WAIT_ACK" else []
        if kind == "CLOSE_REQUEST" and self.phase == "RUNNING" and pair == self.last_acked:
            if self.outstanding:
                if self.issued[0] != pair[0]+1: raise _interrupted("close request skips outstanding renewal")
                self.phase = "CLOSE_WAIT_ACK"
                return []
            return self._grant_close()
        if (kind == "DONE" and self.phase == "CLOSE_GRANTED" and pair == self.issued
                and not self.outstanding and self.last_acked == self.issued):
            self.phase,self.done = "DONE",True
            return []
        raise _interrupted("control order, close request or ACK differs")

    def renew(self,now_tick_ms: int) -> list[dict]:
        self._check(now_tick_ms)
        if self.phase != "RUNNING" or now_tick_ms-self.last_renew < RENEW_MS: return []
        if self.outstanding: raise _interrupted("worker did not acknowledge current lease")
        self.issued = (self.issued[0]+1,min(now_tick_ms+LEASE_MS,self.binding.budget.deadline_tick_ms))
        self.outstanding,self.last_renew = True,now_tick_ms
        return [self._message("RENEW")]

    def require_complete(self,exit_code: int) -> None:
        if (exit_code != 0 or not self.done or not self.close_granted or self.phase != "DONE"
                or self.outstanding or self.last_acked != self.issued):
            raise _interrupted("worker exit lacks exact close handshake")


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
        self.close_granted = self.done_sent = False
        self.close_ready = threading.Event()

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
                if message["kind"] == "CLOSE_GRANT":
                    expected = control_message(self.binding,"CLOSE_GRANT",sequence=self.sequence,
                                               lease_until_tick_ms=self.lease_until,power_released=True)
                    if self.phase != "CLOSING" or self.close_granted or message != expected:
                        raise _interrupted("close grant binding or state differs")
                    self.close_granted = True
                    self.close_ready.set()
                    return
                expected = control_message(self.binding,"RENEW")
                mutable = {"sequence", "lease_until_tick_ms"}
                if (self.close_granted or any(message[k] != expected[k] for k in _FIELDS-mutable)
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
                    end_handshake_verified=self.close_granted and self.done_sent,
                    nonce_sha256=hashlib.sha256(self.binding.nonce.encode()).hexdigest())

    def _close(self):
        with self.lock: self.phase = "CLOSED"
        try:
            if self.watch is not None and not self.api.stop_watchdog(self.watch):
                raise _interrupted("watchdog did not exit")
        finally:
            if self.token is not None: _CURRENT.reset(self.token); self.token = None
            for handle in (self.binding.read_handle,self.binding.write_handle,self.binding.parent_handle):
                self.api.close_handle(handle)

    def _finish(self) -> None:
        with self.lock:
            if self.phase != "PUBLISHING": raise _interrupted("worker did not finish publish phase")
            self.check()
            self.phase = "CLOSING"
            self.api.write_message(self.binding.write_handle,control_message(
                self.binding,"CLOSE_REQUEST",sequence=self.sequence,lease_until_tick_ms=self.lease_until,power_released=True))
        while True:
            with self.lock:
                self.check()
                if self.close_ready.is_set():
                    if not self.close_granted or self.phase != "CLOSING": raise _interrupted("invalid terminal grant state")
                    self.api.write_message(self.binding.write_handle,control_message(
                        self.binding,"DONE",sequence=self.sequence,lease_until_tick_ms=self.lease_until,power_released=True))
                    self.done_sent,self.phase = True,"CLOSED"
                    return
            self.api.pause(.05)

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None: self._finish()
        finally:
            self._close()


def msb_stage_elapsed_seconds() -> float:
    scope = _CURRENT.get()
    if scope is None: raise _interrupted("no inherited MSB worker scope")
    return scope.check()


def _published(packet_path):
    path = Path(packet_path).resolve()
    if path.name != "packet.json" or path.parent.parent.name != "runs":
        raise ValueError("Only an exclusive worker packet is allowed.")
    payload = path.read_bytes()
    if len(payload) > 65536: raise ValueError("Worker metadata exceeds bound.")
    packet = json.loads(payload)
    output = Path(packet["output"]).resolve()
    state = json.loads((output/"stage_state.json").read_text(encoding="utf-8"))
    if (path != output/"runs"/packet["job_id"]/"packet.json" or state.get("status") != "RUNNING"
            or state.get("active_job") != packet["job_id"]
            or any(state.get(k) != packet.get(k) for k in
                   ("run_id","protocol_digest","start_tick_ms","deadline_tick_ms"))
            or state.get("packets",{}).get(packet["job_id"]) != hashlib.sha256(payload).hexdigest()
            or (path.parent/"task_started.json").exists()):
        raise _interrupted("packet is stale, consumed or not published by parent")
    return packet


def _binding_from_metadata(packet, packet_path, handles, worker_pid, api):
    required = {"read_handle","write_handle","parent_handle","nonce"}
    if (not isinstance(handles,dict) or set(handles) != required
            or any(type(handles[k]) is not int or handles[k] <= 0 for k in required-{"nonce"})
            or not isinstance(handles["nonce"],str) or not re.fullmatch(r"[a-f0-9]{64}",handles["nonce"])):
        raise ValueError("Missing inherited worker channel.")
    parent_identity = api.process_identity(handles["parent_handle"])
    if parent_identity != (packet["parent_pid"],packet["parent_created"]):
        raise _interrupted("original parent identity changed")
    return _VerifiedBinding(packet["run_id"],packet["job_id"],packet["protocol_digest"],
                            hashlib.sha256(Path(packet_path).read_bytes()).hexdigest(),handles["nonce"],
                            worker_pid,*parent_identity,handles["parent_handle"],handles["read_handle"],
                            handles["write_handle"],RunBudget(packet["start_tick_ms"],packet["deadline_tick_ms"]),1,0)


def open_worker_scope(packet_path: Path, handles: dict, *, api=None, awake_guard):
    if not isinstance(handles,dict) or set(handles) != {"read_handle","write_handle","parent_handle","nonce"}:
        raise ValueError("Worker cannot start from disk metadata alone.")
    api = NativeMSBAPI() if api is None else api
    binding = None
    try:
        packet = _published(packet_path)
        binding = _binding_from_metadata(packet,packet_path,handles,api.current_pid(),api)
        started = api.tick_ms()
        binding.budget.remaining_seconds(started)
        api.write_message(binding.write_handle,control_message(binding,"HELLO",sequence=0,lease_until_tick_ms=0))
        while True:
            now = api.tick_ms()
            binding.budget.remaining_seconds(now)
            if now-started >= LEASE_MS or not api.process_alive(binding.parent_handle):
                raise _interrupted("parent unavailable before GRANT")
            grant = api.read_message(binding.read_handle)
            if grant is not None:
                _validate_message(grant)
                expected = control_message(binding,"GRANT",lease_until_tick_ms=grant["lease_until_tick_ms"])
                if grant != expected or not now < grant["lease_until_tick_ms"] <= min(now+LEASE_MS,binding.budget.deadline_tick_ms):
                    raise _interrupted("invalid parent GRANT")
                binding = _VerifiedBinding(**{**binding.__dict__,"lease_until_tick_ms":grant["lease_until_tick_ms"]})
                api.write_message(binding.write_handle,control_message(binding,"ACK"))
                return MSBWorkerScope(binding,api=api,awake_guard=awake_guard)
            api.pause(.05)
    except BaseException:
        for name in ("read_handle","write_handle","parent_handle"):
            if type(handles.get(name)) is int and handles[name] > 0: api.close_handle(handles[name])
        raise


def _terminal(error):
    for name in ("TIMED_OUT","SUPERVISION_INTERRUPTED","POWER_RELEASE_FAILED","NUMERIC_RANGE_FAILURE"):
        if name in str(error): return name
    return "FAILED"


@dataclass
class WorkerLaunchOwnership:
    attempt_started: bool = False
    process: object | None = None
    creation_uncertain: bool = False

    def note_attempt(self):
        if self.attempt_started or self.process is not None:
            raise _interrupted("worker launch attempt already consumed")
        self.attempt_started = True

    def note_process(self, process):
        if not self.attempt_started or process is None or (self.process is not None and self.process is not process):
            raise _interrupted("worker process ownership differs")
        self.process = process


class _WorkerStartError(RuntimeError):
    def __init__(self, ownership, reason, *, primary_error=None, restoration_errors=()):
        super().__init__("SUPERVISION_INTERRUPTED: "+reason)
        self.ownership = ownership
        self.spawned_process = ownership.process
        self.primary_error = primary_error
        self.restoration_errors = tuple(restoration_errors)


class MSBParentSupervisor:
    def __init__(self, budget: RunBudget, run_id: str, protocol_digest: str, *, api=None):
        self.budget,self.run_id,self.protocol_digest = budget,run_id,protocol_digest
        self.api = NativeMSBAPI() if api is None else api
        self.guard = None

    def check(self):
        remaining = self.budget.remaining_seconds(self.api.tick_ms())
        if self.guard is not None: self.guard.check()
        return remaining

    def __enter__(self):
        self.check()
        self.guard = self.api.awake_guard()
        self.guard.__enter__()
        try: self.check()
        except BaseException:
            self.guard.__exit__(*sys.exc_info()); raise
        return self

    def __exit__(self,*args):
        if self.guard is not None: self.guard.__exit__(*args)

    def evidence(self): return {} if self.guard is None else self.guard.evidence()

    def _drain_control(self,channel: dict,control: MSBParentControl,binding,*,max_frames: int = 8) -> int:
        if type(max_frames) is not int or max_frames != 8: raise ValueError("Fixed eight-frame drain required")
        if control.binding is not binding: raise _interrupted("drain binding differs")
        count = 0
        for _ in range(max_frames):
            self.check()
            if control.done: return count
            message = self.api.read_message(channel["parent_read"])
            if message is None: return count
            for reply in control.accept(message,self.api.tick_ms()):
                self.check()
                self.api.write_message(channel["parent_write"],reply)
            count += 1
            self.check()
        return count

    def run_job(self, job: dict, protocol: dict) -> dict:
        started = self.api.tick_ms()
        channel, process, kill_job, binding = None,None,None,None
        assigned, control = False,None
        status, error, cleanup_ok = "FAILED","",True
        ownership = WorkerLaunchOwnership()
        guard = None
        try:
            self.check()
            packet = _published(job["packet_path"])
            if (packet["run_id"] != self.run_id or packet["protocol_digest"] != self.protocol_digest
                    or packet["start_tick_ms"] != self.budget.start_tick_ms
                    or packet["deadline_tick_ms"] != self.budget.deadline_tick_ms):
                raise _interrupted("parent run binding differs")
            with self.api.awake_guard() as guard:
                channel = self.api.create_channel()
                kill_job = self.api.create_kill_job()
                child = dict(read_handle=channel["child_read"],write_handle=channel["child_write"],
                             parent_handle=channel["parent_handle"],nonce=secrets.token_hex(32))
                process = self.api.start_worker(job,child,ownership=ownership)
                if process is not ownership.process:
                    ownership.creation_uncertain = True
                    raise _interrupted("launcher returned a different process owner")
                for field in ("child_read","child_write"):
                    self.api.close_handle(channel[field]); channel[field] = None
                binding = _binding_from_metadata(packet,job["packet_path"],child,process.pid,self.api)
                self.check(); guard.check()
                if not self.api.process_alive(binding.parent_handle): raise _interrupted("parent lost before assignment")
                self.api.assign_job(kill_job,process); assigned = True
                while True:
                    self.check(); guard.check()
                    now = self.api.tick_ms()
                    if control is None:
                        if now-started >= LEASE_MS: raise _interrupted("worker HELLO timeout")
                        message = self.api.read_message(channel["parent_read"])
                        if message is not None:
                            _validate_message(message)
                            if message != control_message(binding,"HELLO",sequence=0,lease_until_tick_ms=0):
                                raise _interrupted("worker HELLO identity differs")
                            _published(job["packet_path"])
                            lease = min(now+LEASE_MS,self.budget.deadline_tick_ms)
                            control = MSBParentControl(binding,lease_until_tick_ms=lease,granted_tick_ms=now)
                            self.api.write_message(channel["parent_write"],control_message(binding,"GRANT",lease_until_tick_ms=lease))
                    else:
                        control._check(now)
                        self._drain_control(channel,control,binding)
                    if process.poll() is not None:
                        if control is None: raise _interrupted("worker exited before GRANT")
                        self._drain_control(channel,control,binding)
                        control.require_complete(process.returncode)
                        self.check(); guard.check()
                        status = "COMPLETE"; break
                    if control is not None and not control.done:
                        for renewal in control.renew(self.api.tick_ms()):
                            self.check(); guard.check()
                            self.api.write_message(channel["parent_write"],renewal)
                    self.api.pause(.05)
        except BaseException as failure:
            process = ownership.process
            if ownership.attempt_started and process is None: ownership.creation_uncertain = True
            status,error = _terminal(failure),f"{type(failure).__name__}: {failure}"
        finally:
            cleanup_start = self.api.tick_ms()
            cleanup_ok = not ownership.creation_uncertain
            try:
                if process is not None:
                    if process.poll() is None:
                        if assigned: self.api.terminate_job(kill_job)
                        else: self.api.kill_process(process)
                    cleanup_ok = self.api.wait_process(process,5.) and cleanup_ok
            except BaseException as failure:
                cleanup_ok=False; error += f"; cleanup: {type(failure).__name__}"
            for handle in ([kill_job] + ([] if channel is None else list(channel.values()))):
                if handle is not None:
                    try: self.api.close_handle(handle)
                    except BaseException: cleanup_ok=False
            cleanup_seconds = max(0,self.api.tick_ms()-cleanup_start)/1000.
        power = {} if guard is None else guard.evidence()
        exit_confirmed = process is not None and cleanup_ok and process.returncode is not None
        end_verified = control is not None and control.close_granted and control.done
        if not cleanup_ok: status = "SUPERVISION_INTERRUPTED"
        if status == "COMPLETE" and (not exit_confirmed or not end_verified): status = "SUPERVISION_INTERRUPTED"
        if status == "COMPLETE" and not power.get("power_request_released"):
            status = "POWER_RELEASE_FAILED"
        return dict(job_id=job["job_id"],status=status,attempts=int(ownership.attempt_started),
                    creation_uncertain=ownership.creation_uncertain,
                    pid=None if process is None else process.pid,exit_code=None if process is None else process.returncode,
                    alive=not cleanup_ok or (process is not None and process.returncode is None),error=error,
                    elapsed_seconds=max(0,self.api.tick_ms()-started)/1000.,cleanup_seconds=cleanup_seconds,
                    supervision=power,parent_binding_verified=end_verified,job_assigned=assigned,
                    end_handshake_verified=end_verified,exit_confirmed=exit_confirmed,
                    lease_sequence=1 if control is None else control.issued[0],
                    start_tick_ms=self.budget.start_tick_ms,deadline_tick_ms=self.budget.deadline_tick_ms,
                    final_tick_ms=self.api.tick_ms(),cleanup_confirmed=cleanup_ok,heartbeats=[])


class _BasicLimits(ctypes.Structure):
    _fields_ = [("process_time",ctypes.c_int64),("job_time",ctypes.c_int64),("flags",wintypes.DWORD),
                ("min_working_set",ctypes.c_size_t),("max_working_set",ctypes.c_size_t),
                ("active_processes",wintypes.DWORD),("affinity",ctypes.c_size_t),
                ("priority",wintypes.DWORD),("scheduling",wintypes.DWORD)]


class _IOLimits(ctypes.Structure):
    _fields_ = [(name,ctypes.c_uint64) for name in ("read_ops","write_ops","other_ops","read_bytes","write_bytes","other_bytes")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic",_BasicLimits),("io",_IOLimits),("process_memory",ctypes.c_size_t),
                ("job_memory",ctypes.c_size_t),("peak_process_memory",ctypes.c_size_t),("peak_job_memory",ctypes.c_size_t)]


class NativeMSBAPI:
    """WinAPI adapter. Construction and methods are never used in synthetic tests."""
    def __init__(self):
        if os.name != "nt": raise _interrupted("Windows MSB supervisor required")
        self.kernel = ctypes.WinDLL("kernel32",use_last_error=True)
        signatures = {
            "GetTickCount64": ([],ctypes.c_uint64), "GetCurrentProcessId": ([],wintypes.DWORD),
            "OpenProcess": ([wintypes.DWORD,wintypes.BOOL,wintypes.DWORD],wintypes.HANDLE),
            "GetProcessId": ([wintypes.HANDLE],wintypes.DWORD),
            "GetProcessTimes": ([wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4,wintypes.BOOL),
            "WaitForSingleObject": ([wintypes.HANDLE,wintypes.DWORD],wintypes.DWORD),
            "CreatePipe": ([ctypes.POINTER(wintypes.HANDLE),ctypes.POINTER(wintypes.HANDLE),ctypes.c_void_p,wintypes.DWORD],wintypes.BOOL),
            "CreateJobObjectW": ([ctypes.c_void_p,wintypes.LPCWSTR],wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD],wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE,wintypes.HANDLE],wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE,wintypes.UINT],wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE],wintypes.BOOL),
            "PeekNamedPipe": ([wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p],wintypes.BOOL),
            "ReadFile": ([wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p],wintypes.BOOL),
            "WriteFile": ([wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p],wintypes.BOOL)}
        for name,(args,result) in signatures.items():
            function=getattr(self.kernel,name); function.argtypes=args; function.restype=result
        self.buffers={}; self.writer=threading.Lock()

    def _ok(self,result):
        if not result: raise ctypes.WinError(ctypes.get_last_error())
        return result

    def tick_ms(self): return int(self.kernel.GetTickCount64())
    def current_pid(self): return int(self.kernel.GetCurrentProcessId())

    def current_identity(self):
        handle=self._ok(self.kernel.OpenProcess(0x100000|0x1000,False,self.current_pid()))
        try: return self.process_identity(handle)
        finally: self.close_handle(handle)

    def process_identity(self,handle):
        times=[wintypes.FILETIME() for _ in range(4)]
        self._ok(self.kernel.GetProcessTimes(handle,*(ctypes.byref(t) for t in times)))
        pid=self._ok(self.kernel.GetProcessId(handle))
        return int(pid),(int(times[0].dwHighDateTime)<<32)|int(times[0].dwLowDateTime)

    def process_alive(self,handle):
        result=self.kernel.WaitForSingleObject(handle,0)
        if result not in (0,258): raise _interrupted("parent wait failed")
        return result==258

    def awake_guard(self):
        from lc_execution_safety import WindowsAwakeGuard
        from check_lc_native_environment import NativeEnvironmentAPI
        return WindowsAwakeGuard(api=NativeEnvironmentAPI())

    def create_channel(self):
        handles=[]
        try:
            for _ in range(2):
                read,write=wintypes.HANDLE(),wintypes.HANDLE()
                self._ok(self.kernel.CreatePipe(ctypes.byref(read),ctypes.byref(write),None,MAX_MESSAGE+4))
                handles.extend((read.value,write.value))
            parent=self._ok(self.kernel.OpenProcess(0x100000|0x1000,False,self.current_pid()))
            return dict(parent_read=handles[0],child_write=handles[1],child_read=handles[2],parent_write=handles[3],parent_handle=parent)
        except BaseException:
            for handle in handles: self.close_handle(handle)
            raise

    def start_worker(self,job,child_handles,*,ownership: WorkerLaunchOwnership):
        if (not isinstance(ownership,WorkerLaunchOwnership) or ownership.attempt_started
                or ownership.process is not None or ownership.creation_uncertain):
            raise _interrupted("fresh parent launch ownership required")
        inherited=[child_handles[k] for k in ("read_handle","write_handle","parent_handle")]
        startup=subprocess.STARTUPINFO(); startup.lpAttributeList={"handle_list":inherited}
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                 NUMEXPR_NUM_THREADS="1",PYTHONUNBUFFERED="1",PYTHONUTF8="1")
        process, primary_error, restoration_errors = None, None, []
        try:
            for handle in inherited: os.set_handle_inheritable(handle,True)
            with Path(job["log_path"]).open("x",encoding="utf-8") as log:
                ownership.note_attempt()
                process = subprocess.Popen([*job["command"],"--worker-handles",json.dumps(child_handles)],
                                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                           env=env,close_fds=True,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
                ownership.note_process(process)
        except BaseException as error:
            primary_error = error
        finally:
            if process is not None and ownership.process is None: ownership.note_process(process)
            for handle in inherited:
                try: os.set_handle_inheritable(handle,False)
                except BaseException as error: restoration_errors.append(error)
        if primary_error is not None or restoration_errors:
            if ownership.attempt_started and ownership.process is None: ownership.creation_uncertain = True
            errors = ([] if primary_error is None else [primary_error])+restoration_errors
            reason = "; ".join(f"{type(e).__name__}: {e}" for e in errors)
            raise _WorkerStartError(ownership,reason,primary_error=primary_error,
                                    restoration_errors=restoration_errors) from errors[0]
        return ownership.process

    def create_kill_job(self):
        handle=self._ok(self.kernel.CreateJobObjectW(None,None))
        try:
            limits=_ExtendedLimits(); limits.basic.flags=0x2000
            self._ok(self.kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)))
            return handle
        except BaseException:
            self.close_handle(handle); raise

    def assign_job(self,job_handle,process):
        # The registered CPython 3.11 Windows Popen owns this process handle.
        self._ok(self.kernel.AssignProcessToJobObject(job_handle,int(process._handle)))
    def terminate_job(self,handle): self._ok(self.kernel.TerminateJobObject(handle,1))
    def kill_process(self,process): process.kill()
    def wait_process(self,process,timeout_seconds):
        try: process.wait(timeout=timeout_seconds); return True
        except subprocess.TimeoutExpired: return False
    def close_handle(self,handle): self._ok(self.kernel.CloseHandle(handle))
    def pause(self,seconds): threading.Event().wait(seconds)

    def read_message(self,handle,max_bytes=MAX_MESSAGE):
        if max_bytes != MAX_MESSAGE: raise ValueError("Fixed message bound required.")
        buffer=self.buffers.setdefault(handle,bytearray())
        while True:
            needed=4 if len(buffer)<4 else 4+struct.unpack(">I",buffer[:4])[0]
            if needed>MAX_MESSAGE+4 or needed<=4 and len(buffer)>=4: raise ValueError("Oversize control frame.")
            if len(buffer)==needed and len(buffer)>=4:
                frame=bytes(buffer); buffer.clear(); return decode_message(frame)
            available=wintypes.DWORD()
            self._ok(self.kernel.PeekNamedPipe(handle,None,0,None,ctypes.byref(available),None))
            if not available.value: return None
            count=min(available.value,needed-len(buffer))
            chunk=ctypes.create_string_buffer(count); read=wintypes.DWORD()
            self._ok(self.kernel.ReadFile(handle,chunk,count,ctypes.byref(read),None))
            if read.value!=count: raise _interrupted("truncated pipe read")
            buffer.extend(chunk.raw[:count])

    def write_message(self,handle,message):
        frame=encode_message(message)
        with self.writer:
            written=wintypes.DWORD()
            self._ok(self.kernel.WriteFile(handle,frame,len(frame),ctypes.byref(written),None))
            if written.value!=len(frame): raise _interrupted("truncated pipe write")

    def start_watchdog(self,callback):
        stop=threading.Event()
        def watch():
            while not stop.wait(.05): callback()
        thread=threading.Thread(target=watch,name="msb-parent-watchdog",daemon=True); thread.start()
        return stop,thread
    def stop_watchdog(self,watch):
        stop,thread=watch; stop.set(); thread.join(timeout=1.); return not thread.is_alive()
    def abort_worker(self): os._exit(1)
