"""MSB safety contracts: fake clocks, handles, power and processes only."""
import copy
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import msb_execution_safety as safety
except ModuleNotFoundError:
    safety = None


class FakeAwake:
    def __init__(self): self.acquired = True; self.released = False; self.checks = 0
    def check(self):
        if self.released: raise RuntimeError("Released awake guard used")
        self.checks += 1
        return 0.
    def evidence(self):
        return dict(power_request_acquired=self.acquired, power_request_released=self.released)


class FakeAPI:
    def __init__(self):
        self.now = 1000; self.identity = (7, 99); self.alive = True
        self.inbox = []; self.sent = []; self.closed = []; self.aborted = False
    def tick_ms(self): return self.now
    def process_identity(self, handle): return self.identity
    def process_alive(self, handle): return self.alive
    def read_message(self, handle, max_bytes=8192):
        if self.inbox and isinstance(self.inbox[0], BaseException): raise self.inbox.pop(0)
        return self.inbox.pop(0) if self.inbox else None
    def write_message(self, handle, message): self.sent.append(copy.deepcopy(message))
    def close_handle(self, handle): self.closed.append(handle)
    def start_watchdog(self, callback): self.watchdog = callback; return object()
    def stop_watchdog(self, watch): return True
    def abort_worker(self): self.aborted = True


class ScopeTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(safety, "MSB safety module missing")

    def scope(self):
        api, awake = FakeAPI(), FakeAwake()
        binding = safety._VerifiedBinding(
            run_id="run", job_id="job", protocol_digest="1"*64, packet_digest="2"*64,
            nonce="3"*64, worker_pid=8, parent_pid=7, parent_created=99,
            parent_handle=10, read_handle=11, write_handle=12,
            budget=safety.RunBudget(1000, 7201000), sequence=1, lease_until_tick_ms=31000)
        return safety.MSBWorkerScope(binding, api=api, awake_guard=awake), api, awake

    def publish(self, scope, awake):
        awake.released = True; scope.begin_publish(awake.evidence())

    def test_exact_deadline_and_noninteger_budget_rejected(self):
        budget = safety.RunBudget(1000, 7201000)
        self.assertEqual(budget.remaining_seconds(2000), 7199.)
        with self.assertRaisesRegex(RuntimeError, "TIMED_OUT"): budget.remaining_seconds(7201000)
        with self.assertRaisesRegex(RuntimeError, "SUPERVISION"): budget.elapsed_seconds(999)
        for start, end in ((True,7200001),(0.,7200000),(-1,7199999),(0,7200001),(0,float("nan"))):
            with self.assertRaises(ValueError): safety.RunBudget(start,end)

    def test_new_child_guard_does_not_restart_budget(self):
        scope, api, awake = self.scope(); api.now = 11000
        with scope:
            scope.begin_work()
            self.assertEqual(safety.msb_stage_elapsed_seconds(), 10.)
            self.publish(scope,awake)

    def test_missing_parent_capability_cannot_be_recovered_from_packet(self):
        with self.assertRaises(ValueError):
            safety.MSBWorkerScope(dict(status="RUNNING", verified=True), api=FakeAPI(),awake_guard=FakeAwake())

    def test_reused_pid_with_different_creation_time_rejected(self):
        scope, api, _ = self.scope(); api.identity = (7,100)
        with self.assertRaisesRegex(RuntimeError,"SUPERVISION"): scope.__enter__()

    def test_renewal_cannot_extend_original_deadline_or_replay_sequence(self):
        for invalid in ("deadline", "sequence", "lease"):
            scope, api, awake = self.scope()
            with self.assertRaisesRegex(RuntimeError,"SUPERVISION"):
                with scope:
                    scope.begin_work()
                    message = safety.control_message(scope.binding,"RENEW",sequence=2,lease_until_tick_ms=31000)
                    if invalid == "deadline": message["deadline_tick_ms"] += 1
                    if invalid == "sequence": message["sequence"] = 1
                    if invalid == "lease": message["lease_until_tick_ms"] = 40000
                    api.inbox.append(message); api.watchdog()
                    scope.check()
            self.assertTrue(api.aborted)

    def test_parent_death_disconnect_and_expired_lease_revoke_scope(self):
        for failure in ("dead", "disconnect", "lease", "deadline"):
            scope, api, _ = self.scope()
            with self.assertRaises(RuntimeError):
                with scope:
                    scope.begin_work()
                    if failure == "dead": api.alive = False
                    if failure == "disconnect": api.inbox.append(BrokenPipeError())
                    if failure == "lease": api.now = 31000
                    if failure == "deadline": api.now = 7201000
                    api.watchdog(); scope.check()

    def test_unbound_context_fails_instead_of_zero_elapsed(self):
        with self.assertRaisesRegex(RuntimeError,"SUPERVISION"): safety.msb_stage_elapsed_seconds()

    def test_invalid_framing_and_duplicate_json_keys_fail_closed(self):
        scope, _, _ = self.scope()
        msg = safety.control_message(scope.binding,"ACK")
        frame = safety.encode_message(msg)
        self.assertEqual(safety.decode_message(frame), msg)
        for body in (b'{"kind":"ACK","kind":"DONE"}', b'{"kind":NaN}',b'{}',b'{'):
            with self.assertRaises(ValueError): safety.decode_message(struct.pack(">I",len(body))+body)
        for frame in (b"\x00",struct.pack(">I",8193)+b"x"*8193,struct.pack(">I",10)+b"x"):
            with self.assertRaises(ValueError): safety.decode_message(frame)
        msg["extra"] = "x"
        with self.assertRaises(ValueError): safety.encode_message(msg)

    def test_publish_keeps_deadline_after_awake_release_without_reacquire(self):
        scope, api, awake = self.scope()
        with scope:
            scope.begin_work(); self.publish(scope,awake)
            checks = awake.checks
            api.now = 2000
            self.assertEqual(scope.check(),1.)
            self.assertEqual(awake.checks,checks)
        self.assertEqual(scope.evidence()["phase"],"CLOSED")
        self.assertEqual(api.closed,[11,12,10])
