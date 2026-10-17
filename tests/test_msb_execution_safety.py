"""MSB safety contracts: fake clocks, handles, power and processes only."""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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
    def __enter__(self): self.released=False; return self
    def __exit__(self,*args): self.released=True


class FakeAPI:
    def __init__(self):
        self.now = 1000; self.identity = (7, 99); self.alive = True
        self.inbox = []; self.sent = []; self.closed = []; self.aborted = False
        self.auto_close = True
    def tick_ms(self): return self.now
    def process_identity(self, handle): return self.identity
    def process_alive(self, handle): return self.alive
    def read_message(self, handle, max_bytes=8192):
        if self.inbox and isinstance(self.inbox[0], BaseException): raise self.inbox.pop(0)
        return self.inbox.pop(0) if self.inbox else None
    def write_message(self, handle, message):
        self.sent.append(copy.deepcopy(message))
        if message["kind"]=="CLOSE_REQUEST" and self.auto_close:
            self.inbox.append(dict(message,kind="CLOSE_GRANT"))
    def close_handle(self, handle): self.closed.append(handle)
    def start_watchdog(self, callback): self.watchdog = callback; return object()
    def stop_watchdog(self, watch): return True
    def abort_worker(self): self.aborted = True
    def pause(self,seconds):
        self.now+=int(seconds*1000)
        if hasattr(self,"watchdog"): self.watchdog()


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

    def test_ack_request_done_share_state_lock_and_wait_releases_it(self):
        scope,api,awake=self.scope(); api.auto_close=False
        original_write=api.write_message; original_pause=api.pause
        def write(handle,message):
            self.assertTrue(scope.lock._is_owned(),"control write outside state lock")
            original_write(handle,message)
            if message["kind"]=="CLOSE_REQUEST":
                api.now=16000
                api.inbox.append(safety.control_message(scope.binding,"RENEW",sequence=2,lease_until_tick_ms=46000))
            if message["kind"]=="ACK" and message["sequence"]==2:
                api.inbox.append(safety.control_message(scope.binding,"CLOSE_GRANT",sequence=2,
                                                       lease_until_tick_ms=46000,power_released=True))
        def pause(seconds):
            self.assertFalse(scope.lock._is_owned(),"owner holds state lock while waiting")
            original_pause(seconds)
        api.write_message=write; api.pause=pause
        with scope:
            scope.begin_work(); self.publish(scope,awake)
        self.assertEqual([(m["kind"],m["sequence"]) for m in api.sent],[("CLOSE_REQUEST",1),("ACK",2),("DONE",2)])
        self.assertTrue(scope.evidence()["end_handshake_verified"])

    def test_close_grant_wrong_pair_or_binding_is_rejected(self):
        for invalid in ("lease","nonce","sequence"):
            with self.subTest(invalid=invalid):
                scope,api,awake=self.scope(); api.auto_close=False
                original=api.write_message
                def write(handle,message):
                    original(handle,message)
                    if message["kind"]=="CLOSE_REQUEST":
                        grant=dict(message,kind="CLOSE_GRANT")
                        if invalid=="lease": grant["lease_until_tick_ms"]-=1
                        if invalid=="nonce": grant["nonce"]="4"*64
                        if invalid=="sequence": grant["sequence"]+=1
                        api.inbox.append(grant)
                api.write_message=write
                with self.assertRaises(RuntimeError):
                    with scope:
                        scope.begin_work(); self.publish(scope,awake)
                self.assertNotIn("DONE",[m["kind"] for m in api.sent])

    def test_close_deadline_disconnect_and_watchdog_failure_stop_next_job(self):
        for failure in ("deadline","disconnect","watchdog"):
            with self.subTest(failure=failure):
                scope,api,awake=self.scope()
                original=api.write_message
                def write(handle,message):
                    original(handle,message)
                    if message["kind"]=="CLOSE_REQUEST":
                        if failure=="deadline": api.now=7201000
                        if failure=="disconnect": api.inbox[:]=[BrokenPipeError()]
                api.write_message=write
                if failure=="watchdog": api.stop_watchdog=lambda watch:False
                with self.assertRaises(RuntimeError):
                    with scope:
                        scope.begin_work(); self.publish(scope,awake)
                if failure!="watchdog": self.assertNotIn("DONE",[m["kind"] for m in api.sent])
                self.assertEqual(api.closed,[11,12,10])


class CloseControlTests(unittest.TestCase):
    def control(self):
        cls=getattr(safety,"MSBParentControl",None)
        self.assertTrue(callable(cls),"No explicit parent close state machine")
        scope,_,_=ScopeTests.scope(self)
        return cls(scope.binding,lease_until_tick_ms=31000,granted_tick_ms=1000),scope.binding

    def message(self,binding,kind,q=1,lease=31000):
        return safety.control_message(binding,kind,sequence=q,lease_until_tick_ms=lease,
                                      power_released=kind in ("CLOSE_REQUEST","DONE"))

    def test_request_before_inflight_renewal_waits_exact_ack(self):
        c,b=self.control(); c.accept(self.message(b,"ACK"),1000)
        renewal=c.renew(16000)
        self.assertEqual([(m["kind"],m["sequence"],m["lease_until_tick_ms"]) for m in renewal],[("RENEW",2,46000)])
        self.assertEqual(c.accept(self.message(b,"CLOSE_REQUEST"),16000),[])
        self.assertEqual(c.renew(31000),[])
        self.assertFalse(c.done)
        grants=c.accept(self.message(b,"ACK",2,46000),31001)
        self.assertEqual([(m["kind"],m["sequence"],m["lease_until_tick_ms"]) for m in grants],[("CLOSE_GRANT",2,46000)])
        c.accept(self.message(b,"DONE",2,46000),31002)
        c.require_complete(0)

    def test_renewal_ack_before_close_request_finishes(self):
        c,b=self.control(); c.accept(self.message(b,"ACK"),1000); c.renew(16000)
        c.accept(self.message(b,"ACK",2,46000),16001)
        grants=c.accept(self.message(b,"CLOSE_REQUEST",2,46000),16002)
        self.assertEqual([(m["kind"],m["sequence"]) for m in grants],[("CLOSE_GRANT",2)])
        self.assertEqual(c.renew(32000),[])
        c.accept(self.message(b,"DONE",2,46000),32000); c.require_complete(0)

    def test_old_or_duplicate_close_request_is_rejected(self):
        c,b=self.control(); c.accept(self.message(b,"ACK"),1000); c.renew(16000)
        c.accept(self.message(b,"ACK",2,46000),16001)
        with self.assertRaises(RuntimeError): c.accept(self.message(b,"CLOSE_REQUEST"),16002)
        c.accept(self.message(b,"CLOSE_REQUEST",2,46000),16003)
        with self.assertRaises(RuntimeError): c.accept(self.message(b,"CLOSE_REQUEST",2,46000),16004)

    def test_done_before_close_grant_is_rejected(self):
        c,b=self.control(); c.accept(self.message(b,"ACK"),1000)
        with self.assertRaises(RuntimeError): c.accept(self.message(b,"DONE"),1001)
        with self.assertRaises(RuntimeError): c.require_complete(0)

    def test_closing_without_inflight_ack_cannot_complete(self):
        c,b=self.control(); c.accept(self.message(b,"ACK"),1000); c.renew(16000)
        self.assertEqual(c.accept(self.message(b,"CLOSE_REQUEST"),16000),[])
        with self.assertRaises(RuntimeError): c.require_complete(0)
        with self.assertRaises(RuntimeError): c.accept(self.message(b,"ACK",2,46000),46000)


class FakeProcess:
    pid = 8
    returncode = None
    def poll(self): return self.returncode


class LauncherAPI(FakeAPI):
    def __init__(self, failure=None):
        super().__init__(); self.events=[]; self.failure=failure; self.process=FakeProcess()
    def awake_guard(self): return FakeAwake()
    def current_pid(self): return 7
    def create_channel(self):
        return dict(parent_read=21,parent_write=22,child_read=11,child_write=12,parent_handle=10)
    def create_kill_job(self): self.events.append("create-job"); return 30
    def start_worker(self,job,handles,*,ownership):
        self.events.append("spawn")
        ownership.note_attempt(); ownership.note_process(self.process)
        if self.failure=="restore": raise safety._WorkerStartError(ownership,"inheritance restoration failed")
        if self.failure=="before-assignment": self.alive=False
        packet=json.loads(Path(job["packet_path"]).read_text(encoding="utf-8"))
        binding=safety._binding_from_metadata(packet,job["packet_path"],handles,8,self)
        self.inbox.append(safety.control_message(binding,"HELLO",sequence=0,lease_until_tick_ms=0))
        return self.process
    def assign_job(self,handle,process):
        if self.failure=="assign": raise RuntimeError("assignment failed")
        self.events.append("assigned")
    def terminate_job(self,handle): self.events.append("terminated"); self.process.returncode=1
    def kill_process(self,process): self.events.append("killed"); process.returncode=1
    def wait_process(self,process,timeout_seconds): return self.failure!="wait"
    def pause(self,seconds): self.now+=50
    def write_message(self,handle,message):
        super().write_message(handle,message)
        if message["kind"]=="GRANT":
            self.events.append("body-permitted")
            if self.failure=="ack": return
            self.inbox.append(dict(message,kind="ACK"))
            self.inbox.append(dict(message,kind="CLOSE_REQUEST",power_released=True))
        if message["kind"]=="CLOSE_GRANT":
            self.inbox.append(dict(message,kind="DONE"))
            if self.failure=="done-disconnect": self.inbox.append(BrokenPipeError())
            self.process.returncode=0


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(safety)
        self.assertTrue(hasattr(safety,"MSBParentSupervisor"),"MSB parent launcher missing")
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.output=self.root/"out"
        folder=self.output/"runs"/"job"; folder.mkdir(parents=True)
        self.packet=folder/"packet.json"
        packet=dict(run_id="run",job_id="job",protocol_digest="1"*64,output=str(self.output),
                    start_tick_ms=1000,deadline_tick_ms=7201000,parent_pid=7,parent_created=99)
        self.packet.write_text(json.dumps(packet),encoding="utf-8")
        state=dict(status="RUNNING",active_job="job",run_id="run",protocol_digest="1"*64,
                   start_tick_ms=1000,deadline_tick_ms=7201000,
                   packets={"job":hashlib.sha256(self.packet.read_bytes()).hexdigest()})
        (self.output/"stage_state.json").write_text(json.dumps(state),encoding="utf-8")
        self.job=dict(job_id="job",packet_path=str(self.packet),log_path=str(folder/"worker.log"),command=[])

    def run_fake(self,failure=None):
        api=LauncherAPI(failure)
        with safety.MSBParentSupervisor(safety.RunBudget(1000,7201000),"run","1"*64,api=api) as parent:
            result=parent.run_job(self.job,dict(heartbeat_seconds=15,per_fit_seconds=7200))
        return result,api

    def run_native_failure(self, event, *, restore_error=False, close_error=False):
        api=LauncherAPI()
        process=api.process
        class Log:
            def __enter__(inner):
                if event=="before": raise OSError("primary log open failure")
                return object()
            def __exit__(inner,*args):
                if event=="interrupt": raise KeyboardInterrupt("primary post-spawn interrupt")
                if event=="log": raise OSError("primary log exit failure")
        def inherit(handle,flag):
            if restore_error and handle==11 and not flag: raise OSError("secondary restore failure")
        spawn=Mock(return_value=process)
        if event=="unknown": spawn.side_effect=KeyboardInterrupt("primary Popen interrupted")
        def start(job,handles,**kwargs):
            native=object.__new__(safety.NativeMSBAPI)
            with patch.object(safety.subprocess,"STARTUPINFO",return_value=SimpleNamespace()),patch.object(
                    safety.os,"set_handle_inheritable",side_effect=inherit),patch.object(
                    safety.subprocess,"Popen",spawn),patch.object(Path,"open",return_value=Log()):
                return native.start_worker(job,handles,**kwargs)
        api.start_worker=start
        old_wait=api.wait_process
        def wait(p,timeout):
            self.assertIs(p,process)
            api.events.append("waited")
            return old_wait(p,timeout)
        api.wait_process=wait
        old_close=api.close_handle
        def close(handle):
            old_close(handle)
            if close_error and handle==30: raise OSError("close confirmation failed")
        api.close_handle=close
        with safety.MSBParentSupervisor(safety.RunBudget(1000,7201000),"run","1"*64,api=api) as parent:
            result=parent.run_job(self.job,{})
        return result,api,spawn.call_count

    def test_log_exit_error_after_spawn_still_kills_and_waits(self):
        result,api,count=self.run_native_failure("log")
        self.assertEqual(count,1)
        self.assertEqual(result["attempts"],1)
        self.assertIn("killed",api.events)
        self.assertIn("waited",api.events)
        self.assertTrue(result["cleanup_confirmed"])
        self.assertFalse(result["alive"])
        self.assertEqual(set(api.closed),{10,11,12,21,22,30})

    def test_keyboard_interrupt_after_spawn_retains_ownership(self):
        result,api,count=self.run_native_failure("interrupt")
        self.assertEqual(count,1)
        self.assertEqual(result["attempts"],1)
        self.assertIn("killed",api.events)
        self.assertIn("waited",api.events)
        self.assertFalse(result["alive"])

    def test_restore_error_does_not_hide_primary_startup_error(self):
        result,api,count=self.run_native_failure("log",restore_error=True)
        self.assertEqual(count,1)
        self.assertIn("primary log exit",result["error"])
        self.assertIn("secondary restore",result["error"])
        self.assertIn("waited",api.events)

    def test_attempt_without_recoverable_process_is_uncertain_not_clean(self):
        result,api,count=self.run_native_failure("unknown")
        self.assertEqual(count,1)
        self.assertEqual(result["attempts"],1)
        self.assertTrue(result["creation_uncertain"])
        self.assertTrue(result["alive"])
        self.assertFalse(result["cleanup_confirmed"])
        self.assertEqual(result["status"],"SUPERVISION_INTERRUPTED")

    def test_failure_before_popen_has_zero_attempts(self):
        result,api,count=self.run_native_failure("before")
        self.assertEqual(count,0)
        self.assertEqual(result["attempts"],0)
        self.assertTrue(result["cleanup_confirmed"])
        self.assertNotIn("killed",api.events)

    def test_post_spawn_handle_close_failure_is_not_clean(self):
        result,api,count=self.run_native_failure("log",close_error=True)
        self.assertEqual(count,1)
        self.assertEqual(result["attempts"],1)
        self.assertIn("waited",api.events)
        self.assertFalse(result["cleanup_confirmed"])
        self.assertTrue(result["alive"])
        self.assertEqual(set(api.closed),{10,11,12,21,22,30})

    def test_ownership_rejects_second_attempt_and_process_replacement(self):
        ownership_type=getattr(safety,"WorkerLaunchOwnership",None)
        self.assertTrue(callable(ownership_type),"No single-attempt process ownership")
        ownership=ownership_type()
        ownership.note_attempt()
        with self.assertRaises((RuntimeError,ValueError)): ownership.note_attempt()
        process=FakeProcess(); ownership.note_process(process)
        with self.assertRaises((RuntimeError,ValueError)): ownership.note_process(FakeProcess())
        self.assertIs(ownership.process,process)

    def test_no_grant_before_job_assignment(self):
        result,api=self.run_fake()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertLess(api.events.index("assigned"),api.events.index("body-permitted"))
        self.assertTrue(result["parent_binding_verified"])

    def test_parent_exit_between_publish_and_launch_denies_manual_worker(self):
        with self.assertRaises(ValueError): safety.open_worker_scope(self.packet,{},api=FakeAPI(),awake_guard=FakeAwake())
        self.assertFalse(self.packet.with_name("task_started.json").exists())

    def test_parent_exit_before_assignment_never_enters_body(self):
        result,api=self.run_fake("before-assignment")
        self.assertNotEqual(result["status"],"COMPLETE")
        self.assertNotIn("body-permitted",api.events)

    def test_job_assignment_failure_kills_without_fallback(self):
        result,api=self.run_fake("assign")
        self.assertNotEqual(result["status"],"COMPLETE")
        self.assertNotIn("body-permitted",api.events)
        self.assertIn("killed",api.events)

    def test_lost_ack_does_not_renew_or_launch_next_job(self):
        result,api=self.run_fake("ack")
        self.assertNotEqual(result["status"],"COMPLETE")
        self.assertEqual([m["kind"] for m in api.sent],["GRANT"])

    def test_no_renewal_from_unconditional_background_thread(self):
        result,api=self.run_fake()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertFalse(hasattr(api,"watchdog"))

    def test_job_handle_not_inherited_and_all_channel_ends_closed(self):
        result,api=self.run_fake()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual(set(api.closed),{10,11,12,21,22,30})

    def test_kill_wait_or_release_uncertain_cannot_complete(self):
        result,_=self.run_fake("wait")
        self.assertNotEqual(result["status"],"COMPLETE")
        self.assertTrue(result["alive"])

    def test_cleanup_after_deadline_cannot_grant_compute_time(self):
        api=LauncherAPI(); api.now=7201000
        with self.assertRaisesRegex(RuntimeError,"TIMED_OUT"):
            with safety.MSBParentSupervisor(safety.RunBudget(1000,7201000),"run","1"*64,api=api): pass
        self.assertNotIn("spawn",api.events)

    def test_done_closes_pipe_before_process_exit_is_accepted(self):
        api=LauncherAPI("done-disconnect")
        old_poll=api.process.poll
        polls=[0]
        def poll():
            if api.process.returncode==0 and polls[0]<1:
                polls[0]+=1; return None
            return old_poll()
        api.process.poll=poll
        with safety.MSBParentSupervisor(safety.RunBudget(1000,7201000),"run","1"*64,api=api) as parent:
            result=parent.run_job(self.job,{})
        self.assertEqual(result["status"],"COMPLETE")

    def test_worker_handshake_requires_real_pipe_grant_not_flags(self):
        api=FakeAPI(); api.current_pid=lambda:8
        handles=dict(read_handle=11,write_handle=12,parent_handle=10,nonce="3"*64)
        packet=json.loads(self.packet.read_text())
        binding=safety._binding_from_metadata(packet,self.packet,handles,8,api)
        api.inbox.append(safety.control_message(binding,"GRANT",lease_until_tick_ms=31000))
        awake=FakeAwake()
        with safety.open_worker_scope(self.packet,handles,api=api,awake_guard=awake) as scope:
            scope.begin_work(); awake.released=True; scope.begin_publish(awake.evidence())
        self.assertEqual([m["kind"] for m in api.sent],["HELLO","ACK","CLOSE_REQUEST","DONE"])

    def exit_before_read(self,mode="valid"):
        api=LauncherAPI(); api.pending_done=None; api.empty_seen=False; api.exit_ready=False
        original_start=api.start_worker
        def start(job,handles,**kw):
            api.child_handles=handles
            return original_start(job,handles,**kw)
        api.start_worker=start
        def write(handle,message):
            FakeAPI.write_message(api,handle,message)
            if message["kind"]=="GRANT":
                api.inbox.append(dict(message,kind="ACK"))
                api.inbox.append(dict(message,kind="CLOSE_REQUEST",power_released=True))
            if message["kind"]=="CLOSE_GRANT":
                api.pending_done=dict(message,kind="DONE")
        api.write_message=write
        def read(handle,max_bytes=8192):
            if api.pending_done is not None and not api.empty_seen:
                api.empty_seen=True; api.exit_ready=True; api.events.append("read-empty"); return None
            value=FakeAPI.read_message(api,handle,max_bytes)
            if isinstance(value,dict) and value["kind"]=="DONE": api.events.append("read-DONE")
            return value
        api.read_message=read
        def poll():
            if api.exit_ready:
                api.exit_ready=False; api.process.returncode=0; api.events.append("exit-zero")
                if mode=="valid": api.inbox.append(api.pending_done)
                if mode=="truncated": api.inbox.append(ValueError("truncated control frame"))
                api.pending_done=None
            return api.process.returncode
        api.process.poll=poll
        with safety.MSBParentSupervisor(safety.RunBudget(1000,7201000),"run","1"*64,api=api) as parent:
            result=parent.run_job(self.job,{})
        return result,api

    def test_exit_after_grant_drains_queued_done_before_failure(self):
        result,api=self.exit_before_read()
        self.assertEqual(result["status"],"COMPLETE",result["error"])
        self.assertLess(api.events.index("read-empty"),api.events.index("exit-zero"))
        self.assertLess(api.events.index("exit-zero"),api.events.index("read-DONE"))
        self.assertTrue(result["end_handshake_verified"])
        self.assertTrue(result["exit_confirmed"])

    def test_exit_drain_bound_or_truncated_frame_cannot_complete(self):
        for mode in ("missing","truncated"):
            result,_=self.exit_before_read(mode)
            self.assertNotEqual(result["status"],"COMPLETE")
        control,binding=CloseControlTests.control(self)
        api=LauncherAPI()
        with safety.MSBParentSupervisor(safety.RunBudget(1000,7201000),"run","1"*64,api=api) as parent:
            with self.assertRaises(ValueError):
                parent._drain_control(api.create_channel(),control,binding,max_frames=9)

    def test_post_spawn_restore_failure_still_kills_and_waits(self):
        result,api=self.run_fake("restore")
        self.assertNotEqual(result["status"],"COMPLETE")
        self.assertEqual(result["attempts"],1)
        self.assertIn("killed",api.events)
        self.assertFalse(result["alive"])


class ExecutionIdentityTests(unittest.TestCase):
    def fixture(self):
        from test_run_msb_role_mechanism import RunnerTests,runner
        case=RunnerTests()
        case.setUp(); self.addCleanup(case.doCleanups)
        return case,runner

    def test_v1_contract_is_not_accepted_as_new_execution(self):
        case,runner=self.fixture()
        protocol=case.real_metadata_fixture()
        with patch.object(runner,"ROOT",case.root):
            runner.validate_protocol(case.root,protocol,"fit-mechanism")
            protocol["execution_contract"]="msb_parent_bound_v1"
            with self.assertRaises(ValueError): runner.validate_protocol(case.root,protocol,"fit-mechanism")

    def test_new_documentation_and_code_approval_opens_no_real_input(self):
        case,runner=self.fixture()
        protocol=case.real_metadata_fixture()
        ids=("call_a697651b584d4fb8b2116607dcc82b52","call_2a665eb0bae044b38917a4aa34f5127a",
             "call_34d91895cd594efe92c80a1bf6f41138","call_358af6a530944d5c9a15f0ea8039b545")
        with patch.object(runner,"ROOT",case.root),patch.object(runner,"load_prepared_fold",side_effect=AssertionError("real input")):
            for token in ids:
                with self.subTest(token=token):
                    protocol["authorization"]["user_message_id"]=token
                    with self.assertRaisesRegex(ValueError,"approval"):
                        runner.validate_protocol(case.root,protocol,"fit-mechanism")

    def test_missing_close_evidence_stops_next_job(self):
        case,_=self.fixture()
        state,jobs=case.simulate(end_evidence=False)
        self.assertEqual(state["status"],"SUPERVISION_INTERRUPTED")
        self.assertFalse(state["delivery_verified"])
        self.assertEqual(len(jobs),1)


class NativeAdapterTests(unittest.TestCase):
    def setUp(self): self.assertTrue(hasattr(safety,"NativeMSBAPI"))

    def test_native_adapter_pipe_framing_with_fake_kernel_only(self):
        api=object.__new__(safety.NativeMSBAPI)
        scope,_,_=ScopeTests.scope(self)
        message=safety.control_message(scope.binding,"ACK")
        payload=json.dumps(message).encode()
        remaining=bytearray(struct.pack(">I",len(payload))+payload)
        def peek(handle,unused,size,unused2,available,unused3):
            ctypes.cast(available,ctypes.POINTER(safety.wintypes.DWORD)).contents.value=len(remaining); return 1
        def read(handle,buffer,count,nread,overlapped):
            ctypes.memmove(buffer,bytes(remaining[:count]),count); del remaining[:count]
            ctypes.cast(nread,ctypes.POINTER(safety.wintypes.DWORD)).contents.value=count; return 1
        api.kernel=SimpleNamespace(PeekNamedPipe=peek,ReadFile=read)
        api.buffers={}
        self.assertEqual(api.read_message(11),message)
        self.assertIsNone(api.read_message(11))

    def test_native_spawn_explicit_handles_and_restore_without_os_spawn(self):
        api=object.__new__(safety.NativeMSBAPI)
        handles=dict(read_handle=11,write_handle=12,parent_handle=10,nonce="3"*64)
        with tempfile.TemporaryDirectory() as directory:
            startup=SimpleNamespace()
            with patch.object(safety.subprocess,"STARTUPINFO",return_value=startup),patch.object(
                    safety.os,"set_handle_inheritable") as inherit,patch.object(safety.subprocess,"Popen",return_value=FakeProcess()) as spawn:
                api.start_worker(dict(log_path=str(Path(directory)/"log"),command=["python","runner"]),handles,
                                 ownership=safety.WorkerLaunchOwnership())
            self.assertEqual(startup.lpAttributeList,{"handle_list":[11,12,10]})
            self.assertEqual(inherit.call_args_list[-3:],[((11,False),),((12,False),),((10,False),)])
            self.assertTrue(spawn.call_args.kwargs["close_fds"])
            self.assertEqual(spawn.call_args.kwargs["creationflags"],safety.subprocess.CREATE_NO_WINDOW)

    def test_job_object_kill_on_close_and_no_breakaway_flags(self):
        api=object.__new__(safety.NativeMSBAPI)
        captured=[]
        def configure(handle,kind,pointer,size):
            captured.append((handle,kind,ctypes.cast(pointer,ctypes.POINTER(safety._ExtendedLimits)).contents.basic.flags)); return 1
        api.kernel=SimpleNamespace(CreateJobObjectW=lambda a,b:30,SetInformationJobObject=configure)
        self.assertEqual(api.create_kill_job(),30)
        self.assertEqual(captured,[(30,9,0x2000)])

    def test_native_restore_error_retains_spawned_process_for_cleanup(self):
        api=object.__new__(safety.NativeMSBAPI)
        process=FakeProcess()
        handles=dict(read_handle=11,write_handle=12,parent_handle=10,nonce="3"*64)
        def inherit(handle,flag):
            if handle==11 and flag is False: raise OSError("restore failed")
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(safety.subprocess,"STARTUPINFO",return_value=SimpleNamespace()),patch.object(
                    safety.os,"set_handle_inheritable",side_effect=inherit),patch.object(safety.subprocess,"Popen",return_value=process):
                try: api.start_worker(dict(command=["python"],log_path=str(Path(folder)/"log")),handles,
                                      ownership=safety.WorkerLaunchOwnership())
                except BaseException as error:
                    self.assertIs(getattr(error,"spawned_process",None),process)
                else: self.fail("restore failure was swallowed")
