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
                summary=json.loads((Path(tmp)/"outputs/msb_native_acceptance_repair_synthetic_v1/contract-green/attempt-001/acceptance_summary.json").read_text())
                self.assertEqual((code,summary["status"],summary["skipped"]),(1,"FAILED",1))
            finally:
                os.chdir(previous[0]);tempfile.tempdir=previous[1];sys.path[:]=previous[2]

import msb_execution_safety as safety
from test_msb_execution_safety import FakeAPI, FakeAwake, LauncherAPI


class CaseLauncher(LauncherAPI):
    def __init__(self,case_id,eof=False):
        super().__init__();self.case_id=case_id;self.eof=eof;self.latest=None;self.body_start=None
        self.close_queued=False;self.process._handle=88
    def current_identity(self):return (7,99)
    def process_identity(self,handle):return (8,199) if handle==88 else (7,99)
    def write_message(self,handle,message):
        FakeAPI.write_message(self,handle,message)
        if message["kind"] in ("GRANT","RENEW"):
            self.latest=copy.deepcopy(message)
            if self.body_start is None:self.body_start=self.now
            self.inbox.append(dict(message,kind="ACK"))
            if message["kind"]=="GRANT" and self.case_id=="N2":
                self.process.returncode=23
                if self.eof:
                    error=OSError("native pipe EOF");error.winerror=109;self.inbox.append(error)
            if message["kind"]=="GRANT" and self.case_id=="N3":
                self.inbox.append(dict(message,kind="CLOSE_REQUEST",power_released=True))
        elif message["kind"]=="CLOSE_GRANT":
            self.inbox.append(dict(message,kind="DONE"))
            if self.case_id!="N3":self.process.returncode=0
    def pause(self,seconds):
        self.now+=50
        if (self.case_id=="N1" and self.body_start is not None and not self.close_queued
                and self.now-self.body_start>=45000):
            self.close_queued=True
            self.inbox.append(dict(self.latest,kind="CLOSE_REQUEST",power_released=True))


class WorkerAPI(FakeAPI):
    def __init__(self):
        super().__init__();self.sequence=1;self.next_renew=16000;self.binding=None
    def current_pid(self):return 8
    def current_identity(self):return (8,199)
    def write_message(self,handle,message):
        super().write_message(handle,message)
        if message["kind"]=="HELLO":
            self.binding=message
            self.inbox.append(dict(message,kind="GRANT",sequence=1,lease_until_tick_ms=self.now+30000,job_assigned=True))
    def pause(self,seconds):
        self.now+=max(1,int(seconds*1000))
        if self.binding is not None and self.now>=self.next_renew:
            self.sequence+=1;self.next_renew=self.now+15000
            self.inbox.append(dict(self.binding,kind="RENEW",sequence=self.sequence,lease_until_tick_ms=self.now+30000,job_assigned=True))
        if hasattr(self,"watchdog"):self.watchdog()


class FakeDriverServices:
    def __init__(self,api):
        self.api=api;self.exits=[];self.callbacks=[];self.acks=[];self.traces=[]
    def tick_ms(self):return self.api.tick_ms()
    def monotonic(self):return self.api.tick_ms()/1000.
    def pause(self,seconds):self.api.pause(seconds)
    def awake_guard(self):return FakeAwake()
    def exit_process(self,code):self.exits.append(code);raise SystemExit(code)
    def start_fault(self,callback):self.callbacks.append(callback);return SimpleNamespace(stopped=False)
    def stop_fault(self,watch):watch.stopped=True;return True
    def persist_trace(self,path,events):
        self.traces.append(copy.deepcopy(events))
        path.write_text(json.dumps(events),encoding="utf-8")
    def read_ack(self,case_dir,ready):return dict(ready,collector_verified=True,query_only=True,job_handle_retained=False,
                                               alive_at_ack=True,ack_tick_ms=self.tick_ms())
    def write_ready(self,case_dir,ready):
        self.acks.append(copy.deepcopy(ready))
        (case_dir/"worker_ready.json").write_text(json.dumps(ready),encoding="utf-8")


def case_binding():
    return dict(run_id="native-test",protocol_digest="1"*64,start_tick_ms=1000,
                deadline_tick_ms=7201000,start_monotonic=1.,case_limits=[90,45,75,45])


class WorkerTests(unittest.TestCase):
    def function(self,name):
        value=getattr(native,name,None)
        self.assertTrue(callable(value),"Missing worker implementation: "+name)
        return value

    def case_root(self,root,case_id):
        self.enterContext(patch.object(native,"execution_origins",return_value=observed_origins(root)))
        return Path(root)/"outputs/msb_native_supervision_v1/acceptance_once/cases"/case_id

    def test_controller_reuses_real_parent_and_original_budget(self):
        run=self.function("run_controller")
        api=CaseLauncher("N1");services=FakeDriverServices(api)
        with tempfile.TemporaryDirectory() as tmp:
            output=self.case_root(tmp,"N1");output.mkdir(parents=True)
            row=run(Path(tmp),output,"N1",case_binding(),services=services)
            self.assertEqual(row["msb"]["status"],"COMPLETE")
            self.assertEqual((row["msb"]["start_tick_ms"],row["msb"]["deadline_tick_ms"]),(1000,7201000))
            self.assertGreaterEqual(row["msb"]["lease_sequence"],3)
            self.assertTrue(row["msb"]["end_handshake_verified"])
            self.assertTrue(row["parent_power"]["power_request_released"])
            kinds=[e.get("kind") for e in row["events"]]
            self.assertIn("assigned",kinds);self.assertGreaterEqual(kinds.count("RENEW"),2)
            self.assertTrue(api.close_queued);self.assertFalse(row["msb"]["alive"])

    def test_no_done_and_native_eof_remain_fail_closed_with_exit23(self):
        run=self.function("run_controller")
        for eof,status in ((False,"SUPERVISION_INTERRUPTED"),(True,"FAILED")):
            with self.subTest(eof=eof),tempfile.TemporaryDirectory() as tmp:
                out=self.case_root(tmp,"N2");out.mkdir(parents=True)
                row=run(Path(tmp),out,"N2",case_binding(),services=FakeDriverServices(CaseLauncher("N2",eof)))
                self.assertEqual(row["msb"]["status"],status)
                self.assertEqual(row["msb"]["exit_code"],23)
                self.assertTrue(row["msb"]["cleanup_confirmed"]);self.assertFalse(row["msb"]["end_handshake_verified"])

    def test_done_linger_expires_original_frozen_lease_not_outer_timer(self):
        run=self.function("run_controller");api=CaseLauncher("N3")
        with tempfile.TemporaryDirectory() as tmp:
            out=self.case_root(tmp,"N3");out.mkdir(parents=True)
            row=run(Path(tmp),out,"N3",case_binding(),services=FakeDriverServices(api))
            self.assertEqual(row["msb"]["status"],"SUPERVISION_INTERRUPTED")
            self.assertIn("lease",row["msb"]["error"])
            self.assertTrue(row["msb"]["end_handshake_verified"]);self.assertTrue(row["msb"]["exit_confirmed"])
            self.assertIn("terminated",api.events);self.assertLess(api.now,7201000)

    def test_parent_fault_case_still_uses_real_launch_and_rejects_assignment_failure(self):
        run=self.function("run_controller");api=CaseLauncher("N4");api.failure="assign"
        services=FakeDriverServices(api)
        with tempfile.TemporaryDirectory() as tmp:
            out=self.case_root(tmp,"N4");out.mkdir(parents=True)
            row=run(Path(tmp),out,"N4",case_binding(),services=services)
            self.assertIn("spawn",api.events);self.assertEqual(len(services.callbacks),1)
            self.assertNotEqual(row["msb"]["status"],"COMPLETE")
            self.assertTrue(row["fault_monitor_stopped"])

    def packet(self,tmp,case_id):
        out=self.case_root(tmp,case_id)/"controller";folder=out/"runs"/case_id;folder.mkdir(parents=True)
        packet=dict(case_binding(),root=str(Path(tmp)),output=str(out),case_id=case_id,job_id=case_id,
                    stage="native-supervision-no-data",parent_pid=7,parent_created=99)
        path=folder/"packet.json";path.write_text(json.dumps(packet),encoding="utf-8")
        state=dict(packet,status="RUNNING",active_job=case_id,packets={case_id:__import__("hashlib").sha256(path.read_bytes()).hexdigest()})
        (out/"stage_state.json").write_text(json.dumps(state),encoding="utf-8")
        return path,dict(read_handle=11,write_handle=12,parent_handle=10,nonce="3"*64)

    def test_real_worker_scope_waits45_and_completes_real_close(self):
        run=self.function("run_no_data_worker");api=WorkerAPI();services=FakeDriverServices(api)
        with tempfile.TemporaryDirectory() as tmp:
            path,handles=self.packet(tmp,"N1")
            result=run(path,handles,services=services)
            self.assertGreaterEqual(result["work_elapsed_ms"],45000)
            self.assertTrue(result["worker_execution"]["end_handshake_verified"])
            self.assertTrue(result["worker_power"]["power_request_released"])
            self.assertGreaterEqual(api.sequence,3)
            self.assertEqual([m["kind"] for m in api.sent][-2:],["CLOSE_REQUEST","DONE"])
            self.assertEqual(set(api.closed),{10,11,12})

    def test_worker_exit23_never_publishes_result_or_false_release(self):
        run=self.function("run_no_data_worker");api=WorkerAPI();services=FakeDriverServices(api)
        with tempfile.TemporaryDirectory() as tmp:
            path,handles=self.packet(tmp,"N2")
            with self.assertRaises(SystemExit) as error:run(path,handles,services=services)
            self.assertEqual(error.exception.code,23);self.assertEqual(services.exits,[23])
            self.assertFalse(path.with_name("result.json").exists())
            self.assertNotIn("DONE",[m["kind"] for m in api.sent])

    def test_worker_lingers_only_after_close_and_power_release(self):
        run=self.function("run_no_data_worker");api=WorkerAPI();services=FakeDriverServices(api)
        with tempfile.TemporaryDirectory() as tmp:
            path,handles=self.packet(tmp,"N3")
            result=run(path,handles,services=services)
            self.assertEqual(result["work_elapsed_ms"],0)
            self.assertGreaterEqual(result["linger_elapsed_ms"],35000)
            self.assertTrue(result["worker_execution"]["end_handshake_verified"])
            self.assertTrue(result["worker_power"]["power_request_released"])

    def test_worker_rejects_old_packet_and_missing_live_capability(self):
        run=self.function("run_no_data_worker")
        with tempfile.TemporaryDirectory() as tmp:
            path,handles=self.packet(tmp,"N1");services=FakeDriverServices(WorkerAPI())
            with self.assertRaises(ValueError):run(path,{},services=services)
            self.assertFalse(path.with_name("task_started.json").exists())
            state=json.loads((path.parent.parent.parent/"stage_state.json").read_text())
            state["status"]="COMPLETE"
            (path.parent.parent.parent/"stage_state.json").write_text(json.dumps(state))
            with self.assertRaises(RuntimeError):run(path,handles,services=services)

    def test_parent_fault_requires_real_ready_identity_and_collector_ack(self):
        fault=self.function("parent_fault_after_ready")
        with tempfile.TemporaryDirectory() as tmp:
            gate,services,case,ready=n4_gate_fixture(tmp)
            with self.assertRaises(SystemExit) as error:fault(case,ready,services=services,gate=gate)
            self.assertEqual(error.exception.code,31);self.assertEqual(services.exits,[31])

    def test_private_worker_cli_runs_the_real_scope_with_inherited_handles(self):
        from contextlib import redirect_stdout,redirect_stderr
        import io
        services=FakeDriverServices(WorkerAPI())
        with tempfile.TemporaryDirectory() as tmp:
            path,handles=self.packet(tmp,"N1")
            with patch.object(native,"NativeServices",return_value=services),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                try:code=native.main(["--worker-packet",str(path),"--worker-handles",json.dumps(handles)])
                except SystemExit as error:self.fail("Worker CLI cannot dispatch live-capability mode: "+str(error))
            self.assertEqual(code,0)
            record=json.loads(path.with_name("worker_closed.json").read_text())
            self.assertTrue(record["worker_execution"]["end_handshake_verified"])

    def test_private_worker_cli_rejects_missing_handles_and_public_flag_mix(self):
        from contextlib import redirect_stderr
        import io
        for flags in (["--worker-packet","never-read.json"],
                      ["--worker-packet","never-read.json","--worker-handles","{}","--execute-native"]):
            with self.subTest(flags=flags),patch.object(native,"NativeServices",side_effect=AssertionError("Unauthorized service")) as factory,redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError,SystemExit)):native.main(flags)
                self.assertEqual(factory.call_count,0)

    def test_parent_fault_rejects_job_handle_retention_wrong_identity_and_timeout(self):
        fault=self.function("parent_fault_after_ready")
        for mode in ("retained_job","identity","timeout"):
            services=FakeDriverServices(WorkerAPI())
            ready=dict(pid=8,created=199,working=True,job_assigned=True,case_id="N4",ready_tick_ms=1000)
            def ack(d,r):
                if mode=="timeout":services.api.now=6000;return None
                return dict(r,collector_verified=True,query_only=True,
                            job_handle_retained=mode=="retained_job",created=200 if mode=="identity" else 199)
            services.read_ack=ack
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp,self.assertRaises(RuntimeError):
                fault(Path(tmp),ready,services=services)
            self.assertEqual(services.exits,[])


def power_record():
    return dict(power_request_acquired=True,power_request_released=True)


def case_record(case_id):
    index=("N1","N2","N3","N4").index(case_id)
    def message(kind,tick,sequence=1,lease=31100):
        return dict(kind=kind,tick_ms=tick,sequence=sequence,lease_until_tick_ms=lease,
                    direction="parent_to_worker" if kind in ("GRANT","RENEW","CLOSE_GRANT") else "worker_to_parent",
                    worker_pid=8,parent_pid=7,nonce_sha256="3"*64)
    events=[dict(kind="spawn",tick_ms=1099,pid=8,created=199),dict(kind="assigned",tick_ms=1100,pid=8),
            message("GRANT",1100),message("ACK",1101)]
    ready=dict(pid=8,created=199,case_id=case_id,working=True,job_assigned=True,ready_tick_ms=1102)
    observation=dict(pid=8,created=199,parent_pid=7,parent_created=99,identity_verified=True,
                     query_only=True,job_handle_retained=False,job_assignment_observed=True,
                     exit_observed=True,exit_tick_ms=46110,parent_exit_observed_tick_ms=46105,
                     exit_code=0,handle_closed=True,fallback_before_exit=False,worker_alive=False,
                     ready=ready,ack_tick_ms=1103,monitor_stopped=True)
    outer=dict(status="COMPLETE",exit_code=0,pid=7,attempts=1,alive=False,supervision=power_record())
    msb=dict(status="COMPLETE",pid=8,exit_code=0,attempts=1,alive=False,
             start_tick_ms=1000,deadline_tick_ms=7201000,final_tick_ms=46104,
             cleanup_confirmed=True,exit_confirmed=True,end_handshake_verified=True,lease_sequence=3,error="")
    closed=dict(worker_power=power_record(),worker_execution=dict(end_handshake_verified=True),work_elapsed_ms=45000)
    fault=None;parent=power_record()
    if case_id=="N1":
        events += [message("RENEW",16100,2,46100),message("ACK",16101,2,46100),
                   message("RENEW",31100,3,61100),message("ACK",31101,3,61100),
                   message("CLOSE_REQUEST",46100,3,61100),message("CLOSE_GRANT",46101,3,61100),
                   message("DONE",46102,3,61100)]
    elif case_id=="N2":
        events.append(dict(kind="pipe_error",tick_ms=1104,error="native pipe EOF",winerror=109))
        msb.update(status="FAILED",exit_code=23,final_tick_ms=1105,end_handshake_verified=False,lease_sequence=1,error="OSError: native pipe EOF")
        observation.update(exit_code=23,exit_tick_ms=1105,parent_exit_observed_tick_ms=1106);closed=None
    elif case_id=="N3":
        events += [message("CLOSE_REQUEST",1102),message("CLOSE_GRANT",1103),message("DONE",1104),
                   dict(kind="terminate_job",tick_ms=31100)]
        msb.update(status="SUPERVISION_INTERRUPTED",exit_code=1,final_tick_ms=31101,
                   lease_sequence=1,error="SUPERVISION_INTERRUPTED: parent lease expired")
        observation.update(exit_code=1,exit_tick_ms=31102,parent_exit_observed_tick_ms=31103)
        closed["work_elapsed_ms"]=0
    elif case_id=="N4":
        outer.update(status="FAILED",exit_code=31)
        observation.update(exit_code=1,exit_tick_ms=1110,parent_exit_observed_tick_ms=1109)
        fault=dict(case_id="N4",pid=7,created=99,exit_code=31,tick_ms=1104,collector_ack_verified=True)
        msb=None;closed=None;parent=None
    ready["created"]+=index;observation["created"]+=index;observation["parent_created"]+=index
    events[0]["created"]+=index
    if fault is not None:fault["created"]+=index
    if case_id=="N4":
        observation["alive_at_ack"]=True
        fault.update(worker_pid=observation["pid"],worker_created=observation["created"],
                     alive_before_fault=True,alive_tick_ms=1103,gate_state="FAULT_COMMITTED",
                     cleanup_before_fault=False,trace_complete=True,
                     binding=case_binding(),pre_fault_trace=copy.deepcopy(events))
    return dict(case_id=case_id,binding=case_binding(),outer=outer,msb=msb,events=events,
                observation=observation,worker_closed=closed,parent_power=parent,fault=fault)


def root_evidence():
    return dict(binding=case_binding(),elapsed_seconds=100.,root_power=power_record(),
                initial_settings={"scheme":"same"},final_settings={"scheme":"same"},
                source_identity_verified=True,registration_verified=True,real_data_opened=False,
                final_continuity=dict(verified=True,sample=[100.,100.],gap_seconds=0.,
                                      suspend_delta_seconds=0.,elapsed_seconds=100.))


class CollectorEnvironment:
    def __init__(self,service):
        self.service=service;self.ac=True;self.requests=[];self.samples=[];self.release_error=False
    def set_execution_state(self,flags):
        self.requests.append(flags)
        if len(self.requests)==2:
            if self.release_error:return 0
            mode=self.service.postflight_fault
            if mode=="released_ac":self.ac=False
            if mode=="released_gap":self.service.api.now+=31000
        return 1
    def sample(self):
        if not self.ac:raise RuntimeError("SUPERVISION_INTERRUPTED: synthetic AC disconnected")
        sample=(self.service.api.now/1000.,self.service.api.now/1000.)
        self.samples.append(sample);return sample


class FakeCollectorServices(FakeDriverServices):
    simulated=True
    def __init__(self,fail_case=None):
        super().__init__(WorkerAPI());self.calls=[];self.fail_case=fail_case
        self.registration_ok=True;self.sources_ok=True;self.write_fault=None;self.writes=0
        self.postflight_fault=None;self.last_case_completed=False;self.environment=CollectorEnvironment(self)
    def new_run_id(self):return "native-test"
    def collector_guard(self):
        from lc_execution_safety import WindowsAwakeGuard
        self.guard=WindowsAwakeGuard(api=self.environment);return self.guard
    def power_settings(self):
        if self.last_case_completed:
            if self.postflight_fault=="settings_ac":self.environment.ac=False
            if self.postflight_fault=="settings_gap":self.api.now+=31000
        return {"scheme":"same"}
    def postflight_sample(self):return self.environment.sample()
    def verify_registration(self,root,path,protocol,approval,check):
        check();self.calls.append(("registration",))
        if not self.registration_ok:raise ValueError("Remote identity mismatch")
        return dict(verified=True,commit=approval["registration_commit"])
    def verify_sources(self,root,protocol):
        if not self.sources_ok:raise ValueError("Source identity changed")
        if self.last_case_completed and self.postflight_fault=="source_ac":self.environment.ac=False
        return dict(verified=True)
    def start_observer(self,case_dir,binding):
        self.calls.append(("observe",case_dir.name));return SimpleNamespace(case_dir=case_dir,binding=binding)
    def stop_observer(self,observer):return True
    def run_outer(self,job,protocol,remaining_seconds):
        case_id=job["case_id"];self.calls.append(("launch",case_id,protocol["per_fit_seconds"],remaining_seconds,copy.deepcopy(job["binding"])))
        record=case_record(case_id);record["binding"]=copy.deepcopy(job["binding"])
        if record["fault"] is not None:record["fault"]["binding"]=copy.deepcopy(job["binding"])
        offset=self.api.now-1000
        def shift(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key in ("tick_ms","ready_tick_ms","exit_tick_ms","parent_exit_observed_tick_ms","ack_tick_ms","alive_tick_ms","final_tick_ms","lease_until_tick_ms"):value[key]+=offset
                    else:shift(item)
            elif isinstance(value,list):
                for item in value:shift(item)
        shift(record)
        folder=Path(job["case_dir"])/"controller";folder.mkdir()
        if record["msb"] is not None:
            (folder/"controller_result.json").write_text(json.dumps(record),encoding="utf-8")
        if record["worker_closed"] is not None:
            work=folder/"runs"/case_id;work.mkdir(parents=True)
            (work/"worker_closed.json").write_text(json.dumps(record["worker_closed"]),encoding="utf-8")
        (folder/"trace.json").write_text(json.dumps(record["events"]),encoding="utf-8")
        if record["fault"] is not None:
            (Path(job["case_dir"])/"parent_fault.json").write_text(json.dumps(record["fault"]),encoding="utf-8")
            (Path(job["case_dir"])/"fault_trace.json").write_text(json.dumps(record["fault"]["pre_fault_trace"]),encoding="utf-8")
        elapsed={"N1":45200,"N2":300,"N3":30300,"N4":300}[case_id]
        while elapsed:
            step=min(10000,elapsed);self.api.now+=step;elapsed-=step;self.guard.check()
        if case_id=="N4":self.last_case_completed=True
        self.last_record=copy.deepcopy(record)
        outer=copy.deepcopy(record["outer"])
        if self.fail_case==case_id:outer["status"]="TIMED_OUT"
        return outer
    def finish_observer(self,observer,outer):
        self.calls.append(("finish",observer.case_dir.name))
        return copy.deepcopy(self.last_record["observation"])
    def write_json(self,path,value,*,exclusive=False):
        self.writes+=1
        if self.write_fault is not None:self.write_fault(path,value)
        native._write_json(path,value,exclusive=exclusive)


class CollectorTests(unittest.TestCase):
    def function(self,name):
        value=getattr(native,name,None);self.assertTrue(callable(value),"Missing collector implementation: "+name);return value

    def protocol_file(self,tmp):
        self.enterContext(patch.object(native,"execution_origins",return_value=observed_origins(tmp)))
        root=Path(tmp);path=root/"docs/experiment_protocols/msb_native_supervision_v1.json"
        path.parent.mkdir(parents=True);p=fixture_protocol();path.write_text(json.dumps(p),encoding="utf-8")
        return root,path,p,fixture_approval(p)

    def test_classifier_accepts_only_complete_fixed_four_case_evidence(self):
        classify=self.function("classify_native_cases")
        records=[case_record(c) for c in ("N1","N2","N3","N4")]
        self.assertTrue(classify(records,outer_evidence=root_evidence())["accepted"])
        for invalid in ([],records[:3],records[::-1],records+[records[0]]):
            with self.subTest(size=len(invalid)):self.assertFalse(classify(invalid,outer_evidence=root_evidence())["accepted"])

    def test_classifier_rejects_fake_renewal_or_nonexact_ack(self):
        classify=self.function("classify_native_cases")
        for mode in ("no_renew","bad_ack","post_done_renew"):
            records=[case_record(c) for c in ("N1","N2","N3","N4")]
            if mode=="no_renew":records[0]["events"]=[e for e in records[0]["events"] if e["kind"]!="RENEW"]
            elif mode=="bad_ack":next(e for e in records[0]["events"] if e["kind"]=="ACK" and e["sequence"]==2)["lease_until_tick_ms"]+=1
            else:records[2]["events"].append(copy.deepcopy(records[0]["events"][4]))
            with self.subTest(mode=mode):self.assertFalse(classify(records,outer_evidence=root_evidence())["accepted"])

    def test_classifier_rejects_job_retention_pid_reuse_fallback_and_late_exit(self):
        classify=self.function("classify_native_cases")
        changes=[("job_handle_retained",True),("query_only",False),("created",200),("identity_verified",False),
                 ("job_assignment_observed",False),("fallback_before_exit",True),("exit_observed",False),
                 ("handle_closed",False),("worker_alive",True),("exit_tick_ms",6109)]
        for key,value in changes:
            records=[case_record(c) for c in ("N1","N2","N3","N4")];records[3]["observation"][key]=value
            with self.subTest(key=key):self.assertFalse(classify(records,outer_evidence=root_evidence())["accepted"])

    def test_classifier_never_uses_outer_timeout_or_false_release_as_native_proof(self):
        classify=self.function("classify_native_cases")
        for mode in ("outer_timeout","no_fault","false_parent_release","no_inner","wrong_worker_exit","changed_budget"):
            r=[case_record(c) for c in ("N1","N2","N3","N4")]
            if mode=="outer_timeout":r[3]["outer"]["status"]="TIMED_OUT"
            elif mode=="no_fault":r[3]["fault"]=None
            elif mode=="false_parent_release":r[3]["parent_power"]=power_record()
            elif mode=="no_inner":r[0]["msb"]=None
            elif mode=="wrong_worker_exit":r[1]["msb"]["exit_code"]=1
            else:r[2]["binding"]["start_tick_ms"]=2000
            with self.subTest(mode=mode):self.assertFalse(classify(r,outer_evidence=root_evidence())["accepted"])

    def test_classifier_rejects_global_expiry_source_power_or_setting_failure(self):
        classify=self.function("classify_native_cases");records=[case_record(c) for c in ("N1","N2","N3","N4")]
        for key,value in (("elapsed_seconds",480.),("source_identity_verified",False),("registration_verified",False),
                          ("real_data_opened",True),("root_power",{}),("final_settings",{"scheme":"changed"})):
            evidence=root_evidence();evidence[key]=value
            with self.subTest(key=key):self.assertFalse(classify(records,outer_evidence=evidence)["accepted"])

    def test_collector_launches_four_once_and_passes_original_binding(self):
        self.function("classify_native_cases")
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);services=FakeCollectorServices()
            result=native.run_acceptance(root,path,execute_native=True,approval=a,services=services)
            self.assertEqual(result["status"],"PASS_NATIVE_SUPERVISION_SCOPE",result.get("error",result.get("assessment")))
            launches=[c for c in services.calls if c[0]=="launch"]
            self.assertEqual([c[1] for c in launches],["N1","N2","N3","N4"])
            self.assertEqual([c[2] for c in launches],[90,45,75,45])
            self.assertEqual({(c[4]["start_tick_ms"],c[4]["deadline_tick_ms"]) for c in launches},{(1000,7201000)})
            self.assertFalse(result["native_os_exercised"]);self.assertFalse(result["real_data_opened"])

    def test_collector_stops_after_first_unexpected_case_outcome(self):
        self.function("classify_native_cases")
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);services=FakeCollectorServices("N2")
            result=native.run_acceptance(root,path,execute_native=True,approval=a,services=services)
            self.assertNotEqual(result["status"],"PASS_NATIVE_SUPERVISION_SCOPE")
            self.assertEqual([c[1] for c in services.calls if c[0]=="launch"],["N1","N2"])

    def test_collector_remote_mismatch_never_launches_or_creates_run_directory(self):
        self.function("classify_native_cases")
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);services=FakeCollectorServices();services.registration_ok=False
            with self.assertRaises(ValueError):native.run_acceptance(root,path,execute_native=True,approval=a,services=services)
            self.assertFalse(any(c[0]=="launch" for c in services.calls))
            self.assertFalse((root/p["output_relative"]).exists())

    def test_collector_true_flag_with_code_permission_reads_nothing(self):
        self.function("classify_native_cases")
        a=dict(call_id="call_6961d4dac6b64d7aba8e538f01c5e097",answer="not-native",
               scope="NATIVE_SUPERVISION_NO_DATA",confirmed_from_human=True,registration_commit="a"*40)
        services=Mock()
        with patch.object(Path,"read_text",side_effect=AssertionError("Unauthorized input")):
            with self.assertRaises(ValueError):native.run_acceptance(Path("."),Path("never-read"),execute_native=True,approval=a,services=services)
        self.assertEqual(services.mock_calls,[])

    def test_final_publication_expiry_and_double_failure_preserve_both_errors(self):
        publish=self.function("publish_acceptance")
        with tempfile.TemporaryDirectory() as tmp:
            services=FakeCollectorServices();state=dict(status="PASS_NATIVE_SUPERVISION_SCOPE",delivery_verified=True)
            first=RuntimeError("TIMED_OUT: post write");second=OSError("failure receipt unavailable")
            def writer(path,value,**kw):
                if value["status"]=="PASS_NATIVE_SUPERVISION_SCOPE":services.api.now=481000
                else:raise second
            services.write_json=writer
            continuity=native.FinalContinuity(case_binding(),(1.,1.),services=services)
            with self.assertRaises(RuntimeError) as error:publish(Path(tmp),state,case_binding(),services=services,final_check=continuity.check)
            self.assertIs(error.exception.persistence_error,second)
            self.assertIn("TIMED_OUT",str(error.exception.primary_error))
            self.assertFalse(error.exception.failed_state["delivery_verified"])

    def test_observer_rejects_pid_reuse_without_retaining_job_capability(self):
        observe=self.function("retain_worker_identity")
        query=SimpleNamespace(opened=[],closed=[])
        query.open_query=lambda pid:(query.opened.append(pid) or 88)
        query.query_identity=lambda handle:(8,200)
        query.close_query=lambda handle:query.closed.append(handle)
        with self.assertRaises(RuntimeError):observe(dict(pid=8,created=199),services=query)
        self.assertEqual(query.opened,[8]);self.assertEqual(query.closed,[88])

    def test_native_query_open_uses_only_query_and_sync_rights(self):
        self.function("retain_worker_identity")
        service=object.__new__(native.NativeServices)
        service.api=SimpleNamespace(kernel=SimpleNamespace(OpenProcess=Mock(return_value=88)),_ok=lambda v:v,
                                    process_identity=lambda h:(8,199))
        self.assertEqual(service.open_query(8),88)
        args=service.api.kernel.OpenProcess.call_args.args
        self.assertEqual(args,(0x100000|0x1000,False,8))

    def test_existing_outer_guard_checks_root_guard_on_same_thread(self):
        self.function("retain_worker_identity")
        import lc_execution_safety as lc
        from run_lc_role_adaptation import run_supervised_job
        class Env:
            def __init__(self):self.n=0
            def set_execution_state(self,flags):return 0x80000000
            def sample(self):self.n+=1;return (1.+self.n*.001,1.+self.n*.001)
        env=Env();service=object.__new__(native.NativeServices)
        process=SimpleNamespace(pid=88,returncode=0,poll=lambda:0,wait=lambda timeout:0)
        with tempfile.TemporaryDirectory() as tmp,patch.object(lc,"WindowsExecutionAPI",return_value=env):
            with lc.WindowsAwakeGuard(api=env) as root:
                before=env.n
                with patch("subprocess.Popen",return_value=process),patch("run_lc_role_adaptation.time.monotonic",return_value=1.):
                    row=service.run_outer(dict(job_id="outer",command=["never-launched"],log_path=str(Path(tmp)/"outer.log")),
                                          dict(per_fit_seconds=90,heartbeat_seconds=10),100.)
                self.assertEqual(row["status"],"COMPLETE");self.assertGreater(env.n,before+3)
                self.assertIs(lc._CURRENT_GUARD.get(),root)
            self.assertTrue(root.evidence()["power_request_released"])

    def test_collector_preparation_expiry_launches_no_case(self):
        self.function("classify_native_cases")
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);services=FakeCollectorServices()
            def register(*args):
                services.api.now=481000
                return dict(verified=True,commit="a"*40)
            services.verify_registration=register
            with self.assertRaises(RuntimeError):native.run_acceptance(root,path,execute_native=True,approval=a,services=services)
            self.assertFalse(any(c[0]=="launch" for c in services.calls))
            self.assertFalse((root/p["output_relative"]).exists())

    def test_private_controller_cli_uses_registered_active_claim_and_real_parent(self):
        self.function("classify_native_cases")
        from contextlib import redirect_stdout,redirect_stderr
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);out=root/p["output_relative"];case=out/"cases/N1";case.mkdir(parents=True)
            binding=case_binding();binding["protocol_digest"]=__import__("hashlib").sha256(
                json.dumps(p,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            claim=dict(status="RUNNING",active_case="N1",binding=binding,
                       registration_verified=True,source_identity_verified=True,registered_commit="a"*40)
            (out/"stage_state.json").write_text(json.dumps(claim),encoding="utf-8")
            services=FakeCollectorServices();services.api=CaseLauncher("N1")
            flags=["--root",str(root),"--protocol",str(path),"--execute-native",
                   "--approval-call-id",a["call_id"],"--approval-answer",a["answer"],"--registration-commit","a"*40,
                   "--controller-case","N1","--case-output",str(case),"--binding-json",json.dumps(binding)]
            with patch.object(native,"NativeServices",return_value=services),redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                try:code=native.main(flags)
                except SystemExit as error:self.fail("Controller CLI not wired: "+str(error))
            self.assertEqual(code,0)
            self.assertEqual(json.loads((case/"controller/controller_result.json").read_text())["msb"]["status"],"COMPLETE")

    def test_active_controller_claim_mismatch_denies_before_native_factory(self):
        self.function("classify_native_cases")
        from contextlib import redirect_stderr
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);out=root/p["output_relative"];case=out/"cases/N1";case.mkdir(parents=True)
            (out/"stage_state.json").write_text(json.dumps(dict(status="COMPLETE",active_case="N1")))
            flags=["--root",str(root),"--protocol",str(path),"--execute-native",
                   "--approval-call-id",a["call_id"],"--approval-answer",a["answer"],"--registration-commit","a"*40,
                   "--controller-case","N1","--case-output",str(case),"--binding-json",json.dumps(case_binding())]
            with patch.object(native,"NativeServices",side_effect=AssertionError("Unapproved native")) as factory,redirect_stderr(io.StringIO()):
                with self.assertRaises((ValueError,RuntimeError,SystemExit)):native.main(flags)
            self.assertEqual(factory.call_count,0)

    def test_registered_git_tree_and_protocol_bytes_bind_all_sources(self):
        self.function("retain_worker_identity")
        import base64
        import hashlib
        service=object.__new__(native.NativeServices)
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp)
            tree=[]
            for name in p["source_sha256"]:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);payload=b"synthetic-source\n";target.write_bytes(payload)
                p["source_sha256"][name]=hashlib.sha256(payload).hexdigest()
                tree.append(dict(path=name,type="blob",sha=hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest()))
            path.write_text(json.dumps(p),encoding="utf-8");payload=path.read_bytes()
            tree.append(dict(path="docs/experiment_protocols/msb_native_supervision_v1.json",type="blob",
                             sha=hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest()))
            def api(endpoint,check):
                check()
                if "/git/ref/" in endpoint:return dict(object=dict(sha="a"*40))
                if "/git/commits/" in endpoint:return dict(sha="a"*40,tree=dict(sha="b"*40))
                if "/git/trees/" in endpoint:return dict(truncated=False,tree=tree)
                if "/contents/" in endpoint:return dict(content=base64.b64encode(payload).decode(),encoding="base64")
                raise AssertionError("Unexpected registration endpoint")
            service._api=api
            result=service.verify_registration(root,path,p,a,lambda:None)
            self.assertTrue(result["verified"])
            tree[0]["sha"]="0"*40
            with self.assertRaises(ValueError):service.verify_registration(root,path,p,a,lambda:None)


class ObserverTestServices(FakeDriverServices):
    def __init__(self):
        super().__init__(WorkerAPI());self.api.now=1103;self.opened=[];self.closed=[];self.identity=(8,202)
    def open_query(self,pid):self.opened.append(pid);return 88
    def query_identity(self,handle):return self.identity
    def close_query(self,handle):self.closed.append(handle)
    def query_alive(self,handle):return self.api.now<1200
    def query_exit_code(self,handle):return 1


class ObserverTests(unittest.TestCase):
    def make(self,tmp):
        record=case_record("N4");case=Path(tmp)/"N4";(case/"controller").mkdir(parents=True)
        (case/"worker_ready.json").write_text(json.dumps(record["observation"]["ready"]),encoding="utf-8")
        (case/"controller_ready.json").write_text(json.dumps(dict(pid=7,created=102,binding=case_binding())),encoding="utf-8")
        (case/"controller/trace.json").write_text(json.dumps(record["events"]),encoding="utf-8")
        services=ObserverTestServices()
        return native.NativeObserver(case,case_binding(),services),services,case

    def test_real_observer_retains_query_identity_then_confirms_exit_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer,services,case=self.make(tmp);observer.sample()
            self.assertEqual(services.opened,[8])
            ack=json.loads((case/"collector_ack.json").read_text())
            self.assertTrue(ack["query_only"]);self.assertFalse(ack["job_handle_retained"])
            record=observer.finish(dict(pid=7,exit_code=31,alive=False))
            self.assertTrue(record["exit_observed"]);self.assertTrue(record["handle_closed"])
            self.assertTrue(record["monitor_stopped"]);self.assertFalse(record["worker_alive"])
            self.assertEqual(services.closed,[88])

    def test_real_observer_rejects_creation_reuse_before_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer,services,case=self.make(tmp);services.identity=(8,203)
            with self.assertRaises(RuntimeError):observer.sample()
            observer.close()
            self.assertEqual(services.closed,[88]);self.assertFalse((case/"collector_ack.json").exists())

    def test_partial_ready_write_is_not_misclassified_as_native_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer,services,case=self.make(tmp);ready=case/"worker_ready.json";payload=ready.read_text()
            ready.write_text('{"pid":',encoding="utf-8")
            try:observer.sample()
            except json.JSONDecodeError:self.fail("Transient partial metadata must defer observation without ACK")
            self.assertEqual(services.opened,[]);self.assertFalse((case/"collector_ack.json").exists())
            ready.write_text(payload,encoding="utf-8");observer.sample();observer.finish(dict(pid=7,exit_code=31,alive=False))
            self.assertEqual(services.closed,[88])

    def test_pipe_eof_classification_uses_numeric_cause_not_localized_text(self):
        for winerror,want in ((109,True),(6,False)):
            records=[case_record(c) for c in ("N1","N2","N3","N4")]
            records[1]["msb"]["error"]="OSError: [WinError 109] \u7ba1\u9053\u5df2\u7ed3\u675f"
            next(e for e in records[1]["events"] if e["kind"]=="pipe_error")["winerror"]=winerror
            with self.subTest(winerror=winerror):
                self.assertEqual(native.classify_native_cases(records,outer_evidence=root_evidence())["accepted"],want)

    def test_ack_read_time_counts_toward_the_five_second_cap(self):
        services=FakeDriverServices(WorkerAPI())
        ready=dict(pid=8,created=199,working=True,job_assigned=True,case_id="N4",ready_tick_ms=1000)
        def delayed(case_dir,value):
            services.api.now=6000
            return dict(value,collector_verified=True,query_only=True,job_handle_retained=False)
        services.read_ack=delayed;caught=None
        with tempfile.TemporaryDirectory() as tmp:
            try:native.parent_fault_after_ready(Path(tmp),ready,services=services)
            except BaseException as error:caught=error
        self.assertIsInstance(caught,RuntimeError)
        self.assertEqual(services.exits,[])

    def test_close_rejects_a_stuck_observer_lock_without_unsafe_handle_close(self):
        class Busy:
            def acquire(self,**kwargs):return False
            def release(self):raise AssertionError("unowned lock released")
        with tempfile.TemporaryDirectory() as tmp:
            observer,services,case=self.make(tmp);observer.sample();original=observer.lock;observer.lock=Busy();caught=None
            try:observer.close()
            except BaseException as error:caught=error
            closed_before_cleanup=list(services.closed)
            observer.lock=original;observer.close()
            self.assertIsInstance(caught,RuntimeError)
            self.assertEqual(closed_before_cleanup,[])


    def test_finish_never_samples_while_an_observer_thread_cannot_stop(self):
        class Busy:
            def __init__(self):self.enters=0
            def __enter__(self):self.enters+=1;raise AssertionError("unbounded sampler lock attempted")
            def __exit__(self,*args):return False
            def acquire(self,**kwargs):return False
            def release(self):raise AssertionError("unowned release")
        with tempfile.TemporaryDirectory() as tmp:
            observer,services,case=self.make(tmp);observer.sample()
            original=observer.lock;busy=Busy();observer.lock=busy;observer.watch=SimpleNamespace(stopped=False)
            services.stop_fault=lambda watch:False;caught=None
            try:observer.finish(dict(pid=7,exit_code=31,alive=False))
            except BaseException as error:caught=error
            closed_before_cleanup=list(services.closed)
            observer.lock=original;observer.watch=None;observer.close()
            self.assertIsInstance(caught,RuntimeError)
            self.assertEqual(busy.enters,0)
            self.assertEqual(closed_before_cleanup,[])


def observed_origins(root):
    root=Path(root).resolve()
    return {"driver":str(root/"scripts/check_msb_native_supervision.py"),
            "module:msb_execution_safety:file":str(root/"scripts/msb_execution_safety.py"),
            "module:msb_execution_safety:spec":str(root/"scripts/msb_execution_safety.py")}


class OriginRepairTests(unittest.TestCase):
    def function(self,name):
        value=getattr(native,name,None)
        self.assertTrue(callable(value),"Missing origin repair: "+name)
        return value

    def test_origin_rejects_other_checkout_and_import_alias(self):
        validate=self.function("validate_execution_origins")
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"A",Path(tmp)/"B"
            with self.assertRaises(ValueError):validate(b,observed_origins(a))
            for key in ("module:msb_execution_safety:file","module:msb_execution_safety:spec"):
                origins=observed_origins(a);origins[key]=str(b/"scripts/msb_execution_safety.py")
                with self.subTest(key=key),self.assertRaises(ValueError):validate(a,origins)
            origins=observed_origins(a);del origins["module:msb_execution_safety:file"]
            with self.assertRaises(ValueError):validate(a,origins)

    def test_origin_accepts_same_resolved_root_only(self):
        validate=self.function("validate_execution_origins")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);origins=observed_origins(root)
            result=validate(root/"scripts/..",origins)
            self.assertEqual(Path(result["root"]),root.resolve())
            self.assertEqual(result["driver"],origins["driver"])
            self.assertEqual(result["module_origins"],{k:v for k,v in origins.items() if k!="driver"})

    def test_execution_origins_reports_actual_driver_and_modules(self):
        capture=self.function("execution_origins");origins=capture()
        self.assertEqual(Path(origins["driver"]),Path(native.__file__).resolve())
        self.assertEqual(Path(origins["module:msb_execution_safety:file"]),Path(safety.__file__).resolve())
        self.assertEqual(Path(origins["module:msb_execution_safety:spec"]),Path(safety.__spec__.origin).resolve())

    def test_zero_wait_does_not_turn_tick_overhead_into_work(self):
        services=FakeDriverServices(WorkerAPI());calls=[]
        def tick():services.api.now+=1;return services.api.now
        services.tick_ms=tick;services.pause=lambda seconds:calls.append(seconds)
        self.assertEqual(native._wait_work(0,case_binding(),services),0)
        self.assertEqual(calls,[])

    def test_wait_work_rejects_negative_boolean_or_noninteger(self):
        services=FakeDriverServices(WorkerAPI())
        for value in (-1,True,0.,None):
            with self.subTest(value=value):
                caught=None
                try:native._wait_work(value,case_binding(),services)
                except Exception as error:caught=error
                self.assertIsInstance(caught,ValueError)

    def test_three_entrypoints_reject_wrong_origins_before_native(self):
        from contextlib import redirect_stderr
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=fixture_protocol();approval=fixture_approval(protocol)
            path=root/"docs/experiment_protocols/msb_native_supervision_v1.json"
            flags=["--root",str(root),"--protocol",str(path),"--execute-native",
                   "--approval-call-id",approval["call_id"],"--approval-answer",approval["answer"],
                   "--registration-commit","a"*40,"--controller-case","N1",
                   "--case-output",str(root/protocol["output_relative"]/"cases/N1"),
                   "--binding-json",json.dumps(case_binding())]
            with patch.object(native,"NativeServices",side_effect=AssertionError("Wrong-root native factory")) as factory:
                with patch.object(Path,"read_text",side_effect=AssertionError("Wrong-root protocol read")):
                    with self.subTest(entry="collector"),self.assertRaises(ValueError):
                        native.run_acceptance(root,path,execute_native=True,approval=approval,services=FakeCollectorServices())
                    with self.subTest(entry="controller"),redirect_stderr(io.StringIO()),self.assertRaises(ValueError):native.main(flags)
                packet=root/"packet.json";packet.write_text(json.dumps(dict(root=str(root))),encoding="utf-8")
                handles=dict(read_handle=10,write_handle=11,parent_handle=12,nonce="a"*64)
                with self.subTest(entry="worker"),self.assertRaises(ValueError):
                    native.main(["--worker-packet",str(packet),"--worker-handles",json.dumps(handles)])
                self.assertEqual(factory.call_count,0)

    def test_current_document_and_development_permissions_are_not_native(self):
        for token in ("call_03a86c6e5b064586b7bb980fa04b2a53","call_d538a55e3a7b401fa4709a0d3914bb24",
                      "call_7f515c71f122488da5242db3dc69dd60"):
            protocol=fixture_protocol();protocol["approval_call_id"]=token
            with self.subTest(token=token),self.assertRaises(ValueError):
                native._validate_protocol_fields(protocol,approval=fixture_approval(protocol))


def n4_gate_fixture(tmp):
    record=case_record("N4");case=Path(tmp)/"N4";(case/"controller").mkdir(parents=True)
    services=FakeDriverServices(WorkerAPI());services.api.now=1103
    services.api.current_identity=lambda:(7,102)
    services.open_query=Mock(return_value=88);services.query_identity=Mock(return_value=(8,202))
    services.query_alive=Mock(return_value=True);services.close_query=Mock()
    gate=native.N4FaultGate(case,case_binding(),services=services,events=record["events"])
    return gate,services,case,record["observation"]["ready"]


class N4RepairTests(unittest.TestCase):
    def gate(self,tmp):
        self.assertTrue(callable(getattr(native,"N4FaultGate",None)),"Missing N4FaultGate")
        return n4_gate_fixture(tmp)

    def test_n4_dead_at_ack_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer,services,case=ObserverTests().make(tmp);services.api.now=1300
            caught=None
            try:observer.sample()
            except Exception as error:caught=error
            finally:observer.close()
            self.assertIsInstance(caught,RuntimeError)
            self.assertFalse((case/"collector_ack.json").exists())

    def test_n4_rejects_pre_fault_exit_and_cleanup(self):
        for mode in ("early_exit","terminate_job","kill_process","cleanup_intent","declared_cleanup"):
            records=[case_record(c) for c in ("N1","N2","N3","N4")];r=records[-1]
            if mode=="early_exit":r["observation"]["exit_tick_ms"]=r["fault"]["tick_ms"]-1
            elif mode=="declared_cleanup":r["fault"]["cleanup_before_fault"]=True
            else:
                r["events"].append(dict(kind=mode,tick_ms=1103,pid=8))
                r["fault"]["pre_fault_trace"]=copy.deepcopy(r["events"])
            with self.subTest(mode=mode):
                self.assertFalse(native.classify_native_cases(records,outer_evidence=root_evidence())["accepted"])


    def test_n4_allows_child_observed_before_parent(self):
        records=[case_record(c) for c in ("N1","N2","N3","N4")]
        records[-1]["observation"].update(exit_tick_ms=1110,parent_exit_observed_tick_ms=1120)
        self.assertTrue(native.classify_native_cases(records,outer_evidence=root_evidence())["accepted"])

    def test_n4_fault_exit_same_tick_and_five_second_equality_rejected(self):
        for delta in (0,5000):
            records=[case_record(c) for c in ("N1","N2","N3","N4")]
            r=records[-1];r["observation"]["exit_tick_ms"]=r["fault"]["tick_ms"]+delta
            with self.subTest(delta=delta):
                self.assertFalse(native.classify_native_cases(records,outer_evidence=root_evidence())["accepted"])

    def test_n4_fault_gate_commits_live_identity_snapshot_before_exit31(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate,services,case,ready=self.gate(tmp)
            with self.assertRaises(SystemExit) as caught:
                native.parent_fault_after_ready(case,ready,services=services,gate=gate)
            self.assertEqual(caught.exception.code,31)
            fault=json.loads((case/"parent_fault.json").read_text())
            self.assertTrue(fault["alive_before_fault"])
            self.assertTrue(fault["trace_complete"])
            self.assertEqual(fault["pre_fault_trace"],gate.events)
            self.assertFalse(fault["cleanup_before_fault"])
            self.assertEqual(fault["binding"],case_binding())

    def test_n4_cleanup_fault_race_never_qualifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate,services,case,ready=self.gate(tmp)
            calls=[];services.api.terminate_job=lambda handle:calls.append("terminate")
            trace=native.TraceAPI(services.api,gate.events,case/"controller/trace.json",services,fault_gate=gate)
            trace.terminate_job(9)
            self.assertEqual(calls,["terminate"])
            self.assertTrue((case/"cleanup_intent.json").exists())
            with self.assertRaises(RuntimeError):native.parent_fault_after_ready(case,ready,services=services,gate=gate)
            self.assertEqual(services.exits,[])

    def test_n4_trace_write_failure_never_suppresses_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate,services,case,ready=self.gate(tmp)
            calls=[];services.api.kill_process=lambda process:calls.append(process.pid)
            trace=native.TraceAPI(services.api,gate.events,case/"controller/trace.json",services,fault_gate=gate)
            with patch.object(native,"_write_json",side_effect=OSError("journal failed")):
                trace.kill_process(SimpleNamespace(pid=8))
            self.assertEqual(calls,[8])
            with self.assertRaises(RuntimeError):native.parent_fault_after_ready(case,ready,services=services,gate=gate)
            self.assertEqual(services.exits,[])

    def test_n4_busy_fault_gate_is_bounded_and_fails_closed(self):
        class Busy:
            def __init__(self):self.timeouts=[]
            def acquire(self,*,timeout):self.timeouts.append(timeout);return False
            def release(self):raise AssertionError("Unowned gate release")
        with tempfile.TemporaryDirectory() as tmp:
            gate,services,case,ready=self.gate(tmp);busy=Busy();gate.lock=busy
            with self.assertRaises(RuntimeError):native.parent_fault_after_ready(case,ready,services=services,gate=gate)
            self.assertTrue(busy.timeouts);self.assertTrue(all(0<t<=1 for t in busy.timeouts))
            self.assertEqual(services.exits,[])

    def test_n4_parent_live_query_and_trace_failure_never_exit31(self):
        for mode in ("dead","query_error","journal_error"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                gate,services,case,ready=self.gate(tmp)
                if mode=="dead":services.query_alive=Mock(return_value=False)
                if mode=="query_error":services.query_alive=Mock(side_effect=OSError("query failed"))
                manager=patch.object(native,"_write_json",side_effect=OSError("fault journal failed")) if mode=="journal_error" else __import__("contextlib").nullcontext()
                caught=None
                with manager:
                    try:native.parent_fault_after_ready(case,ready,services=services,gate=gate)
                    except Exception as error:caught=error
                self.assertIsNotNone(caught);self.assertEqual(services.exits,[])

    def test_controller_rechecks_loaded_module_origins_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);good=observed_origins(root);bad=observed_origins(root/"other")
            api=CaseLauncher("N1");services=FakeDriverServices(api)
            with patch.object(native,"execution_origins",side_effect=[good,bad]),self.assertRaises(ValueError):
                native.run_controller(root,root/"outputs/msb_native_supervision_v1/acceptance_once/cases/N1","N1",case_binding(),services=services)
            self.assertEqual(api.events,[])

    def test_n4_cleanup_during_fault_snapshot_cannot_commit(self):
        class GateLock:
            def __init__(self):self.held=False
            def acquire(self,*,timeout):
                if self.held:return False
                self.held=True;return True
            def release(self):self.held=False
        with tempfile.TemporaryDirectory() as tmp:
            gate,services,case,ready=self.gate(tmp);gate.lock=GateLock();calls=[]
            services.api.terminate_job=lambda handle:calls.append("terminate")
            trace=native.TraceAPI(services.api,gate.events,case/"controller/trace.json",services,fault_gate=gate)
            original=native._write_json
            def write(path,value,**kw):
                if Path(path).name=="fault_trace.json":trace.terminate_job(9)
                original(path,value,**kw)
            with patch.object(native,"_write_json",side_effect=write),self.assertRaises(RuntimeError):
                native.parent_fault_after_ready(case,ready,services=services,gate=gate)
            self.assertEqual(calls,["terminate"]);self.assertEqual(services.exits,[])

    def test_n4_missing_live_or_snapshot_evidence_is_rejected(self):
        for field in ("alive_before_fault","trace_complete","pre_fault_trace"):
            records=[case_record(c) for c in ("N1","N2","N3","N4")]
            del records[-1]["fault"][field]
            with self.subTest(field=field):
                self.assertFalse(native.classify_native_cases(records,outer_evidence=root_evidence())["accepted"])

    def test_n4_query_close_failure_still_releases_fault_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate,services,case,ready=self.gate(tmp)
            services.query_alive=Mock(return_value=False)
            services.close_query=Mock(side_effect=OSError("query close failed"))
            caught=None
            try:native.parent_fault_after_ready(case,ready,services=services,gate=gate)
            except Exception as error:caught=error
            self.assertIsInstance(caught,OSError);self.assertEqual(services.exits,[])
            acquired=gate.lock.acquire(blocking=False)
            if acquired:gate.lock.release()
            self.assertTrue(acquired,"Query-close failure must not retain the fault gate")


class FinalRepairTests(unittest.TestCase):
    protocol_file=CollectorTests.protocol_file

    def continuity(self,services,last=(1.,1.)):
        cls=getattr(native,"FinalContinuity",None)
        self.assertTrue(callable(cls),"Missing FinalContinuity")
        return cls(case_binding(),last,services=services)

    def test_postflight_ac_and_gap_failure_never_publish_pass(self):
        for mode in ("source_ac","settings_ac","settings_gap","released_ac","released_gap","after_write_ac","after_write_gap"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices();service.postflight_fault=mode
                if mode.startswith("after_write"):
                    def write_fault(path,value):
                        if value["status"]=="PASS_NATIVE_SUPERVISION_SCOPE":
                            if mode=="after_write_ac":service.environment.ac=False
                            else:service.api.now+=31000
                    service.write_fault=write_fault
                result=native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
                self.assertNotEqual(result["status"],"PASS_NATIVE_SUPERVISION_SCOPE")
                self.assertFalse(result["delivery_verified"])
                self.assertEqual(len([x for x in service.calls if x[0]=="launch"]),4)

    def test_classifier_requires_final_continuity(self):
        records=[case_record(c) for c in ("N1","N2","N3","N4")]
        for value in (None,dict(verified=True),dict(verified=False,sample=[1.,1.],gap_seconds=0.,suspend_delta_seconds=0.,elapsed_seconds=1.)):
            evidence=root_evidence()
            if value is None:del evidence["final_continuity"]
            else:evidence["final_continuity"]=value
            with self.subTest(value=value):
                self.assertFalse(native.classify_native_cases(records,outer_evidence=evidence)["accepted"])

    def test_guard_release_failure_finalizes_and_cli_nonzero(self):
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices();service.environment.release_error=True
            flags=["--root",str(root),"--protocol",str(path),"--execute-native",
                   "--approval-call-id",a["call_id"],"--approval-answer",a["answer"],"--registration-commit","a"*40]
            caught=None;code=None
            with patch.object(native,"NativeServices",return_value=service),patch("time.monotonic",side_effect=service.monotonic) as clock,redirect_stdout(io.StringIO()):
                try:code=native.main(flags)
                except Exception as error:caught=error
            self.assertIsNone(caught,str(caught));self.assertEqual(code,1)
            self.assertEqual(clock.call_count,1)
            state=json.loads((root/p["output_relative"]/"stage_state.json").read_text())
            self.assertEqual(state["status"],"FAILED_NATIVE_SUPERVISION")
            self.assertFalse(state["delivery_verified"])
            self.assertFalse(state["root_power"]["power_request_released"])

    def test_body_release_and_write_failure_keep_all_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices(fail_case="N1")
            service.environment.release_error=True
            def fail(path,value):
                if value["status"]=="FAILED_NATIVE_SUPERVISION":raise OSError("failure receipt unavailable")
            service.write_fault=fail;caught=None
            try:native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
            except Exception as error:caught=error
            self.assertIsInstance(caught,native.AcceptancePublicationError)
            errors=getattr(caught,"primary_errors",[]);self.assertEqual(len(errors),2)
            self.assertIs(caught.primary_error,errors[0]);self.assertIsInstance(caught.persistence_error,OSError)
            self.assertFalse(caught.failed_state["delivery_verified"])
            self.assertIn("POWER_RELEASE_FAILED",str(errors[1]))

    def test_final_check_is_required_for_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=dict(status="PASS_NATIVE_SUPERVISION_SCOPE",delivery_verified=True)
            result=native.publish_acceptance(Path(tmp),state,case_binding(),services=FakeCollectorServices())
            self.assertNotEqual(result["status"],"PASS_NATIVE_SUPERVISION_SCOPE")
            self.assertFalse(result["delivery_verified"])

    def test_final_check_runs_before_and_after_pass_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=FakeCollectorServices();continuity=self.continuity(service);events=[]
            service.write_fault=lambda path,value:events.append("write")
            def check():events.append("check");return continuity.check()
            caught=None;result=None
            try:result=native.publish_acceptance(Path(tmp),dict(status="PASS_NATIVE_SUPERVISION_SCOPE",delivery_verified=True),case_binding(),services=service,final_check=check)
            except Exception as error:caught=error
            self.assertIsNone(caught,str(caught));self.assertEqual(result["status"],"PASS_NATIVE_SUPERVISION_SCOPE")
            self.assertEqual(events,["check","write","check"])
            self.assertEqual(len(result["postflight_checks"]),2)

    def test_final_continuity_bounds_and_invalid_samples(self):
        for sample,valid in (((31.,29.),True),((31.001,31.001),False),((3.001,1.),False),
                             ((.9,1.),False),((1.,math.nan),False),((1.,math.inf),False)):
            service=FakeCollectorServices();continuity=self.continuity(service)
            service.postflight_sample=Mock(return_value=sample);caught=None;result=None
            try:result=continuity.check()
            except Exception as error:caught=error
            with self.subTest(sample=sample):
                if valid:self.assertIsNone(caught,str(caught));self.assertTrue(result["verified"])
                else:self.assertIsNotNone(caught)
        service=FakeCollectorServices();continuity=self.continuity(service);service.api.now=481000
        with self.assertRaises(RuntimeError):continuity.check()

    def test_guard_is_not_reentered_for_postflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices()
            result=native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
            self.assertEqual(result["status"],"PASS_NATIVE_SUPERVISION_SCOPE",result.get("error"))
            self.assertEqual(len(service.environment.requests),2)
            self.assertTrue(result.get("final_continuity",{}).get("verified"))
            self.assertEqual(len(result.get("postflight_checks",[])),2)
