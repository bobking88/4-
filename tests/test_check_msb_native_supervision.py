"""Contracts use scratch metadata and simulated interfaces only."""
import copy
from contextlib import contextmanager, redirect_stdout
import io
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
    return dict(protocol="msb_native_supervision_v2", status="APPROVED_NATIVE_ONLY",
                scope="NATIVE_SUPERVISION_NO_DATA", execution_contract="msb_parent_bound_v2",
                python_launch_contract="DIRECT_CPYTHON311_VENV_V1", runtime_venv_root=r"C:\synthetic\venv",
                approval_call_id="call_"+"f"*32, approval_answer="APPROVED_SINGLE_NATIVE_NO_DATA",
                output_relative="outputs/msb_native_supervision_v2/acceptance_once",
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
            self.assertEqual(result["output_relative"],"outputs/msb_native_supervision_v2/acceptance_once")
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
                    code=harness.main(["--phase","collector-green"])
                summary=json.loads((Path(tmp)/"outputs/msb_trace_publication_retention_synthetic_v1/collector-green/attempt-001/acceptance_summary.json").read_text())
                self.assertEqual((code,summary["status"],summary["skipped"]),(1,"FAILED",1))
            finally:
                os.chdir(previous[0]);tempfile.tempdir=previous[1];sys.path[:]=previous[2]

import msb_execution_safety as safety
from test_msb_execution_safety import FakeAPI, FakeAwake, LauncherAPI, synthetic_python_runtime


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
    capture_runtime=native.NativeServices.capture_runtime
    def __init__(self,api):
        self.api=api;self.exits=[];self.callbacks=[];self.acks=[];self.traces=[]
    def tick_ms(self):return self.api.tick_ms()
    def monotonic(self):return self.api.tick_ms()/1000.
    def pause(self,seconds):self.api.pause(seconds)
    def awake_guard(self):return FakeAwake()
    def exit_process(self,code):self.exits.append(code);raise SystemExit(code)
    def start_fault(self,callback):self.callbacks.append(callback);return SimpleNamespace(stopped=False)
    def stop_fault(self,watch,*,remaining_seconds):
        watch.stopped=True
        if getattr(watch,"stop_result",None) is None:
            watch.stop_result=native.FaultStopResult(True,True,False,remaining_seconds<=0,None)
        return watch.stop_result
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
        return Path(root)/"outputs/msb_native_supervision_v2/acceptance_once/cases"/case_id

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
        root=Path(tmp);path=root/"docs/experiment_protocols/msb_native_supervision_v2.json"
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
            tree.append(dict(path="docs/experiment_protocols/msb_native_supervision_v2.json",type="blob",
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
            services.stop_fault=lambda watch,**kwargs:native.FaultStopResult(False,True,True,False,None);caught=None
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
            path=root/"docs/experiment_protocols/msb_native_supervision_v2.json"
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
                      "call_7f515c71f122488da5242db3dc69dd60",
                      "call_29684c57aed94222832ace218cd392be","call_754f23a7456b4c5cbe8e744ba0dd6114"):
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
                native.run_controller(root,root/"outputs/msb_native_supervision_v2/acceptance_once/cases/N1","N1",case_binding(),services=services)
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


class InitializationErrorRetentionTests(unittest.TestCase):
    protocol_file=CollectorTests.protocol_file

    @contextmanager
    def _initial_fault(self,service,*,mode=None,write_name=None,write_error=None,persistence_error=None):
        from lc_execution_safety import WindowsAwakeGuard
        observed=dict(running=None,check_errors=[],release_errors=[],failure_writes=0,write_failed=False)
        original_factory=service.collector_guard
        original_remaining=native._remaining
        original_exit=WindowsAwakeGuard.__exit__
        original_write=service.write_json
        def remaining(*args):
            try:return original_remaining(*args)
            except BaseException as error:
                if observed["running"] is not None:observed["check_errors"].append(error)
                raise
        def factory():
            guard=original_factory();original_check=guard.check
            def check():
                try:return original_check()
                except BaseException as error:
                    if observed["running"] is not None:observed["check_errors"].append(error)
                    raise
            guard.check=check
            return guard
        def release(guard,*args):
            try:return original_exit(guard,*args)
            except BaseException as error:
                observed["release_errors"].append(error)
                raise
        def write(path,value,*,exclusive=False):
            path=Path(path)
            if value.get("status")=="FAILED_NATIVE_SUPERVISION":
                observed["failure_writes"]+=1
                if persistence_error is not None:raise persistence_error
            if path.name==write_name and not observed["write_failed"]:
                observed["write_failed"]=True
                raise write_error
            original_write(path,value,exclusive=exclusive)
            if (path.name=="stage_state.json" and value.get("status")=="RUNNING"
                    and value.get("active_case") is None and observed["running"] is None):
                observed["running"]=json.loads(path.read_text())
                if mode=="ac":service.environment.ac=False
                elif mode=="gap":service.api.now+=31000
                elif mode=="deadline":service.api.now=value["binding"]["start_tick_ms"]+480000
        with patch.object(service,"collector_guard",side_effect=factory),patch.object(native,"_remaining",side_effect=remaining),\
                patch.object(WindowsAwakeGuard,"__exit__",release),patch.object(service,"write_json",side_effect=write):
            yield observed

    def test_initial_running_check_failure_and_release_preserve_both_causes(self):
        for mode in ("ac","gap","deadline"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices();service.environment.release_error=True
                with self._initial_fault(service,mode=mode) as seen:
                    result=native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
                self.assertEqual(seen["running"]["status"],"RUNNING")
                self.assertIsNone(seen["running"]["active_case"])
                self.assertEqual((len(seen["check_errors"]),len(seen["release_errors"])),(1,1))
                self.assertEqual([e["phase"] for e in result["errors"]],["body","collector_scope_or_release"])
                for row,error in zip(result["errors"],seen["check_errors"]+seen["release_errors"]):
                    self.assertEqual((row["type"],row["message"]),(type(error).__name__,str(error)))
                self.assertEqual(result["status"],"FAILED_NATIVE_SUPERVISION");self.assertFalse(result["delivery_verified"])
                self.assertEqual(result["case_records"],[])
                self.assertFalse(any(c[0] in ("launch","observe") for c in service.calls))
                self.assertFalse(result["root_power"]["power_request_released"])
                self.assertEqual(seen["failure_writes"],1)

    def test_initial_running_triple_failure_preserves_exception_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices();service.environment.release_error=True
            persistence=OSError("synthetic failure receipt unavailable");caught=None
            with self._initial_fault(service,mode="ac",persistence_error=persistence) as seen:
                try:native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
                except BaseException as error:caught=error
            self.assertIsInstance(caught,native.AcceptancePublicationError)
            self.assertEqual(len(caught.primary_errors),2)
            self.assertIs(caught.primary_errors[0],seen["check_errors"][0])
            self.assertIs(caught.primary_errors[1],seen["release_errors"][0])
            self.assertIs(caught.primary_error,caught.primary_errors[0]);self.assertIs(caught.persistence_error,persistence)
            self.assertEqual(seen["failure_writes"],1)
            self.assertEqual(caught.failed_state["status"],"FAILED_NATIVE_SUPERVISION")
            self.assertFalse(caught.failed_state["delivery_verified"])
            self.assertFalse(caught.failed_state["failure_record_persisted"])
            self.assertEqual(json.loads((root/p["output_relative"]/"stage_state.json").read_text())["status"],"RUNNING")

    def test_initial_write_failure_and_release_preserve_both_causes(self):
        for name in ("registered_protocol.json","stage_state.json"):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices();service.environment.release_error=True
                original=OSError("synthetic initialization write failed: "+name)
                with self._initial_fault(service,write_name=name,write_error=original) as seen:
                    result=native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
                self.assertTrue(seen["write_failed"]);self.assertIsNone(seen["running"])
                self.assertEqual([e["phase"] for e in result["errors"]],["body","collector_scope_or_release"])
                self.assertEqual((result["errors"][0]["type"],result["errors"][0]["message"]),("OSError",str(original)))
                self.assertEqual(result["errors"][1]["message"],str(seen["release_errors"][0]))
                self.assertEqual(result["case_records"],[]);self.assertFalse(result["delivery_verified"])
                self.assertFalse(any(c[0] in ("launch","observe") for c in service.calls))
                self.assertEqual(seen["failure_writes"],1)
                self.assertEqual(json.loads((root/p["output_relative"]/"stage_state.json").read_text())["status"],"FAILED_NATIVE_SUPERVISION")

    def test_initial_running_failure_cli_is_nonzero_and_clock_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices();service.environment.release_error=True
            flags=["--root",str(root),"--protocol",str(path),"--execute-native",
                   "--approval-call-id",a["call_id"],"--approval-answer",a["answer"],"--registration-commit","a"*40]
            stdout=io.StringIO()
            with self._initial_fault(service,mode="ac") as seen,patch.object(native,"NativeServices",return_value=service),\
                    patch("time.monotonic",side_effect=service.monotonic) as clock,redirect_stdout(stdout):
                code=native.main(flags)
            self.assertEqual(code,1);self.assertEqual(clock.call_count,1)
            result=json.loads(stdout.getvalue())
            self.assertEqual([e["phase"] for e in result["errors"]],["body","collector_scope_or_release"])
            self.assertEqual(result["errors"][0]["message"],str(seen["check_errors"][0]))
            self.assertFalse(any(c[0] in ("launch","observe") for c in service.calls))

    def test_pre_state_rejection_preserves_no_output_contract(self):
        for mode in ("registration","sources","guard_entry"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root,path,p,a=self.protocol_file(tmp);service=FakeCollectorServices()
                if mode=="registration":service.registration_ok=False
                elif mode=="sources":service.sources_ok=False
                else:service.environment.ac=False
                caught=None
                try:native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
                except BaseException as error:caught=error
                self.assertIsInstance(caught,RuntimeError if mode=="guard_entry" else ValueError)
                self.assertFalse((root/p["output_relative"]).exists());self.assertEqual(service.writes,0)
                self.assertFalse(any(c[0] in ("launch","observe") for c in service.calls))

class ObserverAbortRetentionTests(unittest.TestCase):
    def setup_abort(self,tmp,body,*,callback=None,cached=None,stop_errors=(),cleanup=None,where="outer",alive=False):
        fixture=CollectorTests();self.addCleanup(fixture.doCleanups)
        root,path,protocol,approval=fixture.protocol_file(tmp)
        service=FakeCollectorServices();holder={}
        fault,watch=DirectLaunchBoundaryTests().fault_watch(error=callback,alive=alive)
        fault.api=service.api;service.stop_fault=fault.stop_fault
        def start(case,binding):
            observer=native.NativeObserver(case,binding,service)
            observer.watch=watch;observer.handle=88
            if cached is not None:observer.stop_result=cached;observer.stop_errors=list(stop_errors)
            holder["observer"]=observer
            if where=="check":
                check=service.guard.check
                def fail_check():
                    service.guard.check=check
                    raise body
                service.guard.check=fail_check
            return observer
        service.start_observer=start
        service.stop_observer=Mock(wraps=native.NativeServices.stop_observer.__get__(service))
        service.finish_observer=Mock(side_effect=AssertionError("Abort must not call finish"))
        service.run_outer=Mock(side_effect=body)
        service.close_query=Mock(side_effect=cleanup)
        return SimpleNamespace(root=root,path=path,protocol=protocol,approval=approval,
                               service=service,holder=holder,watch=watch)

    def failed_receipt(self,case):
        persistence=OSError("synthetic receipt unavailable")
        def fail(path,value):
            if value.get("status")=="FAILED_NATIVE_SUPERVISION":raise persistence
        case.service.write_fault=fail
        with self.assertRaises(native.AcceptancePublicationError) as caught:
            native.run_acceptance(case.root,case.path,execute_native=True,
                                  approval=case.approval,services=case.service)
        self.assertIs(caught.exception.persistence_error,persistence)
        return caught.exception

    def test_abort_before_finish_retains_callback_and_body_errors(self):
        for where in ("check","outer"):
            with self.subTest(where=where),tempfile.TemporaryDirectory() as tmp:
                body=RuntimeError("synthetic body");callback=ValueError("synthetic callback")
                case=self.setup_abort(tmp,body,callback=callback,where=where)
                error=self.failed_receipt(case)
                self.assertEqual(len(error.primary_errors),2)
                self.assertIs(error.primary_errors[0],body)
                self.assertIs(error.primary_errors[1],callback)
                self.assertEqual(case.service.stop_observer.call_count,1)
                case.service.finish_observer.assert_not_called()
                self.assertEqual(case.service.run_outer.call_count,int(where=="outer"))
                self.assertEqual(case.watch.thread.joins,[1.])
                observer=case.holder["observer"]
                self.assertIs(observer.stop_result.callback_error,callback)
                record=error.failed_state["observer_diagnostics"]
                self.assertEqual(record["callback_error"]["type"],"ValueError")
                self.assertEqual([e["type"] for e in record["primary_errors"]],["RuntimeError","ValueError"])
                self.assertEqual(error.failed_state["status"],"FAILED_NATIVE_SUPERVISION")

    def test_abort_retains_callback_budget_and_cleanup_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            body=RuntimeError("synthetic body");callback=ValueError("synthetic callback")
            budget=TimeoutError("synthetic budget");cleanup=OSError("synthetic close")
            snapshot=native.FaultStopResult(True,True,False,True,callback)
            case=self.setup_abort(tmp,body,cached=snapshot,stop_errors=[budget],cleanup=cleanup)
            error=self.failed_receipt(case)
            self.assertEqual(len(error.primary_errors),4)
            for actual,expected in zip(error.primary_errors,[body,callback,budget,cleanup]):self.assertIs(actual,expected)
            observer=case.holder["observer"]
            self.assertIs(observer.stop_result,snapshot);self.assertIs(observer.stop_errors[0],budget)
            self.assertEqual(case.watch.thread.joins,[])
            self.assertEqual(case.service.close_query.call_count,1)
            self.assertEqual(case.service.stop_observer.call_count,1)
            case.service.finish_observer.assert_not_called()
            record=error.failed_state["observer_diagnostics"]
            self.assertEqual([e["type"] for e in record["primary_errors"]],["RuntimeError","ValueError","TimeoutError"])
            self.assertEqual([e["type"] for e in record["cleanup_errors"]],["OSError"])

    def test_abort_live_or_expired_observer_never_rejoins_or_closes_live_handle(self):
        for remaining,waits in ((0.,[]),(.25,[.25]),(2.,[1.])):
            with self.subTest(remaining=remaining),tempfile.TemporaryDirectory() as tmp:
                body=RuntimeError("synthetic outer abort")
                case=self.setup_abort(tmp,body,alive=True)
                def abort(*args):
                    case.service.api.now=int(481000-remaining*1000)
                    raise body
                case.service.run_outer.side_effect=abort
                error=self.failed_receipt(case);observer=case.holder["observer"]
                self.assertEqual(case.watch.thread.joins,waits)
                self.assertIs(error.primary_errors[0],body)
                self.assertFalse(observer.stop_result.thread_terminated)
                self.assertFalse(observer.stop_result.stop_ok)
                self.assertEqual(observer.stop_result.deadline_exhausted,remaining==0.)
                self.assertEqual(observer.stop_result.join_timed_out,remaining>0.)
                self.assertFalse(observer.close());self.assertFalse(observer.close())
                self.assertEqual(case.watch.thread.joins,waits)
                case.service.close_query.assert_not_called();self.assertEqual(observer.handle,88)
                case.service.finish_observer.assert_not_called()
                self.assertEqual(case.service.stop_observer.call_count,1)
                self.assertIn("observer_diagnostics",error.failed_state)
                self.assertFalse(error.failed_state["observer_diagnostics"]["stop_ok"])

    def test_abort_missing_snapshot_and_clean_close_still_fail_closed(self):
        for mode in ("missing","healthy","false"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                body=RuntimeError("synthetic body")
                snapshot=None if mode=="missing" else native.FaultStopResult(True,True,False,False,None)
                case=self.setup_abort(tmp,body,cached=snapshot)
                start=case.service.start_observer
                def no_watch(*args):
                    observer=start(*args);observer.watch=None;return observer
                if mode=="missing":case.service.start_observer=no_watch
                if mode=="false":case.service.stop_observer=Mock(return_value=False)
                error=self.failed_receipt(case)
                self.assertIs(error.primary_errors[0],body)
                self.assertEqual(error.failed_state["status"],"FAILED_NATIVE_SUPERVISION")
                self.assertFalse(error.failed_state["delivery_verified"])
                self.assertIn("observer_diagnostics",error.failed_state)
                record=error.failed_state["observer_diagnostics"]
                if mode=="missing":
                    self.assertIsNone(case.holder["observer"].stop_result)
                    self.assertNotIn("thread_terminated",record)
                    self.assertIn("observer stop snapshot missing",str(error.primary_errors[1]))
                elif mode=="false":self.assertIn("observer close not confirmed",str(error.primary_errors[1]))
                else:self.assertEqual(error.primary_errors,[body]);self.assertTrue(record["stop_ok"])
                self.assertEqual(case.service.stop_observer.call_count,1)
                case.service.finish_observer.assert_not_called()
        for token in ("call_a6f6d55427584c8da99bbba75fef5a29","call_9831ed712c9348a19b4a71960443c76d",
                      "call_1772c2605a3d40fabe5266d3b37ba35f"):
            p=fixture_protocol();p["approval_call_id"]=token
            with self.subTest(token=token),tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):
                native.validate_native_protocol(Path(tmp),p,approval=fixture_approval(p))

    def test_abort_errors_survive_release_publication_and_cli(self):
        for write_failure in (True,False):
            with self.subTest(write_failure=write_failure),tempfile.TemporaryDirectory() as tmp:
                body=RuntimeError("synthetic body");callback=ValueError("nonce="+"a"*64)
                budget=TimeoutError("synthetic budget");cleanup=OSError("handle=12345")
                release=OSError("synthetic release")
                snapshot=native.FaultStopResult(True,True,False,True,callback)
                case=self.setup_abort(tmp,body,cached=snapshot,stop_errors=[budget],cleanup=cleanup)
                set_state=case.service.environment.set_execution_state
                def fail_release(flags):
                    if case.service.environment.requests:raise release
                    return set_state(flags)
                case.service.environment.set_execution_state=fail_release
                if write_failure:
                    error=self.failed_receipt(case)
                    self.assertEqual(len(error.primary_errors),5)
                    for actual,expected in zip(error.primary_errors,[body,callback,budget,cleanup,release]):self.assertIs(actual,expected)
                    state=error.failed_state
                else:
                    flags=["--root",str(case.root),"--protocol",str(case.path),"--execute-native",
                           "--approval-call-id",case.approval["call_id"],"--approval-answer",case.approval["answer"],
                           "--registration-commit","a"*40]
                    with patch.object(native,"NativeServices",return_value=case.service),patch("time.monotonic",side_effect=case.service.monotonic),redirect_stdout(io.StringIO()):
                        code=native.main(flags)
                    self.assertEqual(code,1)
                    state=json.loads((case.root/case.protocol["output_relative"]/"stage_state.json").read_text())
                self.assertEqual(state["status"],"FAILED_NATIVE_SUPERVISION")
                self.assertFalse(state["delivery_verified"])
                self.assertEqual([e["phase"] for e in state["errors"]],
                                 ["observer_primary"]*3+["observer_cleanup","collector_scope_or_release"])
                record=state["observer_diagnostics"]
                self.assertEqual(len(record["primary_errors"]),3);self.assertEqual(len(record["cleanup_errors"]),1)
                self.assertNotIn("a"*64,json.dumps(record));self.assertNotIn("12345",json.dumps(record))
                self.assertEqual(case.service.stop_observer.call_count,1)
                case.service.finish_observer.assert_not_called()

    def test_abort_aggregation_deduplicates_identity_without_sampling(self):
        helper=getattr(native,"_observer_abort_error",None)
        self.assertTrue(callable(helper),"Missing pure observer abort aggregator")
        body=RuntimeError("same message");other=RuntimeError("same message");budget=TimeoutError("budget")
        errors=[body,budget,budget,other];original=list(errors)
        snapshot=native.FaultStopResult(True,True,False,True,body)
        with patch.object(native,"_remaining",side_effect=AssertionError("No clock")),\
             patch.object(native.NativeObserver,"sample",side_effect=AssertionError("No sample")),\
             patch.object(native.NativeObserver,"close",side_effect=AssertionError("No close")),\
             patch.object(native.NativeObserver,"finish",side_effect=AssertionError("No finish")),\
             patch.object(native.NativeServices,"stop_fault",side_effect=AssertionError("No stop")):
            result=helper(body,stop_result=snapshot,stop_errors=errors,close_returned=None,close_error=body)
        self.assertIs(result.stop_result,snapshot)
        self.assertEqual(len(result.primary_errors),3)
        for actual,expected in zip(result.primary_errors,[body,budget,other]):self.assertIs(actual,expected)
        self.assertEqual(len(result.cleanup_errors),1);self.assertIs(result.cleanup_errors[0],body)
        self.assertEqual(len(errors),len(original))
        for actual,expected in zip(errors,original):self.assertIs(actual,expected)
        failed=native.FaultStopResult(False,True,True,False,None)
        result=helper(body,stop_result=failed,stop_errors=[],close_returned=False,close_error=None)
        self.assertIn("observer termination not confirmed",str(result.primary_errors[1]))
        result=helper(body,stop_result=None,stop_errors=[],close_returned=None,close_error=other)
        self.assertIsNone(result.stop_result);self.assertIs(result.cleanup_errors[0],other)
        self.assertEqual(len(result.primary_errors),2)
        self.assertIn("observer stop snapshot missing",str(result.primary_errors[1]))


class DirectLaunchBoundaryTests(unittest.TestCase):
    def fault_watch(self,*,error=None,alive=False):
        import threading
        class Event:
            def __init__(self):self.value=False
            def set(self):self.value=True
            def is_set(self):return self.value
        class Thread:
            def __init__(self,*,target,**kw):self.target=target;self.joins=[]
            def start(self):
                if not alive:self.target()
            def join(self,timeout):self.joins.append(timeout)
            def is_alive(self):return alive
        service=object.__new__(native.NativeServices);service.api=FakeAPI()
        service.monotonic=lambda:service.api.now/1000.
        def callback(stop):
            if error is not None:raise error
        with patch.object(threading,"Thread",Thread),patch.object(threading,"Event",Event):
            watch=service.start_fault(callback)
        return service,watch

    def stop_type(self):
        value=getattr(native,"FaultStopResult",None)
        self.assertTrue(callable(value),"Missing typed fault stop result")
        return value

    def test_stopped_fault_with_callback_error_is_not_join_timeout(self):
        self.stop_type();error=ValueError("synthetic callback")
        service,watch=self.fault_watch(error=error)
        result=service.stop_fault(watch,remaining_seconds=2.)
        self.assertTrue(result.thread_terminated);self.assertFalse(result.join_timed_out)
        self.assertIs(result.callback_error,error);self.assertFalse(result.stop_ok)
        self.assertIs(watch.error,error)
        self.assertFalse(result.record()["callback_error_unknown"])

    def test_live_fault_after_join_is_timeout_and_closes_nothing(self):
        self.stop_type();service,watch=self.fault_watch(alive=True)
        with tempfile.TemporaryDirectory() as tmp:
            observer,services,case=ObserverTests().make(tmp);observer.sample()
            services.stop_fault=service.stop_fault;observer.watch=watch
            caught=None
            try:observer.finish(dict(pid=7,exit_code=1,alive=False))
            except BaseException as error:caught=error
            self.assertIsInstance(caught,native.ObserverTerminationError)
            self.assertFalse(caught.stop_result.thread_terminated)
            self.assertTrue(caught.stop_result.join_timed_out)
            self.assertTrue(caught.stop_result.record()["callback_error_unknown"])
            self.assertEqual(services.closed,[])
            self.assertFalse(observer.evidence["monitor_stopped"])

    def test_fault_stop_is_once_and_bounded_by_remaining(self):
        self.stop_type()
        for remaining,waits in ((0.,[]),(.25,[.25]),(2.,[1.])):
            with self.subTest(remaining=remaining):
                service,watch=self.fault_watch()
                first=service.stop_fault(watch,remaining_seconds=remaining)
                again=service.stop_fault(watch,remaining_seconds=100.)
                self.assertIs(first,again);self.assertEqual(watch.thread.joins,waits)
                self.assertEqual(first.deadline_exhausted,remaining==0.)
                if remaining==0.:self.assertFalse(first.join_attempted)
                self.assertEqual(service.api.now,1000)
                expired,expired_watch=self.fault_watch();expired.api.now=481000
                stopped,errors=native._bounded_fault_stop(expired_watch,case_binding(),expired)
                self.assertFalse(stopped.stop_ok);self.assertEqual(expired_watch.thread.joins,[])
                self.assertEqual(len(errors),1);self.assertIn("TIMED_OUT",str(errors[0]))
                with tempfile.TemporaryDirectory() as tmp:
                    observer,services,case=ObserverTests().make(tmp)
                    services.stop_fault=service.stop_fault;observer.watch=watch
                    observer.close();observer.close()
                    self.assertEqual(watch.thread.joins,waits)

    def test_observer_callback_cleanup_release_and_publication_causes_survive(self):
        self.stop_type();self.assertTrue(callable(getattr(native,"ObserverTerminationError",None)))
        for write_failure in (True,False):
            with self.subTest(write_failure=write_failure),tempfile.TemporaryDirectory() as tmp:
                fixture=CollectorTests();self.addCleanup(fixture.doCleanups)
                root,path,p,a=fixture.protocol_file(tmp);service=FakeCollectorServices()
                callback=ValueError("nonce="+"a"*64);cleanup=OSError("handle=12345")
                persistence=OSError("synthetic receipt unavailable")
                fault,watch=self.fault_watch(error=callback)
                fault.api=service.api;service.stop_fault=fault.stop_fault
                def start(case,binding):
                    observer=native.NativeObserver(case,binding,service)
                    observer.watch=watch;observer.handle=88
                    return observer
                service.start_observer=start
                service.finish_observer=native.NativeServices.finish_observer.__get__(service)
                service.stop_observer=native.NativeServices.stop_observer.__get__(service)
                service.close_query=Mock(side_effect=cleanup);service.environment.release_error=True
                if write_failure:
                    def reject(path,value):
                        if value.get("status")=="FAILED_NATIVE_SUPERVISION":raise persistence
                    service.write_fault=reject;caught=None
                    try:native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
                    except BaseException as error:caught=error
                    self.assertIsInstance(caught,native.AcceptancePublicationError)
                    self.assertIs(caught.primary_errors[0],callback)
                    self.assertIs(caught.primary_errors[1],cleanup)
                    self.assertEqual(len(caught.primary_errors),3)
                    self.assertIs(caught.persistence_error,persistence)
                    state=caught.failed_state
                else:
                    flags=["--root",str(root),"--protocol",str(path),"--execute-native",
                           "--approval-call-id",a["call_id"],"--approval-answer",a["answer"],"--registration-commit","a"*40]
                    with patch.object(native,"NativeServices",return_value=service),patch("time.monotonic",side_effect=service.monotonic),redirect_stdout(io.StringIO()):
                        code=native.main(flags)
                    self.assertEqual(code,1)
                    state=json.loads((root/p["output_relative"]/"stage_state.json").read_text())
                self.assertEqual(state["status"],"FAILED_NATIVE_SUPERVISION")
                self.assertEqual(service.close_query.call_count,1)
                record=state["observer_diagnostics"]
                self.assertEqual(record["callback_error"]["type"],"ValueError")
                self.assertEqual(record["cleanup_errors"][0]["type"],"OSError")
                self.assertNotIn("a"*64,json.dumps(record))
                self.assertNotIn("12345",json.dumps(record))

    def test_direct_controller_launch_uses_shared_helper_and_actual_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture=CollectorTests();self.addCleanup(fixture.doCleanups)
            root,path,p,a=fixture.protocol_file(tmp);service=FakeCollectorServices();jobs=[]
            original=service.run_outer
            def run(job,*args):jobs.append(copy.deepcopy(job));return original(job,*args)
            service.run_outer=run
            result=native.run_acceptance(root,path,execute_native=True,approval=a,services=service)
            self.assertEqual(result["status"],"PASS_NATIVE_SUPERVISION_SCOPE")
            self.assertTrue(jobs)
            self.assertEqual(jobs[0].get("python_launch_contract"),"DIRECT_CPYTHON311_VENV_V1")
        import run_lc_role_adaptation as lc
        runtime=synthetic_python_runtime();process=Mock(pid=7,_handle=80,returncode=0)
        process.poll.return_value=0
        api=SimpleNamespace(current_python_runtime=lambda:runtime,process_identity=lambda h:(7,99),
                            process_executable_path=lambda h:runtime.base_executable)
        with tempfile.TemporaryDirectory() as tmp,patch.object(lc,"WindowsAwakeGuard",FakeAwake),patch.object(lc.time,"monotonic",return_value=1.),patch.object(lc.time,"sleep"),patch.object(safety,"NativeMSBAPI",return_value=api) as factory,patch.object(lc.subprocess,"Popen",return_value=process) as spawn:
            job=dict(job_id="N1",command=[runtime.venv_executable,"registered.py"],
                     log_path=str(Path(tmp)/"direct.log"),python_launch_contract="DIRECT_CPYTHON311_VENV_V1")
            row=lc.run_supervised_job(job,dict(per_fit_seconds=90,heartbeat_seconds=10),90)
            self.assertEqual((row["status"],row["pid"],row["python_launch"]["created"]),("COMPLETE",7,"99"))
            self.assertEqual(spawn.call_args.kwargs["executable"],runtime.base_executable)
            self.assertEqual(spawn.call_args.args[0][0],runtime.base_executable)
            ordinary=dict(job,command=["ordinary"],log_path=str(Path(tmp)/"ordinary.log"))
            del ordinary["python_launch_contract"]
            row=lc.run_supervised_job(ordinary,dict(per_fit_seconds=90,heartbeat_seconds=10),90)
            self.assertEqual(row["status"],"COMPLETE");self.assertEqual(factory.call_count,1)
            self.assertNotIn("executable",spawn.call_args.kwargs)
            self.assertEqual(spawn.call_args.args[0],["ordinary"])

    def test_child_runtime_context_mismatch_never_reaches_ready_or_work(self):
        for kind in ("controller","worker"):
            for bad in ("",json.dumps(dict(implementation="pypy"))):
                with self.subTest(kind=kind,bad=bad),tempfile.TemporaryDirectory() as tmp:
                    fixture=WorkerTests();self.addCleanup(fixture.doCleanups)
                    api=CaseLauncher("N1") if kind=="controller" else WorkerAPI()
                    api.current_python_runtime=lambda:synthetic_python_runtime()
                    api.expected_python_runtime=lambda:bad
                    service=FakeDriverServices(api)
                    if kind=="controller":
                        out=fixture.case_root(tmp,"N1");out.mkdir(parents=True)
                        with self.assertRaises(RuntimeError):native.run_controller(Path(tmp),out,"N1",case_binding(),services=service)
                        self.assertFalse((out/"controller_ready.json").exists())
                    else:
                        path,handles=fixture.packet(tmp,"N1")
                        with self.assertRaises(RuntimeError):native.run_no_data_worker(path,handles,services=service)
                        self.assertFalse(path.with_name("task_started.json").exists())
                        self.assertEqual(service.acks,[])

    def test_missing_ready_and_old_permission_never_pass_native_v2(self):
        self.assertEqual(native.FIXED["protocol"],"msb_native_supervision_v2")
        for token in ("call_15418f42581f439da850f0e8cfa0c004",
                      "call_0ed0a8dae29d4d6aaa949f2752ff89ce",
                      "call_3f51cdd2cc0040929106824e0c335950",
                      "call_2790965a504c46dea799f87f8e9be324"):
            p=fixture_protocol();p["approval_call_id"]=token
            with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):
                native.validate_native_protocol(Path(tmp),p,approval=fixture_approval(p))
                self.fail("Consumed or non-native permission accepted")
        with tempfile.TemporaryDirectory() as tmp:
            old=fixture_protocol();old["protocol"]="msb_native_supervision_v1"
            with self.assertRaises(ValueError):native.validate_native_protocol(Path(tmp),old,approval=fixture_approval(old))
            service=ObserverTestServices();service.api.now=1103
            observer=native.NativeObserver(Path(tmp)/"N1",case_binding(),service)
            record=observer.finish(dict(pid=7,exit_code=1,alive=False))
            self.assertFalse(record["identity_verified"])
            self.assertFalse(record["exit_observed"])

    def test_synthetic_scope_covers_runtime_and_thread_adapters(self):
        expected=("current_python_runtime","current_module_executable","process_executable_path")
        self.assertEqual(getattr(harness,"RUNTIME_BOUNDARIES",None),expected)
        import threading
        import time
        import socket
        targets=[lambda:object.__new__(safety.NativeMSBAPI).current_python_runtime(),
                 lambda:object.__new__(safety.NativeMSBAPI).current_module_executable(),
                 lambda:object.__new__(safety.NativeMSBAPI).process_executable_path(80),
                 lambda:threading.Thread(target=lambda:None).start(),
                 lambda:threading.Thread(target=lambda:None).join(),
                 lambda:time.monotonic(),
                 lambda:socket.create_connection(("never-connect.invalid",1))]
        previous=(Path.cwd(),tempfile.tempdir,list(sys.path))
        for invoke in targets+[None]:
            with self.subTest(invoke=invoke),tempfile.TemporaryDirectory() as tmp:
                r=unittest.TestResult();r.testsRun=1
                def modules(names,log):
                    if invoke is not None:invoke()
                    return r
                try:
                    with patch.object(harness,"ROOT",Path(tmp)),patch.object(harness,"_snapshot_sources",return_value={}),patch.object(harness,"_run_modules",side_effect=modules),redirect_stdout(io.StringIO()):
                        code=harness.main(["--phase","collector-green"])
                    data=json.loads((Path(tmp)/"outputs/msb_trace_publication_retention_synthetic_v1/collector-green/attempt-001/acceptance_summary.json").read_text())
                    self.assertEqual(code,0 if invoke is None else 1)
                    self.assertEqual(bool(data["unmocked_boundary_calls"]),invoke is not None)
                finally:
                    os.chdir(previous[0]);tempfile.tempdir=previous[1];sys.path[:]=previous[2]


class TraceMemoryIO:
    def __init__(self,failures=None,*,short_write=False):
        self.files={};self.calls=[];self.failures=failures or {};self.short_write=short_write
        self.on_failure=None

    def fail(self,phase):
        if phase in self.failures:
            if self.on_failure is not None:self.on_failure()
            raise self.failures[phase]

    def open_exclusive(self,path):
        key=str(path);self.calls.append("CREATE");self.fail("CREATE")
        if key in self.files:raise FileExistsError(17,"existing synthetic pending")
        self.files[key]=""
        owner=self
        class Stream:
            def write(self,text):
                owner.calls.append("WRITE");owner.fail("WRITE")
                count=len(text)-1 if owner.short_write else len(text)
                owner.files[key]=text[:count]
                return count
            def close(self):
                owner.calls.append("CLOSE");owner.fail("CLOSE")
        return Stream()

    def replace(self,source,target):
        self.calls.append("REPLACE");self.fail("REPLACE")
        self.files[str(target)]=self.files.pop(str(source))


class TracePublicationRetentionTests(unittest.TestCase):
    def required(self,name):
        value=getattr(native,name,None)
        self.assertTrue(callable(value),"Missing trace-retention behavior: "+name)
        return value

    def helper(self):
        self.required("TracePublicationError")
        return self.required("_write_trace_json")

    def controller(self,tmp,filesystem,case_id="N1"):
        self.required("ControllerTracePublicationError");self.required("TracePublicationState")
        fixture=WorkerTests();self.addCleanup(fixture.doCleanups)
        output=fixture.case_root(tmp,case_id);output.mkdir(parents=True)
        service=FakeDriverServices(CaseLauncher(case_id))
        service.persist_trace=native.NativeServices.persist_trace.__get__(service)
        self.enterContext(patch.object(native,"_TraceFileIO",return_value=filesystem))
        return Path(tmp),output,service

    def test_trace_helper_records_create_failure_and_preserves_existing_pending(self):
        write=self.helper();failure=FileExistsError(17,"synthetic existing")
        fs=TraceMemoryIO({"CREATE":failure});fs.files["trace.json.pending"]="original"
        with self.assertRaises(native.TracePublicationError) as caught:write(Path("trace.json"),[],file_io=fs)
        self.assertIs(caught.exception.first_error,failure)
        self.assertEqual(caught.exception.record()["first_phase"],"CREATE")
        self.assertEqual(fs.files,{"trace.json.pending":"original"});self.assertEqual(fs.calls,["CREATE"])

    def test_trace_helper_records_write_close_replace_and_short_write_failures(self):
        write=self.helper()
        for phase,want in (("WRITE",["CREATE","WRITE","CLOSE"]),("CLOSE",["CREATE","WRITE","CLOSE"]),
                           ("REPLACE",["CREATE","WRITE","CLOSE","REPLACE"]),("SHORT",["CREATE","WRITE","CLOSE"])):
            with self.subTest(phase=phase):
                failure=OSError(5,"synthetic failure");failure.winerror=32
                fs=TraceMemoryIO({phase:failure} if phase!="SHORT" else {},short_write=phase=="SHORT")
                fs.files["trace.json"]="last-good"
                with self.assertRaises(native.TracePublicationError) as caught:write(Path("trace.json"),[],file_io=fs)
                self.assertEqual(caught.exception.record()["first_phase"],"WRITE" if phase=="SHORT" else phase)
                if phase!="SHORT":self.assertIs(caught.exception.first_error,failure)
                self.assertEqual(fs.calls,want);self.assertEqual(fs.files["trace.json"],"last-good")
                self.assertIn("trace.json.pending",fs.files)

    def test_trace_helper_retains_write_and_close_errors_in_order(self):
        write=self.helper();first=OSError("synthetic write");second=OSError("synthetic close")
        fs=TraceMemoryIO({"WRITE":first,"CLOSE":second})
        with self.assertRaises(native.TracePublicationError) as caught:write(Path("trace.json"),[],file_io=fs)
        self.assertEqual([p for p,e in caught.exception.errors],["WRITE","CLOSE"])
        self.assertIs(caught.exception.errors[0][1],first);self.assertIs(caught.exception.errors[1][1],second)
        self.assertIs(caught.exception.first_error,first);self.assertEqual(fs.calls,["CREATE","WRITE","CLOSE"])

    def test_trace_failure_latch_prevents_second_io_and_preserves_pending(self):
        cls=self.required("TracePublicationState");self.helper()
        first=PermissionError(13,"synthetic replacement");fs=TraceMemoryIO({"REPLACE":first})
        def persist(path,events):native._write_trace_json(path,events,file_io=fs)
        service=SimpleNamespace(persist_trace=Mock(side_effect=persist));state=cls()
        with self.assertRaises(native.TracePublicationError) as caught:state.persist(Path("trace.json"),[],services=service)
        original=dict(fs.files);calls=list(fs.calls)
        with self.assertRaises(native.TracePublicationError) as again:state.persist(Path("trace.json"),[{"new":True}],services=service)
        state.finish(Path("trace.json"),[],services=service)
        self.assertIs(again.exception,caught.exception);self.assertIs(state.first_failure.first_error,first)
        self.assertEqual(fs.calls,calls);self.assertEqual(fs.files,original);self.assertEqual(service.persist_trace.call_count,1)
        self.assertEqual(state.record(),dict(state="FAILED",attempts=1,successful_publications=0,
            first_failure=state.first_failure.record(),final_attempted=False,final_skipped_after_error=True))
        unknown=RuntimeError("synthetic adapter")
        other=cls();service.persist_trace.side_effect=unknown
        with self.assertRaises(native.TracePublicationError):other.persist(Path("trace.json"),[],services=service)
        self.assertIs(other.first_failure.first_error,unknown)
        self.assertEqual(other.first_failure.record()["first_phase"],"ADAPTER_UNKNOWN")

    def test_native_persist_trace_uses_real_helper_with_only_fake_file_io(self):
        self.helper();self.required("_TraceFileIO")
        fs=TraceMemoryIO();service=SimpleNamespace()
        with patch.object(native,"_TraceFileIO",return_value=fs),patch.object(Path,"open",side_effect=AssertionError("Live trace I/O")):
            native.NativeServices.persist_trace(service,Path("trace.json"),[])
        self.assertEqual(fs.files,{"trace.json":"[]\n"})
        self.assertEqual(fs.calls,["CREATE","WRITE","CLOSE","REPLACE"])

    def test_trace_success_preserves_serialize_write_close_replace_order(self):
        write=self.helper();fs=TraceMemoryIO()
        events=[{"kind":"spawn"},{"kind":"assigned"},{"kind":"HELLO"},{"kind":"GRANT"},{"kind":"ACK"}]
        write(Path("trace.json"),events,file_io=fs)
        literal='[{"kind": "spawn"}, {"kind": "assigned"}, {"kind": "HELLO"}, {"kind": "GRANT"}, {"kind": "ACK"}]\n'
        self.assertEqual(fs.files,{"trace.json":literal});self.assertEqual(fs.calls,["CREATE","WRITE","CLOSE","REPLACE"])

    def test_controller_first_trace_failure_survives_inner_row_conversion(self):
        self.required("ControllerTracePublicationError")
        first=PermissionError(13,"synthetic trace");first.winerror=32;fs=TraceMemoryIO({"REPLACE":first})
        with tempfile.TemporaryDirectory() as tmp:
            root,output,service=self.controller(tmp,fs)
            with self.assertRaises(native.ControllerTracePublicationError) as caught:
                native.run_controller(root,output,"N1",case_binding(),services=service)
            error=caught.exception;record=error.record()
            self.assertIs(error.trace_publication.first_failure.first_error,first)
            self.assertEqual(record["inner_summary"]["status"],"FAILED")
            self.assertIn("TracePublicationError",record["inner_summary"]["error"])
            self.assertEqual(record["trace_publication"]["first_failure"]["first_phase"],"REPLACE")
            self.assertTrue(record["trace_publication"]["final_skipped_after_error"])
            self.assertEqual(fs.calls,["CREATE","WRITE","CLOSE","REPLACE"])
            self.assertFalse((output/"controller/controller_result.json").exists())
            saved=json.loads((output/"controller/controller_failure.json").read_text())
            self.assertEqual(saved["status"],"FAILED_CONTROLLER_TRACE_PUBLICATION")
            self.assertEqual(saved["inner_summary"],record["inner_summary"])

    def test_controller_final_trace_failure_keeps_inner_row_and_body_errors(self):
        self.required("ControllerTracePublicationError")
        for body in (None,ValueError("synthetic body")):
            with self.subTest(body=body),tempfile.TemporaryDirectory() as tmp:
                fs=TraceMemoryIO({"REPLACE":OSError(5,"synthetic final")});root,output,service=self.controller(tmp,fs)
                row=case_record("N1")["msb"]
                with patch.object(safety.MSBParentSupervisor,"run_job",side_effect=body,return_value=row):
                    with self.assertRaises(native.ControllerTracePublicationError) as caught:
                        native.run_controller(root,output,"N1",case_binding(),services=service)
                error=caught.exception;record=error.record()
                self.assertTrue(record["trace_publication"]["final_attempted"])
                self.assertFalse(record["trace_publication"]["final_skipped_after_error"])
                self.assertEqual(record["parent_power"],dict(power_request_acquired=True,power_request_released=True))
                if body is None:self.assertIs(error.inner_row,row);self.assertEqual(record["inner_summary"]["status"],"COMPLETE")
                else:self.assertIs(error.body_errors[0],body);self.assertIsNone(record["inner_summary"])
                self.assertEqual(fs.calls,["CREATE","WRITE","CLOSE","REPLACE"])

    def test_controller_failure_receipt_error_is_distinct_and_fail_closed(self):
        self.required("ControllerTracePublicationError")
        with tempfile.TemporaryDirectory() as tmp:
            first=OSError(5,"synthetic trace");receipt=OSError("nonce="+"b"*64)
            fs=TraceMemoryIO({"REPLACE":first});root,output,service=self.controller(tmp,fs)
            original=native._write_json;writes=[]
            def write(path,value,**kw):
                if Path(path).name=="controller_failure.json":writes.append(str(path));raise receipt
                return original(path,value,**kw)
            with patch.object(native,"_write_json",side_effect=write):
                with self.assertRaises(native.ControllerTracePublicationError) as caught:
                    native.run_controller(root,output,"N1",case_binding(),services=service)
            error=caught.exception
            self.assertIs(error.receipt_error,receipt);self.assertIs(error.trace_publication.first_failure.first_error,first)
            self.assertEqual(len(writes),1);self.assertEqual(error.record()["receipt_publication"]["status"],"FAILED")
            protocol=fixture_protocol();approval=fixture_approval(protocol)
            protocol_path=root/"docs/experiment_protocols/msb_native_supervision_v2.json"
            protocol_path.parent.mkdir(parents=True);protocol_path.write_text(json.dumps(protocol))
            binding=dict(case_binding(),protocol_digest=native._protocol_digest(protocol))
            claim=dict(status="RUNNING",active_case="N1",binding=binding,registration_verified=True,
                       source_identity_verified=True,registered_commit=approval["registration_commit"])
            (root/protocol["output_relative"]/"stage_state.json").write_text(json.dumps(claim))
            service.verify_sources=Mock();out=io.StringIO()
            flags=["--root",str(root),"--protocol",str(protocol_path),"--execute-native",
                   "--approval-call-id",approval["call_id"],"--approval-answer",approval["answer"],
                   "--registration-commit",approval["registration_commit"],"--controller-case","N1",
                   "--case-output",str(output),"--binding-json",json.dumps(binding)]
            with patch.object(native,"NativeServices",return_value=service),patch.object(native,"run_controller",side_effect=error),redirect_stdout(out):
                code=native.main(flags)
            self.assertEqual(code,1);self.assertEqual(json.loads(out.getvalue())["status"],"FAILED_CONTROLLER_TRACE_PUBLICATION")
            self.assertNotIn("b"*64,out.getvalue());self.assertNotIn("Traceback",out.getvalue())

    def test_collector_retains_controller_failure_but_never_qualifies_case(self):
        self.required("ControllerTracePublicationError")
        record=case_record("N1")
        native._check_case(record,"N1",case_binding())
        failure=dict(status="FAILED_CONTROLLER_TRACE_PUBLICATION")
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp)/"N1";folder=case/"controller";folder.mkdir(parents=True)
            (folder/"trace.json").write_text(json.dumps(record["events"]))
            (folder/"controller_failure.json").write_text(json.dumps(failure))
            loaded=native._load_case_record(case,case_binding(),record["outer"],record["observation"])
            self.assertEqual(loaded["controller_failure"],failure)
        forged=dict(record,controller_failure=failure)
        with self.assertRaisesRegex(ValueError,"trace publication"):native._check_case(forged,"N1",case_binding())

    def test_controller_trace_failure_keeps_existing_cleanup_and_deadline(self):
        self.required("ControllerTracePublicationError")
        for expired in (False,True):
            with self.subTest(expired=expired),tempfile.TemporaryDirectory() as tmp:
                fs=TraceMemoryIO({"REPLACE":OSError("synthetic replacement")});root,output,service=self.controller(tmp,fs)
                if expired:fs.on_failure=lambda:setattr(service.api,"now",481000)
                service.api.failure="wait"
                with patch.object(service.api,"awake_guard",wraps=service.api.awake_guard) as awake,patch.object(native,"_remaining",wraps=native._remaining) as remaining:
                    with self.assertRaises(native.ControllerTracePublicationError) as caught:
                        native.run_controller(root,output,"N1",case_binding(),services=service)
                record=caught.exception.record()
                self.assertFalse(record["inner_summary"]["cleanup_confirmed"])
                self.assertEqual(service.api.events.count("terminated"),1);self.assertEqual(awake.call_count,2)
                self.assertEqual(service.callbacks,[]);self.assertEqual(service.api.closed,[11,12,30,21,22,10])
                self.assertEqual(remaining.call_count,2 if expired else 3)
                self.assertEqual(record["binding"],case_binding())
                self.assertEqual(record["receipt_publication"]["status"],"SKIPPED_BUDGET_EXHAUSTED" if expired else "PERSISTED")
                self.assertEqual((output/"controller/controller_failure.json").exists(),not expired)

    def test_controller_failure_record_redacts_capabilities_and_whitelists_os_codes(self):
        cls=self.required("TracePublicationError");state_cls=self.required("TracePublicationState")
        controller_cls=self.required("ControllerTracePublicationError")
        for errno,winerror,want_errno,want_win in ((5,32,5,32),(0,0,0,0),(4095,65535,4095,65535),
                (True,True,None,None),(4096,65536,None,None),(-1,-1,None,None),(None,None,None,None)):
            with self.subTest(errno=errno,winerror=winerror):
                original=OSError("nonce="+"c"*64+" handles=12345 environment=SECRET")
                original.errno=errno;original.winerror=winerror
                failure=cls((("REPLACE",original),));entry=failure.record()["errors"][0]
                self.assertEqual((entry["errno"],entry["winerror"]),(want_errno,want_win))
                self.assertIs(failure.first_error,original)
                state=state_cls();service=SimpleNamespace(persist_trace=Mock(side_effect=failure))
                with self.assertRaises(cls):state.persist(Path("trace.json"),[],services=service)
                body=RuntimeError("handles=12345 environment=SECRET")
                row=dict(case_record("N1")["msb"],error="nonce="+"c"*64,nonce="hidden")
                error=controller_cls("N1",case_binding(),inner_row=row,parent_power=None,body_errors=[body],
                    fault_stop=None,trace_publication=state)
                text=json.dumps(error.record())
                for secret in ("c"*64,"12345","SECRET","hidden"):self.assertNotIn(secret,text)
                self.assertNotIn("nonce",error.record()["inner_summary"]);self.assertIs(error.body_errors[0],body)
                self.assertNotIn("c"*64,str(error));self.assertNotIn("c"*64,str(failure))

    def test_trace_serialization_failure_opens_no_file(self):
        write=self.helper()
        for events in ([{"bad":math.nan}],[{"bad":object()}]):
            with self.subTest(events=events):
                fs=TraceMemoryIO();fs.files["trace.json.pending"]="original"
                with self.assertRaises(native.TracePublicationError) as caught:write(Path("trace.json"),events,file_io=fs)
                self.assertEqual(caught.exception.record()["first_phase"],"SERIALIZE")
                self.assertEqual(fs.calls,[]);self.assertEqual(fs.files,{"trace.json.pending":"original"})

    def test_normal_and_expected_fault_controller_contracts_remain_unchanged(self):
        self.required("TracePublicationState")
        for case_id,want,exit_code in (("N1","COMPLETE",0),("N2","SUPERVISION_INTERRUPTED",23),("N3","SUPERVISION_INTERRUPTED",1)):
            with self.subTest(case_id=case_id),tempfile.TemporaryDirectory() as tmp:
                fs=TraceMemoryIO();root,output,service=self.controller(tmp,fs,case_id)
                row=native.run_controller(root,output,case_id,case_binding(),services=service)
                self.assertEqual((row["msb"]["status"],row["msb"]["exit_code"]),(want,exit_code))
                self.assertEqual(set(row),{"case_id","msb","events","parent_power","fault_monitor_stopped","real_data_opened"})
                self.assertTrue((output/"controller/controller_result.json").exists())
                self.assertFalse((output/"controller/controller_failure.json").exists())
                self.assertTrue(row["parent_power"]["power_request_released"])
                self.assertFalse(row["real_data_opened"])
