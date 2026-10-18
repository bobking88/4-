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
                summary=json.loads((Path(tmp)/"outputs/msb_native_supervision_synthetic_v1/contract-green/attempt-001/acceptance_summary.json").read_text())
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
                if self.eof:self.inbox.append(OSError("native pipe EOF"))
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
    def read_ack(self,case_dir,ready):return dict(ready,collector_verified=True,query_only=True,job_handle_retained=False)
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
        services=FakeDriverServices(WorkerAPI())
        with tempfile.TemporaryDirectory() as tmp:
            ready=dict(pid=8,created=199,working=True,job_assigned=True,case_id="N4",ready_tick_ms=1000)
            services.read_ack=lambda d,r:dict(r,collector_verified=True,query_only=True,job_handle_retained=False)
            with self.assertRaises(SystemExit) as error:fault(Path(tmp),ready,services=services)
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
