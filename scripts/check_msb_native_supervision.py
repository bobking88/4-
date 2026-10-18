"""Bounded native no-data supervision acceptance; no scientific input or model path."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

SOURCE_FILES = (
    "scripts/analyze_ic_role_mechanism.py",
    "scripts/analyze_lc_role_adaptation.py",
    "scripts/analyze_misclassifications.py",
    "scripts/analyze_msb_role_mechanism.py",
    "scripts/anchor_linear_controls.py",
    "scripts/audit_abmp_candidate_capacity.py",
    "scripts/check_lc_native_environment.py",
    "scripts/frozen_visual_probe.py",
    "scripts/generate_msb_role_figure.py",
    "scripts/hrgv_network.py",
    "scripts/ic_role_audit.py",
    "scripts/ic_role_data.py",
    "scripts/ic_role_model.py",
    "scripts/ic_role_training.py",
    "scripts/lc_execution_safety.py",
    "scripts/lc_role_adapter.py",
    "scripts/lc_role_audit.py",
    "scripts/lc_role_data.py",
    "scripts/lc_role_training.py",
    "scripts/mineral_hierarchy.py",
    "scripts/msb_execution_safety.py",
    "scripts/msb_role_audit.py",
    "scripts/msb_role_controls.py",
    "scripts/msb_role_data.py",
    "scripts/msb_role_model.py",
    "scripts/msb_role_training.py",
    "scripts/run_anchor_linear_controls.py",
    "scripts/run_frozen_visual_probe.py",
    "scripts/run_ic_role_mechanism.py",
    "scripts/run_lc_role_adaptation.py",
    "scripts/run_msb_role_mechanism.py",
    "scripts/run_seen_unseen_gate_study.py",
    "scripts/run_tc_oos_rsg_experiments.py",
    "scripts/run_verifier_strong_controls.py",
    "scripts/run_verifier_trust_development.py",
    "scripts/tc_oos_rsg.py",
    "scripts/train_hierarchical_mineral_classifier.py",
    "scripts/train_hrgv_mineral_classifier.py",
    "scripts/train_mineral_classifier.py",
    "scripts/train_role_aware_mineral_classifier.py",
    "scripts/verifier_risk.py",
    "scripts/verifier_strong_controls.py",
    "scripts/verify_verifier_trust_delivery.py",
    "scripts/check_msb_native_supervision.py",
    "tests/test_check_msb_native_supervision.py",
    "tests/run_msb_native_supervision_synthetic_acceptance.py",
)
CASES = (
    dict(case_id="N1", limit=90, seconds=45),
    dict(case_id="N2", limit=45, exit_code=23),
    dict(case_id="N3", limit=75, linger_seconds=35),
    dict(case_id="N4", limit=45, exit_code=31),
)
NON_NATIVE_IDS = {
    "call_a42aed1b7bab474bb17246c3df69dc4e", "call_6961d4dac6b64d7aba8e538f01c5e097",
    "call_5101ffc4ea2a44459043052c470036ce", "call_0e89f2cb80134c049e4ef3aaf697a7b0",
    "call_aa6d081524a94817b20cceb8fbd25053", "call_a4c9ef81bcea447aa7350f7cd9aff51c",
}
FIXED = dict(protocol="msb_native_supervision_v1", status="APPROVED_NATIVE_ONLY",
             scope="NATIVE_SUPERVISION_NO_DATA", execution_contract="msb_parent_bound_v2",
             output_relative="outputs/msb_native_supervision_v1/acceptance_once",
             max_seconds=480, case_order=["N1","N2","N3","N4"],
             case_limits=[90,45,75,45], require_ac=True, max_attempts=1,
             real_training_authorized=False)


def remaining_acceptance_seconds(start_monotonic: float, start_tick_ms: int, *,
                                 now_monotonic: float, now_tick_ms: int) -> float:
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0
            for v in (start_monotonic,now_monotonic))
            or type(start_tick_ms) is not int or type(now_tick_ms) is not int
            or start_tick_ms<0 or now_tick_ms<start_tick_ms or now_monotonic<start_monotonic):
        raise ValueError("Invalid or backwards acceptance clock")
    remaining=480.-max(now_monotonic-start_monotonic,(now_tick_ms-start_tick_ms)/1000.)
    if remaining<=0: raise RuntimeError("TIMED_OUT: cumulative native acceptance")
    return remaining


def validate_native_protocol(root: Path, protocol: dict, *, approval: dict) -> dict:
    if not isinstance(protocol,dict) or not isinstance(approval,dict):
        raise ValueError("Explicit native protocol and human approval required")
    if set(protocol)!=set(FIXED)|{"approval_call_id","approval_answer","source_sha256"}:
        raise ValueError("No extra input, experiment or execution fields")
    for key,value in FIXED.items():
        if type(protocol[key]) is not type(value) or protocol[key]!=value:
            raise ValueError("Fixed native acceptance field differs: "+key)
    if any(type(v) is not int for v in protocol["case_limits"]):
        raise ValueError("Integer fixed case limits required")
    token=protocol["approval_call_id"]
    if (not isinstance(token,str) or not re.fullmatch(r"call_[A-Za-z0-9]{16,64}",token)
            or token in NON_NATIVE_IDS or not isinstance(protocol["approval_answer"],str)
            or not protocol["approval_answer"].strip()
            or approval.get("call_id")!=token or approval.get("answer")!=protocol["approval_answer"]
            or approval.get("scope")!="NATIVE_SUPERVISION_NO_DATA"
            or approval.get("confirmed_from_human") is not True
            or not re.fullmatch(r"[a-f0-9]{40}",str(approval.get("registration_commit","")))):
        raise ValueError("New exact human native permission and registration required")
    sources=protocol["source_sha256"]
    if (not isinstance(sources,dict) or set(sources)!=set(SOURCE_FILES)
            or any(not isinstance(v,str) or not re.fullmatch(r"[a-f0-9]{64}",v) for v in sources.values())):
        raise ValueError("Exact native acceptance source closure required")
    destination=Path(root).resolve()/protocol["output_relative"]
    if destination.exists(): raise FileExistsError("Exclusive native output consumed; no retry")
    return copy.deepcopy(protocol)


def _write_json(path: Path, value: dict, *, exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open("x",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=True,allow_nan=False);stream.write("\n")
    else:
        temporary=path.with_name(path.name+".pending")
        with temporary.open("x",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=True,allow_nan=False);stream.write("\n")
        temporary.replace(path)


def _case_output(root,case_dir,case_id):
    if case_id not in {c["case_id"] for c in CASES}:raise ValueError("Unknown native case")
    expected=Path(root).resolve()/"outputs/msb_native_supervision_v1/acceptance_once/cases"/case_id
    if Path(case_dir).resolve()!=expected:raise ValueError("Fixed exclusive native case path required")
    return expected


def _binding_budget(binding):
    import msb_execution_safety as safety
    if (not isinstance(binding,dict) or not isinstance(binding.get("run_id"),str)
            or not re.fullmatch(r"[a-f0-9]{64}",str(binding.get("protocol_digest","")))
            or type(binding.get("start_monotonic")) not in (int,float)):
        raise ValueError("Original acceptance binding required")
    budget=safety.RunBudget(binding["start_tick_ms"],binding["deadline_tick_ms"])
    return budget


def _remaining(binding,services):
    return remaining_acceptance_seconds(binding["start_monotonic"],binding["start_tick_ms"],
        now_monotonic=services.monotonic(),now_tick_ms=services.tick_ms())


def _wait_ack(case_dir,ready,services):
    while True:
        now=services.tick_ms()
        if now<ready["ready_tick_ms"] or now-ready["ready_tick_ms"]>=5000:
            raise RuntimeError("SUPERVISION_INTERRUPTED: collector identity ACK timeout")
        ack=services.read_ack(case_dir,ready)
        if ack is not None:
            if (not isinstance(ack,dict) or any(ack.get(k)!=v for k,v in ready.items())
                    or ack.get("collector_verified") is not True or ack.get("query_only") is not True
                    or ack.get("job_handle_retained") is not False):
                raise RuntimeError("SUPERVISION_INTERRUPTED: collector identity ACK differs")
            return ack
        services.pause(.05)


def parent_fault_after_ready(case_dir: Path, ready: dict, *, services):
    if (not isinstance(ready,dict) or ready.get("working") is not True
            or ready.get("job_assigned") is not True or ready.get("case_id")!="N4"
            or any(type(ready.get(k)) is not int or ready[k]<=0 for k in ("pid","created"))
            or type(ready.get("ready_tick_ms")) is not int):
        raise RuntimeError("SUPERVISION_INTERRUPTED: parent fault lacks bound working child")
    _wait_ack(case_dir,ready,services)
    services.exit_process(31)
    raise RuntimeError("Parent fault did not terminate current process")


def _watch_parent_fault(case_dir,binding,services,stopped):
    while not stopped.is_set():
        _remaining(binding,services)
        path=case_dir/"worker_ready.json"
        if path.exists():
            try:ready=json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                services.pause(.05);continue
            parent_fault_after_ready(case_dir,ready,services=services)
            return
        services.pause(.05)


class TraceAPI:
    """Record genuine API returns without fabricating transport or time."""
    def __init__(self,api,events,path,services):
        self.api,self.events,self.path,self.services=api,events,path,services
    def __getattr__(self,name):return getattr(self.api,name)
    def _event(self,kind,**fields):
        self.events.append(dict(kind=kind,tick_ms=self.api.tick_ms(),**fields))
    def start_worker(self,job,handles,*,ownership):
        process=self.api.start_worker(job,handles,ownership=ownership)
        pid,created=self.api.process_identity(int(process._handle))
        self._event("spawn",pid=pid,created=created)
        return process
    def assign_job(self,handle,process):
        self.api.assign_job(handle,process)
        self._event("assigned",pid=process.pid)
    def read_message(self,handle,max_bytes=8192):
        try:message=self.api.read_message(handle,max_bytes)
        except OSError as error:
            self._event("pipe_error",error=repr(error));raise
        if message is not None:self._message("worker_to_parent",message)
        return message
    def write_message(self,handle,message):
        self.api.write_message(handle,message)
        self._message("parent_to_worker",message)
    def _message(self,direction,message):
        self._event(message["kind"],direction=direction,sequence=message["sequence"],
                    lease_until_tick_ms=message["lease_until_tick_ms"],
                    nonce_sha256=hashlib.sha256(message["nonce"].encode()).hexdigest(),
                    worker_pid=message["worker_pid"],parent_pid=message["parent_pid"])
    def pause(self,seconds):
        self.services.persist_trace(self.path,self.events)
        self.api.pause(seconds)
    def terminate_job(self,handle):
        self.api.terminate_job(handle);self._event("terminate_job")
    def kill_process(self,process):
        self.api.kill_process(process);self._event("kill_process",pid=process.pid)


def run_controller(root: Path, output: Path, case_id: str, binding: dict, *, services) -> dict:
    import msb_execution_safety as safety
    case_dir=_case_output(root,output,case_id);budget=_binding_budget(binding)
    _remaining(binding,services)
    folder=case_dir/"controller";folder.mkdir(exist_ok=False)
    work=folder/"runs"/case_id;work.mkdir(parents=True,exist_ok=False)
    parent_pid,parent_created=services.api.current_identity()
    packet=dict(binding,root=str(Path(root).resolve()),output=str(folder),
                stage="native-supervision-no-data",case_id=case_id,job_id=case_id,
                parent_pid=parent_pid,parent_created=parent_created)
    packet_path=work/"packet.json";_write_json(packet_path,packet,exclusive=True)
    state=dict(packet,status="RUNNING",active_job=case_id,
               packets={case_id:hashlib.sha256(packet_path.read_bytes()).hexdigest()})
    _write_json(folder/"stage_state.json",state,exclusive=True)
    events=[];trace=TraceAPI(services.api,events,folder/"trace.json",services)
    command=[sys.executable,str(Path(__file__).resolve()),"--worker-packet",str(packet_path)]
    job=dict(job_id=case_id,packet_path=str(packet_path),log_path=str(work/"worker.log"),command=command)
    watch=None;parent=None;stopped=True
    try:
        if case_id=="N4":
            watch=services.start_fault(lambda stop:_watch_parent_fault(case_dir,binding,services,stop))
        with safety.MSBParentSupervisor(budget,binding["run_id"],binding["protocol_digest"],api=trace) as parent:
            row=parent.run_job(job,{})
    finally:
        if watch is not None:stopped=services.stop_fault(watch)
        services.persist_trace(folder/"trace.json",events)
    if not stopped:raise RuntimeError("SUPERVISION_INTERRUPTED: parent fault monitor not stopped")
    result=dict(case_id=case_id,msb=row,events=events,parent_power=parent.evidence(),
                fault_monitor_stopped=stopped,real_data_opened=False)
    _remaining(binding,services)
    _write_json(folder/"controller_result.json",result,exclusive=True)
    return result


def _wait_work(milliseconds,packet,services,scope=None):
    started=services.tick_ms()
    while services.tick_ms()-started<milliseconds:
        _remaining(packet,services)
        if scope is not None:scope.check()
        services.pause(.05)
    return services.tick_ms()-started


def run_no_data_worker(packet_path: Path, handles: dict, *, services) -> dict:
    import msb_execution_safety as safety
    if not isinstance(handles,dict) or set(handles)!={"read_handle","write_handle","parent_handle","nonce"}:
        raise ValueError("Live inherited capability required before metadata")
    packet=safety._published(packet_path)
    case_id=packet.get("case_id")
    if packet.get("stage")!="native-supervision-no-data":raise ValueError("No scientific stage allowed")
    case_dir=_case_output(Path(packet["root"]),Path(packet["output"]).parent,case_id)
    if Path(packet_path).resolve()!=case_dir/"controller"/"runs"/case_id/"packet.json":
        raise ValueError("Worker packet path differs")
    _binding_budget(packet);_remaining(packet,services)
    awake=services.awake_guard()
    with safety.open_worker_scope(packet_path,handles,api=services.api,awake_guard=awake) as scope:
        with awake:
            scope.begin_work()
            _write_json(Path(packet_path).with_name("task_started.json"),dict(status="ATTEMPT_CONSUMED"),exclusive=True)
            pid,created=services.api.current_identity()
            ready=dict(pid=pid,created=created,case_id=case_id,working=True,
                       job_assigned=scope.evidence()["job_assigned"],ready_tick_ms=services.tick_ms())
            services.write_ready(case_dir,ready)
            if case_id in ("N2","N4"):_wait_ack(case_dir,ready,services)
            if case_id=="N2":
                services.exit_process(23)
                raise RuntimeError("Worker fault did not terminate")
            if case_id=="N4":
                while True:
                    _remaining(packet,services);scope.check();services.pause(.05)
            elapsed=_wait_work(45000 if case_id=="N1" else 0,packet,services,scope)
        power=awake.evidence()
        scope.begin_publish(power)
        candidate=dict(status="NO_DATA_WORKER_CANDIDATE",case_id=case_id,real_data_opened=False,
                       worker_power=power,work_elapsed_ms=elapsed)
        _write_json(Path(packet_path).with_name("result.json"),candidate,exclusive=True)
        scope.check()
    result=dict(candidate,worker_execution=scope.evidence(),linger_elapsed_ms=0)
    _write_json(Path(packet_path).with_name("worker_closed.json"),result,exclusive=True)
    if case_id=="N3":result["linger_elapsed_ms"]=_wait_work(35000,packet,services)
    return result


class NativeServices:
    """Real adapters are constructed only after the separately approved collector gate."""
    def __init__(self):
        import msb_execution_safety as safety
        from check_lc_native_environment import NativeEnvironmentAPI
        self.api=safety.NativeMSBAPI()
        self.environment=NativeEnvironmentAPI()
    def tick_ms(self):return self.api.tick_ms()
    def monotonic(self):
        import time
        return time.monotonic()
    def pause(self,seconds):self.api.pause(seconds)
    def awake_guard(self):return self.api.awake_guard()
    def exit_process(self,code):
        import os
        os._exit(code)
    def persist_trace(self,path,events):_write_json(path,events)
    def write_ready(self,case_dir,ready):_write_json(case_dir/"worker_ready.json",ready,exclusive=True)
    def read_ack(self,case_dir,ready):
        path=case_dir/"collector_ack.json"
        if not path.exists():return None
        try:return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:return None
    def start_fault(self,callback):
        import threading
        from types import SimpleNamespace
        stopped=threading.Event();watch=SimpleNamespace(stopped=stopped,error=None,thread=None)
        def body():
            try:callback(stopped)
            except BaseException as error:watch.error=repr(error)
        watch.thread=threading.Thread(target=body,name="msb-native-parent-fault",daemon=True)
        watch.thread.start()
        return watch
    def stop_fault(self,watch):
        watch.stopped.set();watch.thread.join(timeout=1.)
        return not watch.thread.is_alive() and watch.error is None


def run_acceptance(root: Path, protocol_path: Path, *, execute_native: bool, approval: dict, services=None) -> dict:
    if execute_native is not True:
        raise ValueError("Explicit native execution permission required before any service or input")
    raise RuntimeError("Collector implementation not completed; native execution disabled")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approval-call-id")
    parser.add_argument("--approval-answer")
    parser.add_argument("--registration-commit")
    parser.add_argument("--worker-packet",type=Path)
    parser.add_argument("--worker-handles")
    args=parser.parse_args(argv)
    if args.worker_packet is not None:
        if (args.root is not None or args.protocol is not None or args.execute_native
                or args.approval_call_id or args.approval_answer or args.registration_commit
                or args.worker_handles is None):
            parser.error("Private inherited worker mode cannot mix public flags or omit handles")
        handles=json.loads(args.worker_handles)
        if (not isinstance(handles,dict) or set(handles)!={"read_handle","write_handle","parent_handle","nonce"}
                or any(type(handles[k]) is not int or handles[k]<=0 for k in ("read_handle","write_handle","parent_handle"))
                or not re.fullmatch(r"[a-f0-9]{64}",str(handles["nonce"]))):
            raise ValueError("Exact positive inherited capability required before service construction")
        result=run_no_data_worker(args.worker_packet,handles,services=NativeServices())
        print(json.dumps(result),flush=True)
        return 0
    if args.worker_handles is not None or args.protocol is None:
        parser.error("Explicit protocol required for collector; no detached worker handles")
    approval=dict(call_id=args.approval_call_id,answer=args.approval_answer,
                  registration_commit=args.registration_commit,
                  scope="NATIVE_SUPERVISION_NO_DATA",confirmed_from_human=True)
    result=run_acceptance(args.root or Path(__file__).resolve().parents[1],args.protocol,
                          execute_native=args.execute_native,approval=approval)
    print(json.dumps(result),flush=True)
    return 0 if result.get("status")=="PASS_NATIVE_SUPERVISION_SCOPE" else 1


if __name__=="__main__":
    raise SystemExit(main())
