"""Bounded native no-data supervision acceptance; no scientific input or model path."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
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
    "call_15418f42581f439da850f0e8cfa0c004", "call_0ed0a8dae29d4d6aaa949f2752ff89ce",
    "call_3f51cdd2cc0040929106824e0c335950", "call_2790965a504c46dea799f87f8e9be324",
    "call_29684c57aed94222832ace218cd392be", "call_754f23a7456b4c5cbe8e744ba0dd6114",
    "call_03a86c6e5b064586b7bb980fa04b2a53", "call_d538a55e3a7b401fa4709a0d3914bb24",
    "call_7f515c71f122488da5242db3dc69dd60",
    "call_a42aed1b7bab474bb17246c3df69dc4e", "call_6961d4dac6b64d7aba8e538f01c5e097",
    "call_5101ffc4ea2a44459043052c470036ce", "call_0e89f2cb80134c049e4ef3aaf697a7b0",
    "call_aa6d081524a94817b20cceb8fbd25053", "call_a4c9ef81bcea447aa7350f7cd9aff51c",
    "call_a6f6d55427584c8da99bbba75fef5a29", "call_9831ed712c9348a19b4a71960443c76d",
    "call_1772c2605a3d40fabe5266d3b37ba35f",
}
NATIVE_PROTOCOL_RELATIVE = "docs/experiment_protocols/msb_native_supervision_v3.json"
NATIVE_OUTPUT_RELATIVE = "outputs/msb_native_supervision_v3/acceptance_once"
FIXED = dict(protocol="msb_native_supervision_v3", status="APPROVED_NATIVE_ONLY",
             scope="NATIVE_SUPERVISION_NO_DATA", execution_contract="msb_parent_bound_v2",
             python_launch_contract="DIRECT_CPYTHON311_VENV_V1",
             output_relative=NATIVE_OUTPUT_RELATIVE,
             max_seconds=480, case_order=["N1","N2","N3","N4"],
             case_limits=[90,45,75,45], require_ac=True, max_attempts=1,
             real_training_authorized=False)


def _fault_error_record(error):
    message=str(error)
    if (any(c in message for c in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
            or re.search(r"(?i)\b(?:nonce|handles?|environment|env)\b",message)):
        message="<redacted context>"
    message=re.sub(r"(?i)\b[a-f0-9]{64}\b|\b0x[a-f0-9]+\b|\b[0-9]+\b","<redacted>",message)[:256]
    frames=[];trace=error.__traceback__
    while trace is not None:
        frames.append(dict(file=trace.tb_frame.f_code.co_filename,function=trace.tb_frame.f_code.co_name,line=trace.tb_lineno))
        trace=trace.tb_next
    return dict(type=type(error).__name__,message=message,frames=frames[-8:])


@dataclass(frozen=True)
class FaultStopResult:
    thread_terminated: bool
    join_attempted: bool
    join_timed_out: bool
    deadline_exhausted: bool
    callback_error: BaseException | None

    @property
    def stop_ok(self):
        return self.thread_terminated and self.callback_error is None and not self.deadline_exhausted

    def __bool__(self):raise TypeError("Inspect FaultStopResult.stop_ok explicitly")

    def record(self):
        return dict(thread_terminated=self.thread_terminated,join_attempted=self.join_attempted,
                    join_timed_out=self.join_timed_out,deadline_exhausted=self.deadline_exhausted,
                    callback_error=None if self.callback_error is None else _fault_error_record(self.callback_error),
                    callback_error_unknown=not self.thread_terminated,stop_ok=self.stop_ok)


class ObserverTerminationError(RuntimeError):
    def __init__(self,stop_result,primary_errors,cleanup_errors=()):
        self.stop_result=stop_result;self.primary_errors=list(primary_errors);self.cleanup_errors=list(cleanup_errors)
        super().__init__("SUPERVISION_INTERRUPTED: observer callback, termination or cleanup failed")

    def record(self):
        record={} if self.stop_result is None else self.stop_result.record()
        record.update(primary_errors=[_fault_error_record(e) for e in self.primary_errors],
                      cleanup_errors=[_fault_error_record(e) for e in self.cleanup_errors])
        return record


def _observer_abort_error(
    primary: BaseException,
    *,
    stop_result: FaultStopResult | None,
    stop_errors: list[BaseException],
    close_returned: bool | None,
    close_error: BaseException | None,
) -> ObserverTerminationError:
    snapshot=stop_result if isinstance(stop_result,FaultStopResult) else None
    candidates=[primary]
    if snapshot is not None and snapshot.callback_error is not None:
        candidates.append(snapshot.callback_error)
    candidates.extend(stop_errors)
    errors=[]
    for error in candidates:
        if not any(error is previous for previous in errors):errors.append(error)
    if snapshot is None:
        errors.append(RuntimeError("SUPERVISION_INTERRUPTED: observer stop snapshot missing"))
    elif not snapshot.stop_ok and snapshot.callback_error is None and not stop_errors:
        errors.append(RuntimeError("SUPERVISION_INTERRUPTED: observer termination not confirmed"))
    elif snapshot.stop_ok and close_error is None and close_returned is not True:
        errors.append(RuntimeError("SUPERVISION_INTERRUPTED: observer close not confirmed"))
    return ObserverTerminationError(snapshot,errors,[] if close_error is None else [close_error])


def execution_origins() -> dict[str,str]:
    origins={"driver":str(Path(__file__).resolve())}
    for relative in SOURCE_FILES:
        if not relative.startswith("scripts/"):continue
        name=Path(relative).stem;module=sys.modules.get(name)
        if module is None:continue
        source=getattr(module,"__file__",None)
        origins[f"module:{name}:file"]="" if source is None else str(Path(source).resolve())
        spec=getattr(module,"__spec__",None);origin=getattr(spec,"origin",None)
        if origin is not None:origins[f"module:{name}:spec"]=str(Path(origin).resolve())
    return origins


def validate_execution_origins(root: Path, origins: dict[str,str]) -> dict:
    root=Path(root).resolve();driver=root/"scripts/check_msb_native_supervision.py"
    if (not isinstance(origins,dict) or not isinstance(origins.get("driver"),str)
            or not origins["driver"] or Path(origins["driver"]).resolve()!=driver):
        raise ValueError("Executing driver differs from the registered root")
    project={Path(p).stem:root/p for p in SOURCE_FILES if p.startswith("scripts/")}
    checked={}
    for key,value in origins.items():
        if key=="driver":continue
        match=re.fullmatch(r"module:([A-Za-z0-9_]+):(file|spec)",key)
        if (match is None or match[1] not in project or not isinstance(value,str) or not value
                or Path(value).resolve()!=project[match[1]]
                or f"module:{match[1]}:file" not in origins):
            raise ValueError("Loaded project source differs from registered root: "+str(key))
        checked[key]=str(Path(value).resolve())
    return dict(root=str(root),driver=str(driver),module_origins=checked)


def _execution_root(root):
    return validate_execution_origins(root,execution_origins())


def _worker_packet(packet_path,handles):
    if (not isinstance(handles,dict) or set(handles)!={"read_handle","write_handle","parent_handle","nonce"}
            or any(type(handles[k]) is not int or handles[k]<=0 for k in ("read_handle","write_handle","parent_handle"))
            or not re.fullmatch(r"[a-f0-9]{64}",str(handles["nonce"]))):
        raise ValueError("Exact live inherited capability required before metadata")
    origins=execution_origins();root=Path(origins["driver"]).resolve().parents[1]
    validate_execution_origins(root,origins)
    path=Path(packet_path).resolve();prefix=root/FIXED["output_relative"]/"cases"
    if not path.is_relative_to(prefix):raise ValueError("Worker packet outside executing root")
    parts=path.relative_to(prefix).parts
    if (len(parts)!=5 or parts[0] not in FIXED["case_order"]
            or parts[1:3]!=("controller","runs") or parts[3]!=parts[0] or parts[4]!="packet.json"):
        raise ValueError("Only the fixed no-data worker packet may be read")
    metadata=json.loads(path.read_text(encoding="utf-8"));_execution_root(metadata["root"])
    import msb_execution_safety as safety
    _execution_root(root)
    return safety._published(path)


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


def _validate_protocol_fields(protocol: dict, *, approval: dict) -> dict:
    if not isinstance(protocol,dict) or not isinstance(approval,dict):
        raise ValueError("Explicit native protocol and human approval required")
    if set(protocol)!=set(FIXED)|{"approval_call_id","approval_answer","source_sha256","runtime_venv_root"}:
        raise ValueError("No extra input, experiment or execution fields")
    import ntpath
    if (not isinstance(protocol["runtime_venv_root"],str) or "\0" in protocol["runtime_venv_root"]
            or not ntpath.isabs(protocol["runtime_venv_root"]) or not ntpath.splitdrive(protocol["runtime_venv_root"])[0]):
        raise ValueError("Registered Windows venv root required")
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
    return copy.deepcopy(protocol)


def validate_native_protocol(root: Path, protocol: dict, *, approval: dict) -> dict:
    result=_validate_protocol_fields(protocol,approval=approval)
    destination=Path(root).resolve()/result["output_relative"]
    if destination.exists():raise FileExistsError("Exclusive native output consumed; no retry")
    return result


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


class TracePublicationError(RuntimeError):
    def __init__(self,errors: tuple[tuple[str,BaseException],...]):
        if not errors:raise ValueError("Trace publication requires a first error")
        self.errors=tuple(errors)
        super().__init__("Trace publication failed; inspect redacted diagnostics")

    @property
    def first_error(self):return self.errors[0][1]

    def record(self):
        records=[]
        for phase,error in self.errors:
            item=dict(phase=phase,**_fault_error_record(error))
            for name,maximum in (("errno",4095),("winerror",65535)):
                value=getattr(error,name,None) if isinstance(error,OSError) else None
                item[name]=value if type(value) is int and 0<=value<=maximum else None
            records.append(item)
        return dict(first_phase=self.errors[0][0],errors=records)


class _TraceFileIO:
    def open_exclusive(self,path):return path.open("x",encoding="utf-8")
    def replace(self,source,target):source.replace(target)


def _write_trace_json(path: Path, events: list, *, file_io=None) -> None:
    try:payload=json.dumps(events,ensure_ascii=True,allow_nan=False)+"\n"
    except BaseException as error:raise TracePublicationError((("SERIALIZE",error),)) from None
    path=Path(path);temporary=path.with_name(path.name+".pending")
    adapter=_TraceFileIO() if file_io is None else file_io
    try:stream=adapter.open_exclusive(temporary)
    except BaseException as error:raise TracePublicationError((("CREATE",error),)) from None
    errors=[]
    try:
        count=stream.write(payload)
        if type(count) is not int or count!=len(payload):raise OSError("Trace write length differs")
    except BaseException as error:errors.append(("WRITE",error))
    try:stream.close()
    except BaseException as error:errors.append(("CLOSE",error))
    if errors:raise TracePublicationError(tuple(errors)) from None
    try:adapter.replace(temporary,path)
    except BaseException as error:raise TracePublicationError((("REPLACE",error),)) from None


class TracePublicationState:
    """Latch publication failure before the supervisor turns it into a row."""
    def __init__(self):
        self.first_failure=None;self.attempts=0;self.successful_publications=0
        self.final_attempted=False;self.final_skipped_after_error=False

    def persist(self,path,events,*,services) -> None:
        if self.first_failure is not None:raise self.first_failure from None
        self.attempts+=1
        try:services.persist_trace(path,events)
        except BaseException as error:
            self.first_failure=(error if isinstance(error,TracePublicationError)
                                else TracePublicationError((("ADAPTER_UNKNOWN",error),)))
            raise self.first_failure from None
        self.successful_publications+=1

    def finish(self,path,events,*,services) -> None:
        if self.final_attempted or self.final_skipped_after_error:return
        if self.first_failure is not None:
            self.final_skipped_after_error=True;return
        self.final_attempted=True
        try:self.persist(path,events,services=services)
        except TracePublicationError:pass

    def record(self):
        return dict(state="FAILED" if self.first_failure is not None else "HEALTHY",attempts=self.attempts,
                    successful_publications=self.successful_publications,
                    first_failure=None if self.first_failure is None else self.first_failure.record(),
                    final_attempted=self.final_attempted,final_skipped_after_error=self.final_skipped_after_error)


class ControllerTracePublicationError(RuntimeError):
    def __init__(self,case_id,binding,*,inner_row,parent_power,body_errors,fault_stop,trace_publication,fault_stop_evidence=None):
        self.case_id,self.binding=case_id,copy.deepcopy(binding)
        self.inner_row,self.parent_power=inner_row,parent_power
        self.body_errors=list(body_errors);self.fault_stop=fault_stop;self.trace_publication=trace_publication
        self.fault_stop_evidence=fault_stop_evidence
        self.receipt_error=None;self.receipt_publication=dict(status="NOT_ATTEMPTED",error=None)
        super().__init__("Controller failed trace publication; inspect redacted diagnostics")

    def record(self):
        row=self.inner_row;summary=None
        if isinstance(row,dict):
            summary={k:row[k] for k in ("status","exit_code","cleanup_confirmed","exit_confirmed",
                                       "end_handshake_verified","lease_sequence") if k in row}
            summary["error"]=_fault_error_record(RuntimeError(str(row.get("error",""))))["message"]
        power=None
        if isinstance(self.parent_power,dict):
            names=("wall_elapsed_seconds","max_poll_gap_seconds","max_suspend_delta_seconds",
                   "poll_gap_limit_seconds","suspend_tolerance_seconds","power_request_acquired","power_request_released")
            power={k:v for k,v in self.parent_power.items() if k in names and type(v) in (bool,int,float)
                   and (type(v) is not float or math.isfinite(v))}
        fault_stop=self.fault_stop
        if self.fault_stop_evidence is not None:
            evidence=self.fault_stop_evidence
            result=evidence["returned_result"];cached=evidence["cached_result"]
            stop_error=evidence["stop_exception"];callback=evidence["cached_callback_error"]
            errors=evidence["budget_errors"]
            stopped=(isinstance(fault_stop,dict) and fault_stop.get("stopped") is True
                     and stop_error is None and not errors and isinstance(result,FaultStopResult) and result.stop_ok)
            fault_stop=dict(result=result.record() if isinstance(result,FaultStopResult) else None,
                errors=[_fault_error_record(e) for e in errors],
                exception=None if stop_error is None else _fault_error_record(stop_error),stopped=stopped,
                cached_result=cached.record() if isinstance(cached,FaultStopResult) else None,
                cached_callback_error=None if callback is None else _fault_error_record(callback),
                cached_result_diagnostic_only=True)
        return dict(status="FAILED_CONTROLLER_TRACE_PUBLICATION",case_id=self.case_id,binding=copy.deepcopy(self.binding),
                    inner_summary=summary,parent_power=power,body_errors=[_fault_error_record(e) for e in self.body_errors],
                    fault_stop=copy.deepcopy(fault_stop),trace_publication=self.trace_publication.record(),
                    receipt_publication=copy.deepcopy(self.receipt_publication))


def _case_output(root,case_dir,case_id):
    if case_id not in {c["case_id"] for c in CASES}:raise ValueError("Unknown native case")
    expected=Path(root).resolve()/NATIVE_OUTPUT_RELATIVE/"cases"/case_id
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


def _bounded_fault_stop(watch,binding,services,*,errors: list[BaseException] | None = None) -> tuple[FaultStopResult,list[BaseException]]:
    if errors is None:errors=[]
    try:remaining=_remaining(binding,services)
    except BaseException as error:
        errors.append(error);remaining=0.
    result=services.stop_fault(watch,remaining_seconds=remaining)
    if not errors:
        try:_remaining(binding,services)
        except BaseException as error:errors.append(error)
    return result,errors


def _wait_ack(case_dir,ready,services):
    while True:
        now=services.tick_ms()
        if now<ready["ready_tick_ms"] or now-ready["ready_tick_ms"]>=5000:
            raise RuntimeError("SUPERVISION_INTERRUPTED: collector identity ACK timeout")
        ack=services.read_ack(case_dir,ready)
        if ack is not None:
            accepted_at=services.tick_ms()
            if accepted_at<now or accepted_at-ready["ready_tick_ms"]>=5000:
                raise RuntimeError("SUPERVISION_INTERRUPTED: collector ACK read crossed deadline")
            if (not isinstance(ack,dict) or any(ack.get(k)!=v for k,v in ready.items())
                    or ack.get("collector_verified") is not True or ack.get("query_only") is not True
                    or ack.get("job_handle_retained") is not False):
                raise RuntimeError("SUPERVISION_INTERRUPTED: collector identity ACK differs")
            return ack
        services.pause(.05)


def parent_fault_after_ready(case_dir: Path, ready: dict, *, services, gate=None):
    if (not isinstance(ready,dict) or ready.get("working") is not True
            or ready.get("job_assigned") is not True or ready.get("case_id")!="N4"
            or any(type(ready.get(k)) is not int or ready[k]<=0 for k in ("pid","created"))
            or type(ready.get("ready_tick_ms")) is not int):
        raise RuntimeError("SUPERVISION_INTERRUPTED: parent fault lacks bound working child")
    ack=_wait_ack(case_dir,ready,services)
    if not isinstance(gate,N4FaultGate) or gate.case_dir!=Path(case_dir):
        raise RuntimeError("SUPERVISION_INTERRUPTED: live N4 fault gate required")
    gate.commit_fault(ready,ack,gate.events)


def _watch_parent_fault(case_dir,binding,services,stopped,gate=None):
    while not stopped.is_set():
        _remaining(binding,services)
        path=case_dir/"worker_ready.json"
        if path.exists():
            try:ready=json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                services.pause(.05);continue
            parent_fault_after_ready(case_dir,ready,services=services,gate=gate)
            return
        services.pause(.05)


class N4FaultGate:
    """Order explicit cleanup intent before the bounded parent-fault checkpoint."""
    def __init__(self,case_dir,binding,*,services,events):
        import threading
        self.case_dir,self.binding,self.services=Path(case_dir),copy.deepcopy(binding),services
        self.events=events;self.lock=threading.Lock();self.cleanup_requested=False;self.invalid=False;self.errors=[]

    def _acquire(self):
        if not self.lock.acquire(timeout=min(1.,_remaining(self.binding,self.services))):
            self.invalid=True
            raise RuntimeError("SUPERVISION_INTERRUPTED: bounded N4 gate unavailable")

    def note_cleanup(self,kind,pid=None):
        self.cleanup_requested=True;acquired=False
        try:
            self._acquire();acquired=True
            event=dict(kind="cleanup_intent",action=kind,pid=pid,tick_ms=self.services.tick_ms())
            self.events.append(event)
            _write_json(self.case_dir/"cleanup_intent.json",dict(event,binding=self.binding),exclusive=True)
        except BaseException as error:
            self.invalid=True;self.errors.append(f"{type(error).__name__}: {error}")
        finally:
            if acquired:self.lock.release()

    def commit_fault(self,ready,ack,trace_events):
        self._acquire();handle=None
        try:
            _require(not self.cleanup_requested and not self.invalid,"N4 cleanup intent or gate failure")
            _require(ack.get("alive_at_ack") is True and type(ack.get("ack_tick_ms")) is int,"N4 ACK lacks live observation")
            handle=retain_worker_identity(ready,services=self.services)
            _require(self.services.query_alive(handle) is True,"N4 worker dead before fault")
            alive_tick=self.services.tick_ms();pid,created=self.services.api.current_identity()
            _require(pid!=ready["pid"],"N4 parent/worker identity collision")
            obs=dict(pid=ready["pid"],created=ready["created"],parent_pid=pid,parent_created=created)
            snapshot=copy.deepcopy(trace_events);_check_trace(snapshot,obs)
            _require(not any(e["kind"] in ("cleanup_intent","terminate_job","kill_process") for e in snapshot),"N4 explicit cleanup trace")
            _require(ready["ready_tick_ms"]<=ack["ack_tick_ms"]<=alive_tick,"N4 live ACK ordering")
            _write_json(self.case_dir/"fault_trace.json",snapshot,exclusive=True)
            tick=self.services.tick_ms()
            fault=dict(case_id="N4",pid=pid,created=created,exit_code=31,tick_ms=tick,
                worker_pid=ready["pid"],worker_created=ready["created"],collector_ack_verified=True,
                alive_before_fault=True,alive_tick_ms=alive_tick,gate_state="FAULT_COMMITTED",
                cleanup_before_fault=False,trace_complete=True,binding=self.binding,pre_fault_trace=snapshot)
            _write_json(self.case_dir/"parent_fault.json",fault,exclusive=True)
            _remaining(self.binding,self.services)
            _require(self.services.tick_ms()-ready["ready_tick_ms"]<5000,"N4 fault preparation exceeded ACK window")
            _require(not self.cleanup_requested and not self.invalid,"N4 cleanup raced fault publication")
            _require(self.services.query_alive(handle) is True,"N4 worker exited during fault preparation")
            self.services.exit_process(31)
            raise RuntimeError("Parent fault did not terminate current process")
        except ValueError as error:
            self.invalid=True
            raise RuntimeError("SUPERVISION_INTERRUPTED: "+str(error)) from error
        finally:
            try:
                if handle is not None:self.services.close_query(handle)
            finally:self.lock.release()


class TraceAPI:
    """Record genuine API returns without fabricating transport or time."""
    def __init__(self,api,events,path,services,fault_gate=None,publication=None):
        self.api,self.events,self.path,self.services=api,events,path,services
        self.fault_gate=fault_gate
        self.publication=TracePublicationState() if publication is None else publication
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
            self._event("pipe_error",error=repr(error),winerror=getattr(error,"winerror",None));raise
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
        self.publication.persist(self.path,self.events,services=self.services)
        self.api.pause(seconds)
    def terminate_job(self,handle):
        if self.fault_gate is not None:self.fault_gate.note_cleanup("terminate_job")
        self.api.terminate_job(handle);self._event("terminate_job")
    def kill_process(self,process):
        if self.fault_gate is not None:self.fault_gate.note_cleanup("kill_process",process.pid)
        self.api.kill_process(process);self._event("kill_process",pid=process.pid)


def run_controller(root: Path, output: Path, case_id: str, binding: dict, *, services) -> dict:
    origin=_execution_root(root)
    import msb_execution_safety as safety
    _execution_root(root)
    runtime=safety.validate_inherited_python_runtime(services.api)
    case_dir=_case_output(root,output,case_id);budget=_binding_budget(binding)
    _remaining(binding,services)
    folder=case_dir/"controller";folder.mkdir(exist_ok=False)
    work=folder/"runs"/case_id;work.mkdir(parents=True,exist_ok=False)
    parent_pid,parent_created=services.api.current_identity()
    _write_json(case_dir/"controller_ready.json",dict(pid=parent_pid,created=parent_created,binding=binding,
                python_runtime=safety.python_runtime_record(runtime)),exclusive=True)
    packet=dict(binding,root=str(Path(root).resolve()),output=str(folder),
                stage="native-supervision-no-data",case_id=case_id,job_id=case_id,
                parent_pid=parent_pid,parent_created=parent_created,python_runtime=safety.python_runtime_record(runtime))
    packet_path=work/"packet.json";_write_json(packet_path,packet,exclusive=True)
    state=dict(packet,status="RUNNING",active_job=case_id,
               packets={case_id:hashlib.sha256(packet_path.read_bytes()).hexdigest()})
    _write_json(folder/"stage_state.json",state,exclusive=True)
    events=[]
    gate=N4FaultGate(case_dir,binding,services=services,events=events) if case_id=="N4" else None
    publication=TracePublicationState()
    trace=TraceAPI(services.api,events,folder/"trace.json",services,fault_gate=gate,publication=publication)
    command=[sys.executable,origin["driver"],"--worker-packet",str(packet_path)]
    job=dict(job_id=case_id,packet_path=str(packet_path),log_path=str(work/"worker.log"),command=command)
    watch=None;parent=None;stopped=True;stop_result=None;stop_errors=[]
    row=None;body_error=None;stop_exception=None
    try:
        if case_id=="N4":
            watch=services.start_fault(lambda stop:_watch_parent_fault(case_dir,binding,services,stop,gate))
        with safety.MSBParentSupervisor(budget,binding["run_id"],binding["protocol_digest"],api=trace) as parent:
            row=parent.run_job(job,{})
    except BaseException as error:body_error=error
    finally:
        if watch is not None:
            try:
                stop_result,stop_errors=_bounded_fault_stop(watch,binding,services,errors=stop_errors)
                stopped=stop_result.stop_ok and not stop_errors
            except BaseException as error:stop_exception=error;stopped=False
        publication.finish(folder/"trace.json",events,services=services)
    if publication.first_failure is not None:
        body_errors=[];current=body_error
        while current is not None and len(body_errors)<8 and not any(current is e for e in body_errors):
            body_errors.append(current)
            current=current.__cause__ if current.__cause__ is not None else current.__context__
        body_errors.reverse();parent_power=None
        if parent is not None:
            try:parent_power=parent.evidence()
            except BaseException as error:body_errors.append(error)
        fault_stop=None;fault_stop_evidence=None
        if watch is not None:
            fault_stop=dict(result=None if stop_result is None else stop_result.record(),
                errors=[_fault_error_record(e) for e in stop_errors],
                exception=None if stop_exception is None else _fault_error_record(stop_exception),stopped=stopped)
            fault_stop_evidence=dict(returned_result=stop_result,budget_errors=stop_errors,
                stop_exception=stop_exception,cached_result=getattr(watch,"stop_result",None),
                cached_callback_error=getattr(watch,"error",None))
        failure=ControllerTracePublicationError(case_id,binding,inner_row=row,parent_power=parent_power,
            body_errors=body_errors,fault_stop=fault_stop,trace_publication=publication,fault_stop_evidence=fault_stop_evidence)
        try:_remaining(binding,services)
        except BaseException as error:
            failure.receipt_error=error
            failure.receipt_publication=dict(status="SKIPPED_BUDGET_EXHAUSTED",error=_fault_error_record(error))
        else:
            failure.receipt_publication=dict(status="ATTEMPTED",error=None)
            try:_write_json(folder/"controller_failure.json",failure.record(),exclusive=True)
            except BaseException as error:
                failure.receipt_error=error
                failure.receipt_publication=dict(status="FAILED",error=_fault_error_record(error))
            else:
                try:_remaining(binding,services)
                except BaseException as error:
                    failure.receipt_error=error
                    failure.receipt_publication=dict(status="FAILED_POSTCHECK",error=_fault_error_record(error))
                else:failure.receipt_publication=dict(status="PERSISTED",error=None)
        raise failure from None
    if stop_exception is not None:raise stop_exception.with_traceback(stop_exception.__traceback__)
    if body_error is not None:raise body_error.with_traceback(body_error.__traceback__)
    if not stopped:
        primary=([stop_result.callback_error] if stop_result.callback_error is not None else [])+stop_errors
        if not primary:primary=[RuntimeError("SUPERVISION_INTERRUPTED: parent fault monitor termination not confirmed")]
        raise ObserverTerminationError(stop_result,primary)
    result=dict(case_id=case_id,msb=row,events=events,parent_power=parent.evidence(),
                fault_monitor_stopped=stopped,real_data_opened=False)
    _remaining(binding,services)
    _write_json(folder/"controller_result.json",result,exclusive=True)
    return result


def _wait_work(milliseconds,packet,services,scope=None):
    if type(milliseconds) is not int or milliseconds<0:raise ValueError("Nonnegative integer work duration required")
    if milliseconds==0:return 0
    started=services.tick_ms()
    while services.tick_ms()-started<milliseconds:
        _remaining(packet,services)
        if scope is not None:scope.check()
        services.pause(.05)
    return services.tick_ms()-started


def run_no_data_worker(packet_path: Path, handles: dict, *, services) -> dict:
    packet=_worker_packet(packet_path,handles)
    import msb_execution_safety as safety
    _execution_root(packet["root"])
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

    simulated=False
    def new_run_id(self):
        import uuid
        return uuid.uuid4().hex
    def capture_runtime(self,venv_root):
        import msb_execution_safety as safety
        runtime=self.api.current_python_runtime()
        if safety._windows_path(runtime.venv_prefix)!=safety._windows_path(venv_root):
            raise RuntimeError("SUPERVISION_INTERRUPTED: registered venv root differs")
        return safety.python_runtime_record(runtime)
    def collector_guard(self):
        from lc_execution_safety import WindowsAwakeGuard
        return WindowsAwakeGuard(api=self.environment)
    def power_settings(self):return self.environment.power_settings()
    def postflight_sample(self):return self.environment.sample()
    def write_json(self,path,value,*,exclusive=False):_write_json(path,value,exclusive=exclusive)
    def run_outer(self,job,protocol,remaining_seconds):
        from run_lc_role_adaptation import run_supervised_job
        return run_supervised_job(job,protocol,remaining_seconds)
    def open_query(self,pid):
        return self.api._ok(self.api.kernel.OpenProcess(0x100000|0x1000,False,pid))
    def query_identity(self,handle):return self.api.process_identity(handle)
    def query_alive(self,handle):return self.api.process_alive(handle)
    def close_query(self,handle):self.api.close_handle(handle)
    def query_exit_code(self,handle):
        import ctypes
        from ctypes import wintypes
        function=self.api.kernel.GetExitCodeProcess
        function.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];function.restype=wintypes.BOOL
        value=wintypes.DWORD();self.api._ok(function(handle,ctypes.byref(value)))
        return int(value.value)
    def start_observer(self,case_dir,binding):
        observer=NativeObserver(case_dir,binding,self)
        observer.watch=self.start_fault(observer.loop)
        return observer
    def finish_observer(self,observer,outer):return observer.finish(outer)
    def stop_observer(self,observer):return observer.close()
    def verify_sources(self,root,protocol):
        _execution_root(root)
        for relative,expected in protocol["source_sha256"].items():
            path=(Path(root)/relative).resolve()
            if not path.is_relative_to(Path(root).resolve()) or path.is_symlink():
                raise ValueError("Source escapes registered root")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError("Source identity changed: "+relative)
        return dict(verified=True)
    def _api(self,endpoint,check):
        import subprocess
        check()
        remaining=self._registration_check()
        result=subprocess.run([r"C:\Program Files\GitHub CLI\gh.exe","api",endpoint],
                              capture_output=True,text=True,encoding="utf-8",check=True,timeout=min(15.,remaining))
        check()
        return json.loads(result.stdout)
    def verify_registration(self,root,path,protocol,approval,check):
        import base64
        self.verify_sources(root,protocol)
        self._registration_check=check
        base="repos/bobking88/4-"
        head=self._api(base+"/git/ref/heads/codex/theory-aware-report",check)
        commit_id=approval["registration_commit"]
        if head["object"]["sha"]!=commit_id:raise ValueError("Registered native branch head differs")
        commit=self._api(base+"/git/commits/"+commit_id,check)
        if commit["sha"]!=commit_id:raise ValueError("Registration commit differs")
        tree=self._api(base+"/git/trees/"+commit["tree"]["sha"]+"?recursive=1",check)
        if tree.get("truncated") is not False:raise ValueError("Source tree truncated")
        blobs={r["path"]:r["sha"] for r in tree["tree"] if r.get("type")=="blob"}
        def blob(payload):
            canonical=payload.replace(b"\r\n",b"\n")
            return hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
        for relative in SOURCE_FILES:
            check()
            if blobs.get(relative)!=blob((Path(root)/relative).read_bytes()):
                raise ValueError("Registered source blob differs: "+relative)
        relative=NATIVE_PROTOCOL_RELATIVE
        payload=Path(path).read_bytes()
        if blobs.get(relative)!=blob(payload):raise ValueError("Registered native protocol blob differs")
        content=self._api(base+"/contents/"+relative+"?ref="+commit_id,check)
        if content.get("encoding")!="base64":raise ValueError("Native protocol encoding")
        remote=base64.b64decode(content["content"],validate=False)
        if remote.replace(b"\r\n",b"\n")!=payload.replace(b"\r\n",b"\n"):
            raise ValueError("Registered native protocol content differs")
        self.verify_sources(root,protocol);check()
        return dict(verified=True,commit=commit_id,tree=commit["tree"]["sha"])
    def __init__(self,root: Path):
        self.root=Path(_execution_root(root)["root"])
        import msb_execution_safety as safety
        from check_lc_native_environment import NativeEnvironmentAPI
        _execution_root(self.root)
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
    def persist_trace(self,path,events):_write_trace_json(path,events)
    def write_ready(self,case_dir,ready):_write_json(case_dir/"worker_ready.json",ready,exclusive=True)
    def read_ack(self,case_dir,ready):
        path=case_dir/"collector_ack.json"
        if not path.exists():return None
        try:return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:return None
    def start_fault(self,callback):
        import threading
        from types import SimpleNamespace
        stopped=threading.Event();watch=SimpleNamespace(stopped=stopped,error=None,thread=None,stop_result=None)
        def body():
            try:callback(stopped)
            except BaseException as error:watch.error=error
        watch.thread=threading.Thread(target=body,name="msb-native-parent-fault",daemon=True)
        watch.thread.start()
        return watch
    def stop_fault(self,watch,*,remaining_seconds):
        previous=getattr(watch,"stop_result",None)
        if previous is not None:return previous
        if type(remaining_seconds) not in (int,float) or not math.isfinite(remaining_seconds):
            raise ValueError("Finite inherited stop budget required")
        before=self.monotonic();watch.stopped.set()
        attempted=remaining_seconds>0
        if attempted:watch.thread.join(timeout=min(1.,remaining_seconds))
        terminated=not watch.thread.is_alive();elapsed=self.monotonic()-before
        expired=remaining_seconds<=0 or not math.isfinite(elapsed) or elapsed<0 or elapsed>=remaining_seconds
        watch.stop_result=FaultStopResult(terminated,attempted,attempted and not terminated,expired,watch.error)
        return watch.stop_result


def _protocol_digest(protocol):
    return hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _check_approval(approval):
    if (not isinstance(approval,dict) or approval.get("call_id") in NON_NATIVE_IDS
            or not re.fullmatch(r"call_[A-Za-z0-9]{16,64}",str(approval.get("call_id","")))
            or not isinstance(approval.get("answer"),str) or not approval["answer"].strip()
            or approval.get("scope")!="NATIVE_SUPERVISION_NO_DATA"
            or approval.get("confirmed_from_human") is not True
            or not re.fullmatch(r"[a-f0-9]{40}",str(approval.get("registration_commit","")))):
        raise ValueError("Explicit new native permission required before any input or service")


def _protocol_path(root,path):
    expected=Path(root).resolve()/NATIVE_PROTOCOL_RELATIVE
    if Path(path).resolve()!=expected:raise ValueError("Only the fixed native protocol path may be read")
    return expected


def _read_json_if_present(path):
    path=Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _power_ok(record):
    return (isinstance(record,dict) and record.get("power_request_acquired") is True
            and record.get("power_request_released") is True)


def _require(condition,reason):
    if not condition:raise ValueError(reason)


def _check_trace(events,observation):
    _require(isinstance(events,list) and bool(events),"missing native trace")
    _require(all(type(e.get("tick_ms")) is int for e in events),"invalid trace clocks")
    _require(all(a["tick_ms"]<=b["tick_ms"] for a,b in zip(events,events[1:])),"trace clock/order")
    spawn=[(i,e) for i,e in enumerate(events) if e["kind"]=="spawn"]
    assigned=[i for i,e in enumerate(events) if e["kind"]=="assigned"]
    _require(len(spawn)==1 and len(assigned)==1 and spawn[0][0]<assigned[0],"launch/assignment trace")
    _require((spawn[0][1]["pid"],spawn[0][1]["created"])==(observation["pid"],observation["created"]),"spawn identity")
    grants={};acks={};close=None;done=None;nonce=None;last_sequence=0
    for i,event in enumerate(events):
        kind=event["kind"]
        if kind not in ("HELLO","GRANT","RENEW","ACK","CLOSE_REQUEST","CLOSE_GRANT","DONE"):continue
        _require(event["worker_pid"]==observation["pid"] and event["parent_pid"]==observation["parent_pid"],"message process binding")
        value=event["nonce_sha256"]
        _require(isinstance(value,str) and re.fullmatch(r"[a-f0-9]{64}",value),"nonce digest")
        if nonce is None:nonce=value
        _require(nonce==value,"nonce differs")
        sequence,lease=event["sequence"],event["lease_until_tick_ms"]
        if kind in ("GRANT","RENEW"):
            _require(close is None and done is None,"renew after terminal close")
            _require(i>assigned[0] and sequence==last_sequence+1,"grant sequence")
            _require(kind==("GRANT" if sequence==1 else "RENEW"),"grant type")
            _require(event["direction"]=="parent_to_worker" and lease>event["tick_ms"],"grant lease")
            grants[sequence]=(lease,i);last_sequence=sequence
        elif kind=="ACK":
            _require(sequence in grants and lease==grants[sequence][0] and i>grants[sequence][1],"ACK lease/sequence")
            _require(sequence not in acks,"duplicate ACK")
            acks[sequence]=i
        elif kind=="CLOSE_GRANT":
            _require(close is None and sequence==last_sequence and sequence in acks,"close grant sequence/ACK")
            _require(lease==grants[sequence][0] and acks[sequence]<i,"close grant lease")
            _require(any(e["kind"]=="CLOSE_REQUEST" for e in events[:i]),"missing close request")
            close=(sequence,lease,i)
        elif kind=="DONE":
            _require(done is None and close is not None and (sequence,lease)==close[:2] and i>close[2],"DONE binding")
            done=(sequence,lease,i)
    _require(1 in grants and 1 in acks,"missing initial GRANT/ACK")
    return dict(grants=grants,acks=acks,close=close,done=done,last_sequence=last_sequence)


def _check_case(record,case_id,binding):
    _require(record.get("controller_failure") is None,"controller trace publication failure")
    _require(record.get("case_id")==case_id and record.get("binding")==binding,"case/original binding")
    obs=record["observation"];ready=obs["ready"];outer=record["outer"]
    _require(ready["case_id"]==case_id and ready.get("working") is True and ready.get("job_assigned") is True,"no verified WORKING")
    _require((ready["pid"],ready["created"])==(obs["pid"],obs["created"]),"worker creation identity")
    _require(all(type(obs[k]) is int and obs[k]>0 for k in ("pid","created","parent_pid","parent_created")),"positive process identity")
    for key in ("identity_verified","query_only","job_assignment_observed","exit_observed","handle_closed","monitor_stopped"):
        _require(obs.get(key) is True,"observation missing: "+key)
    for key in ("job_handle_retained","fallback_before_exit","worker_alive"):
        _require(obs.get(key) is False,"unsafe observation: "+key)
    _require(outer.get("attempts")==1 and outer.get("alive") is False and outer.get("pid")==obs["parent_pid"],"outer attempt/identity/exit")
    _require(_power_ok(outer.get("supervision")),"outer power release")
    expected=("FAILED",31) if case_id=="N4" else ("COMPLETE",0)
    _require((outer.get("status"),outer.get("exit_code"))==expected,"outer timeout or wrong controller exit")
    trace=_check_trace(record["events"],obs)
    inner=record.get("msb");closed=record.get("worker_closed")
    if case_id=="N4":
        _require(inner is None and closed is None and record.get("parent_power") is None,"fabricated hard-exit completion/release")
        fault=record["fault"]
        _require(isinstance(fault,dict) and fault.get("case_id")=="N4" and fault.get("exit_code")==31
                 and fault.get("collector_ack_verified") is True,"parent fault evidence")
        _require((fault["pid"],fault["created"])==(obs["parent_pid"],obs["parent_created"]),"parent creation identity")
        _require(obs.get("alive_at_ack") is True and fault.get("alive_before_fault") is True,"N4 live observations missing")
        _require((fault["worker_pid"],fault["worker_created"])==(obs["pid"],obs["created"]),"N4 fault worker identity")
        _require(fault.get("binding")==binding and fault.get("gate_state")=="FAULT_COMMITTED"
                 and fault.get("trace_complete") is True and fault.get("cleanup_before_fault") is False,"N4 fault gate evidence")
        _require(fault.get("pre_fault_trace")==record["events"],"N4 incomplete fault trace")
        _require(record.get("cleanup_intent") is None and not any(e["kind"] in ("cleanup_intent","terminate_job","kill_process") for e in record["events"]),"N4 explicit cleanup before fault")
        _require(ready["ready_tick_ms"]<=obs["ack_tick_ms"]<=fault["alive_tick_ms"]<=fault["tick_ms"]
                 <obs["exit_tick_ms"]<fault["tick_ms"]+5000,"N4 fault/live/exit order or five-second bound")
        _require(trace["done"] is None,"hard exit has DONE")
        return
    _require(isinstance(inner,dict) and inner.get("attempts")==1 and inner.get("alive") is False,"inner attempt/alive")
    _require((inner["pid"],inner["exit_code"])==(obs["pid"],obs["exit_code"]),"inner observed exit differs")
    _require(inner.get("cleanup_confirmed") is True and inner.get("exit_confirmed") is True,"inner cleanup missing")
    _require((inner["start_tick_ms"],inner["deadline_tick_ms"])==(binding["start_tick_ms"],binding["deadline_tick_ms"]),"original budget changed")
    _require(_power_ok(record.get("parent_power")),"parent power release")
    if case_id=="N1":
        _require(inner["status"]=="COMPLETE" and inner["exit_code"]==0 and inner["end_handshake_verified"] is True,"normal completion")
        _require(trace["last_sequence"]>=3 and len(trace["grants"])>=3
                 and set(trace["grants"])==set(trace["acks"]),"two genuine renewal ACKs required")
        _require(trace["done"] is not None and inner["lease_sequence"]==trace["last_sequence"],"normal exact close")
        _require(inner["final_tick_ms"]<trace["close"][1],"normal reception after lease")
        _require(isinstance(closed,dict) and type(closed.get("work_elapsed_ms")) is int and closed["work_elapsed_ms"]>=45000,"45 seconds working missing")
    elif case_id=="N2":
        _require(inner["exit_code"]==23 and trace["done"] is None and closed is None,"worker fault exit/DONE")
        _require(inner.get("end_handshake_verified") is False,"false fault handshake")
        if inner["status"]=="FAILED":
            _require(any(e["kind"]=="pipe_error" and e.get("winerror")==109 for e in record["events"]),"FAILED not native broken-pipe path")
        else:
            _require(inner["status"]=="SUPERVISION_INTERRUPTED" and "close" in inner["error"].lower(),"wrong worker fault cause")
        return
    elif case_id=="N3":
        _require(inner["status"]=="SUPERVISION_INTERRUPTED" and "lease" in inner["error"].lower(),"linger is not MSB lease rejection")
        _require(trace["done"] is not None and inner.get("end_handshake_verified") is True,"linger lacks actual DONE")
        _require(inner["final_tick_ms"]>=trace["close"][1] and any(e["kind"]=="terminate_job" for e in record["events"]),"linger lacks frozen-lease kill")
        _require(isinstance(closed,dict) and closed.get("work_elapsed_ms")==0,"linger unexpectedly did work")
    _require(_power_ok(closed.get("worker_power")) and closed.get("worker_execution",{}).get("end_handshake_verified") is True,"worker actual release/close missing")


def classify_native_cases(case_records: list[dict], *, outer_evidence: dict) -> dict:
    errors=[]
    try:
        _require(isinstance(case_records,list) and [r.get("case_id") for r in case_records]==["N1","N2","N3","N4"],"four fixed cases required")
        binding=outer_evidence["binding"];_binding_budget(binding)
        _require(_power_ok(outer_evidence["root_power"]),"root guard not released")
        elapsed=outer_evidence["elapsed_seconds"]
        _require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<480,"root deadline")
        _require(outer_evidence["initial_settings"]==outer_evidence["final_settings"],"power settings changed")
        _check_final_continuity(outer_evidence["final_continuity"])
        for key in ("source_identity_verified","registration_verified"):
            _require(outer_evidence.get(key) is True,"missing "+key)
        _require(outer_evidence.get("real_data_opened") is False,"real input boundary")
        parents=set();workers=set()
        for case_id,record in zip(("N1","N2","N3","N4"),case_records):
            _check_case(record,case_id,binding)
            obs=record["observation"]
            parents.add((obs["parent_pid"],obs["parent_created"]));workers.add((obs["pid"],obs["created"]))
        _require(len(parents)==4 and len(workers)==4,"replayed process identity")
    except (KeyError,TypeError,ValueError,RuntimeError,IndexError,AttributeError) as error:
        errors.append(str(error))
    return dict(accepted=not errors,errors=errors,scope="FOUR_NATIVE_PATHS_ONLY_NOT_SCIENTIFIC_EFFECT")


def _clock_pair(sample):
    if (not isinstance(sample,(tuple,list)) or len(sample)!=2
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in sample)):
        raise ValueError("Invalid postflight clock sample")
    return tuple(sample)


def _check_final_continuity(record):
    _require(isinstance(record,dict) and record.get("verified") is True,"missing final continuity")
    _clock_pair(record["sample"])
    for name in ("gap_seconds","suspend_delta_seconds","elapsed_seconds"):
        value=record[name]
        _require(type(value) in (int,float) and math.isfinite(value),"invalid final continuity: "+name)
    _require(0<=record["gap_seconds"]<=30 and abs(record["suspend_delta_seconds"])<=2
             and 0<=record["elapsed_seconds"]<480,"postflight continuity limit")


class FinalContinuity:
    """Continue the released root guard's last sample without acquiring a new request."""
    def __init__(self,binding,last_sample,*,services):
        _binding_budget(binding)
        self.binding,self.last,self.services=copy.deepcopy(binding),_clock_pair(last_sample),services
    def check(self):
        _remaining(self.binding,self.services)
        sample=_clock_pair(self.services.postflight_sample())
        wall,active=(sample[i]-self.last[i] for i in (0,1))
        if wall<0 or active<0 or wall>30 or abs(wall-active)>2:
            raise RuntimeError("SUPERVISION_INTERRUPTED: postflight clock discontinuity")
        remaining=_remaining(self.binding,self.services)
        record=dict(verified=True,sample=list(sample),gap_seconds=wall,
                    suspend_delta_seconds=wall-active,elapsed_seconds=480.-remaining)
        _check_final_continuity(record);self.last=sample
        return record


class AcceptancePublicationError(RuntimeError):
    def __init__(self,state,primary,persistence,*,primary_errors=None):
        self.failed_state=copy.deepcopy(state)
        self.primary_errors=list(primary_errors) if primary_errors is not None else [primary]
        self.primary_error=self.primary_errors[0];self.persistence_error=persistence
        super().__init__(f"Native receipt not persisted: primary_type={type(primary).__name__}; persistence_type={type(persistence).__name__}; no verified delivery")


def publish_acceptance(output,state,binding,*,services,final_check=None):
    was_pass=state["status"]=="PASS_NATIVE_SUPERVISION_SCOPE"
    try:
        if was_pass:
            if not callable(final_check):raise RuntimeError("Postflight final_check required for PASS")
            _remaining(binding,services)
            observed=final_check();_check_final_continuity(observed)
            state.setdefault("postflight_checks",[]).append(observed)
        services.write_json(Path(output)/"stage_state.json",state)
        if was_pass:
            _remaining(binding,services)
            observed=final_check();_check_final_continuity(observed)
            state["postflight_checks"].append(observed)
        return state
    except BaseException as primary:
        if not was_pass:
            previous=RuntimeError(state.get("error","Native acceptance was not accepted"))
            state.update(status="FAILED_NATIVE_SUPERVISION",delivery_verified=False,failure_record_persisted=False)
            raise AcceptancePublicationError(state,previous,primary) from primary
        state.update(status="FAILED_NATIVE_SUPERVISION",delivery_verified=False,error=f"{type(primary).__name__}: {primary}")
        state.setdefault("errors",[]).append(dict(phase="publication",type=type(primary).__name__,message=str(primary)))
        try:services.write_json(Path(output)/"stage_state.json",state)
        except BaseException as secondary:
            state["failure_record_persisted"]=False
            raise AcceptancePublicationError(state,primary,secondary) from secondary
        return state


def retain_worker_identity(ready,*,services):
    if (not isinstance(ready,dict) or any(type(ready.get(k)) is not int or ready[k]<=0 for k in ("pid","created"))):
        raise RuntimeError("SUPERVISION_INTERRUPTED: invalid observation identity")
    handle=services.open_query(ready["pid"])
    try:
        if services.query_identity(handle)!=(ready["pid"],ready["created"]):
            raise RuntimeError("SUPERVISION_INTERRUPTED: PID creation identity changed")
        return handle
    except BaseException:
        services.close_query(handle);raise


class NativeObserver:
    def __init__(self,case_dir,binding,services):
        import threading
        self.case_dir,self.binding,self.services=case_dir,binding,services
        self.lock=threading.RLock();self.handle=None;self.watch=None;self.closed=False
        self.stop_result=None;self.stop_errors=[];self.close_error=None
        self.evidence=dict(identity_verified=False,query_only=True,job_handle_retained=False,
                           job_assignment_observed=False,exit_observed=False,handle_closed=False,
                           fallback_before_exit=False,worker_alive=True,monitor_stopped=False)
    def sample(self):
        with self.lock:
            _remaining(self.binding,self.services)
            if self.handle is None:
                try:
                    ready=_read_json_if_present(self.case_dir/"worker_ready.json")
                    controller=_read_json_if_present(self.case_dir/"controller_ready.json")
                    trace=_read_json_if_present(self.case_dir/"controller/trace.json")
                except json.JSONDecodeError:
                    return
                if ready is None or controller is None or trace is None:return
                _require(ready["case_id"]==self.case_dir.name and ready.get("working") is True and ready.get("job_assigned") is True,"observer WORKING")
                _require(controller["binding"]==self.binding,"observer root binding")
                assigned=any(e["kind"]=="assigned" and e.get("pid")==ready["pid"] for e in trace)
                ack=any(e["kind"]=="ACK" and e.get("sequence")==1 and e.get("worker_pid")==ready["pid"]
                        and e.get("parent_pid")==controller["pid"] for e in trace)
                spawn=any(e["kind"]=="spawn" and (e.get("pid"),e.get("created"))==(ready["pid"],ready["created"]) for e in trace)
                if not (assigned and ack and spawn):return
                self.handle=retain_worker_identity(ready,services=self.services)
                if self.services.query_alive(self.handle) is not True:
                    raise RuntimeError("SUPERVISION_INTERRUPTED: observer worker already dead before ACK")
                tick=self.services.tick_ms()
                _require(ready["ready_tick_ms"]<=tick<ready["ready_tick_ms"]+5000,"observer ACK outside ready window")
                self.evidence.update(ready=ready,pid=ready["pid"],created=ready["created"],
                    parent_pid=controller["pid"],parent_created=controller["created"],
                    identity_verified=True,job_assignment_observed=True,alive_at_ack=True,ack_tick_ms=tick)
                _write_json(self.case_dir/"collector_ack.json",dict(ready,collector_verified=True,
                    query_only=True,job_handle_retained=False,alive_at_ack=True,ack_tick_ms=tick),exclusive=True)
            if self.handle is not None and not self.evidence["exit_observed"]:
                alive=self.services.query_alive(self.handle);self.evidence["worker_alive"]=alive
                if not alive:self.evidence.update(exit_observed=True,exit_tick_ms=self.services.tick_ms(),
                                                  exit_code=self.services.query_exit_code(self.handle))
    def loop(self,stopped):
        while not stopped.is_set():
            self.sample();self.services.pause(.05)
    def close(self):
        result=self._stop()
        stopped=result is None or result.stop_ok
        if result is not None and not result.thread_terminated:
            self.evidence["monitor_stopped"]=False
            return False
        if self.close_error is not None:raise self.close_error
        if not self.lock.acquire(timeout=1.):
            self.evidence["monitor_stopped"]=False
            raise RuntimeError("SUPERVISION_INTERRUPTED: observer sampler lock did not release")
        try:
            if self.handle is not None:
                try:self.services.close_query(self.handle)
                except BaseException as error:
                    self.close_error=error
                    raise
                self.handle=None;self.evidence["handle_closed"]=True
            self.evidence["monitor_stopped"]=stopped;self.closed=True
        finally:self.lock.release()
        return stopped
    def _stop(self):
        if self.watch is None:return None
        if self.stop_result is not None:return self.stop_result
        self.stop_result,self.stop_errors=_bounded_fault_stop(self.watch,self.binding,self.services)
        return self.stop_result
    def finish(self,outer):
        observed=self.services.tick_ms()
        self.evidence["parent_exit_observed_tick_ms"]=observed
        result=self._stop();primary=list(self.stop_errors);cleanup=[]
        if result is not None and not result.stop_ok:
            if result.callback_error is not None:primary.insert(0,result.callback_error)
            if not primary:primary.append(RuntimeError("SUPERVISION_INTERRUPTED: observer termination not confirmed"))
        try:
            while not primary:
                self.sample()
                if self.evidence["exit_observed"]:break
                if self.services.tick_ms()-observed>=5000:break
                self.services.pause(.05)
        except BaseException as error:primary.append(error)
        try:self.close()
        except BaseException as error:cleanup.append(error)
        if primary or cleanup:raise ObserverTerminationError(result,primary,cleanup)
        return copy.deepcopy(self.evidence)


def _read_controller_failure(path: Path) -> dict | None:
    try:raw=path.read_text(encoding="utf-8")
    except FileNotFoundError:return None
    except (OSError,UnicodeError) as error:
        return dict(status="INVALID_CONTROLLER_FAILURE",error=_fault_error_record(error))
    try:
        value=json.loads(raw)
        if not isinstance(value,dict) or value.get("status")!="FAILED_CONTROLLER_TRACE_PUBLICATION":
            raise ValueError("Invalid controller failure record shape")
    except ValueError as error:
        return dict(status="INVALID_CONTROLLER_FAILURE",error=_fault_error_record(error))
    return value


def _load_case_record(case_dir,binding,outer,observation):
    failure=_read_controller_failure(case_dir/"controller/controller_failure.json")
    if failure is not None:
        return dict(case_id=case_dir.name,binding=copy.deepcopy(binding),outer=outer,observation=observation,
            msb=None,parent_power=None,events=[],cleanup_intent=None,worker_closed=None,fault=None,
            controller_failure=failure,diagnostic_only=True)
    inner=_read_json_if_present(case_dir/"controller/controller_result.json")
    fault=_read_json_if_present(case_dir/"parent_fault.json")
    events=_read_json_if_present(case_dir/"controller/trace.json") or []
    if case_dir.name=="N4":
        snapshot=_read_json_if_present(case_dir/"fault_trace.json")
        _require(isinstance(fault,dict) and isinstance(snapshot,list) and snapshot==fault.get("pre_fault_trace"),"N4 missing persisted fault snapshot")
        _require(bool(events) and events==snapshot[:len(events)],"N4 raw trace differs from fault snapshot")
        events=snapshot
    return dict(case_id=case_dir.name,binding=copy.deepcopy(binding),outer=outer,observation=observation,
        msb=None if inner is None else inner.get("msb"),
        parent_power=None if inner is None else inner.get("parent_power"),
        events=events,cleanup_intent=_read_json_if_present(case_dir/"cleanup_intent.json"),
        worker_closed=_read_json_if_present(case_dir/"controller/runs"/case_dir.name/"worker_closed.json"),
        fault=fault,controller_failure=None)


def run_acceptance(root: Path, protocol_path: Path, *, execute_native: bool, approval: dict, services=None) -> dict:
    if execute_native is not True:raise ValueError("Native execution requires a separate explicit approval")
    _check_approval(approval)
    if services is None:
        import time
        started=time.monotonic()
    else:started=services.monotonic()
    root=Path(root).resolve();_execution_root(root);protocol_path=_protocol_path(root,protocol_path)
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_native_protocol(root,protocol,approval=approval)
    services=NativeServices(root) if services is None else services
    tick=services.tick_ms()
    binding=dict(run_id=services.new_run_id(),protocol_digest=_protocol_digest(protocol),
                 start_monotonic=started,start_tick_ms=tick,deadline_tick_ms=tick+7200000,
                 case_limits=[90,45,75,45])
    guard=services.collector_guard();output=root/protocol["output_relative"];state=None
    initial=None;errors=[];continuity=None
    def check():
        _remaining(binding,services)
        guard.check()
        return _remaining(binding,services)
    try:
        with guard:
            check();initial=services.power_settings();check()
            registration=services.verify_registration(root,protocol_path,protocol,approval,check)
            _require(registration.get("verified") is True and registration.get("commit")==approval["registration_commit"],"registration not verified")
            services.verify_sources(root,protocol);check()
            runtime=services.capture_runtime(protocol["runtime_venv_root"]);check()
            output.mkdir(parents=True,exist_ok=False)
            state=dict(status="RUNNING",active_case=None,binding=binding,registered_commit=registration["commit"],
                       registration_verified=True,source_identity_verified=True,case_records=[],
                       real_data_opened=False,real_fits=0,model_forwards=0,
                        native_os_exercised=not getattr(services,"simulated",False),delivery_verified=False,
                        python_runtime=runtime)
            try:
                services.write_json(output/"registered_protocol.json",protocol,exclusive=True)
                services.write_json(output/"stage_state.json",state,exclusive=True);check()
                for case in CASES:
                    check();services.verify_sources(root,protocol);check()
                    case_dir=output/"cases"/case["case_id"];case_dir.mkdir(parents=True,exist_ok=False)
                    state["active_case"]=case["case_id"];services.write_json(output/"stage_state.json",state);check()
                    origin=_execution_root(root)
                    command=[sys.executable,origin["driver"],"--root",str(root),
                             "--protocol",str(protocol_path),"--execute-native",
                             "--approval-call-id",approval["call_id"],"--approval-answer",approval["answer"],
                             "--registration-commit",approval["registration_commit"],
                             "--controller-case",case["case_id"],"--case-output",str(case_dir),
                             "--binding-json",json.dumps(binding)]
                    job=dict(job_id=case["case_id"],case_id=case["case_id"],case_dir=str(case_dir),
                             binding=copy.deepcopy(binding),command=command,log_path=str(case_dir/"controller.log"),
                             python_launch_contract=protocol["python_launch_contract"])
                    observer=services.start_observer(case_dir,binding)
                    try:
                        remaining=check()
                        outer=services.run_outer(job,dict(per_fit_seconds=min(case["limit"],remaining),heartbeat_seconds=10),remaining)
                        observation=services.finish_observer(observer,outer)
                    except BaseException as primary:
                        if isinstance(primary,ObserverTerminationError):
                            state["observer_diagnostics"]=primary.record()
                            raise
                        close_returned=None;close_error=None
                        try:close_returned=services.stop_observer(observer)
                        except BaseException as cleanup:close_error=cleanup
                        combined=_observer_abort_error(primary,
                            stop_result=getattr(observer,"stop_result",None),
                            stop_errors=getattr(observer,"stop_errors",[]),
                            close_returned=close_returned,close_error=close_error)
                        state["observer_diagnostics"]=combined.record()
                        raise combined from None
                    check();record=_load_case_record(case_dir,binding,outer,observation);check()
                    state["case_records"].append(record)
                    _check_case(record,case["case_id"],binding);check()
                    services.write_json(output/"stage_state.json",state);check()
                check();services.verify_sources(root,protocol);check()
                state["final_settings"]=services.power_settings();check()
                _require(initial==state["final_settings"],"Power settings changed")
                check()
            except BaseException as error:
                if isinstance(error,ObserverTerminationError):
                    errors.extend(("observer_primary",e) for e in error.primary_errors)
                    errors.extend(("observer_cleanup",e) for e in error.cleanup_errors)
                else:errors.append(("body",error))
    except BaseException as error:
        if state is None:raise
        errors.append(("collector_scope_or_release",error))
    try:
        state["root_power"]=guard.evidence()
        state["initial_settings"]=initial
        elapsed=max(services.monotonic()-started,(services.tick_ms()-tick)/1000.)
        if not math.isfinite(elapsed) or elapsed<0:raise ValueError("Invalid collector elapsed time")
        state["elapsed_seconds"]=elapsed
        if not errors:
            continuity=FinalContinuity(binding,guard.last,services=services)
            state["final_continuity"]=continuity.check()
            state["elapsed_seconds"]=state["final_continuity"]["elapsed_seconds"]
            evidence=dict(binding=binding,elapsed_seconds=state["elapsed_seconds"],root_power=state["root_power"],
                          initial_settings=initial,final_settings=state["final_settings"],registration_verified=True,
                          source_identity_verified=True,real_data_opened=False,final_continuity=state["final_continuity"])
            state["assessment"]=classify_native_cases(state["case_records"],outer_evidence=evidence)
            state["status"]="PASS_NATIVE_SUPERVISION_SCOPE" if state["assessment"]["accepted"] else "INSUFFICIENT_EVIDENCE"
            state["delivery_verified"]=state["assessment"]["accepted"]
    except BaseException as error:
        errors.append(("postflight",error))
    if errors:
        def error_message(phase,error):
            return _fault_error_record(error)["message"] if phase.startswith("observer_") else str(error)
        state.update(status="FAILED_NATIVE_SUPERVISION",delivery_verified=False,
                     error="; ".join(f"{type(e).__name__}: {error_message(p,e)}" for p,e in errors),
                     errors=[dict(phase=p,type=type(e).__name__,message=error_message(p,e)) for p,e in errors])
    try:
        return publish_acceptance(output,state,binding,services=services,
                                  final_check=None if continuity is None else continuity.check)
    except AcceptancePublicationError as error:
        if not errors:raise
        primary=[e for _,e in errors]
        raise AcceptancePublicationError(error.failed_state,primary[0],error.persistence_error,
                                         primary_errors=primary) from error


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
    parser.add_argument("--controller-case",choices=("N1","N2","N3","N4"))
    parser.add_argument("--case-output",type=Path)
    parser.add_argument("--binding-json")
    args=parser.parse_args(argv)
    if args.worker_packet is not None:
        if (args.root is not None or args.protocol is not None or args.execute_native
                or args.approval_call_id or args.approval_answer or args.registration_commit
                or args.worker_handles is None or args.controller_case or args.case_output or args.binding_json):
            parser.error("Private inherited worker mode cannot mix public flags or omit handles")
        handles=json.loads(args.worker_handles)
        if (not isinstance(handles,dict) or set(handles)!={"read_handle","write_handle","parent_handle","nonce"}
                or any(type(handles[k]) is not int or handles[k]<=0 for k in ("read_handle","write_handle","parent_handle"))
                or not re.fullmatch(r"[a-f0-9]{64}",str(handles["nonce"]))):
            raise ValueError("Exact positive inherited capability required before service construction")
        packet=_worker_packet(args.worker_packet,handles)
        result=run_no_data_worker(args.worker_packet,handles,services=NativeServices(Path(packet["root"])))
        print(json.dumps(result),flush=True)
        return 0
    if args.worker_handles is not None or args.protocol is None:
        parser.error("Explicit protocol required for collector; no detached worker handles")
    approval=dict(call_id=args.approval_call_id,answer=args.approval_answer,
                  registration_commit=args.registration_commit,
                  scope="NATIVE_SUPERVISION_NO_DATA",confirmed_from_human=True)
    if args.controller_case:
        if not args.execute_native or args.root is None or args.case_output is None or args.binding_json is None:
            parser.error("Internal controller requires the exact approved collector claim")
        _check_approval(approval)
        root=args.root.resolve();_execution_root(root);path=_protocol_path(root,args.protocol)
        protocol=json.loads(path.read_text(encoding="utf-8"))
        _validate_protocol_fields(protocol,approval=approval)
        binding=json.loads(args.binding_json);_binding_budget(binding)
        _case_output(root,args.case_output,args.controller_case)
        claim=json.loads((root/protocol["output_relative"]/"stage_state.json").read_text(encoding="utf-8"))
        if (claim.get("status")!="RUNNING" or claim.get("active_case")!=args.controller_case
                or claim.get("binding")!=binding or binding["protocol_digest"]!=_protocol_digest(protocol)
                or claim.get("registration_verified") is not True or claim.get("source_identity_verified") is not True
                or claim.get("registered_commit")!=approval["registration_commit"]):
            raise ValueError("Controller lacks the active registered collector claim")
        services=NativeServices(root);services.verify_sources(root,protocol)
        try:result=run_controller(root,args.case_output,args.controller_case,binding,services=services)
        except ControllerTracePublicationError as error:
            try:print(json.dumps(error.record(),allow_nan=False),flush=True)
            except BaseException:return 1
            return 1
        if args.controller_case=="N4":raise RuntimeError("Parent hard-exit scenario returned instead of exiting31")
        inner=result["msb"]
        valid=(inner["status"]=="COMPLETE" if args.controller_case=="N1" else
               inner["status"] in ("FAILED","SUPERVISION_INTERRUPTED") and inner["exit_code"]==23
               if args.controller_case=="N2" else inner["status"]=="SUPERVISION_INTERRUPTED" and "lease" in inner["error"])
        if not valid:raise RuntimeError("Unexpected inner case outcome")
        print(json.dumps(result),flush=True);return 0
    if args.case_output is not None or args.binding_json is not None:
        parser.error("No detached controller arguments")
    result=run_acceptance(args.root or Path(__file__).resolve().parents[1],args.protocol,
                          execute_native=args.execute_native,approval=approval)
    print(json.dumps(result),flush=True)
    return 0 if result.get("status")=="PASS_NATIVE_SUPERVISION_SCOPE" else 1


if __name__=="__main__":
    raise SystemExit(main())
