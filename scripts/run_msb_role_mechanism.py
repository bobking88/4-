"""Exclusive MSB execution: synthetic simulation or separately approved fit-only run."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import torch

from check_lc_native_environment import NativeEnvironmentAPI
from lc_execution_safety import WindowsAwakeGuard, stage_elapsed_seconds
from run_lc_role_adaptation import run_supervised_job, terminal_status
from msb_execution_safety import NativeMSBAPI, MSBParentSupervisor, MSBWorkerScope, RunBudget, open_worker_scope
from run_ic_role_mechanism import SOURCE_FILES as LEGACY_SOURCES
from msb_role_data import (PREPARED_BASE, file_digest, load_prepared_fold, validate_fold_collection,
                           prepared_identity_summary, validate_identity_collection)
from msb_role_training import build_fit_matrix, fit_convex_fold, fit_trajectory

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = "docs/superpowers/specs/2026-10-14_margin_scaled_bilinear_role_evidence_design.md"
SPEC_SHA = "f656d2c73826deda4b850b30a94e9b78f5c3b9dabfdd873157bfcbff6681df42"
REPAIR_SPEC_PATH = "docs/superpowers/specs/2026-10-16-msb-execution-contract-repair-design.md"
REPAIR_SPEC_SHA = "ec55fc59d9f2be2de6f1d0021573094c5d354cae6e3c9cb68d1090ae3b9ec21a"
TERMINAL_SPEC_PATH = "docs/superpowers/specs/2026-10-17-msb-terminal-boundary-repair-design.md"
TERMINAL_SPEC_SHA = "f734ebb2f0d7583b993fabc34a21868ff2bf397e7fad250547193cd89514856f"
SOURCE_FILES = tuple(sorted(set(LEGACY_SOURCES) | {f"scripts/{name}.py" for name in (
    "msb_role_data", "msb_role_model", "msb_role_controls", "msb_role_audit", "msb_role_training",
    "run_msb_role_mechanism", "analyze_msb_role_mechanism", "generate_msb_role_figure", "msb_execution_safety")}))
CODE_ONLY_IDS = ("call_978b15d460804fa2982cb750a1d4b4ad", "call_53a58e64d5ee47fbaba8b7f127ec10b0",
                 "call_b71499819f6c4a8f9f7d2c966238fe40", "call_RHuePD3CQpoUvp4t88mzQnSM",
                 "call_1690e5bbad7a415f8880b0ff473b9521", "call_3d401d8b809843e0b7103995a772caad",
                 "call_bd72917caf2e42eba5bdca8b34c65841", "call_4ada24f023d34e1e8b2768bba9295873",
                 "call_358af6a530944d5c9a15f0ea8039b545", "call_a697651b584d4fb8b2116607dcc82b52",
                 "call_2a665eb0bae044b38917a4aa34f5127a", "call_34d91895cd594efe92c80a1bf6f41138",
                 "call_a4c9ef81bcea447aa7350f7cd9aff51c", "call_aa6d081524a94817b20cceb8fbd25053",
                 "call_0e89f2cb80134c049e4ef3aaf697a7b0")
KINDS = ("load", "trajectory", "convex", "analysis", "final-verify")


def protocol_digest(protocol):
    data = json.dumps(protocol, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _path(root, path):
    root = Path(root).resolve()
    original = Path(path)
    if not original.is_absolute(): original = root/original
    resolved = original.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Path escapes the exclusive MSB scope.")
    relative = original.relative_to(root)
    node = root
    for part in relative.parts:
        node = node/part
        if node.is_symlink(): raise ValueError("Symlink path forbidden in MSB scope.")
    return resolved


def _json(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        return
    temporary = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class MSBTerminalPublicationError(RuntimeError):
    def __init__(self, failed_state: dict, *, primary_error: BaseException | None, persistence_error: BaseException):
        self.failed_state = dict(failed_state)
        self.primary_error = primary_error
        self.persistence_error = persistence_error
        reason = primary_error if primary_error is not None else failed_state.get("error", "unknown prior failure")
        super().__init__(f"Terminal failure record not persisted; status={failed_state.get('status', 'FAILED')}; "
                         f"primary={reason}; persistence={persistence_error}; no verified delivery")


def _publish_terminal_state(output: Path, state: dict, *, budget: RunBudget, api) -> dict:
    primary_error = None
    history = state.setdefault("state_history", [])
    if state["status"] in ("SYNTHETIC_COMPLETE", "MECHANISM_COMPLETE"):
        try:
            budget.remaining_seconds(api.tick_ms())
            if not history or history[-1] != state["status"]: history.append(state["status"])
            state["elapsed_seconds"] = budget.elapsed_seconds(api.tick_ms())
            state["elapsed_sample_phase"] = "before_terminal_publish"
            _json(Path(output)/"stage_state.json", state)
            budget.remaining_seconds(api.tick_ms())
            return state
        except BaseException as error:
            primary_error = error
            state["status"] = terminal_status(error)
            state["error"] = f"{type(error).__name__}: {error}"
            state["delivery_verified"] = False
    state["delivery_verified"] = False
    if not history or history[-1] != state["status"]: history.append(state["status"])
    state["elapsed_seconds"] = max(0, api.tick_ms()-budget.start_tick_ms)/1000.
    state["elapsed_sample_phase"] = "failure_cleanup_before_write"
    try:
        _json(Path(output)/"stage_state.json", state)
    except BaseException as error:
        state["failure_record_persisted"] = False
        raise MSBTerminalPublicationError(state, primary_error=primary_error, persistence_error=error) from error
    return state


def synthetic_protocol():
    return dict(protocol="msb_rre_v1", status="SYNTHETIC_ONLY", execution_contract="msb_parent_bound_v2",
                terminal_spec_sha256=TERMINAL_SPEC_SHA, max_seconds=7200,
                per_fit_seconds=7200, heartbeat_seconds=15, max_neural_trajectories=63,
                max_convex_fits=12, max_temperatures=0, max_updates=400, primary_checkpoint=100,
                max_stage_jobs=80, final_validation_forwards=615, final_train_certificate_forwards=12)


def runtime_environment():
    return dict(python=sys.version.split()[0], torch=str(torch.__version__), dtype="float64", device="cpu")


def _read_protocol(root, protocol_path, stage):
    if stage not in ("synthetic","fit-mechanism"):
        raise ValueError("Unknown execution stage before any file read.")
    if protocol_path is None:
        return synthetic_protocol() if stage == "synthetic" else {}
    path = _path(root,protocol_path)
    relative = path.relative_to(Path(root).resolve()).as_posix()
    if not re.fullmatch(r"docs/experiment_protocols/msb_role_mechanism_v[1-9][0-9]*\.json",relative):
        raise ValueError("Only a scoped MSB protocol JSON may be read.")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(root: Path, protocol: dict, stage: str) -> None:
    if stage not in ("synthetic", "fit-mechanism") or not isinstance(protocol, dict):
        raise ValueError("Only explicit synthetic or fit-mechanism stages exist.")
    expected = synthetic_protocol()
    for key, value in expected.items():
        if key != "status" and (type(protocol.get(key)) is not type(value) or protocol[key] != value):
            raise ValueError("Fixed MSB matrix or cumulative budgets changed.")
    if stage == "synthetic":
        if protocol.get("status") != "SYNTHETIC_ONLY" or set(protocol) != set(expected):
            raise ValueError("Synthetic stage cannot contain real inputs or approvals.")
        return
    allowed = set(expected) | {"authorization", "prepared_entries", "source_sha256", "spec_sha256",
                               "protocol_path", "output_relative", "environment", "repair_spec_sha256"}
    auth = protocol.get("authorization", {})
    if (set(protocol) != allowed or protocol.get("status") != "APPROVED_REAL_ONCE"
            or auth.get("human_approved") is not True or auth.get("scope") != "msb_rre_v1_fit_only_once"
            or not isinstance(auth.get("user_message_id"),str) or not auth["user_message_id"]
            or any(token in auth["user_message_id"] for token in CODE_ONLY_IDS)):
        raise ValueError("Fresh real MSB approval required; old and code-only approvals are invalid.")
    if Path(root).resolve() != ROOT:
        raise ValueError("Real execution must use this registered code checkout.")
    if (not re.fullmatch(r"docs/experiment_protocols/msb_role_mechanism_v[1-9][0-9]*\.json", protocol["protocol_path"])
            or not re.fullmatch(r"outputs/training/msb_role_mechanism_v[1-9][0-9]*/development_fold_0", protocol["output_relative"])):
        raise ValueError("New MSB protocol/output identities required.")
    for field in ("protocol_path", "output_relative"):
        _path(root, protocol[field])
    if set(protocol["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Complete MSB execution source closure required.")
    for name, expected_hash in protocol["source_sha256"].items():
        if file_digest(_path(root, name)) != expected_hash:
            raise ValueError("Registered execution source changed.")
    spec = _path(root, SPEC_PATH).read_text(encoding="utf-8")
    if protocol["spec_sha256"] != SPEC_SHA or hashlib.sha256(spec.encode("utf-8")).hexdigest() != SPEC_SHA:
        raise ValueError("Approved LF-canonical specification changed.")
    repair = _path(root,REPAIR_SPEC_PATH).read_text(encoding="utf-8")
    if protocol["repair_spec_sha256"] != REPAIR_SPEC_SHA or hashlib.sha256(repair.encode("utf-8")).hexdigest() != REPAIR_SPEC_SHA:
        raise ValueError("Approved execution repair specification changed.")
    terminal = _path(root,TERMINAL_SPEC_PATH).read_text(encoding="utf-8")
    if hashlib.sha256(terminal.encode("utf-8")).hexdigest() != TERMINAL_SPEC_SHA:
        raise ValueError("Approved terminal specification changed.")
    entries = protocol["prepared_entries"]
    required = {"fold", "relative_path", "sha256", "registered_train_ids_digest", "registered_validation_ids_digest"}
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("Three registered existing prepared folds required.")
    for fold, entry in enumerate(entries):
        if (set(entry) != required or type(entry.get("fold")) is not int or entry["fold"] != fold
                or entry["relative_path"] != f"{PREPARED_BASE}/prepare_fold{fold}/result.pt"
                or any(not re.fullmatch(r"[a-f0-9]{64}", entry[key]) for key in required if key.endswith("digest") or key == "sha256")):
            raise ValueError("Prepared scope or identity registration changed.")
        _path(root, entry["relative_path"])
    if protocol["environment"] != runtime_environment():
        raise ValueError("Registered runtime environment changed.")


def _registered(root, protocol_path, protocol, *, deadline_check):
    if protocol_path.resolve() != _path(root, protocol["protocol_path"]):
        raise ValueError("Protocol file identity differs from registration.")
    def api(endpoint):
        remaining = deadline_check()
        result = subprocess.run(["C:/Program Files/GitHub CLI/gh.exe", "api", endpoint],
                                capture_output=True, check=True, timeout=min(90,remaining))
        deadline_check()
        return json.loads(result.stdout)
    repo = "repos/bobking88/4-"
    ref = api(repo+"/git/ref/heads/codex/theory-aware-report")["object"]["sha"]
    tree = api(repo+f"/git/trees/{ref}?recursive=1")
    if tree.get("truncated"): raise ValueError("GitHub source tree evidence incomplete.")
    blobs = {row["path"]:row["sha"] for row in tree["tree"] if row["type"] == "blob"}
    payload = api(repo+"/git/blobs/"+blobs[protocol["protocol_path"]])
    remote_protocol = json.loads(base64.b64decode(payload["content"]))
    if protocol_digest(remote_protocol) != protocol_digest(protocol):
        raise ValueError("Real approval is not registered on GitHub.")
    for name in SOURCE_FILES:
        remaining = deadline_check()
        blob = subprocess.run(["git", "hash-object", "--path="+name, str(Path(root)/name)],
                              cwd=root, capture_output=True, text=True, check=True, timeout=min(30,remaining)).stdout.strip()
        deadline_check()
        if blob != blobs.get(name): raise ValueError("Execution source differs from GitHub registration.")
    return dict(commit=ref, protocol_digest=protocol_digest(protocol))


def require_ac_power():
    if NativeEnvironmentAPI().power_status().ACLineStatus != 1:
        raise RuntimeError("SUPERVISION_INTERRUPTED: AC power required")


def _schedule():
    jobs = [dict(kind="load", fold=fold, job_id=f"load_fold{fold}") for fold in range(3)]
    return jobs + build_fit_matrix() + [dict(kind="analysis",job_id="analysis"),dict(kind="final-verify",job_id="final_verify")]


def _job(output, packet):
    folder = output/"runs"/packet["job_id"]
    folder.mkdir(parents=True, exist_ok=False)
    packet_path = folder/"packet.json"
    _json(packet_path, packet, exclusive=True)
    suffix = "result.json" if packet["stage"] == "synthetic" or packet["kind"] in ("load","analysis","final-verify") else "result.pt"
    return dict(job_id=packet["job_id"], log_path=str(folder/"worker.log"),
                command=[sys.executable, str(Path(__file__).resolve()), "--worker-packet", str(packet_path)],
                result_path=str(folder/suffix), packet_path=str(packet_path))


def _preflight_payload(output,protocol,state,check):
    summaries,artifacts = [],[]
    if len(protocol.get("prepared_entries",[])) != 3: raise ValueError("Preflight registration missing.")
    for fold in range(3):
        check()
        relative = f"runs/load_fold{fold}/result.json"
        path = _path(output,relative)
        digest = file_digest(path)
        if digest != state.get("artifacts",{}).get(relative): raise ValueError("Preflight load artifact hash mismatch.")
        result = json.loads(path.read_text(encoding="utf-8"))
        execution = result.get("worker_execution",{})
        if (result.get("status") != "COMPLETE" or result.get("kind") != "load"
                or result.get("inner_fold") != fold or result.get("worker_power",{}).get("power_request_released") is not True
                or execution.get("parent_binding_verified") is not True or execution.get("job_assigned") is not True
                or execution.get("start_tick_ms") != state["start_tick_ms"]
                or execution.get("deadline_tick_ms") != state["deadline_tick_ms"]):
            raise ValueError("Preflight child supervision evidence missing.")
        summary = result.get("identity_summary")
        if not isinstance(summary,dict) or summary.get("prepared_entry") != protocol["prepared_entries"][fold]:
            raise ValueError("Preflight input registration differs.")
        summaries.append(summary); artifacts.append(dict(fold=fold,relative_path=relative,sha256=digest))
    identity = validate_identity_collection(summaries)
    check()
    return dict(schema="msb_joint_preflight_v1",run_id=state["run_id"],protocol_digest=protocol_digest(protocol),
                start_tick_ms=state["start_tick_ms"],deadline_tick_ms=state["deadline_tick_ms"],
                prepared_entries=protocol["prepared_entries"],load_artifacts=artifacts,**identity)


def write_joint_preflight(output: Path, protocol: dict, state: dict, supervisor) -> dict:
    if (state.get("preflight_status") != "PENDING" or state.get("actual_neural_fits") != 0
            or state.get("actual_convex_fits") != 0
            or [j.get("job_id") for j in state.get("jobs",[])] != [f"load_fold{i}" for i in range(3)]):
        raise ValueError("Joint preflight must precede every fit.")
    certificate = _preflight_payload(output,protocol,state,supervisor.check)
    path = _path(output,"joint_preflight.json")
    supervisor.check(); _json(path,certificate,exclusive=True); supervisor.check()
    digest = file_digest(path)
    state["artifacts"]["joint_preflight.json"] = digest
    return dict(preflight_status="PASSED",preflight_sha256=digest)


def require_joint_preflight(output: Path, protocol: dict, packet: dict, scope: MSBWorkerScope) -> dict:
    if not isinstance(scope,MSBWorkerScope) or scope.phase != "WORKING":
        raise RuntimeError("Preflight requires the active MSB worker capability.")
    scope.check()
    state = json.loads((_path(output,"stage_state.json")).read_text(encoding="utf-8"))
    if (state.get("status") != "RUNNING" or state.get("preflight_status") != "PASSED"
            or state.get("active_job") != packet.get("job_id")
            or state.get("protocol_digest") != protocol_digest(protocol)
            or scope.binding.run_id != state.get("run_id")
            or scope.binding.protocol_digest != state.get("protocol_digest")
            or scope.binding.budget.start_tick_ms != state.get("start_tick_ms")
            or scope.binding.budget.deadline_tick_ms != state.get("deadline_tick_ms")
            or packet.get("preflight_sha256") != state.get("preflight_sha256")):
        raise ValueError("Preflight has no valid current run binding.")
    path = _path(output,"joint_preflight.json")
    if file_digest(path) != state.get("preflight_sha256") or state["artifacts"].get("joint_preflight.json") != state["preflight_sha256"]:
        raise ValueError("Joint preflight certificate hash mismatch.")
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate != _preflight_payload(output,protocol,state,scope.check):
        raise ValueError("Joint preflight certificate semantics differ.")
    scope.check()
    return certificate


def run_stage(root, protocol_path, output, stage):
    api = NativeMSBAPI()
    started = api.tick_ms()
    budget = RunBudget(started,started+7_200_000)
    root = Path(root).resolve()
    output = _path(root, output)
    protocol = _read_protocol(root,protocol_path,stage)
    validate_protocol(root, protocol, stage)
    if output.exists(): raise FileExistsError("Exclusive output already exists; no resume or retry.")
    if stage == "fit-mechanism" and output != _path(root, protocol["output_relative"]):
        raise ValueError("Output differs from newly approved protocol.")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="PREPARING", state_history=["PREPARING"], stage=stage,
                 protocol_digest=protocol_digest(protocol), jobs=[], artifacts={}, packets={},
                 actual_neural_fits=0, actual_convex_fits=0, real_data_opened=False,
                 delivery_verified=False, alive=False, attempt=1, registration=None,
                 run_id=uuid.uuid4().hex,start_tick_ms=started,deadline_tick_ms=budget.deadline_tick_ms,
                 preflight_status="PENDING",preflight_sha256=None)
    _json(output/"stage_state.json", state, exclusive=True)
    parent = None
    try:
        remaining = lambda: budget.remaining_seconds(api.tick_ms())
        remaining()
        if stage == "fit-mechanism": state["registration"] = _registered(root, Path(protocol_path), protocol,deadline_check=remaining)
        remaining()
        parent_pid,parent_created = api.current_identity()
        require_ac_power()
        with MSBParentSupervisor(budget,state["run_id"],state["protocol_digest"],api=api) as parent:
            state["status"] = "RUNNING"; state["state_history"].append("RUNNING")
            _json(output/"stage_state.json", state)
            for item in _schedule():
                parent.check()
                if item["kind"] not in ("load",) and state["preflight_status"] == "PENDING":
                    if stage == "fit-mechanism":
                        state.update(write_joint_preflight(output,protocol,state,parent))
                    else:
                        state["preflight_status"] = "SYNTHETIC_NO_DATA"
                    _json(output/"stage_state.json",state)
                packet = dict(item, root=str(root), output=str(output), stage=stage,
                              protocol_path=None if protocol_path is None else str(Path(protocol_path).resolve()),
                              protocol_digest=state["protocol_digest"],run_id=state["run_id"],
                              start_tick_ms=started,deadline_tick_ms=budget.deadline_tick_ms,
                              parent_pid=parent_pid,parent_created=parent_created,
                              preflight_sha256=state["preflight_sha256"])
                job = _job(output, packet)
                state["active_job"] = item["job_id"]
                state["packets"][item["job_id"]] = file_digest(Path(job["packet_path"]))
                _json(output/"stage_state.json", state)
                parent.check()
                result = parent.run_job(job, protocol)
                state["jobs"].append(dict(result, kind=item["kind"], unit=item.get("unit"),
                                          seed=item.get("seed"), fold=item.get("fold")))
                if result.get("status") != "COMPLETE": raise RuntimeError(result.get("status","FAILED")+": worker failed")
                if (result.get("alive") or result.get("attempts") != 1
                        or result.get("parent_binding_verified") is not True or result.get("job_assigned") is not True
                        or result.get("end_handshake_verified") is not True or result.get("exit_confirmed") is not True):
                    raise RuntimeError("SUPERVISION_INTERRUPTED: worker completion identity invalid")
                if not result.get("supervision",{}).get("power_request_released"):
                    raise RuntimeError("POWER_RELEASE_FAILED: worker request not released")
                parent.check()
                destination = _path(output, job["result_path"])
                state["artifacts"][destination.relative_to(output).as_posix()] = file_digest(destination)
                if stage == "fit-mechanism":
                    if item["kind"] == "load": state["real_data_opened"] = True
                    if item["kind"] == "trajectory": state["actual_neural_fits"] += 1
                    if item["kind"] == "convex": state["actual_convex_fits"] += 1
                    if item["kind"] == "final-verify":
                        evidence = json.loads(destination.read_text(encoding="utf-8"))
                        if evidence.get("delivery_verified") is not True: raise ValueError("Final verification evidence missing.")
                        state["delivery_verified"] = True
                state["elapsed_seconds"] = budget.elapsed_seconds(api.tick_ms())
                _json(output/"stage_state.json",state)
                parent.check()
        parent_power = parent.evidence()
        if not parent_power.get("power_request_acquired") or not parent_power.get("power_request_released"):
            raise RuntimeError("POWER_RELEASE_FAILED: parent request evidence incomplete")
        state["status"] = "MECHANISM_COMPLETE" if stage == "fit-mechanism" else "SYNTHETIC_COMPLETE"
        remaining()
        state["power"] = parent_power
    except BaseException as error:
        state["status"] = terminal_status(error)
        state["error"] = f"{type(error).__name__}: {error}"
        state["delivery_verified"] = False
    state["power"] = {} if parent is None else parent.evidence()
    return _publish_terminal_state(output,state,budget=budget,api=api)


def worker(packet_path, *, handles=None):
    api = NativeMSBAPI() if handles else None
    if api is None: raise ValueError("Worker requires an inherited live parent channel.")
    awake = api.awake_guard()
    with open_worker_scope(packet_path,handles,api=api,awake_guard=awake) as scope:
        with awake:
            scope.begin_work()
            result,destination,json_result = _worker_body(packet_path,scope)
        worker_power = awake.evidence()
        scope.begin_publish(worker_power)
        result.update(status="COMPLETE",worker_power=worker_power,worker_execution=scope.evidence())
        scope.check()
        if json_result: _json(destination,result,exclusive=True)
        else:
            with destination.open("xb") as stream: torch.save(result,stream)
        scope.check()
    return result


def _worker_body(packet_path, scope):
    scope.check()
    packet_path = Path(packet_path).resolve()
    if packet_path.suffix != ".json": raise ValueError("Only worker JSON metadata may be read.")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("kind") not in KINDS: raise ValueError("No raw preparation, stop, refit or training replay jobs.")
    root, output = Path(packet["root"]).resolve(), Path(packet["output"]).resolve()
    _path(root,output); _path(output,packet_path)
    stage = packet["stage"]
    protocol = _read_protocol(root,packet.get("protocol_path"),stage)
    validate_protocol(root,protocol,stage)
    state = json.loads((output/"stage_state.json").read_text(encoding="utf-8"))
    if (state.get("status") != "RUNNING" or state.get("stage") != stage
            or state.get("active_job") != packet["job_id"] or state.get("protocol_digest") != protocol_digest(protocol)
            or packet.get("protocol_digest") != state["protocol_digest"]
            or state["packets"].get(packet["job_id"]) != file_digest(packet_path)):
        raise ValueError("Worker requires a published active parent and immutable packet.")
    expected = next((row for row in _schedule() if row["job_id"] == packet["job_id"]),None)
    if expected is None or any(packet.get(k) != v for k,v in expected.items()):
        raise ValueError("Worker schedule changed.")
    folder = _path(output,output/"runs"/packet["job_id"])
    if packet_path != folder/"packet.json": raise ValueError("Worker packet path identity changed.")
    json_result = stage == "synthetic" or packet["kind"] in ("load","analysis","final-verify")
    destination = folder/("result.json" if json_result else "result.pt")
    if destination.exists(): raise FileExistsError("Worker cannot overwrite or replay results.")
    if stage == "fit-mechanism" and packet["kind"] != "load":
        require_joint_preflight(output,protocol,packet,scope)
    _json(folder/"task_started.json",dict(status="ATTEMPT_CONSUMED",kind=packet["kind"]),exclusive=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    scope.check()
    if stage == "synthetic":
        result = dict(status="COMPLETE",scope="SYNTHETIC_CONTROL_PLANE_ONLY",kind=packet["kind"],
                      real_data_opened=False,delivery_verified=False)
    elif packet["kind"] in ("load","trajectory","convex"):
        scope.check()
        entry = protocol["prepared_entries"][packet["fold"]]
        data = load_prepared_fold(root,entry,protocol["authorization"])
        if packet["kind"] == "load":
            result = dict(inner_fold=packet["fold"],identity_summary=prepared_identity_summary(data,packet["fold"],entry))
        elif packet["kind"] == "trajectory":
            result = fit_trajectory(data["train"],data["validation"],packet["unit"],packet["seed"])
        else:
            result = fit_convex_fold(data["train"],data["validation"],packet["unit"],deadline_check=scope.check)
    else:
        from analyze_msb_role_mechanism import collect_saved_predictions, supervised_analysis, verify_saved_mechanism, write_mechanism_report
        with supervised_analysis(root,output,protocol,scope,packet["kind"]):
            if packet["kind"] == "analysis":
                evidence = collect_saved_predictions(root,output,protocol)
                result = write_mechanism_report(evidence,output/"analysis_summary")
            else:
                evidence = verify_saved_mechanism(root,output,protocol)
                result = write_mechanism_report(evidence,output/"verified_summary")
    result["kind"] = packet["kind"]
    return result,destination,json_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-packet",type=Path)
    parser.add_argument("--worker-handles")
    parser.add_argument("--stage",choices=("synthetic","fit-mechanism"))
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.worker_packet:
        if args.stage or args.protocol or args.output: parser.error("Worker and public flags cannot be combined.")
        result = worker(args.worker_packet,handles=None if args.worker_handles is None else json.loads(args.worker_handles))
    else:
        if args.worker_handles: parser.error("Worker handles are internal only.")
        if not args.stage or not args.output: parser.error("Explicit stage and output required.")
        result = run_stage(args.root,args.protocol,args.output,args.stage)
    summary = {key:result[key] for key in ("status","kind","scope","stage","complete","delivery_verified","error") if key in result}
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if result.get("status") in ("COMPLETE","SYNTHETIC_COMPLETE","MECHANISM_COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
