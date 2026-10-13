"""Exclusive bounded mechanism jobs, with no inherited real-experiment approval."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from check_lc_native_environment import NativeEnvironmentAPI
from ic_role_data import ASSIGNMENT_PATH, ASSIGNMENT_SHA, FIT_PATH, FIT_SHA, file_digest, load_fit_bundle, prepare_inner_fold
from ic_role_training import build_fit_matrix, fit_trajectory
from lc_execution_safety import WindowsAwakeGuard, stage_elapsed_seconds
from run_lc_role_adaptation import run_supervised_job, terminal_status

# Local import closure of the execution and analysis roots, including transitive math dependencies.
SOURCE_FILES = tuple(f"scripts/{name}.py" for name in (
    "analyze_ic_role_mechanism", "analyze_lc_role_adaptation", "analyze_misclassifications",
    "anchor_linear_controls", "audit_abmp_candidate_capacity", "check_lc_native_environment",
    "frozen_visual_probe", "hrgv_network", "ic_role_audit", "ic_role_data", "ic_role_model",
    "ic_role_training", "lc_execution_safety", "lc_role_adapter", "lc_role_audit", "lc_role_data",
    "lc_role_training", "mineral_hierarchy", "run_anchor_linear_controls", "run_frozen_visual_probe",
    "run_ic_role_mechanism", "run_lc_role_adaptation", "run_seen_unseen_gate_study",
    "run_tc_oos_rsg_experiments", "run_verifier_strong_controls", "run_verifier_trust_development",
    "tc_oos_rsg", "train_hierarchical_mineral_classifier", "train_hrgv_mineral_classifier",
    "train_mineral_classifier", "train_role_aware_mineral_classifier", "verifier_risk",
    "verifier_strong_controls", "verify_verifier_trust_delivery"))
TERMINAL = ("FAILED", "TIMED_OUT", "SUPERVISION_INTERRUPTED", "POWER_REQUEST_FAILED", "POWER_RELEASE_FAILED")


def _path(root, path):
    root, resolved = Path(root).resolve(), Path(path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Path escapes the exclusive output scope.")
    return resolved


def _json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def synthetic_protocol():
    return dict(protocol="ic_rfa_b_v1", status="SYNTHETIC_ONLY", max_seconds=7200,
                per_fit_seconds=7200, heartbeat_seconds=15, max_trajectories=108,
                max_temperatures=3, max_updates=400, primary_checkpoint=100)


def validate_protocol(root: Path, protocol: dict, stage: str) -> None:
    if stage not in ("synthetic", "fit-mechanism"):
        raise ValueError("Only synthetic or fit-mechanism stages exist.")
    expected = synthetic_protocol()
    if any(protocol.get(k) != v for k, v in expected.items() if k != "status"):
        raise ValueError("Fixed protocol counts, updates or whole-stage budget changed.")
    allowed = set(expected)
    if stage == "synthetic":
        if protocol.get("status") != "SYNTHETIC_ONLY" or set(protocol) != allowed:
            raise ValueError("Synthetic stage cannot name any real input or approval.")
        return
    allowed |= {"authorization", "fit_path", "fit_sha256", "assignment_path", "assignment_sha256",
                "source_sha256", "protocol_path", "output_relative", "environment"}
    authorization = protocol.get("authorization", {})
    if (set(protocol) != allowed or protocol.get("status") != "APPROVED_REAL_ONCE"
            or authorization.get("scope") != "ic_rfa_b_v1_fit_only_mechanism_once"
            or authorization.get("human_approved") is not True
            or not authorization.get("user_message_id") or not authorization.get("approval_utc")):
        raise ValueError("Fresh explicit real approval required; drafts and consumed approvals cannot run.")
    if (protocol["fit_path"] != FIT_PATH or protocol["fit_sha256"] != FIT_SHA
            or protocol["assignment_path"] != ASSIGNMENT_PATH or protocol["assignment_sha256"] != ASSIGNMENT_SHA):
        raise ValueError("Only the pinned fit-only bundle and existing assignment are permitted.")
    for field in ("fit_path", "assignment_path", "protocol_path", "output_relative"):
        _path(root, root/protocol[field])
    if set(protocol["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("Complete execution source identity is required.")
    for relative, expected_hash in protocol["source_sha256"].items():
        if file_digest(_path(root, root/relative)) != expected_hash:
            raise ValueError("Registered execution source changed.")
    environment = dict(python=sys.version.split()[0], torch=str(torch.__version__), dtype="float64", device="cpu")
    if protocol["environment"] != environment:
        raise ValueError("Registered environment changed.")


def _registered(root, protocol_path, protocol):
    relative = protocol_path.resolve().relative_to(root.resolve()).as_posix()
    if relative != protocol["protocol_path"]:
        raise ValueError("Protocol path identity changed.")
    gh = "C:/Program Files/GitHub CLI/gh.exe"
    def api(endpoint):
        result = subprocess.run([gh, "api", endpoint], capture_output=True, check=True, timeout=90)
        return json.loads(result.stdout)
    ref = api("repos/bobking88/4-/git/ref/heads/codex/theory-aware-report")["object"]["sha"]
    remote = api(f"repos/bobking88/4-/contents/{relative}?ref={ref}")
    if hashlib.sha256(base64.b64decode(remote["content"])).hexdigest() != file_digest(protocol_path):
        raise ValueError("Approved protocol is not registered on GitHub.")
    for name in SOURCE_FILES:
        content = api(f"repos/bobking88/4-/contents/{name}?ref={ref}")
        digest = hashlib.sha256(base64.b64decode(content["content"])).hexdigest()
        if digest != protocol["source_sha256"][name]:
            # Git clean filters can change newlines; compare the registered blob instead.
            local = subprocess.run(["git", "hash-object", "--path="+name, str(root/name)],
                                   cwd=root, capture_output=True, text=True, check=True, timeout=30)
            if local.stdout.strip() != content["sha"]:
                raise ValueError("Source does not match GitHub preregistration.")
    return dict(commit=ref, protocol_sha256=file_digest(protocol_path))


def require_ac_power():
    if NativeEnvironmentAPI().power_status().ACLineStatus != 1:
        raise RuntimeError("SUPERVISION_INTERRUPTED: AC connection required")


def _job(output, packet, job_id):
    folder = output/"runs"/job_id
    folder.mkdir(parents=True, exist_ok=False)
    packet_path = folder/"packet.json"
    _json(packet_path, packet, exclusive=True)
    return dict(job_id=job_id, log_path=str(folder/"worker.log"),
                command=[sys.executable, str(Path(__file__).resolve()), "--worker-packet", str(packet_path)],
                result_path=str(folder/("result.json" if packet["stage"] == "synthetic" or packet["kind"] == "analysis" else "result.pt")))


def run_stage(root: Path, protocol_path: Path | None, output: Path, stage: str) -> dict:
    start = time.monotonic()
    root = Path(root).resolve()
    output = _path(root, output)
    protocol = (synthetic_protocol() if stage == "synthetic" and protocol_path is None
                else json.loads(Path(protocol_path).read_text(encoding="utf-8")) if protocol_path else {})
    validate_protocol(root, protocol, stage)
    if output.exists():
        raise FileExistsError("Exclusive output already exists; no resume or retry.")
    registration = None
    if stage == "fit-mechanism":
        if output != _path(root, root/protocol["output_relative"]):
            raise ValueError("Output differs from the newly approved protocol.")
        registration = _registered(root, Path(protocol_path), protocol)
        if file_digest(root/FIT_PATH) != FIT_SHA or file_digest(root/ASSIGNMENT_PATH) != ASSIGNMENT_SHA:
            raise ValueError("Input hashes changed; no deserialization allowed.")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="PREPARING", state_history=["PREPARING"], stage=stage,
                 trajectory_count=0, temperature_count=0, alive=False, registration=registration,
                 real_data_opened=False, jobs=[], artifacts={}, elapsed_seconds=0.)
    _json(output/"stage_state.json", state, exclusive=True)
    guard = None
    try:
        require_ac_power()
        with WindowsAwakeGuard(api=NativeEnvironmentAPI()) as guard:
            state["status"] = "RUNNING"
            state["state_history"].append("RUNNING")
            prepared = {}
            schedule = [dict(kind="prepare", fold=fold, job_id=f"prepare_fold{fold}") for fold in range(3)]
            schedule += [dict(job, kind="trajectory") for job in build_fit_matrix()]
            schedule += [dict(kind="analysis", job_id="analysis")]
            for item in schedule:
                elapsed = max(time.monotonic()-start, stage_elapsed_seconds())
                if elapsed >= protocol["max_seconds"]:
                    raise RuntimeError("TIMED_OUT: whole stage includes preparation and export")
                packet = dict(item, root=str(root), output=str(output), stage=stage,
                              protocol_path=None if protocol_path is None else str(Path(protocol_path).resolve()))
                if item["kind"] == "trajectory" and stage == "fit-mechanism":
                    packet.update(prepared_relative=prepared[item["fold"]],
                                  prepared_sha256=state["artifacts"][prepared[item["fold"]]])
                if item["kind"] == "analysis" and stage == "fit-mechanism":
                    _json(output/"artifact_manifest.json", dict(state, protocol=protocol), exclusive=True)
                job = _job(output, packet, item["job_id"])
                result = run_supervised_job(job, protocol, protocol["max_seconds"]-elapsed)
                state["jobs"].append(dict(result, kind=item["kind"], fold=item.get("fold"),
                                          unit=item.get("unit"), seed=item.get("seed")))
                if result["status"] != "COMPLETE" or result.get("alive") or not result.get("supervision", {}).get("power_request_released"):
                    status = result["status"] if result["status"] in TERMINAL else "FAILED"
                    raise RuntimeError(status+": job failed or supervision incomplete")
                guard.check()
                if stage == "fit-mechanism":
                    path = _path(output, job["result_path"])
                    relative = path.relative_to(output).as_posix()
                    state["artifacts"][relative] = file_digest(path)
                    if item["kind"] == "prepare":
                        prepared[item["fold"]] = relative
                        state["real_data_opened"] = True
                if item["kind"] == "prepare": state["temperature_count"] += 1
                if item["kind"] == "trajectory": state["trajectory_count"] += 1
                state["elapsed_seconds"] = max(time.monotonic()-start, stage_elapsed_seconds())
                _json(output/"stage_state.json", state)
                if max(time.monotonic()-start, stage_elapsed_seconds()) >= protocol["max_seconds"]:
                    raise RuntimeError("TIMED_OUT: cumulative stage budget exhausted")
        state["status"] = "MECHANISM_COMPLETE"
        state["state_history"].append("MECHANISM_COMPLETE")
    except BaseException as error:
        state["status"] = terminal_status(error)
        state["error"] = f"{type(error).__name__}: {error}"
        state["state_history"].append(state["status"])
    state["elapsed_seconds"] = time.monotonic()-start
    state["power"] = {} if guard is None else guard.evidence()
    _json(output/"stage_state.json", state)
    return state


def worker(packet_path: Path) -> dict:
    packet_path = Path(packet_path).resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("kind") not in ("prepare", "trajectory", "analysis"):
        raise ValueError("No stop evaluation, refit or training replay packets.")
    root, output = Path(packet["root"]).resolve(), Path(packet["output"]).resolve()
    _path(root, output)
    _path(output, packet_path)
    stage = packet["stage"]
    suffix = "result.json" if stage == "synthetic" or packet["kind"] == "analysis" else "result.pt"
    destination = packet_path.parent/suffix
    if destination.exists():
        raise FileExistsError("Completed worker task cannot be replayed.")
    protocol = (synthetic_protocol() if stage == "synthetic" else
                json.loads(Path(packet["protocol_path"]).read_text(encoding="utf-8")))
    validate_protocol(root, protocol, stage)
    state = json.loads((output/"stage_state.json").read_text(encoding="utf-8"))
    if state["status"] != "RUNNING" or state["stage"] != stage:
        raise ValueError("Worker requires an active exclusive stage.")
    _json(packet_path.parent/"task_started.json", dict(status="ATTEMPT_CONSUMED", kind=packet["kind"], stage=stage), exclusive=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with WindowsAwakeGuard(api=NativeEnvironmentAPI()):
        if stage == "synthetic":
            result = dict(status="COMPLETE", kind=packet["kind"], scope="control-plane simulation only",
                          real_data_opened=False)
        elif packet["kind"] == "prepare":
            raw, assignment = load_fit_bundle(root)
            result = prepare_inner_fold(raw, assignment, packet["fold"])
        elif packet["kind"] == "trajectory":
            path = _path(output, output/packet["prepared_relative"])
            if file_digest(path) != packet["prepared_sha256"]:
                raise ValueError("Prepared input changed before deserialization.")
            data = torch.load(path, map_location="cpu", weights_only=True)
            if packet["max_updates"] != 400 or packet["primary_checkpoint"] != 100:
                raise ValueError("Unregistered trajectory updates.")
            result = fit_trajectory(data["train"], data["validation"], packet["unit"], packet["seed"])
            result.update(inner_fold=packet["fold"], prepared_relative=packet["prepared_relative"])
        else:
            from analyze_ic_role_mechanism import _verify_active_mechanism, write_mechanism_report
            evidence = _verify_active_mechanism(root, output)
            result = write_mechanism_report(evidence, output/"analysis")
        if destination.exists(): raise FileExistsError("Worker result cannot be overwritten.")
        if suffix.endswith("json"): _json(destination, result, exclusive=True)
        else: torch.save(result, destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-packet", type=Path)
    parser.add_argument("--stage", choices=("synthetic", "fit-mechanism"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_packet: worker(args.worker_packet)
    else:
        result = run_stage(args.root, args.protocol, args.output, args.stage)
        print(json.dumps(result, ensure_ascii=False))
        raise SystemExit(0 if result["status"] == "MECHANISM_COMPLETE" else 1)
