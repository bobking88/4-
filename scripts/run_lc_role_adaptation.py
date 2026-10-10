"""Bounded serial LC-RFA execution; a draft is not real-training authorization."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from lc_role_adapter import ARMS, NumericRangeFailure
from lc_execution_safety import WindowsAwakeGuard, stage_elapsed_seconds
from lc_role_data import SUBSETS, checked_file, fit_preprocessing, load_cache_subset, make_inner_assignment, scoped_file, transform_subset
from lc_role_training import CHECKPOINTS, SEEDS, fit_convex, fit_neural, matched_budget_candidates, refit_neural, select_oof

ROOT = Path(__file__).resolve().parents[1]
SELECTED_ARMS = ARMS + ("P", "E", "S1_matched")
SOURCE_FILES = (
    "scripts/lc_role_data.py", "scripts/lc_role_adapter.py", "scripts/lc_role_audit.py",
    "scripts/lc_role_training.py", "scripts/run_lc_role_adaptation.py",
    "scripts/analyze_lc_role_adaptation.py", "scripts/generate_lc_role_adaptation_figure.py",
    "scripts/anchor_linear_controls.py", "scripts/frozen_visual_probe.py",
    "scripts/verifier_strong_controls.py", "scripts/run_verifier_trust_development.py",
    "scripts/run_frozen_visual_probe.py", "scripts/run_anchor_linear_controls.py",
    "scripts/audit_abmp_candidate_capacity.py", "scripts/tc_oos_rsg.py",
    "scripts/hrgv_network.py", "scripts/run_tc_oos_rsg_experiments.py",
    "scripts/train_mineral_classifier.py",
    "scripts/lc_execution_safety.py",
    "docs/superpowers/specs/2026-10-06-lightweight-role-adaptation-design.md",
    "docs/superpowers/plans/2026-10-06-lightweight-role-adaptation-implementation.md",
)
SPEC_SHA = "aa9fcec06b6190fa6e56969da667c7adbcd07c5eb9ea56574031bd51dabefea0"
REPORT_SHA = "b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c"
REPORT_PATH = "结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_digest(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _json(path: Path, value: dict, *, exclusive=False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def draft_settings() -> dict:
    return dict(protocol="lc_role_adaptation_v1", status="DRAFT_NOT_AUTHORIZED", outer_fold=0,
                subsets=list(SUBSETS), counts={"fit": 340, "stop": 340}, seeds=list(SEEDS),
                regularization=[.001, .01, .1], budgets=[[.25, .25], [.5, .5]],
                checkpoints=list(CHECKPOINTS), threads=1, learning_rate=.001,
                max_fits=422, max_seconds=21600, per_fit_seconds=600, heartbeat_seconds=10,
                training_authorization=None)


def validate_protocol(protocol: dict, stage: str) -> None:
    if stage not in ("preflight", "benchmark", "fit", "evaluate", "verify"):
        raise ValueError("An explicit known execution stage is required.")
    expected = draft_settings()
    if any(protocol.get(key) != value for key, value in expected.items()
           if key not in ("status", "training_authorization")):
        raise ValueError("Unregistered scope, schedule, or compute budget.")
    if protocol.get("status") == "DRAFT_NOT_AUTHORIZED":
        if stage not in ("preflight", "benchmark"):
            raise ValueError("Draft permits implementation preflight/benchmark only, not real training or evaluation.")
    elif protocol.get("status") == "APPROVED_FOR_DEVELOPMENT":
        approval = protocol.get("training_authorization") or {}
        if (approval.get("scope") != "REAL_LC_RFA_FIT_AND_ONCE_STOP"
                or approval.get("max_fits") != 422 or approval.get("max_seconds") != 21600
                or approval.get("stop_evaluations") != 1 or not approval.get("human_text")
                or approval["human_text"] == "认可计划，原生实施代码与合成验收"):
            raise ValueError("Explicit real-development authorization required, not specification/code approval.")
    else:
        raise ValueError("Unknown protocol authorization status.")


def maximum_fit_count(protocol: dict) -> int:
    validate_protocol(protocol, "preflight")
    return 6*6*3*3 + 3*3*3 + 7*3 + 3*3*3+3 + 2*3*3+2


def temperature_fit_count(protocol: dict) -> int:
    validate_protocol(protocol, "preflight")
    return 4


def build_jobs(protocol: dict) -> list[dict]:
    validate_protocol(protocol, "preflight")
    jobs = []
    def add(phase, arm, seed, fold=None, config=None):
        jobs.append(dict(job_id=f"{phase}-{arm}-{len(jobs):04d}", phase=phase,
                         arm=arm, seed=seed, inner_fold=fold, config=config))
    for arm in ARMS:
        budgets = [[0, 0]] if arm == "U1" else protocol["budgets"]
        for regularization, (alpha, beta), fold, seed in itertools.product(protocol["regularization"], budgets, range(3), SEEDS):
            add("cv", arm, seed, fold, dict(regularization=regularization, alpha_max=alpha, beta_max=beta))
    for arm, regularization, fold in itertools.product(("P", "E"), protocol["regularization"], range(3)):
        add("cv", arm, 0, fold, dict(regularization=regularization, alpha_max=0, beta_max=0))
    for arm in ARMS+("P", "E"):
        for seed in ((0,) if arm in ("P", "E") else SEEDS):
            add("final", arm, seed)
    # Deferred templates resolve to R1's fit-selected budget; exact S1 CV jobs are reused.
    for regularization, fold, seed in itertools.product(protocol["regularization"], range(3), SEEDS):
        add("matched_cv", "S1", seed, fold, dict(regularization=regularization, budget_source="R1_fit_selection"))
    for seed in SEEDS:
        add("matched_final", "S1_matched", seed)
    if len(jobs) != 422:
        raise ValueError("Schedule bound differs from the approved plan.")
    return jobs


def runtime_environment() -> dict:
    packages = ("torch", "numpy", "scipy", "scikit-learn", "threadpoolctl", "matplotlib")
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in packages})


def check_sources(root: Path, protocol: dict) -> dict:
    if "runtime_environment" in protocol and protocol["runtime_environment"] != runtime_environment():
        raise ValueError("Registered runtime environment changed.")
    snapshot = {}
    for relative, expected in protocol.get("source_sha256", {}).items():
        path = scoped_file(root, relative)
        actual = file_digest(path)
        if actual != expected:
            raise ValueError(f"Registered source changed: {relative}")
        snapshot[relative] = actual
    return snapshot


def default_protocol(root: Path) -> dict:
    old_path = root/"docs/experiment_protocols/abmp_frozen_visual_probe_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    base = "outputs/training/abmp_frozen_visual_probe_v1/development_fold_0/"
    protocol = dict(draft_settings(), input_hashes=old["input_sha256"],
                    runtime_environment=runtime_environment(),
                    cache_path=base+"frozen_features.pt",
                    cache_sha256="6bbb3221293d5bf0fcadb13bcaa7a3029fdc77e8328e119820665256e3509090",
                    audit_path=base+"image_feature_audit.json")
    protocol["audit_sha256"] = file_digest(root/protocol["audit_path"])
    paths = list(SOURCE_FILES)+[REPORT_PATH, str(old_path.relative_to(root)).replace("\\", "/"),
            base+"frozen_features.pt", base+"image_feature_audit.json", base+"development_summary.json",
            base+"registered_protocol.json", old["feature_extractor"]["checkpoint"]]
    for name in SUBSETS:
        paths.extend([f"outputs/theory/abmp_candidate_capacity_v1/{name}/component_predictions.csv",
                      f"outputs/theory/abmp_candidate_capacity_v1/{name}/routing_evidence.csv",
                      f"outputs/training/tc_oos_rsg_manifests_v1/fold_0/{name}.csv"])
    protocol["source_sha256"] = {p: file_digest(root/p) for p in paths}
    if (protocol["source_sha256"][SOURCE_FILES[-2]] != SPEC_SHA
            or protocol["source_sha256"][REPORT_PATH] != REPORT_SHA
            or protocol["source_sha256"][protocol["cache_path"]] != protocol["cache_sha256"]
            or protocol["source_sha256"][old["feature_extractor"]["checkpoint"]] != old["feature_extractor"]["checkpoint_sha256"]):
        raise ValueError("Approved spec/report/cache/backbone identity mismatch.")
    return protocol


def terminal_status(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "SUPERVISION_INTERRUPTED"
    if isinstance(error, NumericRangeFailure):
        return "NUMERIC_RANGE_FAILURE"
    for status in ("TIMED_OUT", "NUMERIC_RANGE_FAILURE", "SUPERVISION_INTERRUPTED",
                   "POWER_REQUEST_FAILED", "POWER_RELEASE_FAILED"):
        if str(error).startswith(status):
            return status
    return "FAILED"


def elapsed_stage(started: float, *, check=True) -> float:
    return max(time.monotonic()-started, stage_elapsed_seconds(check=check))


def run_supervised_job(job: dict, protocol: dict, remaining_seconds: float) -> dict:
    if remaining_seconds <= 0:
        return dict(job_id=job["job_id"], status="TIMED_OUT", attempts=0, pid=None,
                    exit_code=None, alive=False, elapsed_seconds=0., log_tail="Budget already exhausted")
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    start = time.monotonic()
    budget = min(float(protocol["per_fit_seconds"]), remaining_seconds)
    heartbeats, process, guard = [], None, None
    status, failure = "COMPLETE", ""
    next_heartbeat = float(protocol["heartbeat_seconds"])
    try:
        with log_path.open("x", encoding="utf-8") as log, WindowsAwakeGuard() as guard:
            guard.check()
            try:
                process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT, env=env,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                while True:
                    # Check time continuity before poll, including an already-exited worker.
                    now = max(time.monotonic()-start, guard.check())
                    if now >= budget:
                        status = "TIMED_OUT"
                        break
                    if process.poll() is not None:
                        status = "COMPLETE" if process.returncode == 0 else "NUMERIC_RANGE_FAILURE" if process.returncode == 3 else "FAILED"
                        break
                    if now >= next_heartbeat:
                        item = dict(pid=process.pid, alive=True, elapsed_seconds=now)
                        heartbeats.append(item)
                        print(json.dumps(dict(job_id=job["job_id"], **item)), flush=True)
                        next_heartbeat = now+protocol["heartbeat_seconds"]
                    time.sleep(min(.05, max(0., budget-now)))
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
    except BaseException as error:
        status, failure = terminal_status(error), f"{type(error).__name__}: {error}"
    evidence = {} if guard is None else guard.evidence()
    elapsed = max(time.monotonic()-start, evidence.get("wall_elapsed_seconds", 0.))
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    return dict(job_id=job["job_id"], status=status, attempts=int(process is not None),
                pid=None if process is None else process.pid,
                exit_code=None if process is None else process.returncode,
                alive=process is not None and process.poll() is None, elapsed_seconds=elapsed,
                log_tail=tail, error=failure, supervision=evidence, heartbeats=heartbeats)


def run_job_batch(jobs: list[dict], protocol: dict, ledger: Path, *, consumed_seconds: float) -> list[dict]:
    rows = []
    for job in jobs:
        if consumed_seconds >= protocol["max_seconds"]:
            break
        result = run_supervised_job(job, protocol, protocol["max_seconds"]-consumed_seconds)
        consumed_seconds += result["elapsed_seconds"]
        result["cumulative_seconds"] = consumed_seconds
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, allow_nan=False)+"\n")
        rows.append(result)
        if result["status"] != "COMPLETE":
            break
    return rows


def write_selection_lock(output: Path, lock: dict) -> None:
    path = output/"selection.lock.json"
    temporary = output/"selection.lock.tmp"
    if path.exists() or temporary.exists() or (output/"fit_complete.json").exists():
        raise FileExistsError("Selection lock cannot be overwritten or resumed.")
    _json(temporary, lock, exclusive=True)
    os.replace(temporary, path)
    _json(output/"fit_complete.json", dict(status="FIT_LOCKED", lock_sha256=file_digest(path)), exclusive=True)


def read_selection_lock(output: Path, protocol: dict) -> dict:
    try:
        stage_path = output/"stage_state.json"
        if stage_path.exists() and json.loads(stage_path.read_text(encoding="utf-8")).get("status") != "FIT_LOCKED":
            raise ValueError("Fit stage did not terminate successfully; selection lock is invalid.")
        path = output/"selection.lock.json"
        complete = json.loads((output/"fit_complete.json").read_text(encoding="utf-8"))
        lock = json.loads(path.read_text(encoding="utf-8"))
        if (complete.get("status") != "FIT_LOCKED" or complete.get("lock_sha256") != file_digest(path)
                or lock.get("status") != "FIT_LOCKED" or lock.get("protocol_sha256") != protocol_digest(protocol)
                or set(lock.get("selections", {})) != set(SELECTED_ARMS)
                or not lock.get("artifacts") or not 0 <= lock["consumed_seconds"] <= protocol["max_seconds"]):
            raise ValueError("Stale, incomplete, or tampered fit selection lock.")
        expected = {(arm, seed) for arm in SELECTED_ARMS for seed in ((0,) if arm in ("P", "E") else SEEDS)}
        actual = [(r["arm"], r["seed"]) for r in lock.get("finals", [])]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Missing final groups or seeds.")
        for selection in lock["selections"].values():
            if selection.get("source") != "fit_inner_oof" or "config" not in selection or selection.get("updates") not in CHECKPOINTS:
                raise ValueError("Selection did not originate solely from fit OOF.")
        for relative, digest in lock["artifacts"].items():
            if file_digest(scoped_file(output, relative)) != digest:
                raise ValueError("Locked fit artifact changed.")
        if any(r["state_path"] not in lock["artifacts"] for r in lock["finals"]):
            raise ValueError("Unpinned final model state.")
        return lock
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ValueError("Missing or malformed fit selection lock.") from error


def registered_source_blob_digest(root: Path, relative: str) -> str:
    path = scoped_file(root, relative)
    result = subprocess.run(["git", "hash-object", "--path="+relative, str(path)], cwd=root,
                            capture_output=True, text=True, check=True, timeout=30)
    return result.stdout.strip()


def _registered(root, protocol_path, protocol):
    relative = protocol_path.resolve().relative_to(root.resolve()).as_posix()
    gh = Path("C:/Program Files/GitHub CLI/gh.exe")
    def api(endpoint):
        response = subprocess.run([str(gh), "api", endpoint], capture_output=True, check=True, timeout=90)
        return json.loads(response.stdout)
    ref = api("repos/bobking88/4-/git/ref/heads/codex/theory-aware-report")["object"]["sha"]
    content = api(f"repos/bobking88/4-/contents/{relative}?ref={ref}")
    if hashlib.sha256(base64.b64decode(content["content"])).hexdigest() != file_digest(protocol_path):
        raise ValueError("Development protocol is not the remotely registered file.")
    tree_sha = api(f"repos/bobking88/4-/git/commits/{ref}")["tree"]["sha"]
    tree = api(f"repos/bobking88/4-/git/trees/{tree_sha}?recursive=1")
    entries = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "blob"}
    for path in SOURCE_FILES:
        sha = registered_source_blob_digest(root, path)
        if entries.get(path) != sha:
            raise ValueError(f"Source not present in remote registration commit: {path}")
    return dict(commit=ref, protocol_blob_sha=content["sha"])


def _packet_job(output, packet, job_id):
    packet = dict(packet)
    if "prepared_path" in packet:
        packet["prepared_relative_path"] = Path(packet["prepared_path"]).resolve().relative_to(output.resolve()).as_posix()
    folder = output/"runs"/job_id
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/"packet.pt"
    torch.save(dict(packet, result_path=str(folder/"result.pt")), path)
    return dict(job_id=job_id, log_path=str(folder/"worker.log"),
                command=[sys.executable, str(Path(__file__).resolve()), "--worker-packet", str(path)],
                result_path=str(folder/"result.pt"))


def _worker(path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    packet = torch.load(path, map_location="cpu", weights_only=True)
    start = time.monotonic()
    try:
        kind = packet["kind"]
        if kind == "prepare":
            raw = torch.load(packet["raw_path"], weights_only=True)
            state = fit_preprocessing(raw, packet["train_indices"])
            result = dict(state=state, train=transform_subset(raw, state, packet["train_indices"]),
                          validation=None if packet["validation_indices"] is None else transform_subset(raw, state, packet["validation_indices"]))
        elif kind in ("neural", "convex"):
            data = torch.load(packet["prepared_path"], weights_only=True)
            if kind == "convex":
                result = fit_convex(data["train"], None if packet["final"] else data["validation"], packet["arm"], packet["config"]["regularization"])
            elif packet["final"]:
                result = refit_neural(data["train"], packet["arm"], packet["config"], packet["seed"], packet["updates"])
            else:
                result = fit_neural(data["train"], data["validation"], packet["arm"], packet["config"], packet["seed"])
        elif kind == "evaluate":
            from analyze_lc_role_adaptation import evaluate_locked
            result = evaluate_locked(Path(packet["root"]), packet["protocol"], Path(packet["output"]))
        else:
            raise ValueError("Unknown internal worker kind.")
        result["worker_compute_seconds"] = time.monotonic()-start
        torch.save(result, packet["result_path"])
    except NumericRangeFailure as error:
        torch.save(dict(status="NUMERIC_RANGE_FAILURE", logq=error.logq), packet["result_path"])
        print(str(error), flush=True)
        raise SystemExit(3)


def _execute(job, protocol, ledger, stage_start, previous_seconds=0):
    elapsed = previous_seconds+elapsed_stage(stage_start, check=False)
    result = run_supervised_job(job, protocol, protocol["max_seconds"]-elapsed)
    if result["status"] == "COMPLETE":
        try:
            elapsed_stage(stage_start)
        except BaseException as error:
            result["status"], result["error"] = terminal_status(error), f"{type(error).__name__}: {error}"
    result["cumulative_seconds"] = max(previous_seconds+elapsed_stage(stage_start, check=False), elapsed+result["elapsed_seconds"])
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, allow_nan=False)+"\n")
    if result["status"] != "COMPLETE":
        raise RuntimeError(result["status"]+": "+result.get("error", "")+" "+result["log_tail"])
    fitted = torch.load(job["result_path"], map_location="cpu", weights_only=True)
    for field in ("history", "gradients"):
        if fitted.get(field):
            write_csv(Path(job["result_path"]).parent/(field+".csv"), fitted[field])
    return fitted, result


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fit_stage(root, protocol, output, started, snapshot):
    raw = load_cache_subset(root, protocol, SUBSETS[0])
    if {str(k): int((raw["labels"] == k).sum()) for k in range(4)} != {"0": 66, "1": 130, "2": 97, "3": 47}:
        raise ValueError("Registered fit class counts changed.")
    assignment = make_inner_assignment(raw)
    raw_path = output/"fit_raw.pt"
    torch.save(raw, raw_path)
    _json(output/"inner_assignment.json", {"rows": assignment}, exclusive=True)
    write_csv(output/"inner_assignment.csv", assignment)
    prepared = {}
    ledger = output/"job_ledger.jsonl"
    for fold in (0, 1, 2, "full"):
        train_indices = torch.tensor([i for i, row in enumerate(assignment) if fold == "full" or row["inner_fold"] != fold])
        validation_indices = None if fold == "full" else torch.tensor([i for i, row in enumerate(assignment) if row["inner_fold"] == fold])
        job = _packet_job(output, dict(kind="prepare", raw_path=str(raw_path), train_indices=train_indices,
                                       validation_indices=validation_indices), f"prepare-{fold}")
        _execute(job, protocol, ledger, started)
        prepared[fold] = job["result_path"]
    oof, cv_paths, finals, selections = [], {}, [], {}
    jobs = build_jobs(protocol)
    for schedule in (row for row in jobs if row["phase"] == "cv"):
        arm, fold = schedule["arm"], schedule["inner_fold"]
        packet = dict(kind="convex" if arm in ("P", "E") else "neural", final=False, arm=arm,
                      config=schedule["config"], seed=schedule["seed"], prepared_path=prepared[fold])
        job = _packet_job(output, packet, schedule["job_id"])
        result, _ = _execute(job, protocol, ledger, started)
        if arm not in ("P", "E") and (set(result["checkpoints"]) != set(CHECKPOINTS) or result["optimizer_updates"] != 400):
            raise ValueError("Incomplete CV trajectory cannot enter fit selection.")
        data = torch.load(prepared[fold], weights_only=True)["validation"]
        common = dict(arm=arm, config=schedule["config"], seed=schedule["seed"], inner_fold=fold,
                      image_ids=[r["image_id"] for r in data["records"]], source="fit_inner_oof")
        if arm in ("P", "E"):
            oof.append(dict(common, updates=0, logq=result["validation_logq"], certificate=result))
        else:
            oof.extend(dict(common, updates=step, logq=value["validation_logq"])
                       for step, value in result["checkpoints"].items())
        cv_paths[schedule["job_id"]] = job["result_path"]
    for arm in ARMS+("P", "E"):
        selections[arm] = select_oof([row for row in oof if row["arm"] == arm], assignment)
    selections["S1_matched"] = select_oof(matched_budget_candidates(oof, selections["R1"]), assignment)
    torch.save(oof, output/"oof_predictions.pt")
    # Exact matching S1 CV trajectories already exist for both budgets. No refits or retries.
    _json(output/"matched_cv_reuse.json", dict(reused=True, budget=selections["R1"]["config"],
                                              source="same-protocol S1 CV grid"), exclusive=True)
    final_cache = {}
    for schedule in (row for row in jobs if row["phase"] in ("final", "matched_final")):
        label = schedule["arm"]
        arm = "S1" if label == "S1_matched" else label
        selection = selections[label]
        key = (arm, schedule["seed"], protocol_digest(selection["config"]), selection["updates"])
        if key not in final_cache:
            packet = dict(kind="convex" if arm in ("P", "E") else "neural", final=True, arm=arm,
                          seed=schedule["seed"], config=selection["config"], updates=selection["updates"],
                          prepared_path=prepared["full"])
            job = _packet_job(output, packet, schedule["job_id"])
            _execute(job, protocol, ledger, started)
            final_cache[key] = Path(job["result_path"]).relative_to(output).as_posix()
        finals.append(dict(arm=label, seed=schedule["seed"], state_path=final_cache[key]))
    artifacts = {path.relative_to(output).as_posix(): file_digest(path)
                 for path in output.rglob("*") if path.is_file() and path.name != "stage_state.json"}
    lock = dict(status="FIT_LOCKED", protocol_sha256=protocol_digest(protocol),
                source_snapshot=snapshot, selections=selections, finals=finals,
                full_preprocessing_path=Path(prepared["full"]).relative_to(output).as_posix(),
                cv_paths={key: Path(path).relative_to(output).as_posix() for key, path in cv_paths.items()},
                artifacts=artifacts, consumed_seconds=elapsed_stage(started),
                actual_fit_count=len(cv_paths)+len(final_cache), temperature_fit_count=4)
    if lock["consumed_seconds"] > protocol["max_seconds"] or lock["actual_fit_count"] > protocol["max_fits"]:
        raise RuntimeError("TIMED_OUT: cumulative development budget exhausted before selection lock.")
    check_sources(root, protocol)
    write_selection_lock(output, lock)
    return lock


def _synthetic_raw():
    gen = torch.Generator().manual_seed(20261006)
    labels = torch.arange(120) % 4
    return dict(p=torch.softmax(torch.randn(120, 4, generator=gen, dtype=torch.float64), 1),
                evidence=torch.randn(120, 18, generator=gen, dtype=torch.float64),
                contradiction=torch.rand(120, 1, generator=gen, dtype=torch.float64),
                H=torch.randn(120, 1280, generator=gen, dtype=torch.float64), labels=labels, hashes={},
                records=[dict(image_id=f"synthetic-{i}", split_group_id=f"synthetic-group-{i}",
                              relative_path=f"synthetic/{i}.jpg") for i in range(120)])


def _benchmark(protocol, output, started):
    raw_path = output/"synthetic_raw.pt"
    torch.save(_synthetic_raw(), raw_path)
    ledger = output/"job_ledger.jsonl"
    preparation = _packet_job(output, dict(kind="prepare", raw_path=str(raw_path), train_indices=torch.arange(80),
                                           validation_indices=torch.arange(80, 120)), "synthetic-prepare")
    _, prep_timing = _execute(preparation, protocol, ledger, started)
    timings = []
    for arm in ARMS+("P", "E"):
        packet = dict(kind="convex" if arm in ("P", "E") else "neural", arm=arm, final=False,
                      seed=SEEDS[0], prepared_path=preparation["result_path"],
                      config=dict(regularization=.01, alpha_max=.5, beta_max=.5, checkpoints=[0, 20]))
        job = _packet_job(output, packet, f"synthetic-{arm}")
        result, timing = _execute(job, protocol, ledger, started)
        timings.append(dict(arm=arm, elapsed_seconds=timing["elapsed_seconds"],
                            compute_seconds=result["worker_compute_seconds"]))
    neural = [t for t in timings if t["arm"] in ARMS]
    convex = [t for t in timings if t["arm"] in ("P", "E")]
    scale = 340/80
    # Conservative serial projection includes a fresh import cost for each worker.
    neural_seconds = max(t["elapsed_seconds"]-t["compute_seconds"]+t["compute_seconds"]*20*scale for t in neural)
    convex_seconds = max(t["elapsed_seconds"]-t["compute_seconds"]+t["compute_seconds"]*scale for t in convex)
    projection = 402*neural_seconds+20*convex_seconds+4*prep_timing["elapsed_seconds"]*scale**2
    result = dict(status="TIME_PROJECTION_EXCEEDS_BUDGET" if projection >21600 else "BENCHMARK_COMPLETE",
                  scope="SYNTHETIC_ONLY_NOT_TRAINING_AUTHORIZATION", rows=120, neural_updates=20,
                  per_arm=timings, preparation_seconds=prep_timing["elapsed_seconds"],
                  projected_upper_schedule_seconds=projection, fit_upper_bound=422,
                  temperature_fits=4, uncertainty="Runtime projection, not a guaranteed wall time")
    _json(output/"runtime_projection.json", result, exclusive=True)
    return result


def run_stage(root: Path, protocol_path: Path | None, output: Path, stage: str) -> dict:
    if stage in ("preflight", "benchmark", "fit") and output.exists():
        raise FileExistsError("Existing stage output cannot be overwritten or resumed.")
    if protocol_path is None and stage not in ("preflight", "benchmark"):
        raise ValueError("Real stages require an explicitly registered protocol.")
    protocol = default_protocol(root) if protocol_path is None else json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol, stage)
    snapshot = check_sources(root, protocol)
    if stage in ("fit", "evaluate"):
        if protocol_path is None:
            raise ValueError("A remotely registered development protocol is required.")
        registration = _registered(root, protocol_path, protocol)
    else:
        registration = None
    if stage in ("evaluate", "verify"):
        lock = read_selection_lock(output, protocol)
        if lock["source_snapshot"] != snapshot:
            raise ValueError("Sources differ from the locked fit snapshot.")
        if stage == "verify":
            from analyze_lc_role_adaptation import verify_delivery
            return verify_delivery(root, output)
        if (output/"evaluation").exists():
            raise FileExistsError("Only one stop evaluation is permitted.")
        evaluation = output/"evaluation"
        evaluation.mkdir()
        started = time.monotonic()
        try:
            job = _packet_job(evaluation, dict(kind="evaluate", root=str(root), protocol=protocol, output=str(output)), "evaluate-once")
            with WindowsAwakeGuard() as guard:
                result, timing = _execute(job, protocol, evaluation/"job_ledger.jsonl", started, lock["consumed_seconds"])
                guard.check()
            _json(evaluation/"complete.json", dict(status="EVALUATED", consumed_seconds=timing["cumulative_seconds"],
                                                    result_sha256=file_digest(Path(job["result_path"]))), exclusive=True)
            return result
        except BaseException as error:
            status = terminal_status(error)
            _json(evaluation/"stage_state.json", dict(status=status, attempts=1, error=str(error)), exclusive=True)
            raise
    output.mkdir(parents=True, exist_ok=False)
    _json(output/("draft_protocol.json" if protocol["status"] == "DRAFT_NOT_AUTHORIZED" else "registered_protocol.json"), protocol, exclusive=True)
    started = time.monotonic()
    _json(output/"stage_state.json", dict(status="FITTING" if stage == "fit" else "CHECKING", stage=stage))
    try:
        if stage == "preflight":
            fit = load_cache_subset(root, protocol, SUBSETS[0])
            assignment = make_inner_assignment(fit)
            counts = {str(k): int((fit["labels"] == k).sum()) for k in range(4)}
            if counts != {"0": 66, "1": 130, "2": 97, "3": 47}:
                raise ValueError("Registered fit class counts changed.")
            _json(output/"inner_assignment.json", dict(rows=assignment), exclusive=True)
            _json(output/"schedule.json", dict(jobs=build_jobs(protocol)), exclusive=True)
            result = dict(status="PREFLIGHT_OK", protocol_status=protocol["status"], class_counts=counts,
                          unique_groups=len({r["split_group_id"] for r in fit["records"]}),
                          python=platform.python_version(), torch=torch.__version__, platform=platform.platform(),
                          runtime_environment=runtime_environment(),
                          source_snapshot=snapshot, stop_tables_parsed=False, max_fits=422, temperature_fits=4)
        elif stage == "benchmark":
            with WindowsAwakeGuard() as guard:
                result = _benchmark(protocol, output, started)
                guard.check()
        else:
            with WindowsAwakeGuard() as guard:
                result = _fit_stage(root, protocol, output, started, snapshot)
                guard.check()
        _json(output/"stage_state.json", dict(status=result["status"], stage=stage, registration=registration))
        _json(output/"summary.json", result, exclusive=True)
        return result
    except BaseException as error:
        status = terminal_status(error)
        _json(output/"stage_state.json", dict(status=status, stage=stage, error=str(error), attempts=1))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "benchmark", "fit", "evaluate", "verify"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-packet", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_packet is not None:
        if args.stage is not None or args.protocol is not None or args.output_dir is not None:
            parser.error("Internal worker action cannot be combined with public stage flags.")
        _worker(args.worker_packet)
        return
    if args.stage is None or args.output_dir is None:
        parser.error("--stage and --output-dir are required; no implicit training action.")
    torch.set_num_threads(1)
    print(json.dumps(run_stage(ROOT, args.protocol, args.output_dir, args.stage), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
