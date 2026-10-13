"""One exclusive synthetic timing run; no registered data or real training."""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from analyze_ic_role_mechanism import _verify_trajectory
from check_lc_native_environment import NativeEnvironmentAPI
from ic_role_audit import audit_bounds
from ic_role_data import prepare_inner_fold
from ic_role_model import ICRoleModel, UNITS
from ic_role_training import fit_trajectory
from lc_execution_safety import WindowsAwakeGuard
from run_lc_role_adaptation import run_supervised_job


def budget_estimate(timings):
    keys = ("prepare", "initial_kernel", "update", "checkpoint", "artifact", "verify_export")
    if any(not timings.get(k) or any(not math.isfinite(v) or v < 0 for v in timings[k]) for k in keys):
        return dict(estimated_seconds=None, ready=False, timing_evidence_complete=False,
                    reason="Missing or invalid measured timing component")
    maxima = {k: max(timings[k]) for k in keys}
    estimate = 1.5*(3*maxima["prepare"]+108*(maxima["initial_kernel"]+400*maxima["update"]
                   +9*maxima["checkpoint"]+maxima["artifact"])+maxima["verify_export"])
    complete = timings.get("verification_forward_count") == 108*9
    return dict(estimated_seconds=estimate, maxima_seconds=maxima,
                timing_evidence_complete=complete, ready=complete and estimate <= 7200,
                hard_limit_seconds=7200, safety_factor=1.5,
                scope="one synthetic shape/timing estimate, not a runtime guarantee or real approval")


def _raw():
    gen = torch.Generator().manual_seed(20261006)
    n = 340
    labels = torch.arange(n, dtype=torch.long) % 4
    raw = dict(p=torch.softmax(torch.randn(n, 4, generator=gen, dtype=torch.float64), 1),
        H=torch.randn(n, 1280, generator=gen, dtype=torch.float64),
        evidence=torch.randn(n, 18, generator=gen, dtype=torch.float64),
        contradiction=torch.rand(n, 1, generator=gen, dtype=torch.float64), labels=labels, hashes={},
        records=[dict(image_id=f"synthetic-{i}", split_group_id=f"synthetic-group-{i}",
                      relative_path=f"synthetic/{i}.jpg", four_class_id=str(int(labels[i]))) for i in range(n)])
    assignment = [dict(image_id=r["image_id"], split_group_id=r["split_group_id"],
                       class_id=int(labels[i]), inner_fold=(i//4) % 3) for i, r in enumerate(raw["records"])]
    return raw, assignment


def timing_worker(output):
    output = Path(output)
    if (output/"timing_evidence.json").exists():
        raise FileExistsError("Completed one-shot timing cannot be replayed.")
    with (output/"timing_started.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(status="ATTEMPT_CONSUMED", scope="one synthetic timing only"), stream)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    timings = {k: [] for k in ("prepare", "initial_kernel", "update", "checkpoint", "artifact", "verify_export")}
    units, diagnostics, max_residuals = [], [], {}
    start = time.perf_counter()
    with WindowsAwakeGuard(api=NativeEnvironmentAPI()) as guard, threadpool_limits(limits=1), tempfile.TemporaryDirectory() as temporary:
        raw, assignment = _raw()
        prepared = []
        for fold in range(3):
            begin = time.perf_counter()
            prepared.append(prepare_inner_fold(raw, assignment, fold))
            timings["prepare"].append(time.perf_counter()-begin)
            guard.check()
        train, validation = prepared[0]["train"], prepared[0]["validation"]
        paths = []
        for unit in UNITS:
            trajectory = fit_trajectory(train, validation, unit, 20261002, max_updates=20)
            costs = trajectory["costs"]
            timings["initial_kernel"].append(costs["initial_kernel_seconds"])
            timings["update"].extend(costs["update_seconds"])
            timings["checkpoint"].extend(costs["checkpoint_seconds"])
            artifact_start = time.perf_counter()
            path = Path(temporary)/f"{unit}.pt"
            torch.save(trajectory, path)
            reloaded = torch.load(path, weights_only=True, map_location="cpu")
            timings["artifact"].append(time.perf_counter()-artifact_start)
            # Later-checkpoint cost proxy: this same 20-step saved state, never more updates.
            checkpoint_start = time.perf_counter()
            replay = _verify_trajectory(reloaded, train, validation)
            replay_cost = (time.perf_counter()-checkpoint_start)/len(reloaded["checkpoints"])
            timings["checkpoint"].append(replay_cost)
            diagnostics.append(dict(unit=unit, **replay))
            paths.append(path)
            for point in reloaded["checkpoints"].values():
                for certificate in point["bounds"].values():
                    for key, value in certificate["max_residual_by_check"].items():
                        max_residuals[key] = max(max_residuals.get(key, 0.), value)
            units.append(dict(unit=unit, updates=20, seed=20261002,
                              train_rows=len(train["labels"]), validation_rows=len(validation["labels"])))
            print(json.dumps(dict(unit=unit, completed_updates=20, elapsed=time.perf_counter()-start)), flush=True)
            del trajectory, reloaded
            guard.check()
        verify_start = time.perf_counter()
        count = 0
        # Cost-only 108 x nine slots reuse saved synthetic states. No new fits/checkpoint labels.
        for repeat in range(9):
            for path in paths:
                saved = torch.load(path, weights_only=True, map_location="cpu")
                _verify_trajectory(saved, train, validation)
                count += len(saved["checkpoints"])
                model = ICRoleModel(saved["unit"], seed=20261002)
                model.load_state_dict(saved["checkpoints"][20]["state_dict"])
                for _ in range(4):
                    with torch.no_grad():
                        for batch in (train, validation):
                            result = model(batch["h"], batch["e"], batch["log_anchor"], auxiliary=True)
                            if audit_bounds(model, batch, result)["violation_count"]:
                                raise ValueError("Cost replay certificate failure")
                    count += 1
                del saved
                guard.check()
        timings["verification_forward_count"] = count
        evidence = dict(status="SYNTHETIC_ACCEPTANCE_COMPLETE", synthetic_rows=340,
                        optimizer_trajectory_count=12, optimizer_update_count=240, seeds=[20261002],
                        raw_or_stop_opened=False, real_images_opened=False, units=units,
                        diagnostics=diagnostics, maximum_residual_by_check=max_residuals,
                        timing_scope="one run; 12x20 fits plus cost-only 108x9 saved-state forward replays",
                        real_effectiveness_supported=False, independent_confirmation=False)
        preview = Path(temporary)/"export.json"
        preview.write_text(json.dumps(evidence, allow_nan=False, indent=2), encoding="utf-8")
        json.loads(preview.read_text(encoding="utf-8"))
        timings["verify_export"].append(time.perf_counter()-verify_start)
        evidence.update(timings=timings, estimate=budget_estimate(timings), total_seconds=time.perf_counter()-start)
    evidence["worker_power"] = guard.evidence()
    with (output/"timing_evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, allow_nan=False, indent=2)
    return evidence


def run_once(output: Path):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(per_fit_seconds=7200., heartbeat_seconds=15.)
    job = dict(job_id="one_synthetic_timing", log_path=str(output/"timing_worker.log"),
               command=[sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)])
    native = run_supervised_job(job, protocol, 7200.)
    with (output/"supervision.json").open("x", encoding="utf-8") as stream:
        json.dump(native, stream, allow_nan=False, indent=2)
    if native["status"] != "COMPLETE" or native["alive"] or not native["supervision"].get("power_request_released"):
        return dict(status=native["status"], ready=False, reason="One-shot native supervision failed; no retry")
    timeout_job = dict(job_id="one_short_timeout", log_path=str(output/"timeout_worker.log"),
                       command=[sys.executable, "-c", "import time; time.sleep(30)"])
    cleanup = run_supervised_job(timeout_job, dict(per_fit_seconds=.25, heartbeat_seconds=15.), .25)
    evidence = json.loads((output/"timing_evidence.json").read_text(encoding="utf-8"))
    safe = cleanup["status"] == "TIMED_OUT" and not cleanup["alive"] and cleanup["supervision"].get("power_request_released")
    result = dict(status="SYNTHETIC_ACCEPTANCE_COMPLETE" if safe else "SAFETY_EVIDENCE_FAILED",
                  ready=bool(safe and evidence["estimate"]["ready"]), timeout_cleanup=cleanup,
                  estimate=evidence["estimate"], real_training_authorized=False)
    with (output/"acceptance_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    result = timing_worker(args.output) if args.worker else run_once(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "SYNTHETIC_ACCEPTANCE_COMPLETE" else 1)
