import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import run_lc_role_adaptation as runner
except ModuleNotFoundError:
    runner = None


class ExecutionGateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "LC-RFA supervised runner missing")
        self.protocol = runner.draft_settings()

    def test_scope_schedule_count_and_distinct_unbounded_candidates(self):
        self.assertEqual(runner.maximum_fit_count(self.protocol), 422)
        self.assertEqual(runner.temperature_fit_count(self.protocol), 4)
        jobs = runner.build_jobs(self.protocol)
        self.assertEqual(len(jobs), 422)
        self.assertEqual(len({j["job_id"] for j in jobs}), 422)
        self.assertEqual(sum(j["phase"] == "cv" for j in jobs), 369)
        unbounded = [j for j in jobs if j["phase"] == "cv" and j["arm"] == "U1"]
        self.assertEqual(len(unbounded), 27)
        self.assertTrue(all(j["config"]["alpha_max"] == j["config"]["beta_max"] == 0 for j in unbounded))
        self.assertEqual(sum(j["phase"] == "matched_cv" for j in jobs), 27)

    def test_draft_only_permits_preflight_benchmark_and_fixed_scope(self):
        for stage in ("preflight", "benchmark"):
            runner.validate_protocol(self.protocol, stage)
        for stage in ("fit", "evaluate", "verify", "unknown"):
            with self.assertRaises(ValueError):
                runner.validate_protocol(self.protocol, stage)
        for key, value in (("outer_fold", 1), ("max_fits", 423), ("threads", 2),
                           ("seeds", [20261002]), ("max_seconds", 0), ("budgets", [[1, 1]])):
            bad = copy.deepcopy(self.protocol)
            bad[key] = value
            with self.assertRaises(ValueError):
                runner.validate_protocol(bad, "preflight")

    def test_authorization_cannot_be_inferred_from_code_approval(self):
        bad = dict(self.protocol, status="APPROVED_FOR_DEVELOPMENT",
                   training_authorization={"human_text": "认可计划，原生实施代码与合成验收"})
        with self.assertRaises(ValueError):
            runner.validate_protocol(bad, "fit")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = root / "draft.json"
            protocol.write_text(json.dumps(self.protocol), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.run_stage(root, protocol, root / "result", "fit")
            self.assertFalse((root / "result").exists())

    def test_supervised_failure_has_one_attempt_pid_exit_and_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            job = dict(job_id="crash", command=[sys.executable, "-c", "print('intentional crash'); raise SystemExit(7)"],
                       log_path=str(Path(folder)/"worker.log"))
            result = runner.run_supervised_job(job, self.protocol, 30)
            self.assertEqual(result["attempts"], 1)
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["exit_code"], 7)
            self.assertGreater(result["pid"], 0)
            self.assertIn("intentional crash", result["log_tail"])
            self.assertFalse(result["alive"])

    def test_hard_timeout_kills_worker_and_zero_budget_never_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            job = dict(job_id="timeout", command=[sys.executable, "-c", "import time; print('alive',flush=True); time.sleep(30)"],
                       log_path=str(Path(folder)/"worker.log"))
            result = runner.run_supervised_job(job, dict(self.protocol, per_fit_seconds=.25), 2)
            self.assertEqual(result["status"], "TIMED_OUT")
            self.assertFalse(result["alive"])
            self.assertIsNotNone(result["exit_code"])
            self.assertLess(result["elapsed_seconds"], 5)
            with patch.object(runner.subprocess, "Popen") as spawn:
                stopped = runner.run_supervised_job(job, self.protocol, 0)
                spawn.assert_not_called()
            self.assertEqual(stopped["status"], "TIMED_OUT")
            self.assertEqual(stopped["attempts"], 0)

    def test_job_batch_stops_after_first_failure_or_budget_exhaustion(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = [dict(job_id=f"job-{i}", command=[sys.executable, "-c", "raise SystemExit(9)"],
                         log_path=str(Path(folder)/f"{i}.log")) for i in range(3)]
            ledger = runner.run_job_batch(jobs, self.protocol, Path(folder)/"ledger.jsonl", consumed_seconds=0)
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]["status"], "FAILED")
            self.assertFalse((Path(folder)/"1.log").exists())
            ledger = runner.run_job_batch(jobs, self.protocol, Path(folder)/"spent.jsonl", consumed_seconds=21600)
            self.assertEqual(len(ledger), 0)

    def test_atomic_lock_must_be_complete_current_and_unmodified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                runner.read_selection_lock(root, self.protocol)
            artifact = root / "head.pt"
            artifact.write_bytes(b"synthetic-state")
            lock = dict(status="FIT_LOCKED", protocol_sha256=runner.protocol_digest(self.protocol),
                        selections={arm: {"source": "fit_inner_oof", "config": dict(regularization=.1, alpha_max=.25, beta_max=.25), "updates": 0} for arm in runner.SELECTED_ARMS},
                        artifacts={"head.pt": hashlib.sha256(artifact.read_bytes()).hexdigest()},
                        finals=[dict(arm=arm, seed=seed, state_path="head.pt")
                                for arm in runner.SELECTED_ARMS
                                for seed in ((0,) if arm in ("P", "E") else runner.SEEDS)],
                        consumed_seconds=1, source_snapshot={})
            runner.write_selection_lock(root, lock)
            self.assertEqual(runner.read_selection_lock(root, self.protocol), lock)
            artifact.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                runner.read_selection_lock(root, self.protocol)
            artifact.write_bytes(b"synthetic-state")
            path = root / "selection.lock.json"
            altered = dict(lock, status="FITTING")
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.read_selection_lock(root, self.protocol)
            with self.assertRaises(FileExistsError):
                runner.write_selection_lock(root, lock)

    def test_sources_are_hashed_not_silently_re_registered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"source.txt"
            path.write_text("original", encoding="utf-8")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            protocol = dict(self.protocol, source_sha256={"source.txt": sha})
            self.assertEqual(runner.check_sources(root, protocol), {"source.txt": sha})
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.check_sources(root, protocol)
            protocol["source_sha256"] = {"../outside": sha}
            with self.assertRaises(ValueError):
                runner.check_sources(root, protocol)

    def test_existing_stage_output_is_refused_before_data_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/"existing"
            output.mkdir()
            with patch.object(runner, "default_protocol") as protocol:
                with self.assertRaises(FileExistsError):
                    runner.run_stage(root, None, output, "preflight")
                protocol.assert_not_called()

    def test_numeric_worker_failure_keeps_numeric_terminal_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = dict(self.protocol,source_sha256={})
            with patch.object(runner,"default_protocol",return_value=protocol), patch.object(runner,"_benchmark",side_effect=RuntimeError("NUMERIC_RANGE_FAILURE: synthetic worker underflow")):
                with self.assertRaises(RuntimeError):
                    runner.run_stage(root,None,root/"out","benchmark")
            terminal = json.loads((root/"out"/"stage_state.json").read_text(encoding="utf-8"))
            self.assertEqual(terminal["status"],"NUMERIC_RANGE_FAILURE")

    def test_registered_runtime_records_packages_and_rejects_version_drift(self):
        self.assertTrue(hasattr(runner,"runtime_environment"),"Missing dependency runtime fingerprint")
        environment = runner.runtime_environment()
        self.assertTrue({"torch","numpy","scipy","scikit-learn","threadpoolctl","matplotlib"} <= set(environment["packages"]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = dict(self.protocol,source_sha256={},runtime_environment=environment)
            self.assertEqual(runner.check_sources(root,protocol),{})
            changed = dict(environment,python="different-runtime")
            with patch.object(runner,"runtime_environment",return_value=changed):
                with self.assertRaisesRegex(ValueError,"environment"):
                    runner.check_sources(root,protocol)

    def test_registered_code_identity_uses_git_normalized_blob_bytes(self):
        self.assertIsNotNone(getattr(runner, "registered_source_blob_digest", None), "Missing normalized registration digest")
        root = Path(__file__).resolve().parents[1]
        expected = runner.subprocess.run(["git", "rev-parse", "HEAD:scripts/run_anchor_linear_controls.py"],
                                         cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(runner.registered_source_blob_digest(root, "scripts/run_anchor_linear_controls.py"), expected)

    def test_stop_parse_only_after_valid_lock_and_cumulative_evaluation_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = dict(self.protocol, status="APPROVED_FOR_DEVELOPMENT", source_sha256={},
                            training_authorization=dict(scope="REAL_LC_RFA_FIT_AND_ONCE_STOP", max_fits=422,
                                max_seconds=21600, stop_evaluations=1, human_text="批准本协议真实开发训练"))
            path = root/"protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            output = root/"result"
            output.mkdir()
            with patch.object(runner, "_registered", return_value={}), patch.object(runner, "_execute") as execute:
                with self.assertRaises(ValueError):
                    runner.run_stage(root, path, output, "evaluate")
                execute.assert_not_called()
            artifact = output/"head.pt"
            artifact.write_bytes(b"synthetic-state")
            lock = dict(status="FIT_LOCKED", protocol_sha256=runner.protocol_digest(protocol),
                selections={arm: dict(source="fit_inner_oof", config=dict(regularization=.1, alpha_max=.25, beta_max=.25), updates=0) for arm in runner.SELECTED_ARMS},
                artifacts={"head.pt": hashlib.sha256(artifact.read_bytes()).hexdigest()},
                finals=[dict(arm=arm, seed=seed, state_path="head.pt") for arm in runner.SELECTED_ARMS
                        for seed in ((0,) if arm in ("P", "E") else runner.SEEDS)],
                consumed_seconds=21600, source_snapshot={})
            runner.write_selection_lock(output, lock)
            self.assertTrue((output/"selection.lock.json").exists())
            with patch.object(runner, "_registered", return_value={}), patch.object(runner.subprocess, "Popen") as spawn:
                with self.assertRaises(RuntimeError):
                    runner.run_stage(root, path, output, "evaluate")
                spawn.assert_not_called()
            self.assertTrue((output/"evaluation"/"stage_state.json").exists(), "Missing terminal evaluation state")
            state = json.loads((output/"evaluation"/"stage_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "TIMED_OUT")
            self.assertEqual(state["attempts"], 1)

    def test_fit_orchestration_locks_all_groups_and_reuses_exact_s1_jobs(self):
        from lc_role_fixtures import raw
        source = raw(340)
        source["labels"] = torch.cat([torch.full((count,), label, dtype=torch.long)
                                       for label, count in enumerate((66, 130, 97, 47))])
        for row, label in zip(source["records"], source["labels"]):
            row["four_class_id"] = str(int(label))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            calls = []
            def execute(job, protocol, ledger, started, previous_seconds=0):
                packet = torch.load(Path(job["result_path"]).with_name("packet.pt"), weights_only=True)
                calls.append(packet["kind"])
                if packet["kind"] == "prepare":
                    def batch(indices):
                        n = len(indices)
                        return dict(h=torch.zeros(n, 64, dtype=torch.float64), e=torch.zeros(n, 23, dtype=torch.float64),
                            P=torch.zeros(n, 4, dtype=torch.float64), log_anchor=torch.full((n, 4), -math.log(4), dtype=torch.float64),
                            labels=source["labels"][indices], records=[source["records"][i] for i in indices.tolist()])
                    result = dict(state={"synthetic_orchestration_stub": True}, train=batch(packet["train_indices"]),
                        validation=None if packet["validation_indices"] is None else batch(packet["validation_indices"]))
                else:
                    prepared = torch.load(packet["prepared_path"], weights_only=True)
                    if packet["final"]:
                        result = dict(state_dict={"synthetic": torch.zeros(1)}, train_logq=prepared["train"]["log_anchor"])
                    elif packet["kind"] == "convex":
                        result = dict(validation_logq=prepared["validation"]["log_anchor"], converged=True, gradient_l2=0., gap_upper_bound=0.)
                    else:
                        result = dict(optimizer_updates=400, checkpoints={step: dict(validation_logq=prepared["validation"]["log_anchor"])
                                                                         for step in runner.CHECKPOINTS})
                torch.save(result, job["result_path"])
                return result, {"status": "COMPLETE"}
            import math
            import time
            with patch.object(runner, "load_cache_subset", return_value=source) as loader, patch.object(runner, "_execute", side_effect=execute):
                locked = runner._fit_stage(output, dict(self.protocol, source_sha256={}), output, time.monotonic(), {})
            loader.assert_called_once_with(output, dict(self.protocol, source_sha256={}), runner.SUBSETS[0])
            self.assertEqual(locked["actual_fit_count"], 392)
            self.assertEqual(len(calls), 396)
            self.assertEqual(len(locked["finals"]), 26)
            self.assertEqual(set(locked["selections"]), set(runner.SELECTED_ARMS))
            self.assertTrue((output/"inner_assignment.csv").exists())
            replay = runner.read_selection_lock(output, dict(self.protocol, source_sha256={}))
            self.assertEqual(replay["actual_fit_count"], 392)
            for seed in runner.SEEDS:
                plain = next(r for r in locked["finals"] if r["arm"] == "S1" and r["seed"] == seed)
                matched = next(r for r in locked["finals"] if r["arm"] == "S1_matched" and r["seed"] == seed)
                self.assertEqual(plain["state_path"], matched["state_path"])

    def test_actual_synthetic_worker_packet_and_history_delivery(self):
        import time
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            source = output/"synthetic.pt"
            torch.save(runner._synthetic_raw(), source)
            started = time.monotonic()
            prep = runner._packet_job(output, dict(kind="prepare", raw_path=str(source),
                         train_indices=torch.arange(80), validation_indices=torch.arange(80, 120)), "prepare")
            prepared, _ = runner._execute(prep, self.protocol, output/"ledger.jsonl", started)
            self.assertEqual(prepared["train"]["h"].shape, (80, 64))
            job = runner._packet_job(output, dict(kind="neural", arm="H0", seed=runner.SEEDS[0], final=False,
                         prepared_path=prep["result_path"], config=dict(regularization=.01, alpha_max=.5,
                                                                       beta_max=.5, checkpoints=[0, 20])), "head")
            result, process = runner._execute(job, self.protocol, output/"ledger.jsonl", started)
            self.assertEqual(result["optimizer_updates"], 20)
            self.assertEqual(process["attempts"], 1)
            self.assertFalse(process["alive"])
            self.assertTrue(Path(job["result_path"]).with_name("history.csv").exists())
            self.assertTrue(Path(job["result_path"]).with_name("gradients.csv").exists())


if __name__ == "__main__":
    unittest.main()
