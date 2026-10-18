"""Exclusive synthetic phases for the approved MSB completion contract repair."""
from contextlib import ExitStack, nullcontext
import argparse
import ctypes
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ALL_MODULES = (
    "test_msb_execution_safety", "test_msb_role_data", "test_msb_role_model",
    "test_msb_role_controls", "test_msb_role_audit", "test_msb_role_training",
    "test_run_msb_role_mechanism", "test_analyze_msb_role_mechanism", "test_generate_msb_role_figure",
)
PHASES = {f"{stage}-{color}": (("test_msb_execution_safety",) if stage == "lease" else
          ("test_run_msb_role_mechanism",)) for stage in ("gate", "lease", "publish") for color in ("red", "green")}
PHASES["final"] = ALL_MODULES
SNAPSHOTS = (
    "scripts/msb_execution_safety.py", "scripts/run_msb_role_mechanism.py",
    "tests/test_msb_execution_safety.py", "tests/test_run_msb_role_mechanism.py",
    "tests/run_msb_terminal_repair_acceptance.py",
)


def _acceptance_passed(result, *, forbidden_calls: list, harness_error: str | None) -> bool:
    return (result is not None and result.wasSuccessful() and result.testsRun > 0
            and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses
            and not forbidden_calls and harness_error is None)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=tuple(PHASES))
    args = parser.parse_args(argv)
    phase_root = ROOT/"outputs/msb_completion_repair_synthetic_v1"/args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    legacy_attempt = (phase_root/"acceptance_summary.json").exists()
    for attempt in range(1, 4):
        if legacy_attempt and attempt == 1: continue
        output = phase_root/f"attempt-{attempt:03d}"
        try:
            output.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError("Three synthetic attempts consumed; retain all evidence and stop")
    scratch = ROOT/".superpowers/sdd/2026-10-17-msb-completion-acceptance-repair/test-tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch)
    snapshots = {p: dict(sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest(),
                         source=(ROOT/p).read_text(encoding="utf-8")) for p in SNAPSHOTS}
    (output/"source_snapshot.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT/"tests"), str(ROOT/"scripts")]
    forbidden_calls, tensor_reads = [], []
    log = io.StringIO()
    result = None
    error = None
    bootstrap = {}

    def forbidden(name):
        def reject(*a, **kw):
            forbidden_calls.append(name)
            raise AssertionError("UNMOCKED_BOUNDARY_FORBIDDEN: "+name)
        return reject

    def temporary_load(file, *a, **kw):
        if isinstance(file, io.BytesIO): return original_load(file, *a, **kw)
        path = Path(file if isinstance(file, (str, bytes, os.PathLike)) else file.name).resolve()
        if not path.is_relative_to(scratch.resolve()):
            return forbidden("torch.load outside synthetic scratch")()
        tensor_reads.append(path.relative_to(scratch).as_posix())
        return original_load(file, *a, **kw)

    try:
        # Only installed numeric runtimes bootstrap outside the application guards.
        import torch
        import sklearn
        bootstrap = dict(torch=str(torch.__version__), sklearn=str(sklearn.__version__))
        original_load = torch.load
        with ExitStack() as stack:
            for target, name in ((ctypes, "WinDLL"), (subprocess, "Popen"), (os, "_exit"),
                                 (threading.Thread, "start"), (time, "sleep")):
                stack.enter_context(patch.object(target, name, side_effect=forbidden(name)))
            # Frozen-feature MSB tests must not construct or preprocess a CNN.
            vision = ModuleType("torchvision")
            vision.__path__ = []
            vision.__file__ = "<mocked-unused-torchvision>"
            vision.__version__ = "MOCKED_UNUSED_NOT_LOADED"
            stubs = {"torchvision": vision}
            for name in ("models", "transforms"):
                module = ModuleType("torchvision."+name)
                module.__path__ = []
                module.__file__ = "<mocked-unused-torchvision-"+name+">"
                module.__getattr__ = forbidden("unused torchvision."+name)
                setattr(vision, name, module)
                stubs["torchvision."+name] = module
            stack.enter_context(patch.dict(sys.modules, stubs))
            safety = importlib.import_module("msb_execution_safety")
            for name in ("__init__", "tick_ms"):
                stack.enter_context(patch.object(safety.NativeMSBAPI, name, side_effect=forbidden(name)))
            stack.enter_context(patch.object(torch, "load", side_effect=temporary_load))
            training = importlib.import_module("msb_role_training")
            stack.enter_context(patch.object(training, "threadpool_limits", side_effect=lambda **kw: nullcontext()))
            modules = [importlib.import_module(n) for n in PHASES[args.phase]]
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    except BaseException:
        error = traceback.format_exc()
        log.write(error)
    successful = _acceptance_passed(result, forbidden_calls=forbidden_calls, harness_error=error)
    summary = dict(scope="SYNTHETIC_CONTRACT_ONLY", phase=args.phase, attempt=attempt,
                   bootstrap=bootstrap,
                   unused_dependency_stubs=["torchvision.models", "torchvision.transforms"],
                   status="PASSED" if successful else "FAILED", modules=PHASES[args.phase],
                   tests_run=0 if result is None else result.testsRun,
                   failures=0 if result is None else len(result.failures),
                   errors=0 if result is None else len(result.errors),
                   skipped=0 if result is None else len(result.skipped),
                   expected_failures=0 if result is None else len(result.expectedFailures),
                   unexpected_successes=0 if result is None else len(result.unexpectedSuccesses),
                   unmocked_boundary_calls=forbidden_calls, native_os_exercised=False,
                   real_data_opened=False, real_fits=0, temporary_synthetic_tensor_reads=len(tensor_reads),
                   source_snapshot="DIRECT_PRE_RUN_CAPTURE", harness_error=error)
    (output/"synthetic_tests.log").write_text(log.getvalue(), encoding="utf-8")
    (output/"acceptance_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if not successful: print(log.getvalue()[-10000:], flush=True)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
