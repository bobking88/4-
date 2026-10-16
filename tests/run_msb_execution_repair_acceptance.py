"""Fixed nine-module synthetic acceptance; never a real experiment launcher."""
from contextlib import ExitStack, nullcontext
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "test_msb_execution_safety", "test_msb_role_data", "test_msb_role_model",
    "test_msb_role_controls", "test_msb_role_audit", "test_msb_role_training",
    "test_run_msb_role_mechanism", "test_analyze_msb_role_mechanism", "test_generate_msb_role_figure",
)


def main():
    os.chdir(ROOT)
    scratch = ROOT/".superpowers/sdd/2026-10-16-msb-execution-contract-repair/test-tmp"
    scratch.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir = str(scratch)
    sys.path[:0] = [str(ROOT/"tests"),str(ROOT/"scripts")]
    modules = [importlib.import_module(name) for name in MODULES]
    import torch
    import msb_execution_safety as safety
    output = ROOT/"outputs/msb_execution_repair_synthetic_v3"
    output.mkdir(parents=True,exist_ok=True)
    summary_path = output/"acceptance_summary.json"
    if summary_path.exists(): raise FileExistsError("Do not overwrite an acceptance receipt.")
    forbidden_calls = []
    tensor_reads = []
    original_load = torch.load

    def forbidden(name):
        def reject(*args,**kwargs):
            forbidden_calls.append(name)
            raise AssertionError("UNMOCKED_NATIVE_BOUNDARY_FORBIDDEN: "+name)
        return reject

    def temporary_load(file,*args,**kwargs):
        if isinstance(file,io.BytesIO): return original_load(file,*args,**kwargs)
        path = Path(file if isinstance(file,(str,bytes,os.PathLike)) else file.name).resolve()
        if not path.is_relative_to(scratch.resolve()):
            forbidden_calls.append("torch.load outside synthetic temp")
            raise AssertionError("Real tensor input forbidden in synthetic acceptance")
        tensor_reads.append(path.relative_to(scratch).as_posix())
        return original_load(file,*args,**kwargs)

    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
    with ExitStack() as stack:
        for target,name in ((safety.NativeMSBAPI,"__init__"),(safety.NativeMSBAPI,"tick_ms"),
                            (safety.ctypes,"WinDLL"),(subprocess,"Popen"),(safety.os,"_exit")):
            stack.enter_context(patch.object(target,name,side_effect=forbidden(name)))
        stack.enter_context(patch.object(torch,"load",side_effect=temporary_load))
        # The synthetic tensor algebra stays real; only OS library enumeration is stubbed.
        stack.enter_context(patch("msb_role_training.threadpool_limits",side_effect=lambda **kwargs:nullcontext()))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    successful = result.wasSuccessful() and not forbidden_calls
    summary = dict(scope="SYNTHETIC_CONTRACT_ONLY",status="PASSED" if successful else "FAILED",
                   tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                   skipped=len(result.skipped),modules=list(MODULES),native_os_exercised=False,
                   unmocked_boundary_calls=forbidden_calls,real_data_opened=False,
                   temporary_synthetic_tensor_reads=len(tensor_reads),real_fits=0,
                   historical_clock_deviation_disclosed_separately=True)
    with summary_path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2); stream.write("\n")
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if successful else 1


if __name__ == "__main__": raise SystemExit(main())
