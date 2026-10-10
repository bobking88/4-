"""One approved empty-load Windows acceptance; never authorizes real fitting."""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from lc_execution_safety import WindowsAwakeGuard, WindowsExecutionAPI
from run_lc_role_adaptation import run_supervised_job

SOURCE_FILES = ("scripts/check_lc_native_environment.py", "scripts/lc_execution_safety.py",
                "scripts/run_lc_role_adaptation.py")
SETTINGS = dict(protocol="lc_native_environment_v1", status="APPROVED_NATIVE_ONLY",
                scope="NATIVE_IDLE_SUPERVISION_NO_TRAINING",
                approval_call_id="call_d317f0f1f2fd4fceab4110e48d049e53",
                approval_answer="批准环境验收代码与一次原生运行（推荐）",
                idle_seconds=360, timeout_seconds=.25, max_seconds=480,
                long_worker_seconds=420, heartbeat_seconds=10, standby_seconds=300,
                require_ac=True, max_attempts=1,
                output_dir="outputs/lc_native_environment_v1/acceptance_once")


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint32), ("dwTime", ctypes.c_uint32)]


class NativeEnvironmentAPI(WindowsExecutionAPI):
    def __init__(self):
        super().__init__()
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.power = ctypes.WinDLL("PowrProf", use_last_error=True)
        self.kernel.GetSystemPowerStatus.argtypes = [ctypes.POINTER(SYSTEM_POWER_STATUS)]
        self.kernel.GetSystemPowerStatus.restype = ctypes.c_int
        self.user.GetLastInputInfo.argtypes = [ctypes.POINTER(LASTINPUTINFO)]
        self.user.GetLastInputInfo.restype = ctypes.c_int
        self.power.PowerGetActiveScheme.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.power.PowerGetActiveScheme.restype = ctypes.c_uint32
        for name in ("PowerReadACValueIndex", "PowerReadDCValueIndex"):
            function = getattr(self.power, name)
            function.argtypes = [ctypes.c_void_p]*4+[ctypes.POINTER(ctypes.c_uint32)]
            function.restype = ctypes.c_uint32
        self.kernel.LocalFree.argtypes = [ctypes.c_void_p]
        self.kernel.LocalFree.restype = ctypes.c_void_p

    def power_status(self):
        status = SYSTEM_POWER_STATUS()
        if not self.kernel.GetSystemPowerStatus(ctypes.byref(status)):
            raise RuntimeError("SUPERVISION_INTERRUPTED: AC status unavailable")
        return status

    def sample(self):
        if self.power_status().ACLineStatus != 1:
            raise RuntimeError("SUPERVISION_INTERRUPTED: AC disconnected or unknown")
        return super().sample()

    def snapshot(self):
        power = self.power_status()
        info = LASTINPUTINFO(cbSize=ctypes.sizeof(LASTINPUTINFO))
        if not self.user.GetLastInputInfo(ctypes.byref(info)):
            raise RuntimeError("SUPERVISION_INTERRUPTED: session idle evidence unavailable")
        idle_ms = (self.kernel.GetTickCount64()-info.dwTime) & 0xffffffff
        # A future or ambiguous input tick is not reliable idle-window evidence.
        if idle_ms >= 0x80000000:
            raise RuntimeError("SUPERVISION_INTERRUPTED: ambiguous session idle tick")
        return dict(ac_connected=power.ACLineStatus == 1,
                    battery_percent=None if power.BatteryLifePercent == 255 else int(power.BatteryLifePercent),
                    session_idle_seconds=idle_ms/1000., last_input_tick_ms=int(info.dwTime))

    def power_settings(self):
        scheme = ctypes.c_void_p()
        if self.power.PowerGetActiveScheme(None, ctypes.byref(scheme)) or not scheme.value:
            raise RuntimeError("SUPERVISION_INTERRUPTED: active power scheme unavailable")
        subgroup = (ctypes.c_ubyte*16).from_buffer_copy(uuid.UUID("238c9fa8-0aad-41ed-83f4-97be242c8f20").bytes_le)
        setting = (ctypes.c_ubyte*16).from_buffer_copy(uuid.UUID("29f6c1db-86da-48c5-9fdb-f2b67b1f44da").bytes_le)
        try:
            result = dict(scheme=str(uuid.UUID(bytes_le=ctypes.string_at(scheme, 16))))
            for prefix, name in (("ac", "PowerReadACValueIndex"), ("dc", "PowerReadDCValueIndex")):
                value = ctypes.c_uint32()
                if getattr(self.power, name)(None, scheme, subgroup, setting, ctypes.byref(value)):
                    raise RuntimeError("SUPERVISION_INTERRUPTED: power setting read failed")
                result[prefix+"_standby_seconds"] = int(value.value)
            return result
        finally:
            if self.kernel.LocalFree(scheme):
                raise RuntimeError("SUPERVISION_INTERRUPTED: scheme allocation release failed")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(root, protocol):
    if (set(protocol) != set(SETTINGS) | {"source_sha256"}
            or any(protocol.get(key) != value for key, value in SETTINGS.items())):
        raise ValueError("Native-only approval, scope, or fixed budget does not match.")
    sources = protocol.get("source_sha256", {})
    if set(sources) != set(SOURCE_FILES):
        raise ValueError("Registered source set does not match.")
    for relative in SOURCE_FILES:
        if digest(root/relative) != sources[relative]:
            raise ValueError("Registered source changed: "+relative)


def verify_remote_registration(root, protocol_path):
    gh = "C:/Program Files/GitHub CLI/gh.exe"
    def api(endpoint):
        response = subprocess.run([gh, "api", "repos/bobking88/4-/"+endpoint],
                                  capture_output=True, check=True, timeout=15)
        return json.loads(response.stdout)
    ref = api("git/ref/heads/codex/theory-aware-report")["object"]["sha"]
    relative = protocol_path.resolve().relative_to(root.resolve()).as_posix()
    content = api(f"contents/{relative}?ref={ref}")
    if base64.b64decode(content["content"]) != protocol_path.read_bytes():
        raise ValueError("Native protocol is not remotely registered.")
    tree_id = api(f"git/commits/{ref}")["tree"]["sha"]
    tree = api(f"git/trees/{tree_id}?recursive=1")
    if tree.get("truncated"):
        raise ValueError("Remote registration tree is incomplete.")
    entries = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "blob"}
    for name in SOURCE_FILES:
        payload = (root/name).read_bytes()
        blob = hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest()
        if entries.get(name) != blob:
            raise ValueError("Unregistered native source: "+name)
    return ref


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_acceptance(root, protocol_path, *, execute_native=False):
    if not execute_native:
        raise ValueError("An explicit --execute-native approval gate is required.")
    root = Path(root).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(root, protocol)
    output = root/protocol["output_dir"]
    if output.exists():
        raise FileExistsError("Native acceptance cannot be retried or resumed.")
    # Remote registration and validation are read-only preflight, not acceptance tasks.
    registered_commit = verify_remote_registration(root, protocol_path)
    validate_protocol(root, protocol)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", started_at=datetime.now().astimezone().isoformat(),
                  registered_commit=registered_commit, protocol_sha256=digest(protocol_path),
                  real_training_authorized=False, jobs=[], error="")
    write_json(output/"stage_state.json", result)
    started, guard, api, native_started = time.monotonic(), None, None, None
    try:
        api = NativeEnvironmentAPI()
        native_started = api.sample()[0]
        initial = api.snapshot()
        result["initial"] = initial
        if not initial["ac_connected"]:
            raise RuntimeError("SUPERVISION_INTERRUPTED: AC required")
        settings = api.power_settings()
        result["initial_settings"] = settings
        if settings["ac_standby_seconds"] != protocol["standby_seconds"]:
            raise RuntimeError("SUPERVISION_INTERRUPTED: registered standby timeout changed")
        def elapsed():
            return max(time.monotonic()-started, api.sample()[0]-native_started)
        with WindowsAwakeGuard(api=api) as guard:
            for index, (seconds, budget, expected) in enumerate(((360, 420, "COMPLETE"), (30, .25, "TIMED_OUT"))):
                guard.check()
                remaining = protocol["max_seconds"]-elapsed()
                if remaining <= 0:
                    raise RuntimeError("TIMED_OUT: cumulative native acceptance budget exhausted")
                job = dict(job_id="native-idle" if index == 0 else "native-timeout",
                           command=[sys.executable, "-c", f"import time; time.sleep({seconds})"],
                           log_path=str(output/("idle_child.log" if index == 0 else "timeout_child.log")))
                row = run_supervised_job(job, dict(per_fit_seconds=budget, heartbeat_seconds=10), remaining)
                result["jobs"].append(row)
                row["cumulative_seconds"] = max(time.monotonic()-started, guard.elapsed_seconds)
                with (output/"worker_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                guard.check()
                evidence = row.get("supervision", {})
                if (row["status"] != expected or row["attempts"] != 1 or row["alive"]
                        or row["exit_code"] is None or row["pid"] is None
                        or not evidence.get("power_request_acquired") or not evidence.get("power_request_released")
                        or (index == 0 and (row["exit_code"] != 0 or row["elapsed_seconds"] < 360))
                        or (index == 1 and row["exit_code"] == 0)):
                    raise RuntimeError("SUPERVISION_INTERRUPTED: native child outcome or cleanup invalid")
                if index == 0:
                    result["after_idle"] = api.snapshot()
            result["final_settings"] = api.power_settings()
            if settings != result["final_settings"]:
                raise RuntimeError("SUPERVISION_INTERRUPTED: power settings changed during acceptance")
            if elapsed() >= protocol["max_seconds"]:
                raise RuntimeError("TIMED_OUT: cumulative native acceptance budget exhausted")
            idle = result["after_idle"]
            result["status"] = "PASS_NATIVE_ACCEPTANCE" if idle["session_idle_seconds"] >= 300 else "INCONCLUSIVE_IDLE_WINDOW"
            result["session_idle_evidence_only"] = True
            guard.check()
    except BaseException as error:
        result["status"] = "FAILED_NATIVE_ACCEPTANCE"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["power_request"] = {} if guard is None else guard.evidence()
        elapsed_seconds = max(time.monotonic()-started, result["power_request"].get("wall_elapsed_seconds", 0.))
        if api is not None and native_started is not None:
            try:
                final_clocks = api.sample()
                elapsed_seconds = max(elapsed_seconds, final_clocks[0]-native_started)
                if result["status"] != "FAILED_NATIVE_ACCEPTANCE" and guard is not None:
                    gap = final_clocks[0]-guard.last[0]
                    active_gap = final_clocks[1]-guard.last[1]
                    if (elapsed_seconds >= protocol["max_seconds"] or gap < 0 or active_gap < 0
                            or gap > 30 or abs(gap-active_gap) > 2):
                        raise RuntimeError("SUPERVISION_INTERRUPTED: gap or budget breach at release")
                result["final_clocks"] = dict(sleep_inclusive_seconds=final_clocks[0], active_seconds=final_clocks[1])
            except BaseException as error:
                result["status"] = "FAILED_NATIVE_ACCEPTANCE"
                if not result["error"]:
                    result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_seconds"] = elapsed_seconds
        result["ended_at"] = datetime.now().astimezone().isoformat()
        result["no_retry_no_resume"] = True
        write_json(output/"stage_state.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-native", action="store_true")
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    result = run_acceptance(Path(__file__).resolve().parents[1], args.protocol, execute_native=args.execute_native)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    raise SystemExit(0 if result["status"] == "PASS_NATIVE_ACCEPTANCE" else 1)


if __name__ == "__main__":
    main()
