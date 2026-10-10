# Native Environment Acceptance: Code and Test-Boundary Incident

## Scope and Authorization

The direct human reply to `call_d317f0f1f2fd4fceab4110e48d049e53` approved
independent acceptance code plus one native attempt: one 360-second empty-load
child and one 0.25-second timeout-cleanup child, 480 seconds cumulative native
acceptance budget, AC required, failure stops without retry.

No mineral images, real fitting, stop evaluation, training replay, model/loss/grid
changes, induced sleep, power-plan edits, display keep-awake request, or formal
report changes were authorized. Full real development remains separately gated.

## Implemented Code

`scripts/check_lc_native_environment.py` reuses `WindowsAwakeGuard` and
`run_supervised_job`, without modifying either module. The only child commands
are Python `time.sleep(360)` and `time.sleep(30)` (the latter has a 0.25-second
supervisor limit). A fixed output root refuses retry or resume.

The entry validates the exact approval/schedule and source hashes, verifies
remote protocol/source bytes, then revalidates local source hashes. It checks
AC during clock polling, reads session idle evidence and AC/DC standby settings,
and records acquisition/release and child cleanup. No settings are written.

Successful idle evidence requires at least 300 seconds of current-session idle
at the end of the long child. Insufficient idle is `INCONCLUSIVE_IDLE_WINDOW`,
not success and not permission for an additional attempt. This is session-only
evidence, not a global assertion about all users or a causal proof that the
power request prevented sleep. Other power requests and host behavior can matter.

Native API documentation:

- [SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate): thread-affine temporary request, not prevention of explicit user sleep/lid closure.
- [SYSTEM_POWER_STATUS](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-system_power_status): AC connected must be 1; unknown is rejected.
- [GetLastInputInfo](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getlastinputinfo): current-session input tick only.
- [PowerGetActiveScheme](https://learn.microsoft.com/en-us/windows/win32/api/powersetting/nf-powersetting-powergetactivescheme): read active scheme and free allocated GUID.
- [PowerReadACValueIndex](https://learn.microsoft.com/en-us/windows/win32/api/powrprof/nf-powrprof-powerreadacvalueindex), [PowerReadDCValueIndex](https://learn.microsoft.com/en-us/windows/win32/api/powrprof/nf-powrprof-powerreaddcvalueindex): read, not edit, standby indexes.

## Incident: Not a Valid Native Acceptance

During a failing controller test, two errors combined:

1. The new entry initially omitted the second source-hash validation after remote
   preflight. The test intentionally changed a registered source during preflight.
2. That test omitted the supervised-child boundary double, unlike the passing
   controller tests. The real supervisor therefore started an actual empty-load
   child and used its actual Windows keep-awake API. The outer host and remote
   registration were still simulated; no genuine registered native acceptance
   was executed.

Observed tool evidence:

- Test session `92732`, child PID `20256`, parent PID `1968`.
- Read-only `Win32_Process` inspection confirmed the child executable was the
  project's training-environment Python, with command `-c "import time; time.sleep(360)"`.
- Child creation was `2026-10-10 15:54:33` local time. Console heartbeats reached
  300.719 seconds; they are observations, not a precise measurement of active CPU
  time or a complete native acceptance ledger.
- The first termination request was denied because the PID had not yet been
  independently inspected. After the read-only identity check, targeted
  termination succeeded. No alternative was used to bypass the denial.
- The test session finished with exit 1: 20 tests in 302.325 seconds, two failures.
  The timeout-cleanup child was not launched in this incident.
- A subsequent read-only query found neither child PID 20256 nor parent PID 1968.
- `powercfg /requests` could not be inspected without an elevated administrator
  console (exit 1). Native request release is therefore not claimed as a validated
  outcome of this incident. No elevation or power-plan change was attempted.

The temporary test output was removed by the test fixture's normal cleanup.
This account is based on tool observations; it is not a reconstructed raw native
acceptance log. No mineral data or model execution occurred in the child.

The incident is invalid acceptance, not a successful anti-sleep test. No second
native run is launched under the original one-attempt approval. The native
protocol is on hold and rejects execution; another native attempt requires fresh
explicit approval and preregistration. Real training remains unauthorized.

## Corrective Verification

Controller tests now install the supervised-worker double at fixture level and
block any unmocked `WinDLL` access. Native-bridge tests use explicit DLL-boundary
doubles. The second source-hash validation and post-release clock/budget check
were implemented only after observing their failing tests.

- Initial 18 tests failed because the module was absent.
- Safe final-boundary RED: 25 tests, two expected failures, 0.303 seconds.
- Focused GREEN: 25 tests passed, 0.297 seconds, exit 0.
- Full project GREEN: 508 tests passed, 139.575 seconds, exit 0.
- Review was by the implementation author; no independent reviewer was available.

Logs: `outputs/lc_execution_safety_qa/native_final_boundary_red.log`,
`native_acceptance_green.log`, and `native_acceptance_full_suite.log`.
The on-hold protocol rejects execution before any native request or child launch.

The acceptance code is implemented but effective native idle supervision remains
unverified. Neither this engineering result nor the incident validates LC-RFA-B
model effectiveness, theory superiority, or industrial sorting performance.
