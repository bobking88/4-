# Single Native Acceptance Result: Inconclusive Idle Window

## Outcome

The single attempt authorized by the human's `一次原生验收批准` was executed
once. Its terminal status is **`INCONCLUSIVE_IDLE_WINDOW`**, CLI exit 1. It is
not `PASS_NATIVE_ACCEPTANCE` and will not be retried or resumed.

The executed registration was commit
`1e513665ac3dd1d221a8a2aec827215e64221150` on `codex/theory-aware-report`.
The exact executed protocol SHA-256 is
`da54960efc9adde85ea125a68593505901d43b9cacd924b0a6318d289b73fdb6`.
The pre-execution registration is retained in Git history and copied verbatim
to the evidence directory as `registered_protocol.json`.

Host-recorded interval: `2026-10-11T00:27:16.329979+08:00` to
`2026-10-11T00:33:16.691871+08:00`. Native elapsed time was
**360.359 seconds**, below the registered 480-second limit. Read-only remote
registration preflight preceded this native interval.

## Observed Native Evidence

| Check | Observation |
| --- | --- |
| Long empty-load child | PID 23128; `COMPLETE`; 360.078 seconds; exit 0; one attempt; reaped |
| Short timeout-cleanup child | PID 36100; `TIMED_OUT`; 0.250 seconds; exit 1; one attempt; reaped |
| Root temporary system request | Acquired and released according to native return-value evidence |
| Each child temporary request | Acquired and released according to native return-value evidence |
| Root maximum poll gap | 0.063 seconds, limit 30 seconds |
| Root maximum sleep-inclusive/active clock delta | 0.015003 seconds, tolerance 2 seconds |
| AC | Required throughout clock polling; connected at initial and after-idle snapshots |
| Power settings | Same active scheme and AC/DC standby settings before and after; both standby indexes 300 seconds |
| Final current-session idle | **200.843 seconds**, required minimum 300 seconds |

The last-input tick changed from `609204937` to `609492328` during the long
child. This is evidence of a current-session input timestamp change, not
evidence about who caused it or whether the input was physical or synthesized.
The long task completed under continuous supervision, but that does not replace
the required uninterrupted idle-window evidence.

A subsequent read-only `Win32_Process` query for PIDs 23128 and 36100 returned
no matching process. The outer command session also reached terminal exit 1.
No additional native child, retry, induced sleep, power-plan change, display
keep-awake request, or real training was performed.

## Evidence and Freeze

Raw records are retained under
`outputs/lc_native_environment_v1/acceptance_once/`:

- `stage_state.json`
- `worker_ledger.jsonl`
- `idle_child.log` and `timeout_child.log` (empty children emitted no text)
- `registered_protocol.json`
- `postrun_audit.json` (derived consistency checks, hashes, and freeze checks)

The streamed command output is
`outputs/lc_execution_safety_qa/native_once_console_20261011.log`.
Files are stored without newline normalization to preserve evidence bytes.

The current protocol is marked `CONSUMED_NATIVE_INCONCLUSIVE_NO_RETRY`.
It therefore rejects execution even independently of the existing fixed output
directory. Any further attempt requires fresh human approval and a separately
registered protocol, not relabeling or cleaning this evidence directory.

The safety guard, training runner, LC-RFA-B network and training code, old real
experiment protocol, and formal report remain unchanged. Formal report SHA-256:
`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`.

## Scientific Interpretation

The observed task supervision, request lifecycle, and short timeout cleanup
met their local checks. The **overall idle acceptance did not pass** because its
required idle evidence was insufficient. There is no causal proof here that a
particular request prevented sleep, and no global assertion about other users.

This is execution-safety evidence only, not evidence of recognition improvement,
theoretical superiority, or industrial sorting performance. No mineral images,
real fitting, stop evaluation, training replay, or formal report amendment were
authorized or performed. The complete real scientific experiment remains
separately gated.
