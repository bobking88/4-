# Native Idle Acceptance v2: Inconclusive Input Window

## Terminal Outcome

The independently approved v2 attempt executed once and ended with
**`INCONCLUSIVE_IDLE_WINDOW`**, CLI exit 1. It is not a successful full idle
acceptance. No retry, resume, or extra child was launched.

Executed registration commit:
`4a6f6c7778f15c82fa178c7aca419c5edd5cff49`.
Executed protocol SHA-256:
`7f59cb9e192f9fa4fb89fa63757bc261a459056d6cb1b5285a4adf728b83797a`.
The registered protocol is preserved in Git history and copied verbatim to
`outputs/lc_native_environment_v2/acceptance_once/registered_protocol.json`.

Host-recorded interval: `2026-10-11T11:14:44.517095+08:00` to
`2026-10-11T11:20:44.891109+08:00`. Native elapsed time was
**360.375 seconds**, below the 480-second limit. Remote registration preflight
preceded that native interval.

## Observations

| Item | Native evidence |
| --- | --- |
| Empty-load child | PID 14756; `COMPLETE`; 360.109 seconds; exit 0; one attempt; reaped |
| Short cleanup child | PID 29324; `TIMED_OUT`; 0.250 seconds; exit 1; one attempt; reaped |
| Root and child temporary requests | Acquired and released according to native return-value evidence |
| Root maximum poll gap | 0.063 seconds, allowed 30 seconds |
| Root maximum sleep-inclusive/active delta | 0.014242 seconds, allowed 2 seconds |
| AC | Required throughout polling; connected in initial and after-idle snapshots |
| Power settings | Unchanged active scheme; AC and DC standby indexes both 300 seconds |
| Initial current-session idle | 84.234 seconds |
| Final current-session idle | **129.390 seconds**, below required 300 seconds |

The last-input tick changed from `648097031` to `648412000`. The API reports a
current-session input timestamp; it does not identify the input source, person,
or physical versus synthesized input. The record therefore does not blame the
human or assert that automatic input occurred.

The child supervision and timeout-cleanup checks met their expected outcomes.
The required uninterrupted idle evidence did not. A 360-second supervised task
is not interchangeable with a 300-second current-session idle window.

## Archive and Non-Reuse

Raw `stage_state.json`, `worker_ledger.jsonl`, two empty child logs, the executed
`registered_protocol.json`, and derived `postrun_audit.json` are retained in the
independent v2 output directory. Streamed console output is
`outputs/lc_execution_safety_qa/native_v2_once_console_20261011.log`.
The audit checks terminal/ledger/console consistency, recorded source hashes,
protected scientific files, and a subsequent read-only process query.

The current v2 protocol is marked `CONSUMED_NATIVE_INCONCLUSIVE_NO_RETRY` and
rejects execution. The fixed existing output also refuses reuse. The v1
evidence and consumed protocol are unchanged. No acceptance output is deleted
or relabeled to enable another run.

Safety guard, training runner, LC-RFA-B network/loss/grid, original real
experiment protocol, and formal report remain unchanged. Formal report SHA-256:
`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`.

## Consequences for Research

Two separately authorized attempts have now produced consistent task-supervision
and timeout-cleanup observations but insufficient final idle windows. A further
identical attempt is not launched automatically. The human input clarification
below resolves the previously unanswered question; any further native acceptance
still requires an independent registration and approval.

This engineering result does not prove causal sleep prevention, network
recognition benefit, or theoretical superiority. No mineral images, real fits,
stop evaluation, replay, or report amendment were authorized or performed.
The original development remains `TIMED_OUT`, without a selection lock,
evaluation, or verified performance evidence. Network innovation still requires
the frozen strong-control experiment and theory-to-outcome analysis; neither an
idle test nor synthetic certificate checks substitute for that experiment.

## Human Follow-Up: 2026-10-11

The user subsequently confirmed that keyboard/mouse input occurred during the
six-minute acceptance window, and stated that it had no effect on the experiment.
This is a human self-report, recorded separately from the native API evidence.
It is consistent with the observed input-timestamp change, but does not establish
the source of every input event or explain the earlier v1 attempt.

No real model training or mineral-image evaluation occurred in this acceptance.
Keyboard/mouse activity does not itself change the registered network, loss,
data or protocol. It does reset the session idle timer, however, so the claim of
no effect cannot be used to waive the required 300-second final idle evidence.
The terminal outcome remains `INCONCLUSIVE_IDLE_WINDOW`; the successful child
supervision and timeout-cleanup observations remain unchanged.

The prior question about whether the user operated the keyboard/mouse is now
answered. No automatic-input diagnosis is inferred or initiated from this
confirmation. No raw logs, protocol status, acceptance result, network code or
formal report are relabeled or changed. A new acceptance would require AC power
and no keyboard/mouse operation, including other windows, for its full native
window, under a fresh single-attempt authorization. This follow-up is not that
authorization and does not authorize a real training rerun.
