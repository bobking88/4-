# Renewed Single Native Acceptance Registration

## Direct Human Authorization

The human's direct message on 2026-10-11 (Asia/Shanghai client date) was
`一次原生验收批准`. The registration identifier
`direct_user_2026-10-11_native_once` identifies that typed message; it is not a
tool-call identifier. It supersedes the consumed approval recorded in
`2026-10-10_lc_native_acceptance_code_and_incident.md`. That historical incident
is retained and is not recast as successful acceptance.

The renewed scope is exactly one registered native attempt:

- One Python child sleeping for 360 seconds, with a 420-second worker limit.
- One Python child sleeping for 30 seconds, with a 0.25-second supervisor limit,
  expected to time out and be reaped.
- A 480-second cumulative native runtime limit; AC required and monitored.
- Failure or inconclusive idle evidence stops; no retry, resume, or extra child.

No mineral image reading, real fitting, stop evaluation, training replay,
model/loss/grid changes, induced sleep, power-plan editing, display keep-awake
request, or formal report editing is authorized. A complete real experiment
requires its own renewed approval and registration.

## Unchanged Acceptance Behavior

Only the authorization identity and exact human answer in the acceptance entry
have changed. The native API bridge, budgets, child commands, cleanup logic,
read-only remote registration gate, and fixed non-resumable output root are
unchanged. The underlying safety guard and training runner are byte-identical.

The protocol is `docs/experiment_protocols/lc_native_environment_v1.json`.
The single execution command, after remote registration, is:

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -X utf8 scripts/check_lc_native_environment.py --execute-native --protocol docs/experiment_protocols/lc_native_environment_v1.json
```

The working directory is the existing `codex/theory-aware-report` worktree.
The output root is `outputs/lc_native_environment_v1/acceptance_once` and must
not exist before the attempt. The harness records the remote registration
commit it actually verifies. Read-only registration preflight precedes the
480-second native runtime budget; test and publication time are not native
runtime. No acceptance outcome is asserted by this registration.

## Fresh Verification Before Registration

- Safe authorization RED: 27 tests, exactly two expected authorization failures,
  0.405 seconds; no real native APIs or acceptance children.
- Focused GREEN: 27 tests passed, 0.386 seconds, exit 0.
- Full synthetic project regression: 510 tests passed, 155.471 seconds, exit 0.
- Controller fixtures still block unmocked native DLL calls and use a worker
  double. New tests verify the direct approval and rejection of the old approval.
- Review is author self-review, not independent review.

Fresh logs are in `outputs/lc_execution_safety_qa/`:

- `native_renewal_red_20261011.log`
- `native_renewal_green_20261011.log`
- `native_renewal_full_suite_20261011.log`

Registered SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `scripts/check_lc_native_environment.py` | `40cdd9c154fdcf0916dadff394665f9695d27b87178ac1a27fbb041e120559b7` |
| `scripts/lc_execution_safety.py` | `84089e43cb36133d41647b7576c27e4609058b9c087f4979f595452cd37bc029` |
| `scripts/run_lc_role_adaptation.py` | `b5646792731ded7a4ab537f0c67ce42bed7dbbd26c5b722a0d8e808636d22e68` |

The formal report SHA-256 remains
`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`.
Pre-existing report-figure changes are not part of this registration.

## Interpretation Limits

Success requires at least 300 seconds of current-session idle evidence after
the long child, valid child outcomes and cleanup, unchanged power settings,
continuous supervision, and released temporary requests. Insufficient idle is
`INCONCLUSIVE_IDLE_WINDOW`, not a reason to run again. Native timestamps will be
recorded as reported by the host, separately from the client registration date.

This is an engineering acceptance of idle supervision and timeout cleanup. It
does not prove that the temporary request caused wakefulness, that no other
session was active, or that LC-RFA-B improves mineral recognition. Synthetic
theory checks and real scientific validation remain distinct.
