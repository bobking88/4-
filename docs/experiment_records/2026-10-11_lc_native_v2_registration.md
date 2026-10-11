# Independent Single Native Idle Acceptance v2

## Human Authorization and Scope

The direct human reply to `call_kjhdIG03wWgnfdpyGlkiNaqh` was
`批准上述单次验收`. It approved only authorization/protocol/output metadata
updates, synthetic regression, remote registration, and one new native attempt
in an independent directory. This is not a resume or retry under an old approval.

The preceding v1 outcome remains `INCONCLUSIVE_IDLE_WINDOW`: its final
current-session idle was 200.843 seconds and the input tick changed. That record
and the consumed v1 protocol remain intact. No input source or user is inferred
from the tick change.

The v2 attempt has exactly the same acceptance behavior and budgets:

- One 360-second empty-load child; 420-second worker supervision limit.
- One child sleeping for 30 seconds; 0.25-second supervision limit, expected to
  time out, terminate, and be reaped.
- 480-second cumulative native runtime limit; AC required and monitored.
- At least 300 seconds of current-session idle required after the long child.
- Failure or inconclusive evidence stops; no retry, resume, or extra child.

The human should remain on AC and avoid keyboard/mouse input for the roughly
six-minute native interval. No reply or confirmation is needed during that
interval. No induced sleep, power-plan editing, or display keep-awake request
is performed.

No mineral images, real fitting, stop evaluation, training replay, network/loss/
grid changes, or formal report editing are authorized. A full scientific
experiment remains separately gated even if this engineering acceptance passes.

## Registration and Verification

Protocol: `docs/experiment_protocols/lc_native_environment_v2.json`.
Fresh fixed output: `outputs/lc_native_environment_v2/acceptance_once`.
The existing `codex/theory-aware-report` worktree is reused.

Only four metadata values in the acceptance entry differ from the preceding
commit: protocol identity, approval identity, exact approval answer, and output
root. The supervisor and safety guard remain byte-identical. Source identity
and the exact protocol will be remotely verified before any native request.

Fresh verification:

- Authorization/directory RED: 27 tests, two expected failures, 0.358 seconds.
- Focused GREEN: 27 tests passed, 0.402 seconds, exit 0.
- Full synthetic project regression: 510 tests passed, 156.720 seconds, exit 0.
- Controller fixtures block unmocked native DLL calls and substitute the worker
  boundary. Both consumed historical approvals are explicitly rejected.
- Local metadata audit confirms unchanged old evidence, protected scientific
  files, and an absent new output root. Review is author self-review only.

The raw test logs and audit are `outputs/lc_execution_safety_qa/native_v2_` files
ending in `_20261011.log` or `_20261011.json`.

SHA-256 values:

| File | SHA-256 |
| --- | --- |
| v2 protocol | `7f59cb9e192f9fa4fb89fa63757bc261a459056d6cb1b5285a4adf728b83797a` |
| Acceptance entry | `cd49054276ec758d2c9e1792bb6d50f2ba411d77e1af720727440607525a8ab5` |
| Safety guard | `84089e43cb36133d41647b7576c27e4609058b9c087f4979f595452cd37bc029` |
| Training runner | `b5646792731ded7a4ab537f0c67ce42bed7dbbd26c5b722a0d8e808636d22e68` |
| Formal report | `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c` |

## Single Execution and Interpretation

After the registration commit is synchronized and verified, execute once:

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -X utf8 scripts/check_lc_native_environment.py --execute-native --protocol docs/experiment_protocols/lc_native_environment_v2.json
```

Read-only remote preflight precedes the 480-second native runtime limit. The
runtime record names the registration commit actually verified. The fixed
output root refuses reuse. A result cannot authorize another attempt.

Success is current-session evidence of idle supervision and timeout cleanup,
not proof of causal sleep prevention, hardware deadlines while Windows is not
scheduled, or mineral-recognition benefit. The actual outcome is not known at
registration. No report or paper performance claim is inferred from this test.
