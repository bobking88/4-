# Independent Single Native Idle Acceptance v3

## Authorization and Scope

The human reply to `call_2573049c0a844b2fae6b4e99b2cec3a6` was
`批准，期间接电且不操作键鼠`. This independently authorizes one new registered
empty-load acceptance, after synthetic regression and GitHub registration.
It does not reuse either consumed v1 or v2 approval.

The user confirmed keyboard/mouse operation during v2. Its original
`INCONCLUSIVE_IDLE_WINDOW` outcome remains unchanged. That confirmation is
separately archived in the v2 result record, not inferred from the native API.

The approved v3 behavior and budgets are unchanged:

- One 360-second empty-load child, with a 420-second supervision limit.
- One child sleeping for 30 seconds, with a 0.25-second supervision limit;
  expected timeout, termination and reaping.
- 480-second cumulative native runtime limit; AC required and monitored.
- At least 300 seconds of current-session idle after the long child.
- Failure or insufficient evidence stops; no retry, resume or extra child.

The human agreed to keep AC connected and not operate the keyboard or mouse,
including other windows, during the roughly six-minute native interval.
No reply is required during that interval. No induced sleep, power-plan change
or display keep-awake request is performed.

No mineral images, real training, stop evaluation, replay, network/loss/grid
change or formal-report edit is authorized. Passing this engineering acceptance
does not authorize a full scientific rerun or establish network benefit.

## Pre-Execution Evidence

Protocol: `docs/experiment_protocols/lc_native_environment_v3.json`.
Independent fixed output: `outputs/lc_native_environment_v3/acceptance_once`.
Existing worktree and branch: `codex/theory-aware-report`.
Preparation base: `026f8e1d4e07b52d2b3a0aabefffff3189ec489e`.

Only the protocol identity, approval identity, exact approval answer and output
root changed in the acceptance entry. AST comparison confirms identical runtime
logic after replacing those four metadata values. Safety guard and training
runner remain byte-identical.

Fresh synthetic verification completed before execution:

- RED: 27 tests, two expected authorization failures, 0.436 seconds.
- GREEN: 27 tests passed, 0.382 seconds, exit 0.
- Full project synthetic regression: 510 tests passed, 155.491 seconds, exit 0.
- Controller fixtures replace the worker and reject unmocked native DLL calls.
- All three superseded approval identities are explicitly rejected.
- Historical native evidence and consumed protocols are byte-identical to the
  preparation base; new native output does not yet exist.
- Network, loss, original real-experiment protocol, specification, implementation
  plan and both formal-report copies remain unchanged.
- Author self-review only; no independent reviewer was available.

Raw test logs and the derived scope audit are under
`outputs/lc_execution_safety_qa/native_v3_*_20261011.*`.

| File | SHA-256 |
| --- | --- |
| v3 protocol | `c49bdac02a5a3cedf0acbc452bc1527d51e423bb7e905148ee1aac5e2376996f` |
| Acceptance entry | `6ff6bf40cec8c5f0b78ffc61fd11be8aeadf35880b1ebe9e8112be0b298ac161` |
| Safety guard | `84089e43cb36133d41647b7576c27e4609058b9c087f4979f595452cd37bc029` |
| Training runner | `b5646792731ded7a4ab537f0c67ce42bed7dbbd26c5b722a0d8e808636d22e68` |
| Formal report | `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c` |

## Single Native Execution Gate

After the registration commit is published and verified, execute once:

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -X utf8 scripts/check_lc_native_environment.py --execute-native --protocol docs/experiment_protocols/lc_native_environment_v3.json
```

The entry verifies exact protocol bytes and all three source blobs at a pinned
GitHub branch commit, then revalidates local source hashes. This read-only
registration preflight precedes the 480-second native runtime limit. The result
names the verified registration commit. An existing fixed output refuses reuse.

The outcome is unknown at registration. Success would be current-session
evidence of idle supervision and timeout cleanup, not a counterfactual proof of
causal sleep prevention or a mineral-recognition result. Original real model
development remains failed and unverified; its full rerun is separately gated.
