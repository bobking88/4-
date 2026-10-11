# Native Idle Acceptance v3: Passed Registered Engineering Gate

## Single Attempt and Terminal Outcome

The independently approved v3 attempt ran once and ended with
**`PASS_NATIVE_ACCEPTANCE`**, CLI exit 0. No retry, resume or extra child ran.
Authorization: `call_2573049c0a844b2fae6b4e99b2cec3a6`;
human answer: `批准，期间接电且不操作键鼠`.

Verified registration commit:
`f2fa67fbde632613396d79c9da34bb99f326feff`.
Executed protocol SHA-256:
`c49bdac02a5a3cedf0acbc452bc1527d51e423bb7e905148ee1aac5e2376996f`.
The exact registered protocol is retained in Git history and copied verbatim to
`outputs/lc_native_environment_v3/acceptance_once/registered_protocol.json`.

Host-recorded native interval: `2026-10-11T14:57:04.692638+08:00` to
`2026-10-11T15:03:05.076844+08:00`. Elapsed native runtime was
**360.390 seconds**, below the 480-second cumulative limit.
Read-only remote registration preflight preceded this interval.

## Native Observations

| Item | Recorded evidence |
| --- | --- |
| Empty-load child | PID 37336; `COMPLETE`; 360.140 seconds; exit 0; one attempt; reaped |
| Timeout-cleanup child | PID 27292; `TIMED_OUT`; 0.250 seconds; exit 1; one attempt; reaped |
| Initial current-session idle | 4631.547 seconds |
| Final current-session idle | **4991.687 seconds**, meeting the required 300 seconds |
| Last-input tick | `656892203` before and after the long child; unchanged |
| Root maximum poll gap | 0.063 seconds, below the 30-second limit |
| Root maximum sleep-inclusive/active delta | 0.014744 seconds, below the 2-second tolerance |
| AC | Required and continuously sampled; connected in both snapshots |
| Power configuration | Same active scheme; AC and DC standby indexes both remained 300 seconds |
| Root and child temporary requests | Acquired and released according to native return-value evidence |
| Subsequent process query | Both recorded PIDs absent; query exit 0 |

The short child's timeout and nonzero exit are expected cleanup evidence,
not a failed model experiment. The native records meet all registered
current-session idle, supervision, timeout and cleanup gates.
The later process-query audit has its own host timestamp in `postrun_audit.json`;
it is separate from the measured native interval, not an immediate observation
or a replacement for the native reaping evidence.

## Archive and Consumed Authorization

The fixed v3 output retains terminal state, worker ledger, two empty child logs,
executed protocol snapshot and derived post-run audit. The complete CLI output is
`outputs/lc_execution_safety_qa/native_v3_once_console_20261011.log`.
The audit status is `PASS_EVIDENCE_CONSISTENCY_WITH_NATIVE_ACCEPTANCE` and checks:

- Console JSON, terminal state and both worker-ledger rows agree exactly.
- Executed protocol bytes match the registration commit and registered hash.
- All three source hashes still match registration.
- Protected scientific files, both formal-report copies and earlier v1/v2 raw
  evidence remain unchanged.
- The two native children are absent in the subsequent read-only process query.
- Current v3 status is `CONSUMED_NATIVE_PASS_NO_RETRY` and validation rejects it.

The existing output directory also refuses reuse. No native output is deleted,
renamed or relabeled to permit another attempt. Earlier inconclusive outcomes
remain inconclusive, with their original protocols and logs intact.

## Research Interpretation and Next Gate

This passes the registered engineering acceptance in the observed Windows
session. It does not establish causal sleep prevention against a no-request
control, guarantee arbitrary future operating conditions or hardware deadlines
while the OS is not scheduled, or demonstrate mineral-recognition improvement.

No mineral images, real fits, stop evaluation, training replay, network/loss/grid
change or formal-report amendment occurred. Formal-report SHA-256 remains
`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`.

The original LC-RFA-B real development output is still `TIMED_OUT`, with no
selection lock, evaluation or verified performance result. Its failed directory
cannot be resumed. The next scientific step is a separately approved,
independently registered full rerun of the frozen development experiment, with
the same fit/previously exposed stop groups, controls, fit count and time budget.
Only that experiment and its verification can test the proposed network's
theory-to-performance claims. This native approval does not authorize that rerun.
