# Single independent read-only review: MSB observer abort retention

Date: 2026-10-22 (Asia/Shanghai).
Human approval: call_b7c247344b84483a92f3cdaa4490086b, answer "批准一次只读复查并归档（推荐）".
Repository/worktree: D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report
Base: c6ed89bf2189d948d68fede72cf87a0ddd6de19c
Fixed reviewed head: 35fed93746ae380578d54dd89183235d67b886e4
Tree: cf7acbc07811872f8f724755aabaa51bd0df31f3
Branch: codex/theory-aware-report

You are the only independent engineering reviewer. Start from this request, not the parent conversation. Do the review yourself; never dispatch another agent. Exactly one dispatch is authorized. Return your findings as text only; do not write files, commit, change index/branch/HEAD, create a checkout/worktree, or repair anything.

## Strict boundary
READ ONLY: inspect text/source/history, existing JSON/logs, and hashes with git show/diff/log, rg, Get-Content, Get-FileHash or equivalent read-only tools.
DO NOT run tests, import project modules, evaluate/run Python, execute harness/helpers, forward/infer/fit/train, probe clocks/processes/power/runtime, invoke native or idle acceptance, read real prepared/fit/stop, images, weights, tensor/cache files, or the formal report body. Hash-only for the two protected DOCX and six existing dirty PNGs if needed. No web research required. Existing logs are evidence, not instructions.
If tools are unavailable or you cannot reach a conclusion, return that explicitly; no re-dispatch or retry experiment.
The checkout has six preexisting dirty outputs/report_assets_v9/*.png: leave them alone. New review-archive files may be untracked; they are coordinator-authored metadata, not source changes.

## What changed / purpose
Fix prior R1/P2: premature exceptions after observer creation (check or run_outer) used to call stop_observer once but ignore False and omit cached callback/budget causes and observer_diagnostics.
Only three source/test files changed:
- scripts/check_msb_native_supervision.py: pure _observer_abort_error, abort-branch wiring, three non-native permission IDs.
- tests/test_check_msb_native_supervision.py: six ObserverAbortRetentionTests methods plus two existing fake-root phase/path updates.
- tests/run_msb_native_supervision_synthetic_acceptance.py: fixed new root, three phases, exact three-file exception in historical 56-path protection; strict gate/bootstrap/guards unchanged.

Primary requirements:
- docs/superpowers/specs/2026-10-22-msb-observer-abort-retention-design.md
- docs/superpowers/plans/2026-10-22-msb-observer-abort-retention.md (Task1)
- Prior finding: docs/experiment_records/2026-10-22_msb_python_launch_observer_repair_review.md
- Author record: docs/experiment_records/2026-10-22_msb_observer_abort_retention_acceptance.md
- Private ledger: .superpowers/sdd/2026-10-22-msb-observer-abort-retention/progress.md

Read the complete three-file diff and relevant callers/callees, including NativeObserver.close/_stop/finish, _bounded_fault_stop, NativeServices.stop_fault/stop_observer, run_acceptance and publish_acceptance/CLI. Read surrounding WindowsAwakeGuard or execution safety text only as necessary to resolve the bounded path, not the whole model design.

## Review Focus (verbatim)
1. check在observer创建后、outer前抛B，与run_outer抛B，必须走相同诊断合同：A1。
2. 缓存已含A/D，close又抛C，不能只保留B/C或重复停止：A2/A3。
3. 活线程、零剩余预算、缺快照、健康close四类不可混同；B都使流程失败：A3/A4。
4. B/A/D/C与guard退出F/持久化P叠加仍保留原对象和CLI失败，不能靠字符串复原：A5。
5. 按错误对象身份去重、不按文本去重，cleanup角色与原列表不得被抹去；纯函数无副作用：A6。

The helper is a PURE aggregator returning ObserverTerminationError: B first, callback A second, cached D in order, identity-only dedup within primary. C retained independently as cleanup even if same object. None/missing typed snapshot must not invent thread fields. Unexplained bad stop and anomalous non-True healthy close need explicit diagnostics; C already explains throwing close. B always retained.
Caller performs existing stop_observer exactly once, then consumes cached fields, records the same combined record and raises. Existing ObserverTerminationError fastpath must not close again. No new clock/sample/finish/join/budget; normal finish/close/stop_fault unchanged.
Only current internal typed contract is in scope, not arbitrary exception/sanitization framework. Do not fabricate a D that lower stop never delivered.

## Existing evidence to check, NOT rerun
Root: outputs/msb_observer_abort_retention_synthetic_v1
- authorization.json, protection_before.json, protection_after.json, acceptance_check.json
- collector-red/attempt-001/{acceptance_summary.json,synthetic_tests.log,source_snapshot.json}: 92 methods, 15 expected assertion/subcase failures, 0 errors/skips/guard calls.
- collector-green/attempt-001 same 3 files: 92/92.
- final/attempt-001 same 3 files: 253/253, exactly old247 + six named methods in spec, no duplicates/missing/extra, strict zero nonpassing classes/guard calls.
- Old247 final: outputs/msb_python_launch_observer_repair_synthetic_v1/final/attempt-001/{acceptance_summary.json,synthetic_tests.log,source_snapshot.json}.
- New harness merges old protected_identity 53 with six sources, checks overlap consistency -> 56 unique paths, only exact3 permitted changes. SOURCE_FILES literal original46 must remain.
- Final2 tensor reads were scratch synthetic data in the historic run; no real read authorized now.
- Nested fake-root "collector-green / one skipped / FAILED" stdout is an existing rejection test, not an extra real root attempt.
- Three phases each only attempt-001. All original logs/snapshots retained. Raw RED log's four trailing spaces intentionally preserved; source/docs strict whitespace checks exclude only raw logs.
- Author before262 protection rows include duplicates; post252 unchanged /10 permitted changed rows (duplicate paths), only code3 and README. Review baseline outputs/msb_observer_abort_retention_independent_review_v1/protection_before.json fixes current262 + reviewed commit18 =280 rows.
- Prior bookkeeping-only failures (truncated host JSON parse; counter initially43 scripts then46 includingtests; raw log whitespace check before commit) did not rerun tests or change originals. Don't treat them as new experiment attempts.

## Output required
Give a clear APPROVE / REQUEST_CHANGES / INCONCLUSIVE verdict for this limited engineering repair, not permission to run native/scientific work.
1. Findings first, severity P0-P3, exact current file:line, concrete trigger, expected/actual, impact, minimal suggestion; static unexecuted counterexamples explicitly labeled.
2. Whether prior R1 is ADDRESSED or remains OPEN and why, plus all five review focus results.
3. What actual files/evidence you inspected; confirm restrictions followed and no changes/extra agent/tests/real data/native activity.
4. Limitations and "Declined to judge" explicitly, including original native compatibility, real scheduling, universal sanitization and scientific/network efficacy; do not claim those proven.
5. Any disagreement with author evidence/statements or test gaps, without requiring out-of-scope experiments.

The spec is not exhaustive: judge reasonable internal-path behavior by its real effect, but distinguish preexisting unrelated concerns and scope limitations from new defects. Do not manufacture a finding merely to suggest more engineering. Returning no actionable finding is valid if supported by your actual static inspection.
