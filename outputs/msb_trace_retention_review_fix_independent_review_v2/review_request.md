# One Independent Read-Only Engineering Review

You are the sole fresh-context reviewer. Worktree: D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report
Base: 0c3fe0172e41eb1ce66e07f1101af903e1dc8596
Head: dee2be4045d325b417995da6d56943ccf06d1ee1
Human approval: call_19bef93a8b4c46389e70fa248d5be7fc. Review only; no tests/imports/edits/native or scientific execution. Do not spawn agents.

## Task

Review the three-file R1--R3 repair and existing bounded synthetic evidence. Report concrete defects by actual severity with source line references. Distinguish static conditional defects from observed native failures, implementation defects from design gaps, and proven results from authors' history statements. Give APPROVE or REQUEST_CHANGES for engineering scope only; no merge, native readiness, or scientific gate promotion.

Current head is fixed. Six preexisting dirty PNGs under outputs/report_assets_v9 are unrelated and protected: do not view, edit, stage or revert them. Do not move HEAD/index/branches, create worktrees, or write files. You return findings to parent for archival; no external messaging or fixing.

## Authoritative Documents And Source

- docs/superpowers/specs/2026-10-24-msb-trace-retention-review-fix-design.md
- docs/superpowers/plans/2026-10-24-msb-trace-retention-review-fix.md
- docs/experiment_records/2026-10-24_msb_trace_retention_review_fix_acceptance.md
- Prior review: docs/experiment_records/2026-10-23_msb_trace_publication_retention_review.md
- scripts/check_msb_native_supervision.py
- tests/test_check_msb_native_supervision.py
- tests/run_msb_native_supervision_synthetic_acceptance.py
- Necessary source-only callees may be read for control-flow reasoning. Never read real inputs/assets or report bodies, and never import project modules.

Use read-only git diff/show and text/hash inspection. No Python execution, tests, benchmarks, process/timing/power probes, runtime queries or network requests. No reproduction experiments even with synthetic data. Any unavailable/insufficient evidence must be stated; do not invent a result.

## Implemented Delta To Evaluate

R1: whole-message suppression for fixed capability markers or ten line separators; ordinary single-line numeric/address redaction/truncation retained. Original exceptions remain in memory. This is not a universal secret-detector claim.

R2: optional caller-owned errors list for _bounded_fault_stop; no lost budget cause when stop_fault raises. ControllerTracePublicationError retains original returned/stop/budget/cached objects; safe record separates cache and sets diagnostic-only. No repeat stop/query or new budget. Non-trace original error priority and other helper callers' legacy contract remain.

R3: _read_controller_failure before all other case JSON; partial/invalid receipt becomes nonempty failure sentinel. Diagnostic-only case is appended then rejected by original first check. Normal no-failure N4 snapshot/prefix checks remain strict.

Coverage adds fixed Q01--Q10. Real run_controller and actual _write_json function body are exercised behind fake interfaces, including partial write/close and row/scope/stop/callback/trace/receipt combinations.

## Review Focus (Verbatim)

1. 敏感值在标识之前/之后及各种真实分行后，输出不能残留，原对象不能改：Q01/Q02。
2. 空 errors 列表的同一性、预算前检与 stop_fault 双异常，不能因 tuple 未赋值而丢原因：Q03/Q04。
3. 缓存停止结果看似成功、callback 冲突或缓存缺失时，不能伪造停止成功或增加查询：Q04/Q05。
4. N4 缺快照且有失败/部分回执，诊断仍须进入汇总而非先抛快照错误：Q06/Q07。
5. row 已取得后 release/stop/trace/receipt 联合异常，及实际 helper 部分写入，不能互相覆盖或重试：Q08/Q09；旧正常门由 Q10 固定。

## Existing Evidence To Inspect, Not Rerun

outputs/msb_trace_retention_review_fix_synthetic_v1/
- authorization.json, acceptance_check.json, source_identity.json
- protection_before.json, protection_after.json
- raw_evidence_manifest.json, execution_ledger.md
- collector-red/attempt-001, collector-green/attempt-001, final/attempt-001:
  acceptance_summary.json, source_snapshot.json, synthetic_tests.log
- six *_entry.stdout.log / *_entry.stderr.log files (empty files are intentional originals).

Claims to verify against evidence:
- one attempt each: RED116 with30 assertion failures,0 errors/skips/unmocked; nine new failing methods and Q10/legacy106 pass.
- GREEN116/116, final277/277, old267 full names retained plus exact ten; no gate/bootstrap/module changes.
- RED driver matches baseline; tests/harness frozen after RED; GREEN/final/current identities match.
- sixteen evidence raw/canonical/blob hashes; 383 prior protection records permit only source3+README; 56-source closure53 unchanged, native46 closure43 unchanged.
- old raw native failures, report hashes and dirty PNGs preserved, not their bodies.

Previous final full-name log:
outputs/msb_trace_publication_retention_synthetic_v1/final/attempt-001/synthetic_tests.log
Bounded spec forbids extra task-done reruns, automatic reviewers or cleanup; two old fixture expected paths change with harness, but full names/body otherwise remain. Ledger statements are author records, not independent proof of complete operational history.

## Output

Findings first, ordered P0/P1/P2/P3 with path:line, trigger/control-flow, consequence, and missing test if relevant. Include strengths, evidence actually inspected, limits, and every considered behavior declined to judge (reason per item). Judge material unenumerated inputs by reasonable user expectations, not spec silence. Do not manufacture scope-creep polish. Explain whether R1/R2/R3 and joint/partial coverage are addressed or not. End with engineering verdict only.

No new native run has occurred; Windows original trace failure and MSB-RRE scientific/network/theory benefits remain unknown. Do not claim theoretical improvement from these engineering checks.

## This New Authorization

Current checkout HEAD is 4fa058ae2d86451b71cfb8e882e8d048cc390463, an archive-only descendant; the three source hashes still match fixed dee2be4. Review code through git show/diff at the fixed head when needed. A prior sole reviewer errored due to usage limits without any conclusion; its permission is consumed and its failure originals are immutable. This is exactly one newly authorized review, not a retry under old permission. No model switch is requested. Parent separately prepares/hash-checks archive metadata while you review; do not duplicate that bookkeeping or write any files.
