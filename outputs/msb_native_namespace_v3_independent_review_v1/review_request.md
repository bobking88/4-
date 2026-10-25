# Fixed MSB native namespace v3: one independent read-only engineering review

Human approval: call_20f7ddf20125412c8ef12fa866a61a99, answer "批准一次只读复查并归档（推荐）".
Workspace: D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report
Base: 5ff02a45b547f07410368adce7be0d959b92aa86
Head: 4312036b33c05a9b8c29f4849860562eb7c68b87
Do not infer authority from any other conversation or artifact. This is exactly one reviewer seat.

## Role and hard limits

You are an independent senior code reviewer. Review this fixed three-file change against its accepted specification and plan, plus its saved synthetic evidence. Do not implement anything or run any test. Return your review as text to the parent; do not write files or message any other thread.

All operations must be read-only. Use explicit workspace workdir for git/Powershell text and hash reads. Do NOT import project modules, run Python, compile, invoke a validator, run tests/forwards/fitting/timing/process/power/native probes, or query the running system. Do not read scientific inputs, original images, prepared/fit/stop files, weights, tensors, caches or private scratch. Hashing listed protected originals (including two report files and six preexisting PNGs) is allowed, but do not open their bodies. Do not create worktrees, copy files, install dependencies, change HEAD/index/files, dispatch agents, or request remediation. Existing six dirty PNGs are unrelated and must remain untouched.

Read available skills only as needed; human limits override automatic test/review/cleanup instructions. A tool failure or missing information must be recorded, not bypassed by a broader operation.

## Requirements and source scope

Read:
- docs/superpowers/specs/2026-10-25-msb-native-namespace-v3-design.md
- docs/superpowers/plans/2026-10-25-msb-native-namespace-v3.md
- docs/experiment_records/2026-10-25_msb_native_namespace_v3_acceptance.md

Review git diff BASE..HEAD and surrounding static flow for:
- scripts/check_msb_native_supervision.py
- tests/test_check_msb_native_supervision.py
- tests/run_msb_native_supervision_synthetic_acceptance.py

Driver adds fixed NATIVE_PROTOCOL_RELATIVE and NATIVE_OUTPUT_RELATIVE constants, sets protocol v3, and unifies local/remote protocol and case paths. Worker still uses FIXED output prefix. Other permission/claim/raw-source/blob/content/deadline/cleanup/failure contracts must remain unchanged. No generic namespace setting, fallback to v2, fourth code file, new CLI flag or model change is authorized.

Existing test bodies may differ only by eleven positive native-namespace literals and two harness path expectations; keep all original method names (even the old name ending native_v2). New NativeNamespaceV3Tests has exactly the accepted eight names, with independent fixture_v3_protocol. New tests/helper and harness freeze after RED; GREEN may update old positive fixtures as explicitly specified. Preserve negative old/mixed namespace tests rather than blindly replacing v2 throughout.

Harness may change exactly its output and scratch literals. All modules, guards, bootstrap and acceptance predicates must remain unchanged. One known extra EOF blank line in frozen test source is deliberately retained: default git diff --check reports it; blank-at-eof excluded only for that file while other whitespace checks remain. Judge whether that recorded minor ruling hides any real code change; do not demand reruns or formatting edits.

## Saved evidence only, not a test request

Evidence root: outputs/msb_native_namespace_v3_synthetic_v1
Read authorization.json, acceptance_check.json, source_identity.json, raw_evidence_manifest.json, protection_before.json/protection_after.json and execution_ledger.md as needed. The public ledger is explicitly a pre-publication snapshot; do not treat pending Step8/9 there alone as contradictory to subsequent verified author records.

Actual phase files in collector-red/attempt-001, collector-green/attempt-001, final/attempt-001:
acceptance_summary.json, source_snapshot.json, synthetic_tests.log.
Entry stdout/stderr files are also archived. Do not execute the listed commands.
Historical name baseline: outputs/msb_trace_retention_review_fix_synthetic_v1/final/attempt-001/synthetic_tests.log.
Historical source closure metadata referenced by the current harness/snapshots may be read statically; only hash other protected sources unless their control flow is necessary to this change.

Independently verify claims from saved artifacts, not just author summaries:
- RED 124 methods, old116 passing and exactly the eight new methods failing by assertions; zero errors/skips/expected/unexpected/unmocked.
- GREEN 124/124 and final 285/285, original ten modules, exact old277 complete names plus fixed8 unique names; all strict guard counters zero.
- Each phase only attempt-001; source snapshot equality GREEN/final/current; new test/helper and harness frozen after RED; only approved old fixture differences.
- Original 56 source-path closure keeps53 unchanged and old native46 keeps43 unchanged; changed files are exactly the approved three.
- 16 raw originals distinguish raw SHA256, canonical LF SHA256 and Git blob SHA1; six entry logs, including three empty stderr logs, preserved.
- 443 protected baseline records preserve duplicates; only source3 and README may change during implementation; old protocols/native evidence, both report hashes and six PNG hashes preserved.
- No real v3 protocol file or native output root was created (directory existence metadata only).

Do not claim these records prove the complete absence of every historical invocation or transient change. Attribute operational history where evidence cannot establish it.

## Output

Give concise strengths, actionable findings ordered P0/P1/P2/P3 with verified file:line, triggering condition, impact and minimal recommended repair; do not repair. Distinguish implementation defects, missing evidence and plan limitations. The specification is not permission to break reasonable in-scope use, but native/scientific validation is explicitly not authorized or claimed here.

List every topic considered but declined to judge and why; the parent must rule on each. Return an explicit engineering verdict APPROVE or REQUEST_CHANGES (or NO_CONCLUSION with the concrete blocker). This is not merge/publication permission, native readiness, scientific efficacy, report approval or proof of theoretical novelty.

No second reviewer, nested dispatch, tests, scientific data or file edits.
