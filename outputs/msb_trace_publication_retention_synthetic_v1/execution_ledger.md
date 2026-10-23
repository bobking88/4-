# SDD ledger - plan: docs/superpowers/plans/2026-10-23-msb-trace-publication-retention.md

- BASE: 78000c9f929222e843692200bc66283f86f5b0dc
- Human approval: call_13470536c5a3431aad88e4688fee8725, three source files and bounded synthetic only.
- Task 1: in_progress. Step 1 complete: linked codex/theory-aware-report worktree verified; empty index; new output/workspace absent before claim; six preexisting PNG changes preserved.
- Pre-flight: one task, shared TracePublicationState connects TraceAPI and finally; named helper and failure-record contracts agree between spec and plan.
- Ruling: use apply_patch for this ledger instead of SDD shell helpers; do not run setup/baseline or task-done extra tests. The explicitly approved three-phase cap overrides generic skill runners; cost: ledger verification is authored, not a second runner.
- Ruling: no fresh review agent or workspace deletion. Human requires separate review approval and preservation of every evidence artifact; cost: this work ends author-verified only.
- Steps 2-4 complete. Added exactly fourteen named methods and memory-only trace I/O double; changed only two old path expectations and harness output/scratch paths.
- collector-red/attempt-001: exit 1, 106 tests, fourteen missing-behavior assertion failures, zero errors/skips/expected failures/unexpected successes/unmocked calls; original 92 methods pass. Session 16164 terminal. Evidence retained under the new root.
- Ruling: reuse the existing harness without extra generic skill test commands; the only native executable is the approved synthetic harness entry. Its internal native/process/thread/runtime interfaces stay mocked.
- Steps 5-6 complete. Driver implements trace-only staged I/O, shared failure latch, separate redacted controller failure receipt and explicit collector rejection; generic _write_json and base supervisors unchanged.
- collector-green/attempt-001: exit 0, 106/106, zero failures/errors/skips/expected failures/unexpected successes/unmocked calls. New helper body runs behind memory file I/O; no native services constructor used.
- Ruling: the exclusive on-disk failure receipt says ATTEMPTED; only the in-memory/CLI record can say PERSISTED after write and original-budget postcheck. No second write is allowed, so disk alone never proves delivery; cost: consumers must keep distinguishing persistence from receipt contents.
- Ruling: the already approved ten-module final suite includes synthetic tensor/network contract tests, not scientific data/model inference or real training. Preserve them without expanding scope; do not label those synthetic contracts as scientific forwards or efficacy evidence.
- Steps 7-8 complete: final/attempt-001 exit 0, 267/267, exact old 253 names retained and fourteen planned names added; zero failures/errors/skips/expected failures/unexpected successes/unmocked calls. Two existing scratch synthetic tensor reads only. Session 35285 terminal.
- Author static self-review: generic _write_json unchanged; first-failure latch stops the second trace I/O; one stream close retains ordered dual errors; row/body/stop/trace/receipt records stay separate; collector rejects failure receipts; existing normal and expected-fault tests retained.
- No independent review claimed. No native acceptance or real scientific run. First Windows OS cause remains unknown; network innovation efficacy remains unvalidated.
- Archival note: first Copy-Item wrapper had a PowerShell parser error before executing any copy; it was corrected, with no tests, native calls or output attempts added.
- Task 1 implementation and synthetic acceptance complete; Step 9 scoped commit and remote actual-content readback in progress. Preserve this ledger and every original evidence artifact; do not run task-done or another suite.
