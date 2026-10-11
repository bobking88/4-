# Separately Authorized LC-RFA-B Development Rerun v2

## Authorization

The direct human message `完整开发重跑` answers the pending question approving
one bounded complete development rerun. Authorization identifier:
`direct_user_2026-10-11_full_development_rerun`. This is a direct-message
identifier, not a request-user-input tool-call id. It does not reuse any
consumed native-acceptance or failed real-run approval.

Preparation base: `ed8326be2c98e58117b4dd236d51255f9af1c17c` on
`codex/theory-aware-report`. New protocol:
`docs/experiment_protocols/lc_role_adaptation_v2.json`. New fixed output:
`outputs/training/lc_role_adaptation_v2/development_fold_0`.

The approved scope remains fold 0, 340 existing fit groups and 340 already
exposed stop groups, frozen visual cache, three seeds 20261002/20261003/20261004,
at most 422 fits plus four temperature fits, 600 seconds per supervised worker
and 21600 seconds cumulative fit/evaluation budget. There is no automatic retry,
resume, grid expansion, new image/group access, training replay or formal-report
edit. A failed or insufficient-evidence batch stops. Only a successful atomic
fit lock permits one stop evaluation. Default verification replays inference
and calculations, not optimizer training.

Native v3 passed `PASS_NATIVE_ACCEPTANCE`; its result is archived at the
preparation base. Reuse the accepted safety executor without another native
run. Temporary system-awake requests are released on exit; no power scheme or
display request changes. Supervision discontinuity fails closed. The safety
gate is not scientific evidence of classification benefit or an absolute
deadline while the operating system is not scheduling the process.

## Frozen Scientific Design

The existing LC-RFA-B design, 3024-parameter network, loss, regularization and
budget grid, update checkpoints, model-selection rules, hypotheses and screen
rules are unchanged. AST comparison against the preparation base confirms the
runner changes only the draft protocol identity from v1 to v2. Safety guard,
network/training source, specification and implementation plan are unchanged.

Historical v1 failure remains `TIMED_OUT`; its partial runs are not resumed or
used for selection. All native v1/v2/v3 evidence and both formal-report copies
are preserved. Author self-review only; no independent reviewer was available.

The primary test remains R1 versus T0 and the strongest qualified control,
including S1 and R1-budget-matched S1. Each seed must meet the preregistered
NLL/correction/intrusion thresholds. Optimization truncation or unreliable
controls reject support; no result-dependent relaxation is permitted.

## Fresh Pre-Execution Verification

- RED: 24 tests; expected one superseded-identity acceptance failure and three
  new-identity stage rejections; 21.781 seconds. The raw log is cp936; a UTF-8
  display attempt failed after the tests ended. This is recorded, not discarded.
- GREEN: 24 tests passed, 21.723 seconds, exit 0.
- Full project synthetic regression: 512 tests passed, 137.996 seconds, exit 0.
- Metadata preparation first used the abbreviated native state `PASS`; it
  rejected the actual `PASS_NATIVE_ACCEPTANCE` before any protocol write or
  real run. The subsequent check uses the original state verbatim.
- Fresh preflight: `PREFLIGHT_OK`, 340 unique fit groups, class counts
  66/130/97/47, `stop_tables_parsed=false`. No real fit has started.
- Inner assignment and schedule hashes equal the prior registered artifacts.

Logs and the derived byte-preservation audit:
`outputs/lc_execution_safety_qa/real_v2_*_20261011.*`.

| Artifact | SHA-256 |
| --- | --- |
| New protocol | `2f716188280105361335129c0a2c4c56455755a415b21fe20a84a4ecc3b05555` |
| Runner | `6c29a9f454e1e445d08aa4f6fabdb61bf0145e6d931dc9c23efe81144ae6b4ff` |
| Safety guard | `84089e43cb36133d41647b7576c27e4609058b9c087f4979f595452cd37bc029` |
| Frozen visual cache | `6bbb3221293d5bf0fcadb13bcaa7a3029fdc77e8328e119820665256e3509090` |
| Inner assignment | `e603ba0b603bc903f82649dd19edaebb69f7e171be0dec2092d69048c5469927` |
| Schedule | `ada69bdb1e8a9f6a482ed94f84d4e34f4e19ebcd3b9cf6a01a2e04973bd2a7a7` |
| Fresh preflight summary | `6d6c188cce168cde004cd9578f8b6d20fb1335cff37e4608549063a1a5d90fcd` |
| Scope audit | `6cb6ad1930a97e79a673812f08c127f9b58d90bab90a9c9c11a3a3388815b80b` |
| Formal report, unchanged | `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c` |

## Execution and Evidence Gates

Publish and verify this registration commit before the following single fit:

```powershell
python -X utf8 scripts/run_lc_role_adaptation.py --protocol docs/experiment_protocols/lc_role_adaptation_v2.json --output-dir outputs/training/lc_role_adaptation_v2/development_fold_0 --stage fit
```

Then, only after `FIT_LOCKED`, execute once with `--stage evaluate` and the
same protocol/output, carrying the fit runtime into the six-hour budget.
After successful evaluation, use `--stage verify` without training-replay flags.
Archive all small logs, CSV/JSON and negative findings; retain tensors, frozen
cache and weights locally with hashes. The result is unknown at registration.
Even a successful development screen is not independent confirmation,
statistical significance, an industrial recovery/grade result or a guaranteed
target-recall theorem. The formal report remains read-only in this authorization.
