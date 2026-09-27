# TC-OOS-RSG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and formally evaluate a target-constrained out-of-sample risk projection network whose experiments directly test the proposed target-posterior safety theory and hard-negative correction mechanism.

**Architecture:** Keep the existing EfficientNet-B0 hierarchical expert and residual hard-negative verifiers frozen, produce a verifier-complete equal-fusion fallback posterior and an OOS-gated candidate posterior, then train a lightweight evidence-only projection head. Apply a deterministic per-sample target-posterior cap after both verifier branches, and evaluate the locked system with three-fold group-isolated outer predictions.

**Tech Stack:** Python 3, PyTorch/torchvision, NumPy, scikit-learn metrics, standard-library CSV/JSON/hash utilities, `unittest`, matplotlib, python-docx, Git.

**Spec:** `docs/superpowers/specs/2026-09-27-tc-oos-rsg-design.md`

## Global Constraints

- Do not modify historical OOS-RSG prediction or metric files; all new outputs use a `tc_oos_rsg` namespace.
- Apply target safety projection after each branch has independently passed through the residual Ti-bearing and metallic verifiers.
- The projection head consumes only the 18-dimensional evidence vector fixed in the spec; do not add backbone features during the main experiment.
- Freeze the expert, species/role heads, verifiers, and selected OOS gate while fitting the projection head.
- Pre-register `epsilon_target` as exactly `{0, 0.005, 0.01, 0.02, 0.04, 0.08}`.
- Treat `split_group_id` as indivisible in every outer and inner split.
- Exclude the spent `final_eval` rows from new method development and formal outer evaluation.
- Lock model/configuration hashes before loading outer-evaluation image tensors.
- Commit manifests, configuration, histories, predictions, metrics, figures, and analysis; never commit `.pt` weights or copyrighted image files.
- Preserve all unrelated dirty files in the worktree.

## Review Focus

- `epsilon_target=0` must forbid target-posterior decrease while still permitting candidate routing when the candidate does not reduce target probability; Task 1 pins this behavior.
- NaN, infinite, negative, wrong-width, or non-simplex posteriors must fail clearly rather than silently producing invalid safety claims; Task 1 covers these cases.
- Multi-image duplicate groups and cross-label conflict groups must never cross outer or inner subsets; Task 2 tests group integrity and leakage rejection.
- The 13 development-pool titanomagnetite rows must be distributed 4/4/5 across outer folds when feasible, without claiming that all 35 collected rows entered training; Task 2 and Task 3 pin the count audit.
- Outer-evaluation tensors must remain unread until expert, gate, projector, `epsilon_target`, and file hashes are locked; Task 5 uses an event-trace test to enforce ordering.

---

### Task 1: Core Target-Safe Projection Mathematics

**Files:**
- Create: `scripts/tc_oos_rsg.py`
- Create: `tests/test_tc_oos_rsg.py`

**Interfaces:**
- Produces: `build_projection_evidence(q0, q_candidate, candidate_gate, ti_target_probability, metallic_target_probability) -> torch.Tensor`
- Produces: `TargetRiskProjectionHead(input_dim: int = 18, hidden_dims: tuple[int, int] = (64, 16), dropout: float = 0.10)`
- Produces: `apply_target_safe_projection(q0, q_candidate, raw_route, epsilon_target, target_index: int = 0, eta: float = 1e-12) -> dict[str, torch.Tensor]`; `eta` is only the numerical zero threshold, while positive target drops use the exact cap `min(1, epsilon_target / drop)`.
- Output keys: `route_cap`, `projected_route`, `final_probabilities`, `signed_target_change`, `positive_target_harm`, `identity_residual`.

- [ ] **Step 1: Write failing mathematical-invariant tests**

Add tests named:

```python
test_projection_returns_simplex_probabilities
test_target_drop_never_exceeds_epsilon
test_zero_budget_still_routes_when_candidate_does_not_reduce_target
test_exact_target_change_identity_has_zero_residual
test_projection_head_emits_one_finite_probability_per_row
test_invalid_probabilities_and_shapes_are_rejected
```

Use four-class tensors and assert maximum safety/identity error is at most `1e-7` in float64.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -m unittest tests.test_tc_oos_rsg -v
```

Expected: import failure because `scripts/tc_oos_rsg.py` does not exist.

- [ ] **Step 3: Implement evidence construction, MLP, validation, and projection**

Implement the exact signatures above. Evidence order must be `q0(4), q_candidate(4), abs_difference(4), candidate_gate(1), ti(1), metallic(1), H(q0)(1), H(q_candidate)(1), JS(q0,q_candidate)(1)`.

- [ ] **Step 4: Run focused and existing network tests**

Run the focused command plus:

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -m unittest tests.test_hrgv_network tests.test_tc_oos_rsg -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/tc_oos_rsg.py tests/test_tc_oos_rsg.py
git commit -m "feat: add target-safe risk projection layer"
```

### Task 2: Three-Fold Group-Isolated Manifest Builder

**Files:**
- Create: `scripts/build_tc_oos_rsg_manifests.py`
- Create: `tests/test_build_tc_oos_rsg_manifests.py`

**Interfaces:**
- Consumes: rows from `outputs/training/oos_rsg_confirmation_manifests_v1/partition.csv`.
- Produces: `build_outer_folds(rows, seed: int = 20260927, fold_count: int = 3) -> list[dict[str, str]]`.
- Produces: `build_inner_subsets(outer_train_rows, seed: int) -> list[dict[str, str]]`.
- Produces: `audit_tc_manifests(partitioned_rows) -> dict[str, object]`.
- Adds fields: `outer_fold`, `tc_subset`, `tc_protocol_version`.

- [ ] **Step 1: Write failing split and audit tests**

Add tests asserting:

```python
test_outer_partition_is_deterministic_and_group_intact
test_outer_folds_jointly_cover_every_development_row_once
test_inner_subsets_follow_registered_names_and_group_isolation
test_spent_final_eval_rows_are_rejected
test_titanomagnetite_is_balanced_four_four_five_when_thirteen_groups_exist
test_cross_subset_group_leakage_is_rejected
```

- [ ] **Step 2: Verify failure**

Run `python -m unittest tests.test_build_tc_oos_rsg_manifests -v` with the training venv. Expected: import failure.

- [ ] **Step 3: Implement deterministic grouped stratification**

Use a deterministic greedy assignment over `(mineral_label, four_class_label)` group counts, with stable SHA-256 tie-breaking from `seed|split_group_id`. Inner subset proportions are exactly `60/10/10/10/10` for `expert_fit`, `expert_stop`, `gate_fit`, `gate_stop_projector_fit`, and `projector_stop`.

- [ ] **Step 4: Run focused tests and an in-memory full-data audit**

Expected audit facts: 5,087 development rows, 994 target rows, 13 development titanomagnetite rows, no cross-fold group overlap, and every row assigned once.

- [ ] **Step 5: Commit**

```powershell
git add scripts/build_tc_oos_rsg_manifests.py tests/test_build_tc_oos_rsg_manifests.py
git commit -m "feat: build grouped TC-OOS-RSG folds"
```

### Task 3: Register the Formal Protocol and Generate Locked Manifests

**Files:**
- Create: `docs/experiment_records/2026-09-27_tc_oos_rsg_protocol.md`
- Create: `outputs/training/tc_oos_rsg_manifests_v1/partition.csv`
- Create: `outputs/training/tc_oos_rsg_manifests_v1/audit.json`
- Create: `outputs/training/tc_oos_rsg_manifests_v1/registered_protocol.json`
- Create: `outputs/training/tc_oos_rsg_manifests_v1/fold_{0,1,2}/*.csv`

**Interfaces:**
- Consumes: Task 2 CLI and the approved spec.
- Produces: immutable manifest hashes and exact training/selection rules consumed by Tasks 4 and 5.

- [ ] **Step 1: Add CLI integration test to Task 2 tests**

`test_cli_writes_all_fold_manifests_and_hashes` must assert the expected filenames, SHA-256 entries, subset counts, and absence of `final_eval` image IDs.

- [ ] **Step 2: Run the CLI test and confirm failure before CLI support**

- [ ] **Step 3: Implement CLI and write the protocol before any model run**

Protocol must record seeds, fold/subset policy, epsilon grid, selection rule, metrics, bootstrap count, promotion criteria, titanomagnetite 35/23/13 count definitions, and the rule that `final_eval` is spent.

- [ ] **Step 4: Generate manifests and independently audit them**

Run the builder against the current partition. Re-read every CSV, recompute hashes, and assert exact agreement with `registered_protocol.json`.

- [ ] **Step 5: Commit**

```powershell
git add scripts/build_tc_oos_rsg_manifests.py tests/test_build_tc_oos_rsg_manifests.py docs/experiment_records/2026-09-27_tc_oos_rsg_protocol.md outputs/training/tc_oos_rsg_manifests_v1
git commit -m "experiment: register TC-OOS-RSG protocol"
```

### Task 4: Train and Lock One Expert per Outer Fold

**Files:**
- Create: `scripts/run_tc_oos_rsg_experts.py`
- Create: `tests/test_run_tc_oos_rsg_experts.py`
- Reuse: `scripts/hrgv_network.py`
- Reuse: `scripts/train_hrgv_mineral_classifier.py`
- Reuse: `scripts/train_mineral_classifier.py`

**Interfaces:**
- Consumes: `fold_N/expert_fit.csv`, `fold_N/expert_stop.csv`.
- Produces: `outputs/training/tc_oos_rsg_v1/fold_N/expert/{environment.json,best_validation_metrics.json,metrics_history.csv,selection_lock.json}` and a local untracked `best_model.pt`.
- Produces: a matched single-head EfficientNet-B0 M0 baseline under `fold_N/baseline/`, trained from the same `expert_fit` and selected on the same `expert_stop`, with its own lock and untracked weight.
- Produces: `audit_expert_manifests(fit_records, stop_records, outer_eval_records) -> dict[str, object]`.

- [ ] **Step 1: Write failing isolation and lock tests**

Tests must reject group overlap, reject nonmatching role/species maps, assert the default EfficientNet-B0 configuration for both hierarchical and M0 models, and verify both `selection_lock.json` files contain manifest/config/checkpoint hashes.

- [ ] **Step 2: Verify tests fail**

- [ ] **Step 3: Implement the fold-aware wrapper around the existing trainer**

Do not duplicate model training logic. Pass explicit fit/stop manifests, fixed seed `20260927 + fold`, ImageNet initialization, and existing loss configurations through the established hierarchical and single-head training APIs.

- [ ] **Step 4: Run wrapper tests and a one-epoch CPU synthetic smoke**

Expected: a checkpoint is created locally, lock hashes resolve, and no outer-evaluation record is opened.

- [ ] **Step 5: Commit code and non-weight smoke metadata**

```powershell
git add -- scripts/run_tc_oos_rsg_experts.py tests/test_run_tc_oos_rsg_experts.py outputs/training/tc_oos_rsg_smoke ':(exclude)outputs/training/tc_oos_rsg_smoke/**/*.pt'
git commit -m "feat: train locked TC-OOS-RSG fold experts"
```

### Task 5: OOS Candidate Gate, Projection Training, and Lock-Before-Evaluation

**Files:**
- Create: `scripts/run_tc_oos_rsg_experiments.py`
- Create: `tests/test_run_tc_oos_rsg_experiments.py`
- Reuse: `scripts/run_seen_unseen_gate_study.py`
- Reuse: `scripts/tc_oos_rsg.py`

**Interfaces:**
- Consumes: one locked fold expert plus `gate_fit`, `gate_stop_projector_fit`, `projector_stop`, and `outer_eval` manifests.
- Consumes: the fold-matched M0 baseline lock for direct outer-fold comparison.
- Produces: `train_oos_gate(...) -> SelectedGate`, where `SelectedGate` stores `state_dict`, `epoch`, `stop_metrics`, and `state_sha256`.
- Produces: `train_projection_head(...) -> ProjectionCandidate`, where `ProjectionCandidate` stores `state_dict`, `epoch`, `epsilon_target`, `stop_metrics`, and `state_sha256`.
- Produces: `select_projection_candidate(candidates, baseline_metrics, recall_margin: float = 0.01) -> ProjectionCandidate | FallbackSelection`; `FallbackSelection` stores the baseline metrics and reason `no_safe_candidate`.
- Produces: branch, ablation, and final outer prediction CSVs only after `selection_lock.json` exists.

- [ ] **Step 1: Write failing pipeline tests**

Tests must cover:

```python
test_gate_and_projector_use_disjoint_update_sets
test_candidate_posteriors_apply_verifiers_before_projection
test_selector_uses_recall_then_nll_then_smaller_epsilon
test_selector_returns_explicit_fallback_when_no_candidate_is_safe
test_outer_tensor_loader_runs_only_after_selection_lock
test_all_six_registered_epsilon_values_are_evaluated
```

The ordering test records events and expects `lock_written` before `outer_tensor_loaded`.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement cache extraction and frozen-branch computation**

Reuse `extract_cache` and `final_probabilities`; compute `q0` with gate `0.5` and `q_candidate` with the selected OOS gate, each through residual verifiers before Task 1 projection.

- [ ] **Step 4: Implement projector training, lexicographic selection, ablations, and lock**

Evaluate M1-M6 and component ablations using the same frozen branch caches. Include `epsilon_target`, epoch, seed, manifest hashes, model hashes, and selection reason in the lock.

- [ ] **Step 5: Run focused tests and synthetic end-to-end smoke**

Expected: no safety violations, no outer read before lock, and fallback selection works under deliberately unsafe synthetic candidates.

- [ ] **Step 6: Commit**

```powershell
git add -- scripts/run_tc_oos_rsg_experiments.py tests/test_run_tc_oos_rsg_experiments.py outputs/training/tc_oos_rsg_smoke ':(exclude)outputs/training/tc_oos_rsg_smoke/**/*.pt'
git commit -m "feat: run target-constrained OOS projection"
```

### Task 6: Group-Paired Analysis and Promotion Decision

**Files:**
- Create: `scripts/analyze_tc_oos_rsg_experiments.py`
- Create: `tests/test_analyze_tc_oos_rsg_experiments.py`

**Interfaces:**
- Produces: `calculate_role_metrics(rows) -> dict[str, float]`.
- Produces: `cluster_paired_bootstrap(baseline_rows, method_rows, iterations: int = 10000, seed: int = 20260927) -> dict[str, object]`.
- Produces: `verify_theory_invariants(rows, tolerance: float = 1e-6) -> dict[str, object]`.
- Produces: `assess_tc_promotion(summary, recall_margin: float = 0.01, intrusion_margin: float = 0.01) -> dict[str, object]`.

- [ ] **Step 1: Write failing statistical and promotion tests**

Use deterministic synthetic groups to assert paired resampling, NLL CI direction, one-sided target-recall noninferiority, zero safety violations, at least two favorable NLL folds, one improved intrusion rate, and fallback on any failed mandatory criterion.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement alignment, metrics, group bootstrap, invariant checks, and promotion logic**

Reject any mismatch in image ID, group ID, true label, or fold assignment before statistics.

- [ ] **Step 4: Run focused tests and compare metric calculations against existing analysis code**

Expected differences are at most `1e-12` for shared metrics on the same rows.

- [ ] **Step 5: Commit**

```powershell
git add scripts/analyze_tc_oos_rsg_experiments.py tests/test_analyze_tc_oos_rsg_experiments.py
git commit -m "feat: analyze TC-OOS-RSG theory and promotion"
```

### Task 7: Architecture Figure and Theorem Evidence Artifact

**Files:**
- Create: `scripts/generate_tc_oos_rsg_figure.py`
- Create: `tests/test_generate_tc_oos_rsg_figure.py`
- Create after execution: `outputs/paper_figures_v4/fig_tc_oos_rsg_architecture.{png,pdf,svg}`
- Create after execution: `outputs/paper_figures_v4/fig_tc_oos_rsg_architecture_source.json`
- Create after execution: `outputs/theory/tc_oos_rsg_invariants.json`

**Interfaces:**
- Consumes: locked architecture constants and Task 6 invariant output.
- Produces: an editable vector network figure showing both verifier-complete branches, evidence head, deterministic cap, fallback, and theorem labels.

- [ ] **Step 1: Write failing figure-content tests**

Tests assert all output formats exist, the source JSON identifies the 18 evidence dimensions, and the SVG contains labels for `q0`, `q_phi`, `rho_tilde`, `epsilon_T`, verifier placement, and the target bound.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement figure generation and invariant JSON export**

Use matplotlib for publication PNG/PDF/SVG. Do not use a decorative architecture image; every block and arrow must correspond to implemented tensors.

- [ ] **Step 4: Run tests and visually inspect the PNG**

Use `view_image`; verify legibility at report-column scale and no overlap.

- [ ] **Step 5: Commit**

```powershell
git add scripts/generate_tc_oos_rsg_figure.py tests/test_generate_tc_oos_rsg_figure.py outputs/paper_figures_v4 outputs/theory/tc_oos_rsg_invariants.json
git commit -m "docs: add TC-OOS-RSG architecture evidence"
```

### Task 8: Smoke Run and One-Fold Pilot Gate

**Files:**
- Create after execution: `outputs/training/tc_oos_rsg_smoke/`
- Create after execution: `outputs/training/tc_oos_rsg_v1/fold_0/`
- Create: `docs/experiment_records/2026-09-27_tc_oos_rsg_pilot_results.md`

**Interfaces:**
- Consumes: Tasks 3-7.
- Produces: one complete locked outer-fold result to verify runtime, memory, selection, and theoretical invariants before formal three-fold execution.

- [ ] **Step 1: Run all 264 existing tests plus new tests before GPU work**

Run full `unittest discover`. Expected: all tests pass and the count increases by the new cases.

- [ ] **Step 2: Run a reduced-image, one-epoch end-to-end smoke**

Expected: all M1-M6 outputs, all six epsilon candidates, zero safety violations, and lock-before-evaluation audit success.

- [ ] **Step 3: Run fold 0 pilot with the registered protocol**

Do not edit the epsilon grid or promotion rules after viewing results.

- [ ] **Step 4: Audit the pilot as a method-validity gate**

Check finite losses, nonconstant routes, route-cap activation, exact decomposition residual, group isolation, and whether the learned projector differs from both fallback and unbounded routing. A failed gate triggers debugging, not formal runs.

- [ ] **Step 5: Record and commit non-weight pilot evidence**

```powershell
git add -- docs/experiment_records/2026-09-27_tc_oos_rsg_pilot_results.md outputs/training/tc_oos_rsg_smoke outputs/training/tc_oos_rsg_v1/fold_0 ':(exclude)outputs/training/tc_oos_rsg_smoke/**/*.pt' ':(exclude)outputs/training/tc_oos_rsg_v1/**/*.pt'
git commit -m "experiment: validate TC-OOS-RSG pilot"
```

### Task 9: Formal Three-Fold Experiment and Theory-Linked Analysis

**Files:**
- Create after execution: `outputs/training/tc_oos_rsg_v1/fold_{0,1,2}/`
- Create after execution: `outputs/business_metrics/tc_oos_rsg/formal/`
- Create: `docs/experiment_records/2026-09-27_tc_oos_rsg_formal_results.md`

**Interfaces:**
- Consumes: locked protocol and validated pilot code.
- Produces: pooled OOF predictions, fold metrics, epsilon selections, ablations, 10,000-resample paired intervals, theorem audit, and an explicit promotion decision.

- [ ] **Step 1: Freeze code commit and record its hash in the protocol result header**

- [ ] **Step 2: Run all three outer folds without changing code or registered hyperparameters**

If a run fails operationally, resume the same run from its recorded state; do not replace its seed or fold.

- [ ] **Step 3: Generate pooled OOF analysis and promotion decision**

Report M1-M6, every component ablation, every epsilon, all three fold results, pooled CIs, safety violations, and titanomagnetite sample counts.

- [ ] **Step 4: Interpret results against each theoretical claim**

The results document must contain a table mapping: probability validity, target bound, exact decomposition, OOS necessity, learned projection benefit, verifier-feature benefit, target-recall noninferiority, and hard-negative correction to their direct experimental evidence.

- [ ] **Step 5: Commit all non-weight formal artifacts**

```powershell
git add -- outputs/training/tc_oos_rsg_v1 outputs/business_metrics/tc_oos_rsg/formal docs/experiment_records/2026-09-27_tc_oos_rsg_formal_results.md ':(exclude)outputs/training/tc_oos_rsg_v1/**/*.pt'
git commit -m "experiment: evaluate TC-OOS-RSG theory"
```

### Task 10: Update Technical Report and Paper Foundation

**Files:**
- Create: `tools/append_tc_oos_rsg_to_official_report.py`
- Create: `tests/test_append_tc_oos_rsg_to_official_report.py`
- Modify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx`
- Modify or create: `docs/paper_drafts/tc_oos_rsg_methods_theory_zh.md`
- Modify or create: `docs/paper_drafts/tc_oos_rsg_experiment_results_zh.md`

**Interfaces:**
- Consumes: Task 7 figures and Task 9 locked analysis.
- Produces: report sections for motivation, network, formulas, proofs, protocol, ablations, results, limitations, and promotion outcome.

- [ ] **Step 1: Write failing report-builder tests**

Tests assert one TC-OOS-RSG appendix/section title, formulas and theorem labels, one architecture figure, tables for M1-M6 and epsilon sensitivity, the 35/23 titanomagnetite wording, and no claim of industrial grade prediction or hard-recall guarantee.

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement idempotent report append/update logic and manuscript sections**

The report must state the actual promotion outcome. If the method fails, retain the theory and negative empirical evidence without rewriting it as success.

- [ ] **Step 4: Render and visually verify the DOCX**

Use the document skill render workflow when available. Inspect every added page for formula clipping, table overflow, duplicate headings, figure readability, and page breaks. If page rendering remains unavailable, report structural QA only and do not claim visual QA.

- [ ] **Step 5: Run full tests and commit**

```powershell
git add tools/append_tc_oos_rsg_to_official_report.py tests/test_append_tc_oos_rsg_to_official_report.py docs/paper_drafts 结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx
git commit -m "docs: integrate TC-OOS-RSG theory and evidence"
```

### Task 11: Final Reproducibility Audit and GitHub Synchronization

**Files:**
- Modify: `README.md`
- Modify or create: `docs/reproducibility/tc_oos_rsg_reproduction.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: exact commands, environment information, artifact hashes, excluded weights statement, and verified branch state.

- [ ] **Step 1: Run the complete test suite and `git diff --check`**

Expected: all tests pass; no whitespace errors.

- [ ] **Step 2: Recompute hashes and verify every documented result against source JSON/CSV**

Check report values, figure source JSON, pooled OOF row count, group uniqueness, prediction alignment, selected epsilon values, and promotion status.

- [ ] **Step 3: Confirm no weights or raw images are staged**

Run `git diff --cached --name-only` and reject `.pt`, `.pth`, `.jpg`, `.jpeg`, `.png` dataset images, or unrelated dirty assets. Paper figures are allowed.

- [ ] **Step 4: Commit reproducibility documentation**

```powershell
git add README.md docs/reproducibility/tc_oos_rsg_reproduction.md
git commit -m "docs: document TC-OOS-RSG reproduction"
```

- [ ] **Step 5: Push `codex/theory-aware-report` and verify the remote commit**

Run `git push origin codex/theory-aware-report`; then compare local and remote commit hashes. If the network fails, retain the verified local commits and report the exact failure without recreating them.
