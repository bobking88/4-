# HRGV-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, train, and document HRGV-Net as the network-level innovation for the technical report and first paper.

**Architecture:** EfficientNet-B0 supplies one shared visual representation to a direct four-role expert, a 17-species expert mapped to roles, a learned reliability gate, and two asymmetric target-verification heads. The final posterior fuses both role experts and attenuates target evidence unless it passes the titanium-bearing and metallic hard-negative verifiers.

**Tech Stack:** Python 3.11, PyTorch, torchvision EfficientNet-B0, scikit-learn metrics, unittest/pytest, CSV/JSON experiment artifacts, python-docx report generation.

**Spec:** `docs/superpowers/specs/2026-08-19-hrgv-network-design.md`

## Global Constraints

- Preserve the fixed four-role label order and existing dataset split.
- Preserve all existing baseline scripts and outputs.
- Train formal configurations with seeds `20260727`, `20260728`, and `20260729`.
- Select checkpoints on validation Macro F1 only; never tune from the test split.
- Use final risk-gated posterior `q` for validation, testing, confidence, and prediction export.
- Do not claim industrial sorting, grade, vanadium content, or recovery performance.
- Do not add or commit `结题/技术报告_配图_总体技术路线.png` unless separately requested.

---

### Task 1: Mathematical probability and verifier primitives

**Files:**
- Create: `scripts/hrgv_network.py`
- Create: `tests/test_hrgv_network.py`

**Interfaces:**
- Produces: `normalized_entropy(probabilities, torch) -> Tensor`
- Produces: `jensen_shannon_divergence(first, second, torch) -> Tensor`
- Produces: `mix_role_experts(direct, mapped, gate) -> Tensor`
- Produces: `apply_target_verifiers(fused, ti_target_probability, metallic_target_probability, target_index=0) -> Tensor`
- Produces: `masked_verifier_loss(logits, role_labels, negative_role_id, criterion, target_role_id=0) -> tuple[Tensor, int]`

- [ ] **Step 1: Write failing probability-invariant tests**

Add tests that require normalized entropy and Jensen-Shannon divergence to return `[batch, 1]`, require zero divergence for identical distributions, and reject mismatched tensor shapes.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_hrgv_network.py -v`

Expected: import failure because `hrgv_network.py` does not exist.

- [ ] **Step 3: Implement the probability primitives**

Implement finite, epsilon-clamped entropy and Jensen-Shannon functions without detaching tensors.

- [ ] **Step 4: Add failing expert-mixture and verifier-monotonicity tests**

Tests must cover gate endpoints, probability normalization, verifier identity at probabilities equal to one, and monotonic non-increase of target probability as either verifier probability decreases.

- [ ] **Step 5: Run the focused tests and verify RED**

Expected: missing `mix_role_experts` or `apply_target_verifiers` behavior.

- [ ] **Step 6: Implement minimal mixture and verifier correction**

Use convex mixture for experts and multiplicative target evidence followed by row normalization.

- [ ] **Step 7: Add failing masked-verifier-loss tests**

Test eligible sample count, target/negative binary label orientation, exclusion of unrelated roles, and differentiable zero for an empty eligible subset.

- [ ] **Step 8: Implement masked verifier loss and run focused tests**

Run: `python -m pytest tests/test_hrgv_network.py -v`

Expected: all Task 1 tests pass.

- [ ] **Step 9: Commit Task 1**

Commit message: `feat: add HRGV probability and verifier primitives`

### Task 2: HRGV-Net model and complete loss

**Files:**
- Modify: `scripts/hrgv_network.py`
- Modify: `tests/test_hrgv_network.py`

**Interfaces:**
- Produces: `HierarchicalRiskGatedVerificationNet(models, role_matrix, pretrained, embedding_dim, gate_hidden_dim)`
- Produces: `compute_hrgv_losses(outputs, role_labels, species_labels, mapping, criteria, weights, temperature, torch) -> tuple[Tensor, dict[str, Tensor]]`
- Model output keys: `role_logits`, `species_logits`, `direct_role_probabilities`, `mapped_role_probabilities`, `gate`, `fused_role_probabilities`, `ti_verifier_logits`, `metallic_verifier_logits`, `ti_target_probability`, `metallic_target_probability`, `final_role_probabilities`, `embeddings`, `expert_js_divergence`.

- [ ] **Step 1: Write a failing model-output-contract test**

Instantiate with a synthetic `4 x 17` mapping, forward two `64 x 64` images, and assert every key, shape, probability invariant, and registered mapping buffer.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: missing model class.

- [ ] **Step 3: Implement the model minimally**

Reuse torchvision EfficientNet-B0 features, add direct/species heads, a two-layer scalar gate, two binary verifier heads, and the existing normalized projection head.

- [ ] **Step 4: Write failing backward-pass and loss-decomposition tests**

Require finite total loss, all named terms, correct verifier eligible counts, and non-null gradients for backbone, gate, direct role head, species head, both verifiers, and projection head.

- [ ] **Step 5: Implement the complete objective**

Use weighted NLL on final probabilities, auxiliary direct role CE, species CE, hierarchy KL, masked verifier CEs, and the existing role-aware contrastive loss.

- [ ] **Step 6: Run focused and existing hierarchical tests**

Run: `python -m pytest tests/test_hrgv_network.py tests/test_hierarchical_mineral_classifier.py -v`

Expected: all pass.

- [ ] **Step 7: Commit Task 2**

Commit message: `feat: implement hierarchical risk gated verification network`

### Task 3: Training, evaluation, and artifact export

**Files:**
- Create: `scripts/train_hrgv_mineral_classifier.py`
- Create: `tests/test_train_hrgv_mineral_classifier.py`

**Interfaces:**
- Consumes: all Task 2 model outputs and loss functions.
- Produces: `run_epoch(...)` returning loss terms, role/species predictions, confidences, gate values, divergences, and verifier probabilities.
- Produces: CLI flags `--lambda-direct`, `--lambda-species`, `--lambda-consistency`, `--lambda-verifier`, `--lambda-contrast`, `--gate-hidden-dim`, `--disable-verifiers`, and `--fixed-gate`.

- [ ] **Step 1: Write failing CLI-default and validation tests**

Require the spec defaults, reject negative loss weights, and reject a fixed gate outside `[0,1]`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_train_hrgv_mineral_classifier.py -v`

Expected: missing training module.

- [ ] **Step 3: Implement CLI, data loading, criteria, and run loop**

Follow `train_hierarchical_mineral_classifier.py`, but use the final posterior for metrics and preserve all HRGV diagnostics.

- [ ] **Step 4: Write failing artifact-schema tests**

Require environment mapping/weights, every history diagnostic, final/direct/mapped prediction fields, both verifier probabilities, and checkpoint mapping.

- [ ] **Step 5: Implement artifact writing**

Write UTF-8-SIG CSV and deterministic JSON using existing repository helpers.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_train_hrgv_mineral_classifier.py -v`

Expected: all pass.

- [ ] **Step 7: Commit Task 3**

Commit message: `feat: add reproducible HRGV training pipeline`

### Task 4: Smoke training and experiment launcher

**Files:**
- Create: `scripts/run_hrgv_experiments.py`
- Create: `tests/test_run_hrgv_experiments.py`
- Create after execution: `outputs/training/smoke_hrgv_seed20260727/*`

**Interfaces:**
- Produces: a dry-run command matrix for six configurations and three seeds.
- Produces: subprocess execution with fail-fast behavior and per-run output directories.

- [ ] **Step 1: Write failing command-matrix tests**

Assert exact seeds, unique output directories, manifest/dataset paths, and ablation flags.

- [ ] **Step 2: Run tests and verify RED**

Expected: missing launcher.

- [ ] **Step 3: Implement command generation and dry-run mode**

Keep formal execution opt-in; default launcher mode prints the reproducible commands.

- [ ] **Step 4: Run a CUDA smoke experiment**

Run `train_hrgv_mineral_classifier.py` with `--smoke-run --device cuda` and the fixed manifest. Verify the output schemas and finite metrics.

- [ ] **Step 5: Run the full unit suite**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 6: Commit Task 4**

Commit message: `test: verify HRGV smoke training and experiment matrix`

### Task 5: Formal three-seed experiments and paired inference

**Files:**
- Create through execution: `outputs/training/formal_hrgv_*/*`
- Create: `scripts/analyze_hrgv_experiment.py`
- Create: `tests/test_analyze_hrgv_experiment.py`
- Create through execution: `outputs/business_metrics/hrgv_network/*`
- Create through execution: `outputs/paper_experiments_v4/statistical_inference/*`
- Create: `docs/experiment_records/2026-08-19_hrgv_network.md`

**Interfaces:**
- Consumes: three-seed prediction CSVs from complete HRGV and required ablations.
- Produces: three-seed summary, role metrics, target proxy metrics, gate-by-role summary, verifier ROC-AUC, paired cluster bootstrap intervals, and corrected paired-test table.

- [ ] **Step 1: Write failing analysis tests with synthetic predictions**

Test both intrusion definitions, target recall, gate summaries, verifier eligible subsets, and paired cluster resampling by `split_group_id` within true role.

- [ ] **Step 2: Implement analysis and verify focused tests**

Run: `python -m pytest tests/test_analyze_hrgv_experiment.py -v`.

- [ ] **Step 3: Run complete HRGV for all three seeds**

Use the fixed manifest, pretrained EfficientNet-B0, CUDA, validation early stopping, and spec defaults.

- [ ] **Step 4: Run required gate/verifier ablations for all three seeds**

Run gate-only, equal-fusion, no-contrast, and complete HRGV configurations.

- [ ] **Step 5: Generate paired statistical inference**

Compare complete HRGV against the existing hierarchical model and weighted-CE baseline using the same seed and test records.

- [ ] **Step 6: Record actual outcomes without promotional filtering**

Write the experiment record with commands, environment, stopping epochs, point estimates, confidence intervals, adjusted p-values, acceptance-criterion status, and limitations.

- [ ] **Step 7: Commit Task 5**

Commit message: `exp: add three seed HRGV network evaluation`

### Task 6: Architecture figure, formulas, report, and paper package

**Files:**
- Create: `scripts/generate_hrgv_architecture_figure.py`
- Create: `tests/test_generate_hrgv_architecture_figure.py`
- Create: `outputs/paper_figures_v2/fig_hrgv_architecture.png`
- Create: `outputs/paper_figures_v2/fig_hrgv_architecture.svg`
- Create: `docs/hrgv_theory_and_method.md`
- Modify: `scripts/update_formal_report_v3.py`
- Modify: `tests/test_update_formal_report_v2.py`
- Modify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx`

**Interfaces:**
- Produces: editable SVG and 300-dpi PNG network figure.
- Produces: theorem assumptions, proof sketches, experiment linkage, and cautious innovation wording.
- Produces: updated formal report with architecture, formulas, ablations, confidence intervals, and explicit scope limits.

- [ ] **Step 1: Write failing figure-structure tests**

Require all seven visual modules, both verifier paths, mapping matrix, gate, final posterior, and valid SVG/PNG dimensions.

- [ ] **Step 2: Implement and generate the network figure**

Use repository figure conventions; do not depend on Word-to-PDF rendering.

- [ ] **Step 3: Write the theory and method supplement**

Include all formulas and proofs from the spec, distinguish theorem from empirical hypothesis, and cite related-work boundaries.

- [ ] **Step 4: Add failing report-content tests**

Require HRGV-Net name, complete loss, both propositions, three-seed results, architecture figure, limitations, and acceptance-criterion conclusion.

- [ ] **Step 5: Update the report generator and formal DOCX**

Use python-docx structural verification and embedded-image checks; do not install a Word-to-PDF renderer.

- [ ] **Step 6: Run report and figure tests**

Run: `python -m pytest tests/test_generate_hrgv_architecture_figure.py tests/test_update_formal_report_v2.py tests/test_formal_report_v3.py -v`.

- [ ] **Step 7: Commit Task 6**

Commit message: `docs: integrate HRGV theory network and experiments`

### Task 7: Final verification and GitHub publication

**Files:**
- Modify: `README.md`
- Modify: `训练说明.md`
- Modify: `docs/experiment_records/2026-08-19_hrgv_network.md`

**Interfaces:**
- Produces: reproducibility commands, artifact index, scope statement, and branch publication.

- [ ] **Step 1: Update reproduction documentation**

List smoke command, formal command matrix, analysis command, figure generation, report generation, hardware, seed policy, and non-distributed image/weight constraints.

- [ ] **Step 2: Run fresh full verification**

Run: `python -m pytest -q`.

Run the HRGV artifact analyzer against every formal output.

Inspect `git diff --check`, `git status --short`, report paragraph/table/relationship counts, embedded image dimensions, and all required output files.

- [ ] **Step 3: Audit every spec requirement against evidence**

Record pass/fail evidence for architecture outputs, loss terms, invariants, three seeds, ablations, statistical inference, report formulas, figure, and scope language.

- [ ] **Step 4: Commit and push intentionally**

Exclude the unrelated untracked technical-route PNG. Push `codex/theory-aware-report` and verify the remote commit.

- [ ] **Step 5: Report actual status**

State measured results, acceptance-criterion outcome, tests run, report path, GitHub branch/commit, and any remaining limitation.

## Self-Review

- Spec coverage: all architecture, loss, inference, output, theory, experiment, report, and publication requirements map to Tasks 1-7.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation step is used.
- Type consistency: all probability functions return batch tensors; the model output contract is shared by Tasks 2-5; all formal analyses consume exported final predictions and diagnostics.
- Execution mode: inline execution is selected because the user explicitly instructed the current task to continue after approving the architecture.
