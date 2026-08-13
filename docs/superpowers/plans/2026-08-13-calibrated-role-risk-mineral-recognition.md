# Calibrated Role-Risk Mineral Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible theoretical and experimental evidence for species-to-role risk contraction, dual-head consistency, validation-calibrated selective recognition, source-held-out generalization, and magnetite-proxy sensitivity.

**Architecture:** Reuse the trained hierarchical EfficientNet-B0 checkpoints and fixed manifest. First export validation/test logits and both role pathways, then analyze the coarse-role and KL/TV propositions without retraining. Build calibration and source/proxy manifests as isolated units; only source-held-out and proxy-ablation comparisons require new training runs.

**Tech Stack:** Python 3.11, PyTorch/torchvision, NumPy, SciPy, scikit-learn, matplotlib, Pillow, unittest, CSV/JSON.

## Global Constraints

- Keep the current 8,529-image fixed manifest unchanged and preserve all `split_group_id` boundaries.
- Never fit temperature, thresholds, or risk targets on the test set.
- Split validation records by `split_group_id`; a group cannot cross `calibration_fit` and `risk_certification`.
- Do not commit raw images, model weights, local caches, or downloaded papers.
- Report three-seed mean, sample standard deviation, and paired seed differences; do not claim statistical significance from three seeds.
- Interpret source holdout only as public-specimen photographer/source generalization.
- State that 35 titanomagnetite images were downloaded, 12 cross-label conflicts were excluded, and 23 entered the final dataset.
- Keep stage-conditioned processing actions, grade, recovery, XRF, and cost optimization outside the implemented contribution.

---

### Task 1: Export hierarchical validation and test probabilities

**Files:**
- Create: `scripts/export_hierarchical_probabilities.py`
- Create: `tests/test_export_hierarchical_probabilities.py`
- Modify: `scripts/train_hierarchical_mineral_classifier.py`

**Interfaces:**
- Produces: `load_hierarchical_checkpoint(checkpoint_path, mapping, dependencies, device) -> HierarchicalRoleAwareEfficientNet`.
- Produces: `build_probability_rows(records, role_logits, species_logits, mapping, torch) -> list[dict[str, str]]`.
- CLI writes `validation_probabilities.csv` and `test_probabilities.csv` with `image_id`, labels, split group, role logits/probabilities, species logits/probabilities, mapped role probabilities, role prediction, species prediction, confidence, locality, and photographer.

- [ ] **Step 1: Write failing probability-row tests**

```python
def test_probability_rows_include_both_role_paths(self):
    rows = build_probability_rows(records, role_logits, species_logits, mapping, torch)
    self.assertIn("role_probability_target_mineral", rows[0])
    self.assertIn("mapped_role_probability_target_mineral", rows[0])
    self.assertAlmostEqual(sum(float(rows[0][f"role_probability_{r}"]) for r in CLASS_LABELS), 1.0)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `.venv-training\Scripts\python.exe -m unittest tests.test_export_hierarchical_probabilities -v`

Expected: FAIL because the exporter does not exist.

- [ ] **Step 3: Implement checkpoint loading and row construction**

Reuse `HierarchicalRoleAwareEfficientNet`, `aggregate_role_probabilities`, frozen class order, and evaluation transforms. Validate checkpoint species labels exactly match the current mapping before inference.

- [ ] **Step 4: Add split inference CLI**

Arguments: `--checkpoint`, `--manifest`, `--dataset-root`, `--audit-manifest`, `--output-dir`, `--device`, `--batch-size`, `--num-workers`. Export validation and test outputs without augmentation or gradient calculation.

- [ ] **Step 5: Run tests and export six runs**

Export three complete hierarchical and three no-consistency checkpoints from the main workspace weights into `outputs/paper_experiments_v2/probabilities/<run_name>/`.

- [ ] **Step 6: Commit code and compact outputs**

Commit scripts, tests, CSV probabilities, and environment/provenance JSON; exclude checkpoints.

### Task 2: Analyze role-risk contraction and KL consistency

**Files:**
- Create: `scripts/analyze_hierarchy_consistency.py`
- Create: `tests/test_analyze_hierarchy_consistency.py`
- Create: `outputs/paper_experiments_v2/hierarchy_consistency/`

**Interfaces:**
- Produces: `calculate_hierarchy_metrics(rows) -> dict[str, object]`.
- Produces: `total_variation(first, second) -> float`.
- Produces: `kl_divergence(mapped, direct) -> float` using epsilon-clamped probabilities.

- [ ] **Step 1: Write failing theorem-metric tests**

```python
def test_species_error_can_preserve_role_correctness(self):
    summary = calculate_hierarchy_metrics(make_rows())
    self.assertEqual(summary["species_wrong_role_correct_count"], 1)

def test_pinsker_bound_holds(self):
    self.assertLessEqual(total_variation(p, q), (0.5 * kl_divergence(p, q)) ** 0.5 + 1e-12)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `.venv-training\Scripts\python.exe -m unittest tests.test_analyze_hierarchy_consistency -v`

- [ ] **Step 3: Implement per-image and aggregate metrics**

Calculate direct-role accuracy, mapped-role accuracy, species accuracy, species-wrong/role-correct rate, mean KL, mean TV, Pinsker slack, head disagreement rate, confidence gap, and KL-decile disagreement.

- [ ] **Step 4: Aggregate full versus no-consistency seeds**

Write per-seed CSV, mean/sample-std JSON, paired seed differences, and a Markdown interpretation that KL controls distribution discrepancy but does not imply accuracy improvement.

- [ ] **Step 5: Generate figures**

Create KL-versus-disagreement decile plot and full/no-consistency metric comparison with source CSV files.

- [ ] **Step 6: Run tests, analysis, and commit**

### Task 3: Calibrate probabilities and certify selective risk

**Files:**
- Create: `scripts/calibrate_selective_recognition.py`
- Create: `tests/test_calibrate_selective_recognition.py`
- Create: `outputs/paper_experiments_v2/calibrated_selective_recognition/`

**Interfaces:**
- Produces: `split_validation_groups(rows, seed=20260813) -> tuple[list, list]`.
- Produces: `fit_temperature(logits, labels) -> float`.
- Produces: `clopper_pearson_upper(errors, total, confidence) -> float`.
- Produces: `select_certified_threshold(rows, thresholds, delta, alpha) -> dict[str, object]`.

- [ ] **Step 1: Write failing split, calibration, and certificate tests**

```python
def test_validation_subsets_do_not_share_groups(self):
    fit, certify = split_validation_groups(rows)
    self.assertTrue(groups(fit).isdisjoint(groups(certify)))

def test_temperature_is_positive_and_reduces_fit_nll(self):
    temperature = fit_temperature(logits, labels)
    self.assertGreater(temperature, 0.0)
    self.assertLessEqual(nll(logits / temperature, labels), nll(logits, labels) + 1e-8)

def test_no_certificate_is_reported_when_bound_exceeds_target(self):
    result = select_certified_threshold(rows, (0.0, 0.5), delta=0.01, alpha=0.05)
    self.assertEqual(result["status"], "no_certified_threshold")
```

- [ ] **Step 2: Run focused tests and verify failure**

- [ ] **Step 3: Implement grouped validation split and temperature scaling**

Use deterministic role-stratified group allocation. Optimize scalar `log_temperature` with LBFGS on `calibration_fit` NLL, then freeze it.

- [ ] **Step 4: Implement simultaneous risk certificate**

Use thresholds `0.00, 0.05, ..., 0.95`, Bonferroni-adjusted one-sided Clopper-Pearson bounds, `alpha=0.05`, and deltas `0.10`, `0.15`, `0.20`. Choose the certified threshold with maximal certification coverage; never fall back to test optimization.

- [ ] **Step 5: Add calibration and selective metrics**

Report pre/post NLL, Brier Score, 15-bin ECE, AURC, coverage, selective risk, target miss, titanium intrusion, and metallic intrusion. Generate reliability and risk-coverage figures with source CSV.

- [ ] **Step 6: Run three seeds and commit reproducible outputs**

### Task 4: Build and audit a photographer-held-out manifest

**Files:**
- Create: `scripts/build_source_holdout_manifest.py`
- Create: `tests/test_build_source_holdout_manifest.py`
- Create: `outputs/paper_experiments_v2/source_holdout/`

**Interfaces:**
- Produces: `normalize_source_group(value: str) -> str`.
- Produces: `allocate_source_groups(rows, target_test_ratio, seed) -> list[dict[str, str]]`.
- Produces: `validate_source_holdout(rows) -> dict[str, object]`.

- [ ] **Step 1: Write failing zero-overlap and reproducibility tests**

```python
def test_photographer_groups_never_cross_splits(self):
    assigned = allocate_source_groups(rows, 0.15, 20260813)
    validate_source_holdout(assigned)
    self.assertEqual(cross_split_group_count(assigned), 0)
```

- [ ] **Step 2: Run focused tests and verify failure**

- [ ] **Step 3: Implement normalization and deterministic group allocation**

Join final records to audit metadata by `image_id`. Exclude missing-photographer records from this strict experimental subset. Allocate complete photographer groups while minimizing role-distribution and test-ratio deviation.

- [ ] **Step 4: Enforce acceptance gates**

Require zero source overlap, zero `split_group_id` overlap, all four roles in train/val/test, at least 30 test images per role, valid paths, and a deterministic manifest hash. If any gate fails, write a rejected audit and do not train.

- [ ] **Step 5: Produce summary tables and commit the accepted manifest**

### Task 5: Add magnetite-proxy sensitivity manifest

**Files:**
- Create: `scripts/build_proxy_ablation_manifest.py`
- Create: `tests/test_build_proxy_ablation_manifest.py`
- Create: `outputs/paper_experiments_v2/proxy_ablation/`

**Interfaces:**
- Produces: `build_proxy_ablation_rows(final_rows, audit_rows) -> list[dict[str, str]]`.
- Produces: `summarize_titanomagnetite_provenance(audit_rows, final_rows) -> dict[str, object]`.

- [ ] **Step 1: Write failing proxy-removal and provenance tests**

```python
def test_titanomagnetite_counts_are_traced(self):
    summary = summarize_titanomagnetite_provenance(audit, final)
    self.assertEqual(summary["downloaded"], 35)
    self.assertEqual(summary["excluded_cross_label_conflict"], 12)
    self.assertEqual(summary["final"], 23)
```

- [ ] **Step 2: Run focused tests and verify failure**

- [ ] **Step 3: Build an ablation manifest without ordinary magnetite proxy images**

Preserve original splits for all remaining images and reject any accidental removal of ilmenite or titanomagnetite records. Report per-role and per-species counts before/after removal.

- [ ] **Step 4: Validate paths, groups, and class coverage**

- [ ] **Step 5: Commit script, tests, manifest, and provenance summary**

### Task 6: Run required new training experiments

**Files:**
- Reuse: `scripts/train_mineral_classifier.py`
- Reuse: `scripts/train_hierarchical_mineral_classifier.py`
- Create: `outputs/paper_experiments_v2/training_summaries/`

**Interfaces:**
- Consumes accepted source-held-out and proxy-ablation manifests from Tasks 4 and 5.
- Produces test metrics, predictions, confusion matrices, histories, and environment records for each run.

- [ ] **Step 1: Run smoke training on each new manifest**

Run one baseline and one hierarchical smoke run per manifest. Expected: checkpoints and metrics are written, no missing class or path errors occur.

- [ ] **Step 2: Run photographer-held-out baseline for three seeds**

Use seeds `20260727`, `20260728`, `20260729`, same optimizer, image size, pretrained weights, and early stopping as the fixed-split baseline.

- [ ] **Step 3: Run photographer-held-out hierarchical model for three seeds**

- [ ] **Step 4: Run proxy-ablation hierarchical model for three seeds**

Do not reinterpret this as true titanomagnetite performance; it is sensitivity to ordinary-magnetite proxy removal.

- [ ] **Step 5: Summarize paired model and split differences**

- [ ] **Step 6: Verify all expected outputs and commit summaries only**

### Task 7: Update paper figures, technical report, and reproduction guide

**Files:**
- Modify: `scripts/generate_paper_figures.py`
- Modify: `tests/test_generate_paper_figures.py`
- Modify: `scripts/build_technical_report.py`
- Modify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx`
- Modify: `README.md`
- Create: `docs/experiment_records/2026-08-13_calibrated_role_risk_experiments.md`

**Interfaces:**
- Consumes all verified JSON/CSV summaries from Tasks 2, 3, and 6.
- Produces manuscript-ready figures and a report whose numerical claims trace to files.

- [ ] **Step 1: Add theorem-to-experiment tables and figures**

Include role-risk contraction, KL/TV consistency, reliability, certified risk-coverage, source holdout, and proxy sensitivity. Each figure has a source CSV.

- [ ] **Step 2: Revise theoretical contribution wording**

State assumptions and proof scope explicitly. Replace the old synthetic-candidate experiment as the central theorem evidence; retain it only as an illustrative logical check or appendix result.

- [ ] **Step 3: Update the formal technical report**

Add formulas, proof outlines, experiment protocols, verified results, and limitations. Preserve the existing baseline and network diagrams.

- [ ] **Step 4: Render and inspect the DOCX**

Use the documents renderer. Inspect every page for figure legibility, equation clipping, table overflow, and numbering consistency. If LibreOffice is unavailable, perform structural checks and disclose the limitation.

- [ ] **Step 5: Run full tests and data-integrity checks**

Run: `.venv-training\Scripts\python.exe -m unittest discover -s tests -v`

Also run `git diff --check`, scan tracked files for checkpoints/raw images, and verify every reported number is present in source JSON/CSV.

- [ ] **Step 6: Commit and push the final experiment/report update**

## Self-Review

**Spec coverage:** Tasks 1–3 implement all three theoretical results and prevent calibration/test leakage. Task 4 covers source-held-out evidence. Task 5 traces the 35/12/23 titanomagnetite counts and constructs the proxy ablation. Task 6 performs only the retraining required by new manifests. Task 7 updates figures, report, and reproducibility documentation.

**Placeholder scan:** The plan contains no `TBD`, `TODO`, or unspecified implementation instruction. Each task defines files, interfaces, tests, outputs, and acceptance behavior.

**Type consistency:** Probability rows from Task 1 feed Tasks 2 and 3. Source and proxy manifests from Tasks 4 and 5 feed Task 6. All final summaries feed Task 7.

