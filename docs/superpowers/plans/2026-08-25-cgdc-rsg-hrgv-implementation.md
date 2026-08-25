# CGDC-RSG-HRGV-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Implement and formally evaluate CGDC-RSG-HRGV-Net, a decomposed cross-granularity expert architecture with disagreement-triggered bounded posterior calibration.

**Architecture:** Preserve current RSG-HRGV behavior unless CGDC is enabled. CGDC uses two residual adapters after EfficientNet-B0, drives role and species heads from their respective paths, applies a tanh-bounded residual scaled by Jensen-Shannon disagreement to the RSG-fused posterior, and finally uses the existing residual verifiers.

**Tech Stack:** Python 3.11, PyTorch, torchvision EfficientNet-B0, NumPy, matplotlib, python-docx and unittest.

**Spec:** docs/superpowers/specs/2026-08-25-cgdc-rsg-hrgv-design.md

## Global Constraints

- Existing RSG-HRGV output behavior remains equivalent when CGDC is disabled.
- Use the frozen four-role, seventeen-species mapping and fixed manifest split.
- Inference cannot use labels, stage, grade, recovery or cost metadata.
- Bound the calibrator residual with tanh and scale it by rho = 1 - exp(-JS).
- Formal claims require the registered three seeds and paired split_group_id Bootstrap.
- Do not claim industrial sorting, grade prediction, recovery, OOD ability or universal accuracy gain.

---

## Task 1: Probability Primitives

**Files:**
- Modify: scripts/hrgv_network.py
- Test: tests/test_hrgv_network.py

**Interfaces:**
- Produce disagreement_gain(direct, mapped, torch) with shape [batch, 1].
- Produce apply_disagreement_calibration(fused, residual_logits, gain).

- [ ] **Step 1: Write failing tests**

~~~python
def test_disagreement_calibration_is_identity_under_matching_experts(self):
    direct = torch.tensor([[0.70, 0.10, 0.10, 0.10]])
    residual = torch.tensor([[0.9, -0.7, 0.2, -0.1]])
    gain = disagreement_gain(direct, direct, torch)
    calibrated = apply_disagreement_calibration(direct, residual, gain)
    self.assertTrue(torch.allclose(gain, torch.zeros_like(gain)))
    self.assertTrue(torch.allclose(calibrated, direct, atol=1e-7))

def test_disagreement_calibration_bounds_log_odds_shift(self):
    fused = torch.tensor([[0.45, 0.25, 0.20, 0.10]])
    residual = torch.tensor([[2.0, -2.0, 0.3, -0.3]])
    gain = torch.tensor([[0.35]])
    calibrated = apply_disagreement_calibration(fused, residual, gain)
    shift = torch.log(calibrated[0, 0] / calibrated[0, 1]) - torch.log(fused[0, 0] / fused[0, 1])
    self.assertLessEqual(abs(float(shift)), 2.0 * float(gain[0, 0]) + 1e-6)
~~~

- [ ] **Step 2: Verify RED**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest tests.test_hrgv_network.HRGVProbabilityPrimitiveTests -v

Expected: import errors because the two functions are absent.

- [ ] **Step 3: Implement minimal primitives**

~~~python
def disagreement_gain(direct, mapped, torch):
    return 1.0 - torch.exp(-jensen_shannon_divergence(direct, mapped, torch))
~~~

Validate dimensions and probabilities. Tanh-bound residual logits, add gain times residual to log(fused), then softmax.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command. Expected: both tests pass.

- [ ] **Step 5: Commit**

Run: git add scripts/hrgv_network.py tests/test_hrgv_network.py

Run: git commit -m "feat: add disagreement bounded calibration primitives"

## Task 2: Decomposed Expert Adapters

**Files:**
- Modify: scripts/hrgv_network.py
- Test: tests/test_hrgv_network.py

**Interfaces:**
- Extend constructor with enable_cgdc=False, adapter_dim=256 and calibration_hidden_dim=256.
- Return calibrated_role_probabilities, cgdc_gain, cgdc_residual_logits, direct_adapter_delta and species_adapter_delta.

- [ ] **Step 1: Write failing tests**

~~~python
def test_disabled_cgdc_preserves_rsg_fused_posterior(self):
    model = build_test_model(enable_cgdc=False)
    outputs = model(torch.randn(2, 3, 224, 224))
    self.assertTrue(torch.allclose(
        outputs["calibrated_role_probabilities"],
        outputs["fused_role_probabilities"], atol=1e-7))

def test_enabled_cgdc_exports_paths_and_probability_simplex(self):
    model = build_test_model(enable_cgdc=True)
    outputs = model(torch.randn(2, 3, 224, 224))
    self.assertEqual(outputs["direct_adapter_delta"].shape, (2, 1280))
    self.assertEqual(outputs["species_adapter_delta"].shape, (2, 1280))
    self.assertEqual(outputs["cgdc_gain"].shape, (2, 1))
    self.assertTrue(torch.allclose(
        outputs["calibrated_role_probabilities"].sum(dim=1), torch.ones(2), atol=1e-6))
~~~

- [ ] **Step 2: Verify RED**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest tests.test_hrgv_network.HRGVModelTests -v

Expected: constructor and output-key errors.

- [ ] **Step 3: Implement adapters and calibrator**

~~~python
adapter = nn.Sequential(
    nn.Linear(feature_dim, adapter_dim),
    nn.ReLU(inplace=True),
    nn.Linear(adapter_dim, feature_dim),
)
u_d = features + direct_adapter(features)
u_s = features + species_adapter(features)
~~~

Drive role/species heads from u_d/u_s. Assemble conflict features exactly as in the spec. Use Task 1 calibration before existing verifiers. When disabled, retain existing features, set calibrated posterior equal to fused posterior and emit zero diagnostics.

- [ ] **Step 4: Verify GREEN**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest tests.test_hrgv_network -v

Expected: adapter and baseline-equivalence tests pass.

- [ ] **Step 5: Commit**

Run: git add scripts/hrgv_network.py tests/test_hrgv_network.py

Run: git commit -m "feat: add decomposed CGDC expert paths"

## Task 3: Losses, CLI and Prediction Evidence

**Files:**
- Modify: scripts/hrgv_network.py
- Modify: scripts/train_hrgv_mineral_classifier.py
- Test: tests/test_hrgv_network.py
- Test: tests/test_train_hrgv_mineral_classifier.py

**Interfaces:**
- Extend HRGVLossWeights with decomposition and calibration weights defaulted to 0.
- Add flags --enable-cgdc, --adapter-dim, --calibration-hidden-dim, --lambda-decomposition and --lambda-calibration.
- Export cgdc_gain, calibrated_true_probability, pre_calibration_predicted_label and calibrated_predicted_label.

- [ ] **Step 1: Write failing tests**

~~~python
def test_cgdc_losses_are_zero_when_disabled(self):
    outputs = build_disabled_cgdc_outputs()
    _, terms = compute_hrgv_losses(...)
    self.assertEqual(float(terms["decomposition_loss"]), 0.0)
    self.assertEqual(float(terms["calibration_loss"]), 0.0)

def test_cgdc_cli_rejects_nonpositive_adapter_dimension(self):
    args = parse_args([...,"--enable-cgdc","--adapter-dim","0"])
    with self.assertRaisesRegex(ValueError, "adapter"):
        validate_args(args)
~~~

- [ ] **Step 2: Verify RED**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest tests.test_hrgv_network tests.test_train_hrgv_mineral_classifier -v

Expected: loss-term and CLI-validation failures.

- [ ] **Step 3: Implement and verify GREEN**

Use epsilon-stabilized cosine-squared loss on adapter deltas and calibration NLL from calibrated posterior. Include both only when CGDC is enabled. Persist losses and added fields. Run the Step 2 command; expected PASS and unchanged disabled loss.

- [ ] **Step 4: Commit**

Run: git add scripts/hrgv_network.py scripts/train_hrgv_mineral_classifier.py tests/test_hrgv_network.py tests/test_train_hrgv_mineral_classifier.py

Run: git commit -m "feat: train and audit CGDC losses"

## Task 4: Experiment Matrix and Statistics

**Files:**
- Create: scripts/run_cgdc_rsg_experiments.py
- Create: scripts/analyze_cgdc_rsg_experiments.py
- Test: tests/test_run_cgdc_rsg_experiments.py
- Test: tests/test_analyze_cgdc_rsg_experiments.py

**Interfaces:**
- Register rsg_complete, cgdc_complete, cgdc_shared_features, cgdc_unconditional and cgdc_no_decomposition_loss.
- Use formal seeds 20260727, 20260728 and 20260729.
- Produce paired Bootstrap JSON, multi-class Brier score, ten-bin ECE and a manifest.

- [ ] **Step 1: Write failing tests**

~~~python
def test_formal_matrix_has_five_configurations_and_three_seeds(self):
    commands = build_experiment_commands(...)
    self.assertEqual(len(commands), 15)
    self.assertIn("cgdc_complete", {item.configuration for item in commands})

def test_calibration_summary_reports_brier_and_ece(self):
    result = summarize_calibration(rows)
    self.assertIn("brier_score", result)
    self.assertIn("expected_calibration_error", result)
~~~

- [ ] **Step 2: Verify RED**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest tests.test_run_cgdc_rsg_experiments tests.test_analyze_cgdc_rsg_experiments -v

Expected: import failure because runner and analyzer are absent.

- [ ] **Step 3: Implement and verify GREEN**

Use frozen configuration flags. Reuse established two-stage seed/group resampling. Run the Step 2 command; expected PASS with fifteen commands and calibration keys.

- [ ] **Step 4: Commit**

Run: git add scripts/run_cgdc_rsg_experiments.py scripts/analyze_cgdc_rsg_experiments.py tests/test_run_cgdc_rsg_experiments.py tests/test_analyze_cgdc_rsg_experiments.py

Run: git commit -m "feat: register CGDC formal experiment matrix"

## Task 5: Architecture Figure

**Files:**
- Modify: scripts/generate_hrgv_architecture_figure.py
- Test: tests/test_generate_hrgv_architecture_figure.py

**Interfaces:** Export outputs/paper_figures/cgdc_rsg_hrgv_architecture.png and supported vector forms. Description must name adapters, mapping, RSG gate, disagreement gain, bounded calibrator and residual verifiers.

- [ ] **Step 1: Write failing test**

~~~python
def test_cgdc_architecture_figure_exports_required_modules(self):
    outputs = export_cgdc_architecture_figure(tmp_path)
    self.assertTrue(outputs["png"].exists())
    self.assertIn("disagreement-triggered calibrator", outputs["description"].lower())
~~~

- [ ] **Step 2: Verify RED**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest tests.test_generate_hrgv_architecture_figure -v

Expected: missing exporter.

- [ ] **Step 3: Implement, verify and commit**

Draw backbone, adapters, experts/mapping, RSG fusion, JS-to-rho scale, tanh residual, calibrated posterior and verifiers. Run the Step 2 command and require nonempty output. Then run: git add scripts/generate_hrgv_architecture_figure.py tests/test_generate_hrgv_architecture_figure.py outputs/paper_figures/cgdc_rsg_hrgv_architecture.*

Run: git commit -m "docs: add CGDC architecture figure"

## Task 6: Smoke and Formal Matrix

**Files:**
- Create: docs/experiment_records/2026-08-25_cgdc_rsg_hrgv.md
- Create: outputs/training/cgdc_formal
- Create: outputs/business_metrics/cgdc_rsg_hrgv/formal

- [ ] **Step 1: Run smoke**

Run: .\.venv-training\Scripts\python.exe scripts\run_cgdc_rsg_experiments.py --manifest 数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv --dataset-root 数据集\mindat_manual_positive_v1 --output-root outputs\training\cgdc_smoke --stage pilot --config cgdc_complete --device cuda --execute

- [ ] **Step 2: Validate smoke**

Require test_metrics.json, test_predictions.csv, all CGDC fields, finite values and matching row counts. Record it as execution validation only.

- [ ] **Step 3: Run formal matrix**

Run: .\.venv-training\Scripts\python.exe scripts\run_cgdc_rsg_experiments.py --manifest 数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv --dataset-root 数据集\mindat_manual_positive_v1 --output-root outputs\training\cgdc_formal --stage formal --device cuda --execute

- [ ] **Step 4: Analyze**

Run: .\.venv-training\Scripts\python.exe scripts\analyze_cgdc_rsg_experiments.py --config-root rsg_complete=outputs\training\rsg_controlled --config-root cgdc_complete=outputs\training\cgdc_formal --config-root cgdc_shared_features=outputs\training\cgdc_formal --config-root cgdc_unconditional=outputs\training\cgdc_formal --config-root cgdc_no_decomposition_loss=outputs\training\cgdc_formal --output-dir outputs\business_metrics\cgdc_rsg_hrgv\formal --bootstrap-replicates 2000

- [ ] **Step 5: Record and commit**

Add means, sample standard deviations, intervals and a claim boundary for each proposition. Intervals crossing zero are inconclusive.

Run: git add outputs/training/cgdc_formal outputs/business_metrics/cgdc_rsg_hrgv/formal docs/experiment_records/2026-08-25_cgdc_rsg_hrgv.md

Run: git commit -m "exp: evaluate CGDC formal matrix"

## Task 7: Report, Paper and Verification

**Files:**
- Create: scripts/update_formal_report_v9.py
- Modify: 结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx
- Modify: docs/paper_v1/manuscript_core_draft.md
- Test: tests/test_formal_report_v9.py

- [ ] **Step 1: Write failing document test**

~~~python
def test_cgdc_appendix_has_theory_figure_and_evidence_boundary(self):
    update_report(source, output)
    text = "\n".join(p.text for p in Document(output).paragraphs)
    self.assertIn("附录 D CGDC-RSG-HRGV 网络理论与实验", text)
    self.assertIn("一致恒等性", text)
    self.assertIn("不等同于工业分选", text)
~~~

- [ ] **Step 2: Verify RED, implement and verify GREEN**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest tests.test_formal_report_v9 -v

Insert propositions, proofs, figure and only formal metrics. Rerun the command and require PASS.

- [ ] **Step 3: Full verification and sync**

Run: .\.venv-training\Scripts\python.exe -X utf8 -m unittest discover -s tests -v

Run: git diff --check

Attempt DOCX rendering. If LibreOffice is still absent by user choice, preserve that conversion failure and instead check DOCX headings, tables, relationships and embedded image dimensions.

Run: git add scripts/update_formal_report_v9.py tests/test_formal_report_v9.py docs/paper_v1/manuscript_core_draft.md 结题 outputs/report_assets_v9

Run: git commit -m "docs: integrate CGDC theory and formal evidence"

Run: git push origin codex/theory-aware-report

## Plan Self-Review

The specification maps directly to this plan: probability and proof primitives are Task 1; adapters and disabled equivalence are Task 2; objective/CLI/audit exports are Task 3; the experiment matrix and statistics are Task 4 and Task 6; the architecture figure is Task 5; report/paper evidence is Task 7. The output names introduced by Task 2 are consumed by Task 3, and Task 6 consumes the runner/analyzer introduced in Task 4. All tests, commands and artifact paths are explicit.
