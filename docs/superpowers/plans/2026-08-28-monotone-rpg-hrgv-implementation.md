# M-RPG-HRGV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate the candidate-capacity-normalized, between-role-uncertainty-monotone M-RPG gate as the final network innovation for the first paper.

**Architecture:** Keep the existing EfficientNet-B0 backbone, direct role expert, 17-species expert, frozen species-to-role map, RSG gate-regret supervision and residual verifiers. M-RPG computes exact raw partitioned entropy diagnostics, normalizes role-relevant and within-role uncertainty by their support capacities, then replaces only the fusion gate with a positive-coefficient between-role term.

**Tech Stack:** Python 3.11, PyTorch, torchvision EfficientNet-B0, NumPy, CSV/JSON, matplotlib, python-docx, unittest.

**Spec:** `docs/superpowers/specs/2026-08-28-monotone-rpg-hrgv-design.md`

## Global Constraints

- Retain the public-specimen closed-set four-role task and all established claim boundaries.
- Preserve RSG, CGDC, and original RPG behavior bit-for-bit when `enable_mrpg=False`.
- Use fixed seeds `20260727`, `20260728`, and `20260729`; do not alter split groups, image transformations, backbone, optimizer, verifiers, or loss weights.
- Do not start M-RPG GPU work until the current 12-run RPG matrix finishes and its metrics are analyzed.
- Do not commit original images, checkpoints, or `outputs/training/**`; commit code, tests, summary CSV/JSON, figure assets, and report/document sources only.
- State M-R1 through M-R4 as formula-level properties. State empirical improvement only if paired seed-and-cluster Bootstrap intervals support it.

---

### Task 1: Add normalized partition primitives and prove their contracts

**Files:**
- Modify: `scripts/hrgv_network.py`
- Modify: `tests/test_hrgv_network.py`

**Interfaces:**
- `role_partitioned_uncertainty(...)` gains `within_capacity` and two `[batch,1]` normalized values: `normalized_between_role_entropy`, `normalized_within_role_entropy`.
- The normalized diagnostics are added to `role_partitioned_uncertainty(...)` so the original raw fields and the new normalized fields are emitted from one validated partition calculation.
- `monotone_role_gate(base_logit, normalized_between, raw_coefficient, torch) -> Tensor` returns `sigmoid(base_logit + softplus(raw_coefficient)*normalized_between)`.

- [ ] **Step 1: Write failing tests for M-R2 and M-R3**

```python
def test_capacity_normalized_partitioned_uncertainties_are_bounded(self) -> None:
    values = role_partitioned_uncertainty(self.species, self.role_matrix, self.torch)
    self.assertTrue(self.torch.all(values["normalized_between_role_entropy"] >= 0))
    self.assertTrue(self.torch.all(values["normalized_between_role_entropy"] <= 1))
    self.assertTrue(self.torch.all(values["normalized_within_role_entropy"] >= 0))
    self.assertTrue(self.torch.all(values["normalized_within_role_entropy"] <= 1))

def test_monotone_role_gate_never_decreases_with_between_role_uncertainty(self) -> None:
    first = monotone_role_gate(self.torch.tensor([[0.2]]), self.torch.tensor([[0.1]]), self.torch.tensor([0.3]), self.torch)
    second = monotone_role_gate(self.torch.tensor([[0.2]]), self.torch.tensor([[0.9]]), self.torch.tensor([0.3]), self.torch)
    self.assertGreaterEqual(float(second), float(first))
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network -v`

Expected: failures because the normalized diagnostic keys and monotone gate do not exist.

- [ ] **Step 3: Implement raw/normalized separation without changing RPG paths**

Compute `within_capacity=sum_r p_m(r) log(n_r)` from role cardinalities, preserve raw `total_species_entropy`, `between_role_entropy`, and `within_role_entropy`, and add the two normalized values with epsilon-safe zero-denominator handling. Implement the positive `softplus` coefficient function separately so its test does not depend on EfficientNet.

- [ ] **Step 4: Add M-R1 and M-R4 regression tests, then run focused tests**

```python
def test_raw_partitioned_entropy_identity_is_preserved_after_normalization(self) -> None:
    values = role_partitioned_uncertainty(self.species, self.role_matrix, self.torch)
    self.assertTrue(self.torch.allclose(values["total_species_entropy"], values["between_role_entropy"] + values["within_role_entropy"], atol=1e-6))

def test_monotone_gate_fusion_stays_inside_expert_envelope(self) -> None:
    fused = mix_role_experts(self.direct, self.mapped, self.gate)
    self.assertTrue(self.torch.all(fused >= self.torch.minimum(self.direct, self.mapped)))
    self.assertTrue(self.torch.all(fused <= self.torch.maximum(self.direct, self.mapped)))
```

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add scripts/hrgv_network.py tests/test_hrgv_network.py
git commit -m "feat: add normalized monotone RPG primitives"
```

### Task 2: Integrate the M-RPG architecture and reproducibility metadata

**Files:**
- Modify: `scripts/hrgv_network.py`
- Modify: `scripts/train_hrgv_mineral_classifier.py`
- Modify: `tests/test_hrgv_network.py`
- Modify: `tests/test_train_hrgv_mineral_classifier.py`

**Interfaces:**
- Constructor receives `enable_mrpg: bool=False` and `mrpg_between_mode: Literal["monotone", "unconstrained", "disabled"]="monotone"`.
- CLI exposes `--enable-mrpg` and `--mrpg-between-mode`.
- `enable_cgdc`, `enable_rpg`, and `enable_mrpg` are pairwise exclusive.
- M-RPG output exports raw and normalized entropy diagnostics plus `mrpg_between_coefficient`.

- [ ] **Step 1: Write failing model and CLI tests**

```python
def test_mrpg_outputs_normalized_diagnostics_and_positive_coefficient(self) -> None:
    model = HierarchicalRiskGatedVerificationNet(self.models, self.role_matrix, pretrained=False, embedding_dim=8, gate_hidden_dim=16, enable_mrpg=True)
    outputs = model(self.torch.randn(2, 3, 64, 64))
    self.assertEqual(outputs["normalized_between_role_entropy"].shape, (2, 1))
    self.assertTrue(self.torch.all(outputs["mrpg_between_coefficient"] >= 0))

def test_cli_rejects_mrpg_with_rpg(self) -> None:
    args = parse_args(["--manifest", "m.csv", "--dataset-root", "data", "--output-dir", "out", "--enable-rpg", "--enable-mrpg"])
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
        validate_args(args)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network tests.test_train_hrgv_mineral_classifier -v`

Expected: failure because M-RPG flags and output fields are absent.

- [ ] **Step 3: Implement only the gate replacement**

For the monotone mode, build `base_gate_network` over `[h,H(p_d),u_within]`; add `softplus(beta)*u_between` before sigmoid. For `unconstrained`, use a scalar unconstrained `alpha*u_between`. For `disabled`, omit the between term. Keep the existing RPG gate path unchanged. Thread configuration and diagnostic columns through model metadata, prediction CSV, metrics JSON, and model names.

- [ ] **Step 4: Run tests and one CPU smoke job**

Run focused tests above, then:

```powershell
..\..\.venv-training\Scripts\python.exe scripts\train_hrgv_mineral_classifier.py `
  --manifest 'D:\成信工科研\人工智能选矿\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv' `
  --dataset-root 'D:\成信工科研\人工智能选矿\数据集\mindat_manual_positive_v1' `
  --output-dir outputs\training\smoke_mrpg --epochs 1 --batch-size 2 --num-workers 0 `
  --image-size 64 --patience 1 --device cpu --smoke-run --no-pretrained --enable-mrpg
```

Expected: `test_metrics.json` and `test_predictions.csv` include normalized values and the M-RPG model name.

- [ ] **Step 5: Commit Task 2**

```powershell
git add scripts/hrgv_network.py scripts/train_hrgv_mineral_classifier.py tests/test_hrgv_network.py tests/test_train_hrgv_mineral_classifier.py
git commit -m "feat: add monotone role partitioned gate"
```

### Task 3: Register, analyze, and visualize M-RPG experiments

**Files:**
- Create: `scripts/run_mrpg_hrgv_experiments.py`
- Create: `scripts/analyze_mrpg_hrgv_experiments.py`
- Modify: `scripts/generate_hrgv_architecture_figure.py`
- Create: `tests/test_run_mrpg_hrgv_experiments.py`
- Create: `tests/test_analyze_mrpg_hrgv_experiments.py`
- Modify: `tests/test_generate_hrgv_architecture_figure.py`

**Interfaces:**
- Registered configurations: `mrpg_complete`, `mrpg_unconstrained_between`, `mrpg_without_between`, each using the three formal seeds.
- Analyzer compares RSG and RPG complete baselines with each M-RPG configuration using the established group-aware paired Bootstrap utilities.
- `generate_mrpg_architecture_figure(prefix)` exports PNG/SVG/PDF/TIFF and source text for M-R1 through M-R4.

- [ ] **Step 1: Write failing runner, analyzer, and figure tests**

```python
def test_mrpg_formal_matrix_has_three_configurations_and_three_seeds(self) -> None:
    commands = build_experiment_commands(
        PROJECT_ROOT, MANIFEST, DATASET_ROOT, OUTPUT_ROOT,
        Path(sys.executable), "cuda", TORCH_HOME, "formal",
    )
    self.assertEqual(len(commands), 9)
    self.assertIn("mrpg_unconstrained_between", CONFIGURATION_FLAGS)

def test_mrpg_analysis_declares_all_comparators(self) -> None:
    self.assertEqual(REQUIRED_CONFIGURATIONS[:2], ("rsg_complete", "rpg_complete"))
    self.assertIn("mrpg_complete", REQUIRED_CONFIGURATIONS)

def test_mrpg_figure_contains_monotonicity_equation(self) -> None:
    outputs = generate_mrpg_architecture_figure(Path(temp_dir) / "mrpg")
    self.assertIn("partial g", outputs["source_text"])
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_run_mrpg_hrgv_experiments tests.test_analyze_mrpg_hrgv_experiments tests.test_generate_hrgv_architecture_figure -v`

Expected: import failures for the new runner/analyzer and missing figure generator.

- [ ] **Step 3: Implement the 9-run formal contract and analysis**

Reuse the RSG residual-verifier flags and formal seed validation. The analyzer must reject missing baselines, align predictions by image ID and split group, export three-seed summaries, deltas, and paired Bootstrap JSON for M-RPG versus RSG and versus RPG complete. No training output directory is added to Git.

- [ ] **Step 4: Generate the architecture figure and verify focused tests**

The figure must show: `p_s(17)`, frozen `M`, raw chain identity, capacity normalization, direct expert `p_d`, constrained coefficient `softplus(beta)>=0`, monotone gate, convex fusion, and residual verifiers.

Run all three focused files. Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add scripts/run_mrpg_hrgv_experiments.py scripts/analyze_mrpg_hrgv_experiments.py scripts/generate_hrgv_architecture_figure.py tests/test_run_mrpg_hrgv_experiments.py tests/test_analyze_mrpg_hrgv_experiments.py tests/test_generate_hrgv_architecture_figure.py
git commit -m "feat: register monotone RPG experiments"
```

### Task 4: Produce formal evidence and integrate the report and paper

**Files:**
- Modify: `scripts/update_formal_report_v9.py`
- Modify: `tests/test_update_formal_report_v9.py`
- Modify: `docs/paper_drafts/cgdc_rsg_hrgv_methods_theory.md`
- Modify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx`
- Create: `outputs/business_metrics/mrpg_hrgv/formal/*.{csv,json}`
- Create: `outputs/paper_figures/mrpg_hrgv_architecture.{png,svg,pdf,tiff}`

**Interfaces:**
- `load_formal_mrpg_evidence(...)` verifies RSG, RPG, and all three M-RPG configurations each have three fixed seeds.
- The report updater idempotently appends one appendix after RPG, containing M-R1 through M-R4, the figure, metrics, paired evidence, and claim boundaries.

- [ ] **Step 1: Wait for and analyze the existing RPG matrix**

Run the existing RPG analyzer only after all 12 registered metrics exist. Inspect paired CIs before deciding whether RPG is a comparator or only a negative ablation.

- [ ] **Step 2: Run M-RPG only after Task 3 tests pass**

Run the 9 GPU jobs serially. Require `test_metrics.json` and `test_predictions.csv` for all three seeds of every configuration before analysis.

- [ ] **Step 3: Write failing report/paper integration tests**

```python
def test_appends_mrpg_appendix_once_with_monotone_boundaries(self) -> None:
    update_report(source, output, cgdc_dir, rpg_dir, mrpg_dir)
    text = "\n".join(paragraph.text for paragraph in Document(output).paragraphs)
    self.assertEqual(text.count("附录 F M-RPG-HRGV 候选容量归一化单调门控理论与实验"), 1)
    self.assertIn("partial g", text)
    self.assertIn("不等同于工业分选", text)
```

- [ ] **Step 4: Update report/paper and run all tests**

The appendix and draft must identify raw entropy identity separately from normalized gate variables, call M-R3 a deterministic structure property, and qualify all empirical claims with paired interval evidence.

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest discover -s tests -v`

Expected: PASS. Do not install a Word renderer; retain structural DOCX validation only.

- [ ] **Step 5: Commit selected reproducibility artifacts and push**

```powershell
git add scripts tests docs\paper_drafts "结题\基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx" outputs\business_metrics\mrpg_hrgv\formal outputs\paper_figures\mrpg_hrgv_architecture.*
git commit -m "report: integrate monotone RPG evidence"
git push origin codex/theory-aware-report
```

## Self-Review

- Spec coverage: Task 1 covers M-R1/M-R2/M-R3/M-R4 primitives; Task 2 isolates the new network path; Task 3 creates reproducible comparisons and a structure figure; Task 4 requires formal evidence before report or paper claims.
- Placeholder scan: every task declares paths, interfaces, tests, and command-level acceptance evidence.
- Type consistency: Task 1 returns diagnostics consumed by Task 2; Task 2 flags define Task 3 configurations; Task 3 JSON/CSV outputs satisfy Task 4 evidence loading.
