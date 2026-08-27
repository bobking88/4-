# RPG-HRGV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a Role-Partitioned Granularity Gate on the existing RSG-HRGV network, with exact entropy-partition tests and a three-seed, fixed-split ablation matrix.

**Architecture:** The species posterior is aggregated through the frozen species-to-role matrix. A new primitive computes role-level entropy and expected within-role species entropy, which are exposed separately to a new regret-supervised gate. The final decision, hard-negative residual verifiers, data split, backbone, and loss remain those of RSG-HRGV so the experiment isolates the gate's structural contribution.

**Tech Stack:** Python 3.11, PyTorch, torchvision EfficientNet-B0, NumPy, CSV/JSON artifacts, python-docx, matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-08-27-rpg-hrgv-design.md`

## Global Constraints

- Keep the task as closed-set recognition of public mineral specimen images under the fixed four-role protocol.
- Do not claim grade prediction, recovery, industrial sorting, mineral chemistry, stage-conditioned action prediction, OOD behavior, or deployment.
- Do not combine RPG and CGDC in the main model; CGDC remains an independently analyzed ablation.
- Use seeds `20260727`, `20260728`, and `20260729`, the existing fixed manifest, split groups, EfficientNet-B0 backbone, RSG regret flags, and residual verifier policy.
- Do not overwrite `outputs/training/cgdc_formal`; use `outputs/training/rpg_formal` for new runs.
- Reuse the finished `formal_rsg_complete_*` runs from `outputs/training/cgdc_formal` as the baseline rather than retraining them.
- State P-R1 through P-R4 as formula-level properties. State empirical improvements only when paired seed-and-cluster Bootstrap intervals support them.
- Do not add original Mindat images or `best_model.pt` checkpoints to Git. Commit code, metrics, JSON/CSV summaries, figures, and report source only.

---

### Task 1: Implement and test role-partitioned uncertainty primitives

**Files:**
- Modify: `scripts/hrgv_network.py`
- Modify: `tests/test_hrgv_network.py`

**Interfaces:**
- Produces: `role_partitioned_uncertainty(species_probabilities, role_matrix, torch) -> dict[str, Tensor]`.
- Returned tensors have shape `[batch, 1]` for `between_role_entropy`, `within_role_entropy`, and `total_species_entropy`; `mapped_role_probabilities` has shape `[batch, roles]`.
- `within_role_entropy` is the posterior-weighted conditional entropy and is zero for a role with one species.
- Produces: `role_partitioned_gate_features(direct_entropy, partition, mode, torch) -> Tensor` with exactly three scalar columns: partitioned `[U_between, U_within, |H(p_d)-U_between|]`; `without_within` replaces only the middle column with zero; `without_between` replaces the first and third columns with zero; `total_only` returns `[H(S), 0, 0]`.

- [ ] **Step 1: Write failing tests for P-R1 and P-R2**

```python
def test_partitioned_entropy_obeys_chain_rule(self) -> None:
    from hrgv_network import role_partitioned_uncertainty
    species = self.torch.tensor([[0.20, 0.30, 0.10, 0.40]])
    role_matrix = self.torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    values = role_partitioned_uncertainty(species, role_matrix, self.torch)
    self.assertTrue(self.torch.allclose(
        values["total_species_entropy"],
        values["between_role_entropy"] + values["within_role_entropy"],
        atol=1e-6,
    ))

def test_within_role_redistribution_preserves_mapped_role_posterior(self) -> None:
    from hrgv_network import role_partitioned_uncertainty
    role_matrix = self.torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
    first = role_partitioned_uncertainty(
        self.torch.tensor([[0.20, 0.30, 0.10, 0.40]]), role_matrix, self.torch
    )
    second = role_partitioned_uncertainty(
        self.torch.tensor([[0.45, 0.05, 0.25, 0.25]]), role_matrix, self.torch
    )
    self.assertTrue(self.torch.allclose(
        first["mapped_role_probabilities"], second["mapped_role_probabilities"]
    ))
```

- [ ] **Step 2: Run the focused test file and verify the new tests fail**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network -v`

Expected: FAIL with an import error for `role_partitioned_uncertainty`.

- [ ] **Step 3: Add the validated probability and entropy implementation**

```python
def role_partitioned_uncertainty(species_probabilities, role_matrix, torch):
    _validate_probability_matrix(species_probabilities, "species_probabilities")
    if role_matrix.ndim != 2 or role_matrix.shape[1] != species_probabilities.shape[1]:
        raise ValueError("role_matrix must have one column per species.")
    if not torch.allclose(role_matrix.sum(dim=0), torch.ones_like(role_matrix.sum(dim=0))):
        raise ValueError("Every species must map to exactly one role.")
    epsilon = torch.finfo(species_probabilities.dtype).eps
    species = species_probabilities.clamp_min(epsilon)
    species = species / species.sum(dim=1, keepdim=True)
    mapped = species @ role_matrix.T
    mapped = mapped / mapped.sum(dim=1, keepdim=True).clamp_min(epsilon)
    total = -(species * species.log()).sum(dim=1, keepdim=True)
    between = -(mapped.clamp_min(epsilon) * mapped.clamp_min(epsilon).log()).sum(dim=1, keepdim=True)
    conditional = (
        species.unsqueeze(1) * role_matrix.unsqueeze(0)
        / mapped.unsqueeze(2).clamp_min(epsilon)
    )
    conditional_entropy = -(
        conditional * conditional.clamp_min(epsilon).log()
    ).sum(dim=2)
    within = (mapped * conditional_entropy).sum(dim=1, keepdim=True)
    return {"mapped_role_probabilities": mapped, "total_species_entropy": total,
            "between_role_entropy": between, "within_role_entropy": within}
```

- [ ] **Step 4: Add a singleton-role regression test and run focused tests**

```python
def test_singleton_role_has_zero_conditional_entropy(self) -> None:
    from hrgv_network import role_partitioned_uncertainty
    values = role_partitioned_uncertainty(
        self.torch.tensor([[0.25, 0.75]]), self.torch.eye(2), self.torch
    )
    self.assertTrue(self.torch.allclose(
        values["within_role_entropy"], self.torch.zeros((1, 1))
    ))

def test_gate_feature_modes_do_not_conflate_partitioned_uncertainties(self) -> None:
    from hrgv_network import role_partitioned_gate_features
    partition = {
        "between_role_entropy": self.torch.tensor([[0.4]]),
        "within_role_entropy": self.torch.tensor([[0.3]]),
        "total_species_entropy": self.torch.tensor([[0.7]]),
    }
    features = role_partitioned_gate_features(self.torch.tensor([[0.6]]), partition, "partitioned", self.torch)
    self.assertTrue(self.torch.allclose(features, self.torch.tensor([[0.4, 0.3, 0.2]])))
    self.assertTrue(self.torch.allclose(
        role_partitioned_gate_features(self.torch.tensor([[0.6]]), partition, "without_between", self.torch),
        self.torch.tensor([[0.0, 0.3, 0.0]]),
    ))
```

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network -v`

Expected: PASS.

- [ ] **Step 5: Commit the primitive and proofs**

Run: `git add scripts/hrgv_network.py tests/test_hrgv_network.py`

Run: `git commit -m "feat: add role partitioned uncertainty primitives"`

### Task 2: Add the RPG gate while preserving RSG and CGDC behavior

**Files:**
- Modify: `scripts/hrgv_network.py`
- Modify: `scripts/train_hrgv_mineral_classifier.py`
- Modify: `tests/test_hrgv_network.py`
- Modify: `tests/test_train_hrgv_mineral_classifier.py`

**Interfaces:**
- Constructor adds `enable_rpg: bool = False` and `rpg_entropy_mode: str = "partitioned"`.
- Valid RPG modes are `partitioned`, `without_within`, `without_between`, and `total_only`.
- RPG outputs include `between_role_entropy`, `within_role_entropy`, and `total_species_entropy`; non-RPG output contracts remain unchanged.
- CLI adds `--enable-rpg` and `--rpg-entropy-mode`; RPG and CGDC are mutually exclusive.

- [ ] **Step 1: Write failing model and CLI tests**

```python
def test_rpg_model_exposes_separated_entropy_and_valid_fusion(self) -> None:
    model = HierarchicalRiskGatedVerificationNet(
        self.models, self.role_matrix, pretrained=False,
        embedding_dim=8, gate_hidden_dim=16, enable_rpg=True,
    )
    outputs = model(self.torch.randn(3, 3, 64, 64))
    self.assertEqual(outputs["between_role_entropy"].shape, (3, 1))
    self.assertEqual(outputs["within_role_entropy"].shape, (3, 1))
    self.assertTrue(self.torch.allclose(
        outputs["fused_role_probabilities"].sum(dim=1), self.torch.ones(3), atol=1e-6
    ))

def test_cli_rejects_cgdc_and_rpg_together(self) -> None:
    args = parse_args(["--manifest", "a.csv", "--dataset-root", "root", "--output-dir", "out",
                       "--enable-cgdc", "--enable-rpg"])
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
        validate_args(args)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network tests.test_train_hrgv_mineral_classifier -v`

Expected: FAIL because RPG arguments and outputs are not implemented.

- [ ] **Step 3: Implement RPG-specific gate inputs without modifying baseline gate inputs**

```python
if self.enable_rpg:
    partition = role_partitioned_uncertainty(species_probabilities, self.role_matrix, torch)
    entropy_inputs = role_partitioned_gate_features(
        direct_entropy, partition, self.rpg_entropy_mode, torch
    )
    gate_inputs = torch.cat([features, direct_entropy, entropy_inputs], dim=1)
else:
    gate_inputs = torch.cat([features, direct_entropy, mapped_entropy, expert_js_divergence], dim=1)
```

`role_partitioned_gate_features` must return exactly three scalar columns. Gate input dimensions must be constructed from the selected architecture, not hard-coded.

- [ ] **Step 4: Thread configuration through training artifacts and verify behavior**

Add the two flags to parsing, validation, model construction, output metadata, and model-name construction. Keep `role_probability_*` exports as final four-role posteriors. Add entropy diagnostic CSV columns only when RPG is enabled.

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network tests.test_train_hrgv_mineral_classifier -v`

Expected: PASS.

- [ ] **Step 5: Run a CPU smoke training pass and commit**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\train_hrgv_mineral_classifier.py --manifest "数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv" --dataset-root "D:\成信工科研\人工智能选矿\数据集\mindat_manual_positive_v1" --output-dir "outputs\training\rpg_smoke" --enable-rpg --device cpu --smoke-run --no-pretrained`

Expected: writes `test_metrics.json` and `test_predictions.csv` with RPG entropy diagnostics.

Run: `git add scripts/hrgv_network.py scripts/train_hrgv_mineral_classifier.py tests/test_hrgv_network.py tests/test_train_hrgv_mineral_classifier.py`

Run: `git commit -m "feat: add role partitioned granularity gate"`

### Task 3: Register the isolated RPG experiment matrix

**Files:**
- Create: `scripts/run_rpg_hrgv_experiments.py`
- Create: `tests/test_run_rpg_hrgv_experiments.py`

**Interfaces:**
- `CONFIGURATION_FLAGS` contains `rpg_complete`, `rpg_without_within`, `rpg_without_between`, and `rpg_total_entropy_only`.
- `build_experiment_commands(...)` returns 12 unique formal commands using seeds `20260727`, `20260728`, and `20260729`.
- Every RPG configuration contains `--enable-rpg`, residual verifier flags, and the detached gate feature policy used by `rsg_complete`.

- [ ] **Step 1: Write the failing matrix test**

```python
def test_formal_rpg_matrix_has_four_configurations_and_three_seeds(self) -> None:
    from run_rpg_hrgv_experiments import CONFIGURATION_FLAGS, build_experiment_commands
    commands = build_experiment_commands(PROJECT_ROOT, MANIFEST, DATASET_ROOT, OUTPUT_ROOT,
                                         Path(sys.executable), "cuda", TORCH_HOME, "formal")
    self.assertEqual(set(CONFIGURATION_FLAGS), {
        "rpg_complete", "rpg_without_within", "rpg_without_between", "rpg_total_entropy_only"
    })
    self.assertEqual(len(commands), 12)
    self.assertIn("--enable-rpg", commands[0].arguments)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_run_rpg_hrgv_experiments -v`

Expected: FAIL with an import error for `run_rpg_hrgv_experiments`.

- [ ] **Step 3: Implement the runner by following the formal command pattern**

Create an `ExperimentCommand` dataclass and `build_experiment_commands` matching `run_cgdc_rsg_experiments.py`. Define:

```python
RSG_FLAGS = (
    "--verifier-mode", "residual", "--lambda-gate-regret", "0.1",
    "--gate-regret-temperature", "0.2", "--gate-gap-temperature", "0.5",
    "--detach-gate-features",
)
CONFIGURATION_FLAGS = {
    "rpg_complete": (*RSG_FLAGS, "--enable-rpg"),
    "rpg_without_within": (*RSG_FLAGS, "--enable-rpg", "--rpg-entropy-mode", "without_within"),
    "rpg_without_between": (*RSG_FLAGS, "--enable-rpg", "--rpg-entropy-mode", "without_between"),
    "rpg_total_entropy_only": (*RSG_FLAGS, "--enable-rpg", "--rpg-entropy-mode", "total_only"),
}
```

- [ ] **Step 4: Run the matrix test and dry-run**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_run_rpg_hrgv_experiments -v`

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\run_rpg_hrgv_experiments.py --manifest "数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv" --dataset-root "D:\成信工科研\人工智能选矿\数据集\mindat_manual_positive_v1" --output-root "outputs\training\rpg_formal" --stage formal --device cuda --dry-run`

Expected: PASS; dry-run prints exactly twelve unique commands.

- [ ] **Step 5: Commit the registered matrix**

Run: `git add scripts/run_rpg_hrgv_experiments.py tests/test_run_rpg_hrgv_experiments.py`

Run: `git commit -m "feat: register RPG formal ablations"`

### Task 4: Analyze RPG results with matched RSG baseline and make the architecture figure

**Files:**
- Create: `scripts/analyze_rpg_hrgv_experiments.py`
- Modify: `scripts/generate_hrgv_architecture_figure.py`
- Create: `tests/test_analyze_rpg_hrgv_experiments.py`
- Modify: `tests/test_generate_hrgv_architecture_figure.py`

**Interfaces:**
- `analyze_rpg_hrgv_experiments.py` accepts `--config-root` entries, including `rsg_complete=outputs/training/cgdc_formal`, and writes `rpg_three_seed_summary.{csv,json}`, `rpg_ablation_deltas.csv`, per-configuration paired JSON, and `analysis_manifest.json`.
- It reuses the same seed-and-cluster Bootstrap conventions and Brier/ECE calculations as CGDC analysis.
- `generate_rpg_architecture_figure(prefix: Path)` writes PNG, SVG, PDF, and TIFF, showing the species posterior, frozen role partition, `U_between`, `U_within`, RPG gate, RSG fusion, and hard-negative residual verification.

- [ ] **Step 1: Write failing analysis and figure tests**

```python
def test_rpg_analysis_includes_partitioned_ablation_names(self) -> None:
    from analyze_rpg_hrgv_experiments import REQUIRED_CONFIGURATIONS
    self.assertEqual(REQUIRED_CONFIGURATIONS[0], "rsg_complete")
    self.assertIn("rpg_without_within", REQUIRED_CONFIGURATIONS)

def test_rpg_figure_exports_partitioned_uncertainty_modules(self) -> None:
    outputs = generate_rpg_architecture_figure(Path(temp_dir) / "rpg")
    self.assertTrue(outputs["png"].is_file())
    self.assertIn("U_between", outputs["source_text"])
    self.assertIn("U_within", outputs["source_text"])
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_analyze_rpg_hrgv_experiments tests.test_generate_hrgv_architecture_figure -v`

Expected: FAIL because RPG analyzer and figure generator are absent.

- [ ] **Step 3: Implement analysis by reusing only tested generic metrics**

Import `summarize_calibration`, `paired_calibration_bootstrap`, `parse_config_roots`, `load_cgdc_configuration`, and paired routing functions; declare the five required configuration names explicitly. Reject missing configurations and any run that does not contain exactly the three formal seeds. Preserve the metric directions from `FAVORABLE_DIRECTIONS`.

- [ ] **Step 4: Implement the publication figure and run tests**

Draw a left-to-right figure with these labeled blocks: `EfficientNet-B0`, `p_s (17 species)`, `M p_s (4 roles)`, `U_between`, `U_within`, `p_d`, `RPG regret gate`, `p_f`, and `residual Ti/metallic verifiers`. Add a caption source text that states P-R1's entropy identity and P-R3's convex evidence envelope.

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_analyze_rpg_hrgv_experiments tests.test_generate_hrgv_architecture_figure -v`

Expected: PASS.

- [ ] **Step 5: Commit analyzer and figure code**

Run: `git add scripts/analyze_rpg_hrgv_experiments.py scripts/generate_hrgv_architecture_figure.py tests/test_analyze_rpg_hrgv_experiments.py tests/test_generate_hrgv_architecture_figure.py`

Run: `git commit -m "feat: analyze and visualize RPG ablations"`

### Task 5: Run formal RPG evidence and append it to the technical report

**Files:**
- Modify: `scripts/update_formal_report_v9.py`
- Modify: `tests/test_update_formal_report_v9.py`
- Modify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx`
- Create: `outputs/business_metrics/rpg_hrgv/formal/rpg_three_seed_summary.json`
- Create: `outputs/business_metrics/rpg_hrgv/formal/rpg_ablation_deltas.csv`
- Create: `outputs/business_metrics/rpg_hrgv/formal/paired_*.json`
- Create: `outputs/paper_figures/rpg_hrgv_architecture.{png,svg,pdf,tiff}`

**Interfaces:**
- `load_formal_rpg_evidence(analysis_dir: Path) -> dict[str, object]` validates all five configurations and their three registered seeds.
- `update_report(...)` idempotently adds one `附录 E 角色分区不确定性门控网络理论与实验` section after the CGDC appendix.

- [ ] **Step 1: Execute the twelve registered GPU jobs only after all earlier tests pass**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\run_rpg_hrgv_experiments.py --manifest "数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv" --dataset-root "D:\成信工科研\人工智能选矿\数据集\mindat_manual_positive_v1" --output-root "outputs\training\rpg_formal" --stage formal --device cuda --execute`

Expected: every `formal_rpg_*_seed20260727|20260728|20260729` directory contains `test_metrics.json` and `test_predictions.csv`.

- [ ] **Step 2: Generate analysis against the reused RSG baseline**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\analyze_rpg_hrgv_experiments.py `
  --config-root rsg_complete=outputs\training\cgdc_formal `
  --config-root rpg_complete=outputs\training\rpg_formal `
  --config-root rpg_without_within=outputs\training\rpg_formal `
  --config-root rpg_without_between=outputs\training\rpg_formal `
  --config-root rpg_total_entropy_only=outputs\training\rpg_formal `
  --output-dir outputs\business_metrics\rpg_hrgv\formal `
  --bootstrap-replicates 2000
```

Expected: three-seed summary, deltas, and four paired JSON artifacts.

- [ ] **Step 3: Write a failing report-integration test and then implement the appendix**

```python
def test_appends_rpg_appendix_once_with_entropy_identity_and_boundaries(self):
    update_report(source, output, cgdc_analysis_dir, rpg_analysis_dir)
    document = Document(output)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    self.assertEqual(text.count("附录 E 角色分区不确定性门控网络理论与实验"), 1)
    self.assertIn("H(S|x)=H(R|x)+H(S|R,x)", text)
    self.assertIn("不等同于工业分选", text)
```

`load_formal_rpg_evidence` must raise `ValueError` if any required configuration or seed is absent. The appendix must place P-R1 through P-R4 beside the RPG structure figure and label all performance statements as three-seed empirical evidence.

- [ ] **Step 4: Run document structural tests and full suite**

Run: `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest discover -s tests -v`

Expected: PASS. Do not install a Word renderer; validate DOCX package relationships, image relationships, unique captions, headings, and claim-boundary text only.

- [ ] **Step 5: Commit reproducible artifacts and sync GitHub**

Run: `git add scripts/update_formal_report_v9.py tests/test_update_formal_report_v9.py "结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx" outputs/business_metrics/rpg_hrgv/formal outputs/paper_figures/rpg_hrgv_architecture.*`

Run: `git commit -m "report: integrate RPG formal evidence"`

Run: `git push origin codex/theory-aware-report`

## Self-Review

- Spec coverage: Tasks 1-2 implement P-R1 through P-R4 and the isolated RPG gate; Task 3 fixes the three-seed experimental contract; Task 4 covers paired analysis and the structure figure; Task 5 creates report evidence only after formal output exists.
- Placeholder scan: no task uses TODO/TBD or generic test wording; every code change has a named test and command.
- Type consistency: Task 1's returned dictionary keys are used by Task 2; Task 2's CLI flags are registered by Task 3; Task 3's configuration names are enforced by Task 4 and consumed by Task 5.

## Execution Handoff

The plan is ready at `docs/superpowers/plans/2026-08-27-rpg-hrgv-implementation.md`. Execute inline after the ongoing CGDC matrix has completed, so GPU jobs remain serialized and the CGDC result is preserved as an independent comparator.
