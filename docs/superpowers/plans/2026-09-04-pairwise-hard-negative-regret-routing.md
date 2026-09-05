# PHR-Routing-Net Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate a pairwise hard-negative regret routing extension of RSG-HRGV that targets target-mineral versus Ti-bearing and metallic-hard-negative boundaries without altering the registered baseline protocol.

**Architecture:** Keep the shared backbone, direct-role expert, species-mapped role expert, RSG global gate, and binary verifier heads. Add two pair gates that mix expert target-versus-negative log-odds; project both fused margins into the four-class posterior through the closed-form minimum-norm three-node correction. Train pair gates from detached label-directed margin advantage targets while final classification loss continues to train visual experts.

**Tech Stack:** Python 3, PyTorch, torchvision, CSV/JSON experiment artefacts, unittest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-02-pairwise-hard-negative-regret-routing-design.md`

## Global Constraints

- Use the frozen manifest, image-ID/near-duplicate group split, augmentation, optimizer, early stopping rule, and role order `[target_mineral, ti_bearing_negative, gangue_waste, metallic_hard_negative]`.
- Keep `--enable-phr` false by default. Old RSG-HRGV calls must retain their output values and legacy prediction fields.
- In PHR mode retain verifier losses and verifier probabilities as detached pair-gate features, but replace residual or multiplicative verifier output correction with PHR log-odds correction.
- Pair-gate features and pair-regret targets are detached by default. Only `--couple-phr-gate-features` restores their feature gradients for a registered ablation.
- Select PHR hyperparameters on validation data only. Do not use fixed-test metrics for structural or hyperparameter selection.
- Preserve user-created uncommitted outputs. Stage only files explicitly named in a task.

## File Structure

- `scripts/hrgv_network.py`: pair-margin helpers, PHR loss helpers, model branch, loss integration.
- `tests/test_hrgv_network.py`: mathematical invariants, model contract, gradient isolation, legacy regression.
- `scripts/train_hrgv_mineral_classifier.py`: flags, validation, metrics, prediction fields, environment metadata.
- `tests/test_train_hrgv_mineral_classifier.py`: CLI and prediction contracts.
- `scripts/run_phr_hrgv_experiments.py`: registered screening and formal runner.
- `scripts/analyze_phr_hrgv_experiments.py`: aggregate metrics and grouped bootstrap.
- `tests/test_run_phr_hrgv_experiments.py` and `tests/test_analyze_phr_hrgv_experiments.py`: runner and analysis contracts.
- `scripts/generate_phr_routing_figure.py`: evidence-backed network and result figures.
- `tests/test_generate_phr_routing_figure.py`: figure source and output tests.
- `docs/experiment_records/2026-09-04_phr_hrgv.md`: formal experiment record, created only after runs exist.

## Task 1: Pairwise Math and Invariants

**Files:**
- Modify: `scripts/hrgv_network.py:204-319`
- Modify: `tests/test_hrgv_network.py:354-542`

**Interfaces:**
- `pairwise_log_odds(probabilities, target_index, negative_indices, torch) -> torch.Tensor`, shape `[batch, 2]`.
- `pairwise_routing_targets(direct_margins, mapped_margins, role_labels, target_index, negative_indices, target_temperature, gap_temperature, torch, hard_target=False, unweighted=False) -> dict[str, torch.Tensor]`.
- `pairwise_margin_routing_diagnostics(pair_gates, direct_margins, mapped_margins, role_labels, target_index, negative_indices, torch) -> dict[str, torch.Tensor]`.
- `apply_pairwise_log_odds_correction(base_probabilities, fused_margins, target_index, ti_index, metallic_index, torch) -> dict[str, torch.Tensor]`.

- [x] **Step 1: Write failing mathematical tests**

Add tests for direct/mapped role probabilities ordered `[T,I,G,M]`. Test target versus Ti and target versus metallic log odds, label-direction reversal for an `I` label, and the exact per-edge regret identity:

```python
oracle_margin - fused_margin == (
    (gate - hard_oracle_gate).abs()
    * (direct_utility - mapped_utility).abs()
)
```

Add a logistic loss test for `0 <= phi(fused) - phi(oracle) <= margin_regret`, and a correction test that both final log-odds equal requested fused margins.

- [x] **Step 2: Run targeted tests and verify they fail**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network.HRGVNetworkTests.test_pairwise_targets_reverse_the_preferred_expert_for_negative_labels -v
```

Expected: FAIL because PHR helper functions do not exist.

- [x] **Step 3: Implement the pure helpers**

Add them immediately after `regret_gate_targets`. Clamp probability before `log`. For edge $q$, set `eligible = (role_labels == target_index) | (role_labels == negative_index)` and direction to `+1` for target, `-1` for negative. Build detached utility gap, sigmoid soft target, tanh gap weight, hard oracle, and differentiable zero for an empty eligible set.

Apply correction with:

```python
delta_ti = fused_margins[:, 0:1] - (base_logits[:, 0:1] - base_logits[:, 1:2])
delta_m = fused_margins[:, 1:2] - (base_logits[:, 0:1] - base_logits[:, 3:4])
a_t = (delta_ti + delta_m) / 3.0
a_ti = (delta_m - 2.0 * delta_ti) / 3.0
a_m = (delta_ti - 2.0 * delta_m) / 3.0
```

Keep the gangue logit unchanged, return corrected probabilities, base margins, deltas, and adjustments.

- [x] **Step 4: Verify all pure invariants**

Add assertions that the three logit adjustments sum to zero and match the explicit least-norm constrained solution. Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```powershell
git add scripts/hrgv_network.py tests/test_hrgv_network.py
git commit -m "feat: add pairwise hard-negative routing math"
```

## Task 2: PHR Model Branch and Baseline Regression

**Files:**
- Modify: `scripts/hrgv_network.py:434-788`
- Modify: `tests/test_hrgv_network.py:543-949`

**Interfaces:**
- Extend `HierarchicalRiskGatedVerificationNet.__init__` with `enable_phr: bool = False`, `phr_gate_hidden_dim: int = 128`, and `detach_phr_gate_features: bool = True`.
- In PHR mode emit `phr_pair_gates`, `phr_direct_margins`, `phr_mapped_margins`, `phr_fused_margins`, `phr_base_margins`, `phr_margin_deltas`, and `phr_logit_adjustments`.

- [x] **Step 1: Write failing PHR model tests**

Create a PHR-enabled EfficientNet-B0 model test:

```python
model = HierarchicalRiskGatedVerificationNet(
    models=models, role_matrix=role_matrix, pretrained=False,
    backbone_name="efficientnet_b0", enable_phr=True,
    detach_phr_gate_features=True,
)
outputs = model(torch.randn(2, 3, 64, 64))
self.assertEqual(outputs["phr_pair_gates"].shape, (2, 2))
```

Assert final probabilities form a simplex and their two log-odds equal `phr_fused_margins`. Add a seeded non-PHR regression test that constructs models with identical weights and checks existing keys and values are unchanged.

- [x] **Step 2: Run the model test and verify failure**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network.HRGVNetworkTests.test_phr_model_exposes_pairwise_outputs_and_exact_margins -v
```

Expected: FAIL because `enable_phr` is not accepted.

- [x] **Step 3: Implement the model branch**

Create `self.phr_gate_networks` as `nn.ModuleDict({"ti": ..., "metallic": ...})`. Each gate consumes `feature_dim + 6`: visual feature, direct/mapped margin, absolute gap, direct/mapped pair entropy, and matching verifier target probability. In PHR mode compute margins, pair gates, fused margins, and call the correction helper on `fused_role_probabilities`.

Keep verifier heads and output probabilities; do not invoke residual/multiplicative verifier post-processing in PHR mode. Reject PHR combined with CGDC, RPG, or M-RPG in this first version to preserve a single interpretable evidence chain.

- [x] **Step 4: Test gradient isolation and backward compatibility**

Create a PHR-only loss from pair-gate BCE targets and call `backward()`. Assert pair gate gradients exist while backbone, role head, species head, and verifier head gradients are absent with detached features. Add the coupled-feature ablation assertion that backbone gradients reappear. Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```powershell
git add scripts/hrgv_network.py tests/test_hrgv_network.py
git commit -m "feat: integrate pairwise hard-negative routing model"
```

## Task 3: Loss, CLI, Metrics, and Predictions

**Files:**
- Modify: `scripts/hrgv_network.py:789-900`
- Modify: `scripts/train_hrgv_mineral_classifier.py:76-1068`
- Modify: `tests/test_hrgv_network.py:798-949`
- Modify: `tests/test_train_hrgv_mineral_classifier.py:25-310`

**Interfaces:**
- Add `pairwise_regret: float = 0.0` to `HRGVLossWeights`.
- Extend `compute_hrgv_losses` with PHR temperatures and target/weight ablation arguments; return `phr_pairwise_regret_loss`, `phr_ti_gate_loss`, and `phr_metallic_gate_loss`.
- Add flags `--enable-phr`, `--lambda-phr`, `--phr-target-temperature`, `--phr-gap-temperature`, `--phr-hard-gate-target`, `--phr-unweighted`, `--phr-gate-hidden-dim`, and `--couple-phr-gate-features`.

- [x] **Step 1: Write failing loss and CLI tests**

Test that zero PHR weight preserves the old total loss; that Ti loss only uses target/Ti rows and metallic loss only target/metallic rows; and that parsing accepts:

```python
args = parse_args([
    "--manifest", "manifest.csv", "--dataset-root", "images", "--output-dir", "out",
    "--enable-phr", "--lambda-phr", "0.10", "--phr-target-temperature", "0.20",
])
self.assertTrue(args.enable_phr)
self.assertAlmostEqual(args.lambda_phr, 0.10)
```

Test prediction fields for both pair gates, three margin variants, two pair regrets, and correction values.

- [x] **Step 2: Run targeted test and verify failure**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_train_hrgv_mineral_classifier.HRGVTrainerTests.test_cli_exposes_phr_switches -v
```

Expected: FAIL because parser lacks PHR flags.

- [x] **Step 3: Implement trainer plumbing**

Compute PHR target dictionaries only when PHR outputs exist; otherwise use differentiable zero losses. Thread flags into model construction, `HRGVLossWeights`, epoch JSONL, `environment.json`, `metrics.json`, and CSV rows. In PHR mode write `phr_replaces_verifier_postprocessor: true` in environment metadata while preserving the binary verifier loss.

Add `summarize_pairwise_routing` that reports eligible count, hard selection accuracy, mean margin regret, weighted routing error, and sign-preservation rate independently for the Ti and metallic edges.

- [x] **Step 4: Verify trainer path**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network tests.test_train_hrgv_mineral_classifier -q
```

Run existing synthetic smoke fixtures with `--enable-phr --lambda-phr 0.10 --smoke-run --no-pretrained`; verify checkpoint, metrics, environment, and PHR prediction fields exist.

- [x] **Step 5: Commit**

```powershell
git add scripts/hrgv_network.py scripts/train_hrgv_mineral_classifier.py tests/test_hrgv_network.py tests/test_train_hrgv_mineral_classifier.py
git commit -m "feat: train and audit pairwise routing"
```

## Task 4: Registered Runner and Analysis

**Files:**
- Create: `scripts/run_phr_hrgv_experiments.py`
- Create: `scripts/analyze_phr_hrgv_experiments.py`
- Create: `tests/test_run_phr_hrgv_experiments.py`
- Create: `tests/test_analyze_phr_hrgv_experiments.py`

**Interfaces:**
- Runner accepts `--mode screen|formal`, `--manifest`, `--dataset-root`, `--output-root`, and `--device`.
- Screen uses seed `20260728`; formal runs only baseline and complete PHR for seeds `20260727`, `20260728`, `20260729`.
- Analysis writes `summary.csv`, `paired_cluster_bootstrap.csv`, `pairwise_routing_summary.csv`, and `analysis.json`.

- [x] **Step 1: Write failing runner and analysis tests**

Mock trainer invocation and assert screen has exactly these configurations: `rsg_reference`, `phr_complete`, `phr_fixed_half`, `phr_hard_target`, `phr_unweighted`, `phr_coupled_features`, `phr_ti_only`, and `phr_metallic_only`. Assert formal has six runs and no ablation flags.

Use synthetic aligned prediction CSVs with shared `split_group_id`; assert grouped bootstrap reports paired Macro F1, target recall, two intrusion-rate deltas, and pair metrics remain separate.

- [x] **Step 2: Verify expected failure**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_run_phr_hrgv_experiments tests.test_analyze_phr_hrgv_experiments -q
```

Expected: FAIL because runner and analysis scripts do not exist.

- [x] **Step 3: Implement registered screening**

Build commands as explicit lists given to `subprocess.run(check=True)`. Write `registered_configurations.json` before launching. Formal mode must require a validation-only `screen_decision.json` containing selected configuration, criterion identifier, seed, timestamp, and explanation.

- [x] **Step 4: Implement evidence analysis**

Load each seed's metrics and aligned prediction CSVs. Report mean, sample standard deviation, and per-seed values for Accuracy, Macro F1, target recall, both intrusions, both pair regrets, and sign-preservation rates. Bootstrap split groups, not images. Set `formal_evidence_supports_claim` in `analysis.json` only when the registered three-seed criterion is met; otherwise populate `claim_boundary` with required negative-result wording.

- [ ] **Step 5: Verify and commit**

Run the Task 4 unit tests then:

```powershell
git add scripts/run_phr_hrgv_experiments.py scripts/analyze_phr_hrgv_experiments.py tests/test_run_phr_hrgv_experiments.py tests/test_analyze_phr_hrgv_experiments.py
git commit -m "feat: register pairwise routing experiments"
```

## Task 5: Execute and Audit Experiments

**Files:**
- Create: `outputs/training/phr_hrgv_screen/registered_configurations.json`
- Create: `outputs/training/phr_hrgv_formal/<configuration>/<seed>/...`
- Create: `outputs/business_metrics/phr_hrgv/summary.csv`
- Create: `outputs/business_metrics/phr_hrgv/paired_cluster_bootstrap.csv`
- Create: `outputs/business_metrics/phr_hrgv/pairwise_routing_summary.csv`
- Create: `outputs/business_metrics/phr_hrgv/analysis.json`
- Create: `docs/experiment_records/2026-09-04_phr_hrgv.md`

- [ ] **Step 1: Verify immutable inputs**

Record trainer package versions, CUDA availability, manifest SHA-256, and Git commit in `registered_configurations.json`. Verify `data/dataset_split_manifest.csv` row count and all three split names.

- [ ] **Step 2: Run registered screening**

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\run_phr_hrgv_experiments.py --mode screen --manifest data\dataset_split_manifest.csv --dataset-root data\images --output-root outputs\training\phr_hrgv_screen --device auto
```

Do not change parameters while runs are active. Retain failure logs and record failed configurations rather than silently restarting them with modified settings.

- [ ] **Step 3: Record a validation-only decision**

Read only validation metrics. Write `screen_decision.json` using one registered criterion. If none qualifies, write a no-promotion decision, create the negative-result record, and do not launch formal seeds.

- [ ] **Step 4: Run formal seeds if promoted**

Run formal mode with its exact selected configuration, then run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\analyze_phr_hrgv_experiments.py --experiment-root outputs\training\phr_hrgv_formal --output-dir outputs\business_metrics\phr_hrgv --bootstrap-replicates 2000
```

- [ ] **Step 5: Write and commit evidence**

Record data version, commands, seeds, results, bootstrap intervals, pairwise evidence, failures, and permitted claims in the experiment record. Verify JSON/CSV readability, seed-aligned predictions, and `git diff --check`. Do not commit checkpoints or original images.

```powershell
git add docs/experiment_records/2026-09-04_phr_hrgv.md outputs/business_metrics/phr_hrgv scripts tests
git commit -m "results: add pairwise routing experiments"
```

## Task 6: Figures, Report, Paper, and GitHub

**Files:**
- Create: `scripts/generate_phr_routing_figure.py`
- Create: `tests/test_generate_phr_routing_figure.py`
- Create: `outputs/paper_figures_v3/fig_phr_routing_architecture.{png,pdf,svg,tiff}`
- Create: `outputs/paper_figures_v3/fig_phr_edge_regret.{png,pdf,svg,tiff}`
- Modify: `docs/paper_v1/manuscript_core_draft.md`
- Modify: `docs/paper_v1/theoretical_appendix.md`
- Modify: `scripts/update_formal_report_v9.py`
- Modify: `tests/test_update_formal_report_v9.py`

- [ ] **Step 1: Write failing figure and report-gating tests**

Assert figure generation rejects missing/malformed analysis, emits PNG/PDF/SVG/TIFF plus source JSON from valid fixtures, and labels oracle values as diagnostics. Assert report builder cannot state success if `formal_evidence_supports_claim` is false.

- [ ] **Step 2: Implement and inspect figures**

Draw one compact architecture panel for backbone, experts, global gate, pair gates, and constrained log-odds correction. Draw one result panel for four-class risk metrics and both edge regrets with intervals. Render PNG and visually inspect before accepting vector and TIFF exports.

- [ ] **Step 3: Integrate evidence-bound prose**

Insert equations, propositions, protocol, and results only from `analysis.json`. For a failed promotion, label PHR as a registered exploratory/negative result and retain RSG-HRGV as the main model. Preserve public-specimen and non-industrial limitations.

- [ ] **Step 4: Verify, commit, and synchronize**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network tests.test_train_hrgv_mineral_classifier tests.test_run_phr_hrgv_experiments tests.test_analyze_phr_hrgv_experiments tests.test_generate_phr_routing_figure tests.test_update_formal_report_v9 -q
```

Inspect generated figures and `git diff --check`. Stage only PHR code, tests, compact evidence, figures, report/paper updates, and documents. Push with `git push origin codex/theory-aware-report`, then verify local and remote commit IDs match.
