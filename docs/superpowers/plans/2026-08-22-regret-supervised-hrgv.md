# RSG-HRGV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a regret-supervised gate to HRGV, verify its mathematical properties and routing behavior, run controlled experiments, and update the report and manuscript only when the predefined evidence gates are met.

**Architecture:** The existing direct-role and species-mapped experts remain unchanged. During training, detached per-sample expert log losses define a soft oracle gate and a gap weight; a weighted soft binary-cross-entropy term trains the existing inference gate without labels at inference. Gate inputs are detached only in the RSG configuration so routing supervision cannot reshape the backbone through the gate path.

**Tech Stack:** Python 3.11, PyTorch, torchvision EfficientNet-B0, standard-library `unittest`, CSV/JSON experiment artifacts, matplotlib, python-docx.

**Spec:** `docs/superpowers/specs/2026-08-22-regret-supervised-hrgv-design.md`

## Global Constraints

- Preserve the fixed 8,529-image manifest and the existing train/validation/test split.
- Do not tune hyperparameters on the test set.
- Preserve the original HRGV behavior when `gate_regret=0` and `detach_gate_features=False`.
- Training labels may construct gate targets only during training/analysis; inference remains image-only.
- Main claims require three seeds and paired cluster bootstrap; a one-seed pilot is structural screening only.
- Keep stage-conditioned beneficiation decisions and OOD acquisition outside this implementation.
- Never describe public specimen performance as ore grade, recovery, or industrial sorting performance.

---

### Task 1: Regret-gate probability primitives

**Files:**
- Modify: `scripts/hrgv_network.py`
- Modify: `tests/test_hrgv_network.py`

**Interfaces:**
- Produces: `regret_gate_targets(direct, mapped, role_labels, target_temperature, gap_temperature, torch, hard_target=False, unweighted=False) -> dict[str, Tensor]`
- Produces: `weighted_soft_gate_loss(gate, target, weight, torch) -> Tensor`
- Produces: `gate_routing_diagnostics(gate, direct, mapped, role_labels, torch) -> dict[str, Tensor]`
- Consumes: role probabilities shaped `[batch, roles]`, gate shaped `[batch, 1]`, labels shaped `[batch]`.

- [ ] **Step 1: Write failing tests for target direction and gap weighting**

Add tests with direct true probabilities `[0.8, 0.2]` and mapped true probabilities `[0.2, 0.8]`. Assert the first soft target is greater than `0.5`, the second is less than `0.5`, hard targets are `[1, 0]`, and a larger absolute log-loss gap receives a larger weight.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_hrgv_network.RegretGatePrimitiveTests -v
```

Expected: import or attribute failure because the new functions do not exist.

- [ ] **Step 3: Implement target construction**

Gather the true-class probabilities, clamp by dtype epsilon, compute detached negative log likelihoods, and return these exact keys:

```python
{
    "direct_true_probability": direct_true,
    "mapped_true_probability": mapped_true,
    "expert_loss_gap": delta,
    "soft_oracle_gate": soft_target,
    "hard_oracle_gate": hard_target,
    "gate_gap_weight": gap_weight,
}
```

Validate positive temperatures and exact tensor shapes.

- [ ] **Step 4: Write failing tests for weighted soft BCE and zero-weight stability**

Assert matching gates have lower loss than reversed gates. Assert all-zero weights return a differentiable finite zero whose backward pass leaves a defined gate gradient.

- [ ] **Step 5: Implement weighted soft gate loss**

Use clamped gate probabilities and normalize by `weight.sum().clamp_min(epsilon)`. Return `gate.sum() * 0.0` when the effective weight is zero.

- [ ] **Step 6: Write and pass theorem-property tests**

Test 200 deterministic random probability pairs for:

```text
0 <= -log(p_g) + log(max(a,b))
     <= |g-g_oracle| |a-b| / epsilon
|g_soft-g_oracle| <= exp(-|delta| / temperature)
```

- [ ] **Step 7: Implement routing diagnostics**

Return hard gate selection correctness, fused true probability, per-row NLL routing regret, weighted gate error, and expert disagreement masks without changing model predictions.

- [ ] **Step 8: Run the HRGV test module**

Expected: all existing and new HRGV tests pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add scripts/hrgv_network.py tests/test_hrgv_network.py
git commit -m "feat: add regret-gate objectives and bounds"
```

---

### Task 2: Integrate RSG into the HRGV model and loss

**Files:**
- Modify: `scripts/hrgv_network.py`
- Modify: `tests/test_hrgv_network.py`

**Interfaces:**
- Extends: `HRGVLossWeights` with `gate_regret: float = 0.0`.
- Extends: `HierarchicalRiskGatedVerificationNet(..., detach_gate_features: bool = False)`.
- Extends: `compute_hrgv_losses(..., gate_regret_temperature: float = 0.20, gate_gap_temperature: float = 0.50, hard_gate_target: bool = False, unweighted_gate_regret: bool = False)`.
- Produces loss keys: `gate_regret_loss`, `mean_gate_gap_weight`, `gate_target_alignment`.

- [ ] **Step 1: Write a failing backward-isolation test**

Construct a small fake backbone as in the existing tests. Backpropagate only `gate_regret_loss` with `detach_gate_features=True`. Assert gate-network weights receive nonzero gradients while backbone, role head, and species head receive no gradients from this loss.

- [ ] **Step 2: Run the focused test and verify failure**

Expected: constructor or loss signature does not accept the new options.

- [ ] **Step 3: Add backward-compatible model configuration**

When `detach_gate_features=True`, pass `gate_inputs.detach()` to `gate_network`. Keep the constructor default `False` so old HRGV checkpoints and tests preserve their computational graph.

- [ ] **Step 4: Add the new loss term**

Call `regret_gate_targets` with detached expert probabilities, calculate `weighted_soft_gate_loss`, and add `weights.gate_regret * gate_regret_loss` to the total objective. Do not alter verifier or contrastive losses.

- [ ] **Step 5: Add exact-reproduction test**

With `gate_regret=0` and `detach_gate_features=False`, compare total loss and all original loss components against the pre-change formula to numerical tolerance `1e-7`.

- [ ] **Step 6: Run HRGV model tests**

Expected: all tests pass, including complete loss backward coverage.

- [ ] **Step 7: Commit Task 2**

```powershell
git add scripts/hrgv_network.py tests/test_hrgv_network.py
git commit -m "feat: integrate regret-supervised HRGV gate"
```

---

### Task 3: CLI, validation, training metrics, and prediction artifacts

**Files:**
- Modify: `scripts/train_hrgv_mineral_classifier.py`
- Modify: `tests/test_train_hrgv_mineral_classifier.py`

**Interfaces:**
- Adds CLI arguments: `--lambda-gate-regret`, `--gate-regret-temperature`, `--gate-gap-temperature`, `--disable-gate-regret`, `--hard-gate-target`, `--unweighted-gate-regret`, `--couple-gate-features`.
- Adds epoch outputs: `mean_gate_regret_loss`, `gate_selection_accuracy`, `mean_routing_regret_nll`, `mean_weighted_gate_error`.
- Adds prediction CSV columns specified in the design.

- [ ] **Step 1: Write failing parser-default and validation tests**

Assert the RSG formal defaults are `lambda=0.10`, `T_r=0.20`, `T_w=0.50`, soft targets, weighted loss, and detached gate features. Assert nonpositive temperatures and negative loss weights raise `ValueError`.

- [ ] **Step 2: Implement CLI options without changing legacy commands**

`--disable-gate-regret` forces the effective gate-regret weight to zero. `--couple-gate-features` passes `detach_gate_features=False`; otherwise RSG commands pass `True`.

- [ ] **Step 3: Write failing prediction-row tests**

Provide two synthetic samples and assert all nine routing-diagnostic columns are present, parseable, and aligned with `image_id`.

- [ ] **Step 4: Extend epoch collection and CSV export**

Store direct, mapped, and fused role probability rows. Compute label-dependent routing diagnostics only after labels are available; never feed them back into inference predictions.

- [ ] **Step 5: Extend metrics JSON and environment record**

Record all RSG hyperparameters, gate detachment state, routing metrics, and oracle complementarity ceiling.

- [ ] **Step 6: Run training-configuration and artifact tests**

Expected: all focused tests pass.

- [ ] **Step 7: Run a one-batch CPU smoke training**

Use `--smoke-run --epochs 1 --batch-size 4 --device cpu` with a temporary output directory. Verify model, metrics JSON, prediction CSV, and confusion matrix are produced.

- [ ] **Step 8: Commit Task 3**

```powershell
git add scripts/train_hrgv_mineral_classifier.py tests/test_train_hrgv_mineral_classifier.py
git commit -m "feat: export RSG routing diagnostics"
```

---

### Task 4: Reproducible pilot and formal experiment runner

**Files:**
- Create: `scripts/run_rsg_hrgv_experiments.py`
- Create: `tests/test_run_rsg_hrgv_experiments.py`

**Interfaces:**
- Produces deterministic command records for configurations `rsg_complete`, `rsg_hard_target`, `rsg_unweighted`, `rsg_coupled_gate`, and `hrgv_reference`.
- Uses seeds `20260727`, `20260728`, `20260729` and unique output directories.

- [ ] **Step 1: Write failing matrix tests**

Assert configuration names are unique, pilot commands use only seed `20260728`, formal commands use all three fixed seeds, and every output directory is unique.

- [ ] **Step 2: Implement the command registry and dry-run mode**

Follow the structure of `scripts/run_hrgv_experiments.py`. Include `--stage pilot|formal`, `--config`, and `--dry-run`.

- [ ] **Step 3: Test exact ablation flags**

Verify hard target, unweighted target, coupled gate features, and disabled gate regret map to their intended CLI flags.

- [ ] **Step 4: Run the runner tests and dry run**

Expected: tests pass and dry run prints reproducible commands without starting training.

- [ ] **Step 5: Commit Task 4**

```powershell
git add scripts/run_rsg_hrgv_experiments.py tests/test_run_rsg_hrgv_experiments.py
git commit -m "feat: add RSG-HRGV experiment matrix"
```

---

### Task 5: Routing and classification analysis

**Files:**
- Create: `scripts/analyze_rsg_hrgv_experiment.py`
- Create: `tests/test_analyze_rsg_hrgv_experiment.py`

**Interfaces:**
- Consumes one prediction CSV per configuration and seed.
- Produces `rsg_three_seed_summary.csv/json`, `rsg_routing_metrics.csv`, `rsg_ablation_deltas.csv`, and paired cluster bootstrap results.

- [ ] **Step 1: Write failing metric tests**

Use a four-row fixture where direct and mapped experts each uniquely solve one row. Assert disagreement rate, selection accuracy, oracle ceiling, NLL routing regret, and complementarity recovery are exact.

- [ ] **Step 2: Implement routing metrics**

Define complementarity recovery as

```text
(fused_accuracy - max(direct_accuracy, mapped_accuracy)) /
(oracle_accuracy - max(direct_accuracy, mapped_accuracy))
```

Return `None` when the denominator is zero; never divide by zero or silently substitute zero.

- [ ] **Step 3: Reuse paired cluster bootstrap**

Import the existing grouped bootstrap utilities instead of duplicating them. Orient lower intrusion and lower routing regret as improvements.

- [ ] **Step 4: Add three-seed completeness checks**

Formal summaries must reject missing or duplicate seeds.

- [ ] **Step 5: Run analysis tests**

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
git add scripts/analyze_rsg_hrgv_experiment.py tests/test_analyze_rsg_hrgv_experiment.py
git commit -m "feat: analyze RSG routing regret"
```

---

### Task 6: Run pilot, apply the evidence gate, and run formal seeds

**Files:**
- Create outputs under: `outputs/training/formal_rsg_*`
- Create summaries under: `outputs/business_metrics/rsg_hrgv/`
- Update: `docs/experiment_records/2026-08-22_hrgv_network.md`

**Interfaces:**
- Consumes the fixed manifest and existing pretrained cache.
- Produces checkpoints locally, but stages only metrics, predictions, confusion matrices, and summaries for Git.

- [ ] **Step 1: Run the seed-20260728 pilot matrix**

Run each of the five configurations through the experiment runner. Do not change parameters after viewing test-set metrics.

- [ ] **Step 2: Analyze pilot validation metrics**

Select the formal RSG configuration from validation Macro F1, target recall, both intrusion rates, and routing regret. Record the decision rule and selected configuration before formal test analysis.

- [ ] **Step 3: Evaluate the predefined pilot gate**

If none of the three design criteria is met, stop formal training and retain the result as a negative mechanism study. If a criterion is met, run all three formal seeds.

- [ ] **Step 4: Run formal three-seed experiments**

Use the selected configuration and the frozen commands. Do not modify the test split or training schedule.

- [ ] **Step 5: Generate paired statistics**

Compare RSG-HRGV with original HRGV and equal fusion using the same image IDs and split groups. Use at least 2,000 bootstrap replicates with a fixed bootstrap seed.

- [ ] **Step 6: Update the experiment record**

Record commands, environment, runtime, pilot decision, all metrics, uncertainty intervals, failures, and claim boundaries.

- [ ] **Step 7: Commit result artifacts without model weights**

Stage prediction CSVs, metrics JSON, confusion matrices, analysis summaries, and the experiment record. Exclude `best_model.pt`, caches, and logs.

---

### Task 7: Network figure, report, and manuscript integration

**Files:**
- Modify: `scripts/generate_hrgv_architecture_figure.py`
- Modify: `tests/test_generate_hrgv_architecture_figure.py`
- Create or modify: report update script under `scripts/`
- Modify: `tests/test_formal_report_v4.py` or create the next report-version test
- Modify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx`
- Modify: `docs/paper_v1/manuscript_core_draft.md`
- Modify: `docs/paper_v1/paper_outline_and_evidence_map.md`

**Interfaces:**
- Figure adds a dashed training-only route from expert losses to soft oracle target and gate loss.
- Report and paper consume only finalized three-seed summaries and paired statistics.

- [ ] **Step 1: Write figure-content and report-content tests**

Assert the figure source contains `expert regret`, `soft oracle gate`, `training only`, and no inference label path. Assert the report contains the new formula, theorem, routing metrics, and evidence-boundary language.

- [ ] **Step 2: Generate publication figure bundle**

Export PNG, SVG, PDF, TIFF, and source JSON. Preserve the existing inference architecture while visually separating the training-only supervision branch.

- [ ] **Step 3: Update theory and results text**

Add the three propositions, routing diagnostics, ablation table, paired intervals, and a paragraph stating whether the predefined claim gate passed.

- [ ] **Step 4: Update abstract and conclusion conditionally**

If the evidence gate passes, describe RSG-HRGV as the final model. If it fails, retain HRGV as the main model and present regret supervision as a mechanism analysis or negative result.

- [ ] **Step 5: Run DOCX structural verification**

Check relationships, image widths, unique figure/table numbering, formulas, required sections, and prohibited overclaims. Do not install a rendering component.

- [ ] **Step 6: Commit Task 7**

Stage the report, manuscript, figures, source JSON, update script, and tests.

---

### Task 8: Full verification and GitHub synchronization

**Files:**
- Verify all modified code, tests, outputs, documents, and Git state.

- [ ] **Step 1: Run the complete test suite**

```powershell
D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run repository checks**

```powershell
git diff --check
git status --short
```

Confirm the unrelated `结题/技术报告_配图_总体技术路线.png` remains untouched and unstaged.

- [ ] **Step 3: Audit claim-to-evidence coverage**

For every new theorem and empirical claim, point to its test, prediction artifact, summary, figure, and report paragraph. Downgrade any claim without complete evidence.

- [ ] **Step 4: Push the branch**

```powershell
git push
```

- [ ] **Step 5: Report actual status**

State test counts, selected model, measured effects, uncertainty, document paths, commit hashes, GitHub sync status, and remaining limitations.
