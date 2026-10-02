# Verifier Trust Development Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Record RED/GREEN and full-suite results in the ledger.

**Goal:** Execute the human-approved Fold 0 development comparison of fixed, free-strength and conditional-risk verification.

**Architecture:** Reuse frozen probability tables and the existing 18-64-16-1 evidence head. Separate analytic probability transformations from source validation, training and delivery. A4 fits unweighted target BCE and projects its estimated probability into the verification interval; A3 fits final four-class NLL.

**Tech Stack:** Existing Python, PyTorch, unittest; CPU float64. No new dependencies.

**Spec:** `docs/experiment_records/2026-10-02_abmp_conditional_risk_derivation.md`; human approval "批准开发实验".

## Global Constraints

- Only Fold 0 gate_stop_projector_fit and projector_stop; frozen expert and OOS gate.
- No formal report edit, outer-fold access, sweep, deployment claim or industrial cost assumption.
- Three seeds, 30 epochs each, batch 256, AdamW lr 0.001, weight decay 0.0001, dropout 0.1.
- Same head, initialization, shuffle/dropout seed and stopping NLL criterion for both learned methods.
- A2 is the fit-only exact convex scalar optimum, including endpoints.
- Refuse output overwrite; pin source hashes; keep all seeds and controls, not only best results.

## Review Focus

- Zero contradiction and probability endpoints: identity and explicit validation, never silent posterior clipping.
- CSV reorder, labels, duplicates or changed hashes: fail before fitting.
- Checkpoint leakage: train only fit; choose earliest minimum stopping four-class NLL.
- Dynamic strength versus mere calibration: fit-mean replay on stop, same controls for every seed.
- Failure or negative results: no promotion; report and outer folds remain unchanged.

## Task 1: Analytic Risk Layer

Files: `scripts/verifier_risk.py`, `tests/test_verifier_risk.py`.
Interfaces: `verification_family(p, contradiction, gamma)`, `conditional_risk_projection(p, contradiction, target_probability)`, `optimal_global_strength(p, contradiction, labels)`.

- [x] Write hand-derived endpoint/ratio/zero-contradiction, gradients, regret-bound and convex-optimum tests.
- [x] Run unittest on the new file. Expected: missing implementation assertions fail.
- [x] Implement the three interfaces with strict input contracts and finite float64 behavior.
- [x] Run focused tests and full suite. Expected: green.

## Task 2: Registered Experiment Runner

Files: `scripts/run_verifier_trust_development.py`, `tests/test_verifier_trust_development.py`, `docs/experiment_protocols/abmp_verifier_trust_development_v1.json`.
Interfaces: `load_subset(root, name, input_hashes)`, `fit_head(fit, stop, mode, seed, budget)`, `development_gate(metrics, controls, diagnostics, thresholds)`, CLI with `--root --protocol --output`.

- [x] Write source-integrity, real training/checkpoint/replay, batch-independence, complete-controls and gate tests.
- [x] Observe missing-feature failures before implementation.
- [x] Implement without pixel/expert training or outer data access. Save histories, per-image outputs, initial/selected state hashes and replay diagnostics.
- [x] Run focused tests and full suite. Expected: green.

## Task 3: Real Development and Delivery

- [x] Run the pre-registered CLI once. Expected: six head fits and three unchanged fixed controls, all results retained.
- [x] Reload all six checkpoints and replay predictions, verify input and report immutability.
- [x] Record honest results and update the research plan; do not update the formal report.
- [x] Perform separate whole-change self-review; no subagent tool is available.
- [ ] Commit only named deliverables and attempt authorized branch upload if approval/network permits; never bypass a failed approval.

Development thresholds are frozen in the protocol before training. Any passing result is development-only, not independent confirmation.
