# ABMP Candidate Capacity Audit Implementation Plan

> Execute inline using superpowers:executing-plans. This is the diagnostic next step of the approved ABMP research programme, not a new network claim.

**Goal:** Identify which existing expert, verifier or routing restriction removes target-recovery capacity before designing another network.

**Architecture:** Reuse the frozen Fold 0 expert and OOS gate. Extract only gate_stop_projector_fit and projector_stop images. Compare direct, mapped, fixed-average, OOS and original-gate posteriors before and after shared residual verification. Solve target-winning intervals on the full direct/mapped segment and the restricted q0/qphi segment.

**Tech Stack:** Existing training Python, PyTorch, unittest, CSV and JSON.

**Spec:** Section 20 of docs/superpowers/specs/2026-09-30-abmp-rsg-net-v2-design.md and next-step section of docs/experiment_records/2026-10-01_abmp_rsg_v2_development.md.

## Global Constraints

- No Fold 1/2 access and no outer_eval image loading, optimisation or selection.
- Read inner manifest identities only for group separation and validate hashes against the existing Fold 0 lock.
- Do not modify frozen weights or previous outputs; refuse an existing output directory.
- Label-aware attainable bounds are diagnostics, not deployable classifier performance.
- Use the source verifier thresholds and strengths; do not assume all checkpoints share defaults.
- Save probabilities and provenance, not image pixels, embeddings or checkpoint files.

## Review Focus

- Two non-target endpoints can have a target-winning interior; endpoint union is not an exact capacity calculation.
- Constant slopes and empty intervals must not divide by zero or produce false feasibility.
- Shared target scaling is positive and normalised; effective mixture weights need their normalisation factors.
- Verifier penalties can remove true and false targets; report both, not just apparent intrusion reduction.
- Missing classes must produce explicit undefined metrics, not NaN JSON or invented rates.

## Task 1 Component Geometry and Extraction

Files: scripts/audit_abmp_candidate_capacity.py and tests/test_audit_abmp_candidate_capacity.py.

Interfaces:
- target_segment_interval(first, second, lower=0, upper=1, minimum_margin=1e-9) returns lower, upper and feasible tensors for q(g)=g*first+(1-g)*second.
- build_component_branches(cache, candidate_gate, verifier_config) returns branch posteriors, common scale and equivalent verified-space gate.
- analyze_component_cache(cache, candidate_gate, verifier_config) returns serialisable metrics, verifier effects and segment capacities.
- validate_sources(protocol_dir, source_dir) validates inner-only manifests, group separation and frozen source hashes.

- [x] Write and observe failing tests for interior target feasibility, constant directions, verifier commutation, lost true/false targets, restricted-segment capacity and source provenance guards.
- [x] Implement the diagnostic script using the existing frozen expert loader and extraction APIs.
- [x] Run the dedicated tests and extract the two permitted inner subsets once.

## Task 2 Evidence and Research Decision

- [x] Save component_predictions.csv, per-image routing evidence and audit_summary.json with source hashes and no promotion decision.
- [x] Verify normalised common target scaling, effective-gate equivalence and target-set monotonicity on real data.
- [x] Derive the finite-sample nested-segment and one-way-verifier properties with assumptions and numerical residuals.
- [x] Write an experiment record explaining which next network design is supported by observed capacity; retain the possibility that no current routing change can help.
- [x] Run full regression tests and whole-change adversarial review. Ruling: ARS disallows automatic delegation without explicit permission; review is inline and not represented as independent external review.
- [ ] Create a scoped commit and make one bounded GitHub retry; record upload failure honestly if the network is unavailable.

Research efficacy remains unproved until the subsequent design and independent experiments succeed.
