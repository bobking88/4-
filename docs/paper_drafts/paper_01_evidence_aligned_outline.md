# First Paper: Evidence-Aligned Outline and Decision Record

## 1. One-Sentence Argument

In closed-set four-role recognition of vanadium-titanium-magnetite-related minerals from public specimen images, we study an auditable cross-granularity risk-gated network and a capacity-normalized monotone role-partitioned extension, then use registered multi-seed experiments to distinguish its exact theoretical constraints from empirical effects not yet shown to be stable.

## 2. Terminology Ledger

| Canonical term | First-use definition | Do not substitute with |
| --- | --- | --- |
| public-specimen four-role task | Closed-set recognition of `target_mineral`, `ti_bearing_negative`, `gangue_negative`, and `metallic_hard_negative` from Mindat specimen images | industrial sorting, grade prediction |
| HRGV-Net | Hierarchical Risk-Gated Verification Network with direct-role and species-mapped-role experts | generic mixture of experts |
| RSG | regret-supervised gate for HRGV-Net | risk-aware loss without definition |
| RPG | raw role-partitioned granularity gate | hierarchical classifier |
| M-RPG | capacity-normalized monotone role-partitioned granularity gate | universally superior gate |
| target-class recall | Recall of `target_mineral` under the fixed four-role test protocol | recovery rate |
| target intrusion rate | A non-target role predicted as `target_mineral` under the fixed test protocol | industrial contamination rate |
| split-group clustered Bootstrap | Resampling registered random seeds and whole `split_group_id` clusters | independent-image significance test |

## 3. Evidence Ledger

### Evidence that can be stated directly

- The data pipeline fixes a 17-species-to-four-role mapping, controls exact and near duplicate leakage, and preserves split groups.
- HRGV-Net retains both a direct-role posterior and a species-mapped-role posterior, then fuses them with a constrained gate.
- RPG preserves the raw entropy identity `H(S|x)=H(R|x)+H(S|R,x)` under the frozen mapping.
- M-RPG preserves the raw identity, bounds the normalized uncertainty inputs in `[0,1]`, gives a nonnegative derivative of direct-expert allocation with respect to between-role uncertainty, and keeps the fused posterior in the expert envelope.
- In the registered five-seed extension, M-RPG target recall was 73.47% versus 72.35% for RSG; the paired clustered Bootstrap difference was +1.12 percentage points with 95% CI [-1.83, 3.90].

### Evidence that must be stated as inconclusive

- In the five-seed extension, M-RPG Macro F1 was 74.03% versus 74.01% for RSG; the difference was +0.02 percentage points with 95% CI [-1.44, 1.49].
- Five-seed target recall, Ti-bearing and metallic target-intrusion, Brier-score, and ECE intervals all crossed zero.
- Direct M-RPG ablations did not isolate a stable empirical contribution of the monotone coefficient or between-role uncertainty input.
- RSG versus the no-regret-gate HRGV reference did not show a stable Macro F1 or target-recall advantage in the current three-seed Bootstrap analysis.

### Claims explicitly excluded

- Industrial separation, grade prediction, recovery-rate improvement, elemental-content identification, field generalization, and unknown-mineral rejection.
- Generic-first claims about mixture-of-experts, entropy decomposition, monotone networks, calibration, or cost-sensitive learning.

## 4. Manuscript Architecture

### Title candidates

1. *Auditable Cross-Granularity Risk-Gated Recognition of Vanadium-Titanium-Magnetite-Related Minerals from Public Specimen Images*
2. *Role-Partitioned Uncertainty for Closed-Set Recognition of Vanadium-Titanium-Magnetite-Related Minerals*

### Introduction

1. State the narrow problem: visual role recognition from public specimen images, not grade or industrial sorting.
2. Identify the data problem: mixed specimens, fine-grained labels mapped to coarse roles, duplicate and source leakage risks.
3. Identify the modeling problem: direct role prediction and species-derived role evidence may disagree; a single entropy conflates role-relevant and within-role ambiguity.
4. State three bounded contributions: auditable dataset protocol; HRGV/RSG architecture; M-RPG theory with exact constraints and registered multi-seed falsification of an unqualified performance claim.

### Methods

1. Dataset acquisition, audit, frozen labels, duplicate/group control, and fixed split.
2. HRGV-Net: EfficientNet-B0 backbone, two experts, convex fusion, residual verifiers, and RSG loss.
3. RPG and M-RPG: mapping matrix, raw chain-rule identity, capacity normalization, monotone allocation derivative, and convex-envelope property.
4. Registered configurations, seeds, metrics, and split-group clustered Bootstrap.

### Results

1. Dataset composition and leakage-control evidence.
2. Baseline and role-risk metrics, with confusion patterns.
3. RSG formal comparison: retain confidence intervals that cross zero.
4. RPG and M-RPG formal comparisons: show the three-seed trend, five-seed extension, and all inconclusive overall/component effects together.
5. Failure modes: mixed specimens, proxy minerals, class-internal visual variability, and public-image source bias.

### Discussion

1. Explain that the preliminary target-recall trend was not stable after the five-seed extension, while the theory still gives interpretable constraints on the network.
2. Explain why formula-level monotonicity does not imply isolated empirical superiority.
3. Contrast a public-specimen image task with actual mineral processing data needs.
4. State the required next evidence: source-held-out and controlled real-image tests, expert visual audit, and a pre-registered stronger baseline comparison.

## 5. Pre-Submission Gate

The current report can retain the full M-RPG appendix as a theory-and-evidence record. A paper claiming algorithmic advantage should not be submitted until at least one of the following is complete:

1. A source-held-out evaluation that uses a defensible source or photographer grouping and reports the resulting confidence intervals.
2. A small, independently curated real-specimen external set with expert labels, used only as an external test and not mixed into the public-image split.
3. A reformulated method whose direct ablations identify a stable contribution beyond the current deterministic M-RPG constraints.

The five-seed extension is complete and did not confirm a stable M-RPG advantage. The most practical next experiment is item 1: it tests whether the public-image result survives a provenance shift without collecting new images. Item 2 should follow if an independently curated set can be obtained; otherwise, the first paper should be positioned as a rigorous technical-report-derived methods and reproducibility study rather than an algorithm-superiority paper.
