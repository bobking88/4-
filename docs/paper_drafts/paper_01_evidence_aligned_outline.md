# First Paper: Evidence-Aligned Outline and Decision Record

## 1. One-Sentence Argument

In closed-set four-role recognition of vanadium-titanium-magnetite-related minerals from public specimen images, we study an auditable cross-granularity regret-supervised gated verification network. Registered multi-seed, photographer-held-out, and backbone-replacement tests distinguish its directly verified routing mechanism from empirical effects that are not yet shown to be stable at the whole-classification level.

## 2. Terminology Ledger

| Canonical term | First-use definition | Do not substitute with |
| --- | --- | --- |
| public-specimen four-role task | Closed-set recognition of `target_mineral`, `ti_bearing_negative`, `gangue_negative`, and `metallic_hard_negative` from Mindat specimen images | industrial sorting, grade prediction |
| HRGV-Net | Hierarchical Risk-Gated Verification Network with direct-role and species-mapped-role experts | generic mixture of experts |
| RSG | regret-supervised gate for HRGV-Net | risk-aware loss without definition |
| routing regret | Excess negative log-likelihood relative to the expert with the higher true-role probability under the two-expert convex fusion | accuracy regret, industrial loss |
| M-RPG / PHR | Registered theory-driven candidate extensions | proven main method, universally superior gate |
| target-class recall | Recall of `target_mineral` under the fixed four-role test protocol | recovery rate |
| target intrusion rate | A non-target role predicted as `target_mineral` under the fixed test protocol | industrial contamination rate |
| split-group clustered Bootstrap | Resampling registered random seeds and whole `split_group_id` clusters | independent-image significance test |

## 3. Evidence Ledger

### Evidence that can be stated directly

- The data pipeline fixes a 17-species-to-four-role mapping, controls exact and near duplicate leakage, and preserves split groups.
- HRGV-Net retains both a direct-role posterior and a species-mapped-role posterior, then fuses them with a constrained gate.
- For the RSG gate, two-expert convex fusion gives an exact routing-regret decomposition and a probability lower-bound-based upper bound; the soft target converges exponentially toward hard expert selection as the true-role expert gap grows.
- In the three-seed fixed-test comparison, RSG reduced mean routing regret by 1.77 percentage points relative to matched HRGV, with paired split-group clustered Bootstrap 95% CI [-2.86, -0.69].
- In the independent photographer-held-out three-seed confirmation and a matched ResNet50 backbone replacement, RSG reduced mean routing regret by 3.53 and 3.33 percentage points, respectively; both corresponding intervals excluded zero.
- Nine high-precision checkpoint replays over 10,242 images produced zero numerical violations of the registered fusion identity and routing bounds within the stated tolerance.

### Evidence that must be stated as inconclusive

- RSG versus the no-regret-gate HRGV reference did not show a stable Macro F1 or target-recall advantage in the current three-seed Bootstrap analysis.
- The five-seed M-RPG extension did not establish stable gains in Macro F1, target recall, target intrusion, Brier score, or ECE; its direct ablations also did not isolate a stable component contribution.
- The validation-only, eight-configuration PHR screen reduced its two local pairwise regrets but reduced Macro F1 and target recall while increasing both registered target-intrusion rates. It therefore did not advance to a formal multi-seed test.

### Claims explicitly excluded

- Industrial separation, grade prediction, recovery-rate improvement, elemental-content identification, field generalization, and unknown-mineral rejection.
- Generic-first claims about mixture-of-experts, entropy decomposition, monotone networks, calibration, or cost-sensitive learning.

## 4. Manuscript Architecture

### Title candidates

1. *Auditable Cross-Granularity Risk-Gated Recognition of Vanadium-Titanium-Magnetite-Related Minerals from Public Specimen Images*
2. *Cross-Granularity Regret-Supervised Gated Verification for Visual Role Recognition of Vanadium-Titanium-Magnetite-Related Minerals*

### Introduction

1. State the narrow problem: visual role recognition from public specimen images, not grade or industrial sorting.
2. Identify the data problem: mixed specimens, fine-grained labels mapped to coarse roles, duplicate and source leakage risks.
3. Identify the modeling problem: direct role prediction and species-derived role evidence may disagree, but a learned fusion should be assessed by its own routing objective rather than inferred from an unqualified overall-score claim.
4. State three bounded contributions: auditable dataset protocol; RSG-HRGV architecture with true-role-gap-derived soft routing supervision; theory-to-implementation evidence that separates routing-regret improvement from inconclusive whole-classification effects.

### Methods

1. Dataset acquisition, audit, frozen labels, duplicate/group control, and fixed split.
2. HRGV-Net: EfficientNet-B0 backbone, two experts, convex fusion, residual verifiers, and RSG loss.
3. RSG routing: regret target, gap weighting, local gradient isolation, convex-fusion regret identity and soft-target approximation bound.
4. Residual hard-negative verifiers, registered configurations, seeds, metrics, split-group clustered Bootstrap, and high-precision formula replay.

### Results

1. Dataset composition and leakage-control evidence.
2. Baseline and role-risk metrics, with confusion patterns.
3. RSG formal comparison: retain confidence intervals that cross zero.
4. Mechanism verification: fixed test, photographer holdout, ResNet50 replacement, routing-regret component ablations, and the explicit limits of global metrics.
5. Candidate-extension boundaries: M-RPG five-seed inconclusive result and PHR validation-screen non-promotion.
6. Failure modes: mixed specimens, proxy minerals, class-internal visual variability, and public-image source bias.

### Discussion

1. Explain why a verified reduction in the pre-defined routing regret does not imply stable Macro F1, recall, or intrusion improvement.
2. Explain why exact convex-fusion and soft-target properties are calculation-graph constraints rather than generalization theorems.
3. Contrast a public-specimen image task with actual mineral processing data needs.
4. State the required next evidence: independently curated real-image tests, expert visual audit, and a pre-registered external comparison; stage-conditioned decision learning remains a second-stage question.

## 5. Pre-Submission Gate

The current report can retain M-RPG and PHR as theory-and-evidence appendices, but neither is a first-paper performance contribution. A paper may report RSG-HRGV as a bounded routing-method study; it must not claim a general algorithmic advantage until at least one of the following is complete:

1. A source-held-out evaluation that uses a defensible source or photographer grouping and reports the resulting confidence intervals.
2. A small, independently curated real-specimen external set with expert labels, used only as an external test and not mixed into the public-image split.
3. A newly pre-registered method evaluated on a genuinely independent split or external set, with a pre-specified global risk objective and direct ablations that identify a stable contribution.

The photographer-held-out experiment is now complete and supports the limited RSG routing-mechanism claim, not industrial or cross-site generalization. The next high-value evidence is therefore item 2: an independently curated, expert-confirmed external set. Without it, the first paper should be positioned as a rigorous technical-report-derived methods and reproducibility study rather than an algorithm-superiority paper.
