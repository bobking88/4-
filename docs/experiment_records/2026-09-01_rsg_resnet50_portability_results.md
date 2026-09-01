# RSG-HRGV ResNet50 Backbone Portability Check

## Purpose

This registered three-seed check tests the empirical boundary of the RSG backbone-invariance proposition. It does not compare backbone quality. Both configurations use the same ImageNet-pretrained ResNet50 visual backbone, split, augmentation, optimizer, early stopping, residual hard-negative verifiers, training budget, and seeds. The only comparison is HRGV reference versus RSG-HRGV routing.

## Protocol

- Dataset: fixed four-role Mindat specimen-image protocol.
- Seeds: `20260727`, `20260728`, and `20260729`.
- Reference: `hrgv_reference`.
- Comparison: `rsg_complete`.
- Statistics: paired two-stage bootstrap over seeds and `split_group_id` image clusters, 2,000 replicates, seed `20260822`.
- Reproducibility artifacts: `outputs/business_metrics/rsg_hrgv/resnet50_portability/` and `outputs/paper_figures_v3/fig_rsg_theory_evidence_portability_*`.

## Results

| Configuration | Accuracy, mean | Macro F1, mean | Mean routing regret, mean |
| --- | ---: | ---: | ---: |
| HRGV reference | 75.36% | 73.29% | 11.84% |
| RSG-HRGV | 76.64% | 74.28% | 8.51% |

The predefined RSG minus HRGV routing-regret difference is -3.33 percentage points, with a 95% bootstrap interval of [-4.86, -1.98] percentage points. Each seed has a negative routing-regret difference. This reproduces reduced predefined routing regret after replacing EfficientNet-B0 with ResNet50.

Accuracy difference is +1.27 percentage points (95% CI [-0.44, +3.12]) and Macro F1 difference is +0.99 percentage points (95% CI [-0.75, +2.88]). Both intervals cross zero. Target recall and the two intrusion metrics also have intervals crossing zero. Therefore this check does not establish stable classification superiority, target-recall superiority, or intrusion-control superiority.

## Interpretation Boundary

The result empirically supports the portability boundary of the RSG routing mechanism under the tested ResNet50 output contract. The deterministic Theorems B.1--B.3 remain fusion-layer statements conditional on valid expert probability outputs, convex fusion, and stop-gradient routing supervision. The experiment does not prove that every visual backbone will improve, nor does it support industrial sorting, grade prediction, recovery prediction, elemental-content inference, real external generalization, or unknown-mineral rejection claims.
