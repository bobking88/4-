# RSG-HRGV theory-component ablation (2026-08-25)

## Purpose and boundary

This registered follow-up tests the separately removable components of the RSG-HRGV regret-supervision branch on the frozen public-specimen image protocol. It is a visual-recognition mechanism study, not an industrial sorting, grade-prediction, recovery-rate, or OOD result.

All configurations use the same EfficientNet-B0 backbone, split manifest, three formal seeds (`20260727`, `20260728`, `20260729`), coupled residual verifier architecture, image augmentation, optimizer, early stopping, and training budget. Only the named regret-branch component changes.

## Configurations

| Configuration | Change from complete RSG | Theoretical relation |
|---|---|---|
| `rsg_complete` | Soft target, gap weight, and detached gate evidence | Frozen complete method |
| `rsg_hard_target` | Replaces `sigmoid(Delta / Tr)` with `I(Delta >= 0)` | Soft-target approximation bound |
| `rsg_unweighted` | Replaces `tanh(abs(Delta) / Tw)` with `1` | Gap-aware regret bound proxy |
| `rsg_coupled_gate` | Allows regret-gate evidence to backpropagate to experts/backbone | Local gradient-isolation boundary |

The frozen complete settings are `lambda_gate_regret=0.1`, `Tr=0.2`, and `Tw=0.5`.

## Reproduction

```powershell
\.venv-training\Scripts\python.exe scripts\run_rsg_hrgv_experiments.py `
  --manifest 数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv `
  --dataset-root 数据集\mindat_manual_positive_v1 `
  --output-root outputs\training\rsg_theory_ablation `
  --stage formal `
  --config rsg_hard_target `
  --config rsg_unweighted `
  --config rsg_coupled_gate `
  --device cuda --execute
```

The complete RSG reference is retained under `outputs/training/rsg_controlled`. Analyze all roots together with:

```powershell
\.venv-training\Scripts\python.exe scripts\analyze_rsg_theory_ablation.py `
  --config-root rsg_complete=outputs\training\rsg_controlled `
  --config-root rsg_hard_target=outputs\training\rsg_theory_ablation `
  --config-root rsg_unweighted=outputs\training\rsg_theory_ablation `
  --config-root rsg_coupled_gate=outputs\training\rsg_theory_ablation `
  --output-dir outputs\business_metrics\rsg_hrgv\theory_ablation `
  --bootstrap-replicates 2000
```

## Three-seed summary

| Configuration | Accuracy | Macro F1 | One-right gate selection | Mean routing regret |
|---|---:|---:|---:|---:|
| Complete RSG | 76.45% +/- 0.98% | 74.76% +/- 0.90% | 58.01% +/- 1.76% | 8.09% +/- 0.17% |
| Hard target | 76.38% +/- 0.20% | 74.56% +/- 0.40% | 53.24% +/- 4.44% | 10.02% +/- 0.65% |
| Unweighted | 75.62% +/- 1.16% | 73.77% +/- 1.23% | 58.21% +/- 4.78% | 7.88% +/- 0.45% |
| Coupled gate | 74.95% +/- 1.25% | 73.17% +/- 1.35% | 51.64% +/- 8.83% | 7.57% +/- 0.21% |

All percentages are evaluated on the same 1,284-image fixed test set. Standard deviations are across the three independently trained seeds.

## Paired, group-cluster Bootstrap findings

For each seed, predictions are aligned by `image_id`; bootstrap sampling resamples the three seed records and test `split_group_id` clusters. The reported interval is the 95% two-stage bootstrap interval of comparison minus complete RSG.

| Comparison | Routing-regret difference | 95% interval | Directional probability | Interpretation |
|---|---:|---:|---:|---|
| Hard target - complete | +1.94 pp | [+1.30, +2.60] pp | 0.000 unfavorable | The hard target increases routing regret. This supports the continuous target as the tested choice. |
| Unweighted - complete | -0.20 pp | [-0.62, +0.12] pp | 0.860 favorable | Interval crosses zero. Gap weighting remains theory-motivated but has no stable independent empirical gain here. |
| Coupled gate - complete | -0.51 pp | [-0.85, -0.16] pp | 1.000 favorable | Regret decreases, but Macro F1 changes by -1.58 pp with interval [-4.10, +0.35] pp and main-task trade-off must be reported. |

## Claims that are supported

1. On the frozen public-specimen protocol, the continuous regret target lowers the defined routing regret relative to its hard-threshold ablation.
2. The stop-gradient construction has an exact local gradient property: `Lreg` does not backpropagate through gate evidence to the backbone or experts. This is a statement about the implemented computation graph, not a universal performance theorem.
3. The unweighted and coupled-gate ablations reveal data-dependent trade-offs. Neither may be presented as a categorical empirical proof that gap weighting or gradient isolation always improves overall classification.

## Claims that remain unsupported

- Any improvement in industrial sorting, recovery, concentrate grade, or V/Ti content.
- OOD/unknown-mineral performance; no independently scored unknown-mineral set exists yet.
- A universal Accuracy or Macro-F1 advantage of RSG-HRGV.
