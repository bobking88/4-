# RSG-HRGV photographer-holdout generalization plan (2026-08-24)

## Purpose

The fixed random split controls duplicate images but cannot fully rule out source-style dependence. This experiment evaluates whether the frozen regret-supervised routing mechanism remains interpretable when photographers are held out as whole groups. It is a source-held-out validation, not an independently collected industrial or OOD dataset.

## Frozen data protocol

- Manifest: `outputs/paper_experiments_v2/source_holdout/photographer_holdout_manifest.csv`
- Audit: `outputs/paper_experiments_v2/source_holdout/photographer_holdout_audit.json`
- Dataset root: project-root `数据集/mindat_manual_positive_v1` (the training command must pass its absolute path because this manifest is executed from the isolated worktree)
- Included rows: 5,639
- Split: train 3,947 / validation 846 / test 846
- Grouping: photographer-or-credit and duplicate group never cross splits
- Excluded before split: 2,890 rows without photographer metadata

## Frozen model comparison

The only comparison is the same coupled residual HRGV reference versus the frozen complete RSG-HRGV setting:

- HRGV reference: `--verifier-mode residual --couple-verifier-features`
- RSG complete: reference flags plus `--lambda-gate-regret 0.1 --gate-regret-temperature 0.2 --gate-gap-temperature 0.5 --detach-gate-features`

Both use the identical source-held-out manifest, pretrained EfficientNet-B0 initialization, optimizer schedule, and early stopping.

## Pilot gate

Run the reference and complete RSG configuration once at seed `20260728`. Continue to an independent three-seed confirmation set only if both runs complete and the RSG routing-regret direction is non-worsening. The pilot is used to check protocol execution and output integrity, not to establish an effect. The formal confirmation set is `20260727`, `20260729`, and `20260730`; it deliberately excludes the screening seed `20260728`.

## Claim boundary

If the three-seed comparison is completed, report source-held-out macro F1, target recall, both intrusion metrics, one-right-one-wrong routing selection, and routing regret with the same grouped paired Bootstrap protocol. Do not call this external industrial validation, independent OOD detection, or proof of production-line generalization.

## Completed confirmation (2026-08-25)

The frozen screening pilot at seed `20260728` was not included in the formal estimate. The independent confirmation set used seeds `20260727`, `20260729`, and `20260730`.

| Setting | Accuracy | Macro F1 | Target recall | Ti intrusion | Metallic intrusion | One-right gate selection | Mean routing regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Coupled residual HRGV | 71.20% ± 0.69% | 68.99% ± 1.17% | 71.25% ± 6.31% | 12.69% ± 1.83% | 11.88% ± 3.96% | 45.19% ± 8.34% | 11.77% ± 0.63% |
| Complete RSG-HRGV | 71.47% ± 1.14% | 69.25% ± 0.99% | 72.04% ± 2.64% | 11.40% ± 2.10% | 14.19% ± 3.02% | 53.69% ± 4.02% | 8.24% ± 0.80% |

The paired cluster Bootstrap interval crosses zero for Accuracy, Macro F1, target recall, and both intrusion measures. RSG minus HRGV mean routing regret is `-3.53` percentage points with a 95% interval of `[-4.75, -2.17]` percentage points and a directional resampling probability of `1.0000`. Therefore the evidence supports a lower defined routing regret under photographer holdout; it does not establish an overall classification improvement or industrial generalization.

Artifacts: `outputs/training/rsg_source_holdout/`, `outputs/business_metrics/rsg_hrgv/source_holdout/`, and `outputs/paper_figures_v2/fig_rsg_source_holdout.png`.
