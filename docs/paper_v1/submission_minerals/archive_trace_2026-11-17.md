# Historical Configuration Trace and Baseline Consolidation

Date: 17 November 2026. Input documentation commit: `34cb85c55c41056dbc4c7649ce56a8a3a9609a5e`. This identifies the documentation input, not the source used for historical training. Scope: one bounded pass over existing protocol descriptions, commands, aggregate tables, environment metadata and available console logs. No training, model execution, metric recomputation, raw-image reading, prediction/weight inspection or official-report editing was performed.

## 1. Outcome

- Main Table 5 now contains seven completed baseline/method rows with archived Accuracy and Macro F1 means and sample standard deviations.
- All 18 formal environment-file hashes match the existing snapshot. This establishes identity with that snapshot, not historical run completeness or the identity of executed training source.
- No complete per-run optimizer, batch, transform or hardware record was recovered from the scoped material. Those fields remain unresolved; no current default or later-run value was substituted.
- An old plain-EfficientNet target-recall dispersion conflict is disclosed, not silently resolved. This column is not included in the new overview.
- The claim remains bounded: lower defined routing regret is supported; stronger network expressivity and stable overall classification superiority are not established.

## 2. Bounded Search and Evidence Strength

| Material inspected | What it supports | What it does not support |
|---|---|---|
| Existing 18-run snapshot and corresponding environment files | Identities, saved coefficients/dimensions/flags, seeds, CUDA and PyTorch string | Missing optimizer arguments, exact transforms, GPU model, dependency lock or original source commit |
| Fixed-split RSG note | Runner path, formal stage, paired configurations, dataset-path references and CUDA request | Resolved batch, learning rate, weight decay, epoch limit or patience |
| Photographer-holdout RSG note | Paired flags, ImageNet initialization, shared optimizer/early-stopping policy, formal seeds | Complete per-run command or exact optimization values |
| ResNet50 portability note | Backbone and IMAGENET1K_V2 initialization, paired flags, seeds, validation macro-F1 selection | Full optimizer/transform/hardware settings |
| Available ResNet50 stdout/stderr | Stdout contains skipped completed runs, epoch progress, early-stop messages and aggregate outputs; stderr is empty | A stopping epoch is not configured patience or maximum epochs; no resolved command/configuration dump was found |
| Baseline instructions and README examples | Example commands use 30 epochs and batch size 16; baseline summary describes AdamW | Proof that later RSG runs used these example settings |
| Baseline and component aggregate tables | Existing means and sample standard deviations; documented task/split/seeds | New paired uncertainty, harmonized historical compute, causal attribution or state-of-the-art status |

The console-log check searched for command/Namespace, batch, learning-rate, optimizer, weight-decay, patience, image-size, Python, CUDA and GPU fields. It returned no matching configuration lines. The search was restricted to the three principal RSG output roots and named documentation, not a forensic search of the computer. Missing records might exist elsewhere.

Instructions and commands copied into notes are not execution receipts. Outputs depending on source defaults cannot identify those defaults without source-to-run binding. Later experiments' explicit settings were not borrowed. Historical records were not modified.

## 3. Table 5 Row Binding

Values are transcribed from existing aggregate documents, not recomputed. Units are percent; dispersion is sample standard deviation across the documented three seeds.

| Row | Accuracy | Macro F1 | Source ID |
|---|---:|---:|---|
| ResNet50, weighted CE | 74.84 +/- 1.56 | 72.06 +/- 0.92 | B1 |
| EfficientNet-B0, weighted CE | 74.69 +/- 1.28 | 72.87 +/- 1.21 | B1 |
| EfficientNet-B0, weighted Focal | 75.21 +/- 0.74 | 73.22 +/- 0.83 | B2 |
| Role-aware precursor | 74.79 +/- 0.84 | 73.02 +/- 1.09 | B3 |
| Hierarchical precursor | 75.42 +/- 2.59 | 73.41 +/- 2.40 | B4 |
| Coupled residual HRGV | 76.48 +/- 0.23 | 74.59 +/- 0.34 | P1 |
| RSG-HRGV | 76.45 +/- 0.98 | 74.76 +/- 0.90 | P1 |

Split counts and seeds come from historical notes, not a new manifest read. B3's hierarchical extension uses role-to-mapped KL; HRGV/RSG uses mapped-to-direct KL. They must not be collapsed into one loss definition. H1 limits ordinary CNN comparison to summaries because per-image predictions were not retained. Table 5 is descriptive context; Tables 1-3 remain the matched routing evidence. Comparing standard deviations is not a significance test.

## 4. Unresolved Dispersion Conflict

B1 and the first result tables of B2/B3 report plain EfficientNet-B0 target recall as `71.98 +/- 0.83`. The binary proxy tables in B2/B3 report `71.98 +/- 0.46`. Under identical labels, predictions and seed aggregation, collapsing non-target classes leaves target recall unchanged. The two dispersion values require a provenance explanation; these tables do not provide one.

Action: preserve both records, disclose the conflict in Supplementary Note S6, and add neither dispersion to Table 5. No cause or numerical correction is asserted. The conflicting baseline is distinct from the main HRGV/RSG comparison. This audit does not independently reproduce all other baseline fields.

## 5. Source Identity Ledger

Paths are relative to repository root. Hashes were read from current files. M1 already itemizes the original per-run environment hashes; all 18 were checked against their current files.

| ID | Path | SHA-256 |
|---|---|---|
| B1 | `outputs/training/formal_experiment_summary_v1.md` | `051c02d5f68d2f406c51fddb31c25bf51c2d06d7dc87a30e61eebfcb44ca287d` |
| B2 | `docs/experiment_records/2026-07-30_focal_loss_three_seed_ablation.md` | `49c080d3f52b1c9b0040154343d0e5face653feca49ab3a27723915d6b226a6e` |
| B3 | `docs/experiment_records/2026-07-30_role_aware_hard_negative_learning.md` | `589dc30123502a224d516189a17ad43eca854c9ca4afed969ed21519e2511df3` |
| B4 | `outputs/business_metrics/hierarchical_efficientnet_b0/hierarchical_metrics_three_seed.md` | `9c755587b6dd6f54e53d74c19a67df668974fbe0e18f0092e38ebe42355746a5` |
| H1 | `docs/experiment_records/2026-08-22_hrgv_network.md` | `3f5ef163ecb7be55b694e525c040a875c0980dbc29b2295e9e53515a5ae331d0` |
| P1 | `docs/experiment_records/2026-08-22_rsg_hrgv.md` | `8585df55f1b3da189de18d74103b36a8ff648fa8d40f7f126963b6552c362f49` |
| P2 | `docs/experiment_records/2026-08-24_rsg_source_holdout.md` | `8277090eccc14060ea597d92bf85e8d311f3893792b62d4cc056ab9b59c5e217` |
| P3 | `docs/experiment_records/2026-08-31_rsg_resnet50_portability.md` | `93ef4577e0c1d8f10bf82fcb07cdc4a129613b200b57c6e6e8db26395d947386` |
| M1 | `docs/paper_v1/materials_2026-11-03/run_configuration_snapshot.json` | `4b20224c150f0d41e85f617e014ac4808f55d11d563da32e5afcff8f9b677c39` |
| L1 | `outputs/training/rsg_resnet50_portability/logs/resnet50_formal_stdout.log` | `848433569734cbb47739888c76a0e91820557afb4c414b6166c91dcf1af12c2a` |
| L2 | `outputs/training/rsg_resnet50_portability/logs/resnet50_formal_stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| I1 | `训练说明.md` | `b4f49e968955bfd8eb2763d1931169c8d4303472e5dd29e50f2a5146c33b7767` |

## 6. Closure Decision

The bounded archive-trace task is closed, with unresolved fields retained. This is not closure of reproducibility or of the broader network-innovation objective. Next is submission-source/reference cleanup and editable manuscript typesetting, not another model sweep. Author facts and original missing run records, if available, must come from their owners. Submission decisions must account for unresolved reproducibility risk.

No source/model/protocol file, original aggregate table, Chinese core draft or official technical report was edited. No independent review or new experimental validation is claimed. The files remain working manuscript material, not an accepted or submitted article.
