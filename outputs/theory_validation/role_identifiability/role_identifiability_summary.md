# Controlled Candidate-Set Role Identifiability Validation

This is a **controlled logical validation** of candidate-set identifiability. It uses the frozen species-to-role mapping and does not infer visual labels from raw images or model predictions.

## Configuration

- Manifest: `数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv`
- Dataset root: `D:\成信工科研\人工智能选矿\数据集\mindat_manual_positive_v1`
- Fixed seed: `20260801`
- Candidate sizes: `2, 3, 4`
- Species count: `17`
- Fixed split consumed unchanged: `{'test': 1284, 'train': 5961, 'val': 1284}`

## Results

| Scenario | Rows | Species unique rate | Role unique rate |
|---|---:|---:|---:|
| role_consistent | 48 | 0.00% | 100.00% |
| role_conflict | 51 | 0.00% | 0.00% |

## Interpretation

Role-consistent candidate sets are expected to preserve role identifiability even when species identity is ambiguous. Role-conflict candidate sets intentionally include another role and therefore do not guarantee a unique role.

This result validates the stated logical condition only. It is not a visual-label audit, a claim about image truth, or an industrial separation or recovery estimate.
