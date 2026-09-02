# RSG-HRGV 双专家互补性诊断（2026-09-02）

## 目的

RSG-HRGV 的门控仅在两位专家给出不同且至少有一位提供更好证据时才有实际意义。本记录从已经冻结的三随机种子正式汇总中提取双专家、融合与逐图 oracle 的诊断量，用于回答：

1. 直接角色专家与种类映射角色专家是否完全冗余；
2. 存在多少“恰有一位专家正确”的可路由样本；
3. 完整 RSG 门控在该条件子集上的专家选择是否高于随机二选一的 50% 参考。

## 固定输入

| 协议 | 汇总文件 |
|---|---|
| 固定测试 | `outputs/business_metrics/rsg_hrgv/formal/rsg_three_seed_summary.json` |
| 摄影者留出 | `outputs/business_metrics/rsg_hrgv/source_holdout/rsg_three_seed_summary.json` |
| ResNet50 跨主干 | `outputs/business_metrics/rsg_hrgv/resnet50_portability/rsg_three_seed_summary.json` |

只读取每个文件的 `rsg_complete` 三随机种子均值；不重新训练、不重新划分数据，也不改动既有统计比较。

## 诊断结果

| 协议 | 直接角色 Accuracy | 种类映射 Accuracy | 融合 Accuracy | 真值 oracle Accuracy | 专家预测分歧 | 一对一错 | 条件门控选对率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 固定测试 | 75.96% | 76.64% | 76.45% | 78.71% | 5.61% | 4.83% | 58.01% |
| 摄影者留出 | 71.16% | 71.55% | 71.47% | 73.76% | 6.11% | 4.81% | 53.69% |
| ResNet50 跨主干 | 76.30% | 76.43% | 76.64% | 78.35% | 4.88% | 3.97% | 51.98% |

图文件：`outputs/paper_figures_v3/fig_rsg_expert_complementarity.png`，并同时导出 SVG、PDF、TIFF 和源数据说明 JSON。

## 解释与边界

- 两位专家在每个协议中均存在非零的预测分歧和一对一错子集，因而门控不是在完全相同的两条概率上进行无意义的权重学习。
- 一对一错子集只占约 4% 至 5%，是门控能够直接修复专家分类错误的有限空间；其余样本中两位专家要么同对、要么同错。
- 完整 RSG 的条件门控选对率在三个协议中略高于 50%，与平均路由后悔降低的正式成对 Bootstrap 结果共同构成机制证据；该比例本身不是新的独立显著性检验。
- oracle 使用每张图像的真实角色标签选择更优专家，部署时不可获得，只能表示当前双专家的诊断上限。融合 Accuracy 未达到 oracle 并不等同于模型失效，也不能据此推出更大模型、更多专家或工业现场收益。
- 本诊断不构成总体分类优越性、开集识别、工业分选、品位预测、回收率或元素含量结论。

## 复现

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' scripts\generate_rsg_expert_complementarity_figure.py `
  --fixed-summary-json outputs\business_metrics\rsg_hrgv\formal\rsg_three_seed_summary.json `
  --holdout-summary-json outputs\business_metrics\rsg_hrgv\source_holdout\rsg_three_seed_summary.json `
  --portability-summary-json outputs\business_metrics\rsg_hrgv\resnet50_portability\rsg_three_seed_summary.json `
  --prefix outputs\paper_figures_v3\fig_rsg_expert_complementarity
```
