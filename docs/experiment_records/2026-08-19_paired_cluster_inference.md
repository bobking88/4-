# 固定测试集成对簇 Bootstrap 与 McNemar 推断

## 目的

比较普通 EfficientNet-B0 与矿物种类—选矿角色分层模型时，不能只报告三个随机种子的均值。两种模型在同一固定测试集上逐图配对，且同一原图、重复图或近重复图可能属于同一个 `split_group_id`。本实验同时考虑训练随机种子差异和测试图像簇相关性，估计模型效应区间，并检查逐图正确/错误变化是否在三个种子上保持一致。

## 输入

- 基线：`outputs/training/formal_role_aware_efficientnet_b0_seed*/test_predictions.csv`
- 分层模型：`outputs/training/formal_hierarchical_efficientnet_b0_seed*/test_predictions.csv`
- 随机种子：`20260727`、`20260728`、`20260729`
- 每个种子固定测试图像：1,284 张
- 配对键：`image_id`
- 簇键：`split_group_id`

脚本首先验证两种模型的 `image_id`、`true_label` 和 `split_group_id` 完全一致。若任一字段不一致，分析直接终止。

## 方法

对第 `b` 次 Bootstrap 重复：

1. 从三个训练随机种子中有放回抽取三个种子；
2. 在每个被抽中的种子内，按真实四类角色分层；
3. 在每个角色内对 `split_group_id` 簇有放回抽样，并保留簇内全部图像；
4. 计算分层模型减去基线模型的指标差，再对抽中的种子取平均。

正式实验使用 10,000 次重复和 95% 百分位区间。Accuracy、Macro F1 和目标召回越高越好；目标漏选率以及两类干扰误入目标的比例越低越好。另对每个种子的逐图正确/错误变化执行双侧精确 McNemar 检验，并对三个 p 值作 Holm 校正。

## 结果

| 指标 | 基线均值 | 分层模型均值 | 差值/百分点 | 95%簇区间/百分点 | 解释 |
|---|---:|---:|---:|---:|---|
| Accuracy | 74.79% | 75.42% | +0.62 | [-2.39, 3.22] | 区间跨 0 |
| Macro F1 | 73.02% | 73.41% | +0.39 | [-2.46, 3.04] | 区间跨 0 |
| 目标召回 | 68.39% | 77.56% | +9.16 | [5.18, 13.28] | 稳定改善 |
| 目标漏选率 | 31.61% | 22.44% | -9.16 | [-13.28, -5.18] | 稳定下降 |
| 含钛干扰误入目标 | 7.74% | 10.93% | +3.19 | [0.07, 6.79] | 不利上升 |
| 金属光泽干扰误入目标 | 9.66% | 14.39% | +4.73 | [1.33, 8.14] | 不利上升 |

McNemar 检验结果：

| 种子 | 仅基线正确 | 仅分层模型正确 | 精确 p 值 | Holm 校正 p 值 |
|---|---:|---:|---:|---:|
| 20260727 | 135 | 101 | 0.0315 | 0.0945 |
| 20260728 | 83 | 110 | 0.0610 | 0.1101 |
| 20260729 | 107 | 138 | 0.0551 | 0.1101 |

## 结论

分层模型稳定提高目标召回并降低目标漏选，但同时稳定增加两类困难负样本误入目标的比例。Accuracy 与 Macro F1 的区间均跨 0，三个种子的 McNemar 校正结果也均大于 0.05。因此，现有证据支持“风险重分配”而不是“总体性能全面提高”：辅助层级任务使模型更倾向于保留目标类，但这一倾向必须由概率校准、选择性拒识或类别相关风险阈值约束。

## 解释边界与常见误判检查

- 不把“目标召回提高”偷换成“工业回收率提高”；当前没有品位、质量流量或生产线数据。
- 不把三个随机种子当成充分的大样本统计；种子层不确定性仍较粗糙。
- 不把 Bootstrap 区间外推到未观测矿区、摄影者、网站或真实碎矿域。
- 不把未校正的单种子 McNemar p 值作为稳定优势证据。
- 不根据同一测试集反复选择最终模型或阈值；风险阈值仍按独立验证子集确定。
- 目标漏选率与目标召回互为补数，只作为同一风险结果的两种业务表达，不视为两个独立发现。

## 复现

```powershell
.\.venv-training\Scripts\python.exe .\scripts\analyze_paired_cluster_statistics.py `
  --training-root .\outputs\training `
  --output-dir .\outputs\paper_experiments_v3\statistical_inference `
  --figure-prefix .\outputs\paper_figures_v3\fig_paired_cluster_effects `
  --bootstrap-replicates 10000 `
  --rng-seed 20260819
```

主要输出：

- `paired_cluster_bootstrap_summary.json`
- `paired_cluster_bootstrap_summary.csv`
- `bootstrap_distribution.csv`
- `mcnemar_exact.csv`
- `paired_seed_metrics.csv`
- `outputs/paper_figures_v3/fig_paired_cluster_effects.{png,svg,pdf,tiff}`
