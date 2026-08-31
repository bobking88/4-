# RSG-HRGV ResNet50 跨骨干可迁移性确认（2026-08-31）

## 目的

RSG-HRGV 的主实验使用 EfficientNet-B0。为检验后悔监督门控是否仅依赖于某一视觉主干，本确认实验将共享主干替换为 ImageNet 预训练 ResNet50，其余层级映射、双专家、残差验证器、训练增强、优化器、早停及固定四角色划分均保持不变。

本实验不用于比较 EfficientNet-B0 与 ResNet50 的优劣，也不将 ResNet50 作为新的主方法。唯一的机制问题是：在第二种结构不同的 CNN 主干上，RSG 相对严格匹配的 HRGV 参考模型是否仍降低预先定义的平均路由后悔。

## 冻结设计

- 主干：`resnet50`，使用 `ResNet50_Weights.IMAGENET1K_V2`；
- 数据：`数据集/dataset_final_v1/dataset_split_manifest_v1_0.csv`，图像根目录 `数据集/mindat_manual_positive_v1`；
- 参考模型：HRGV，残差验证器、验证器特征耦合、`--disable-gate-regret`；
- RSG 模型：参考模型设置加 `--lambda-gate-regret 0.1 --gate-regret-temperature 0.2 --gate-gap-temperature 0.5 --detach-gate-features`；
- 种子：20260727、20260728、20260729；
- 选择准则：验证集 Macro F1 早停；仅保留对应最佳检查点进行测试；
- 输出根目录：`outputs/training/rsg_resnet50_portability/resnet50/`。

## 预先规定的分析

以每个种子的测试集逐图诊断计算平均路由后悔：

\[
\bar r=\frac{1}{N}\sum_{i=1}^{N}
\left[-\log p_f(y_i\mid x_i)+\log\max\{p_d(y_i\mid x_i),p_m(y_i\mid x_i)\}\right].
\]

主要比较为 RSG 减 HRGV 的三种子均值与按 `split_group_id` 聚类的成对 Bootstrap 区间。只有当区间整体小于零时，才将结果描述为“跨骨干下支持较低的预定义平均路由后悔”。Accuracy、Macro F1、目标类召回、两类误入目标比例、Brier 和 ECE 均作为边界性指标同步报告；若其区间跨零，则不主张总体分类性能提升。

在结果产生前，不更新技术报告的任何数值结论。若出现不稳定或不支持主要指标的结果，应如实记录为 RSG 的主干依赖边界。

## 可复现命令

```powershell
& .\.venv-training\Scripts\python.exe .\scripts\run_rsg_hrgv_experiments.py `
  --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv `
  --dataset-root .\数据集\mindat_manual_positive_v1 `
  --output-root .\outputs\training\rsg_resnet50_portability `
  --python-executable .\.venv-training\Scripts\python.exe `
  --device cuda --torch-home .\.torch-cache --stage formal `
  --backbone resnet50 --config hrgv_reference --config rsg_complete --execute
```
