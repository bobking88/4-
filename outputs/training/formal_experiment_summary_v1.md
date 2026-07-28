# 钒钛矿相关矿物图像四分类正式基线实验汇总（v1.0）

## 1. 实验目的

在固定的数据版本和固定的训练/验证/测试划分下，对 ResNet50 与 EfficientNet-B0 进行四分类基线对比。该实验用于验证：基于公开矿物标本图像构建的钒钛矿相关矿物数据集，是否能支持目标矿物、含钛干扰矿物、脉石/废石和金属光泽干扰矿物的初步视觉区分。

## 2. 数据与实验设置

| 项目 | 设置 |
|---|---|
| 数据版本 | `dataset_final_v1` |
| 图像总数 | 8,529 张 |
| 类别 | `target_mineral`、`ti_bearing_negative`、`gangue_negative`、`metallic_hard_negative` |
| 训练/验证/测试 | 5,961 / 1,284 / 1,284 张 |
| 数据划分 | 按 Mindat 图片编号分组，训练、验证和测试集之间无图片编号泄漏 |
| 数据质控 | 人工分层复核 823 张；排除 32 张；2 张 `needs_expert` 样本未进入训练 |
| 模型 | ResNet50、EfficientNet-B0，均使用 ImageNet 预训练权重 |
| 训练策略 | AdamW、类别逆频率权重、早停；每种模型使用 3 个随机种子独立训练 |
| 评价指标 | Accuracy、Macro Precision、Macro Recall、Macro F1、各类别 Recall |

最终训练清单：`数据集/dataset_final_v1/dataset_split_manifest_v1_0.csv`。

## 3. 总体结果

表中数值为 3 次独立训练的均值 ± 样本标准差，单位为 %。

| 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| ResNet50 | 74.84 ± 1.56 | 73.32 ± 1.87 | 71.63 ± 0.78 | 72.06 ± 0.92 |
| EfficientNet-B0 | 74.69 ± 1.28 | 72.71 ± 1.28 | 73.17 ± 1.12 | **72.87 ± 1.21** |

EfficientNet-B0 的平均 Macro F1 比 ResNet50 高 0.67 个百分点，且平均 Accuracy 基本相当。因此，后续误差分析和改进实验可将 EfficientNet-B0 作为主基线，ResNet50 作为对照基线。

## 4. 各类别召回率

| 模型 | 目标矿物 | 含钛干扰矿物 | 脉石/废石 | 金属光泽干扰 |
|---|---:|---:|---:|---:|
| ResNet50 | 71.58 ± 5.18 | 77.53 ± 8.32 | **83.61 ± 0.82** | 53.79 ± 4.26 |
| EfficientNet-B0 | **71.98 ± 0.83** | 74.34 ± 2.15 | 81.97 ± 2.85 | **64.39 ± 3.47** |

结论：

- 脉石/废石最容易区分，说明其颜色、透明度和晶体形貌与目标矿物存在较明显视觉差异。
- 金属光泽干扰是最难类别。黄铁矿、赤铁矿、针铁矿、黄铜矿等在暗色、反光和块状形态上与目标矿物存在相似性，尤其容易形成混淆。
- EfficientNet-B0 对金属光泽干扰的平均召回率高于 ResNet50 10.60 个百分点，是其优于 ResNet50 的主要原因。
- 目标矿物召回率约为 72%，说明该公开标本图像数据集能够支持“钒钛矿相关成分矿物”的初步识别；但它不能直接等同于工业传送带或现场矿石分选性能。

## 5. 单次试验记录

| 模型 | 随机种子 | Accuracy | Macro F1 | 最佳验证 Macro F1 |
|---|---:|---:|---:|---:|
| ResNet50 | 20260727 | 75.93 | 72.95 | 74.99 |
| ResNet50 | 20260728 | 73.05 | 71.12 | 73.23 |
| ResNet50 | 20260729 | 75.55 | 72.12 | 74.88 |
| EfficientNet-B0 | 20260727 | 75.47 | 73.32 | 73.75 |
| EfficientNet-B0 | 20260728 | 75.39 | 73.80 | 73.25 |
| EfficientNet-B0 | 20260729 | 73.21 | 71.50 | 72.80 |

## 6. 可写入论文和结题报告的表述

在经人工分层复核后的 8,529 张公开矿物标本图像上，本文构建了面向钒钛矿相关矿物识别的四分类任务。固定训练集、验证集和测试集划分，并采用按 Mindat 图片编号分组的策略避免相同来源图像跨集合泄漏。结果表明，EfficientNet-B0 在三次独立训练中的平均 Accuracy 为 74.69%，平均 Macro F1 为 72.87%，略优于 ResNet50 的 72.06%。模型对脉石/废石具有较高的识别能力，而对金属光泽干扰矿物的区分仍是主要挑战。该结果证明了公开矿物图像构建钒钛矿相关成分矿物视觉识别基线的可行性，并为后续引入真实矿石颗粒图像、开展领域适配和工业分选验证提供了基础。

## 7. 输出文件位置

- ResNet50 三次结果：`outputs/training/formal_resnet50_seed20260727`、`formal_resnet50_seed20260728`、`formal_resnet50_seed20260729`
- EfficientNet-B0 三次结果：`outputs/training/formal_efficientnet_b0_seed20260727`、`formal_efficientnet_b0_seed20260728`、`formal_efficientnet_b0_seed20260729`
- 每次试验均包含 `best_model.pt`、`metrics_history.csv`、`test_metrics.json` 和 `confusion_matrix.csv`。
