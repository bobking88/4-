# 论文与技术报告图注（v1.0）

## 图 1 模型总体性能对比

**中文图注：** ResNet50 与 EfficientNet-B0 在固定测试集上的总体性能对比。柱形表示 3 个随机种子独立训练结果的均值，误差线表示样本标准差。EfficientNet-B0 的平均 Macro F1 为 72.87%，略高于 ResNet50 的 72.06%。

**English legend:** Overall test-set comparison between ResNet50 and EfficientNet-B0. Bars show the mean of three independent seeds and error bars indicate sample standard deviation. EfficientNet-B0 achieved a slightly higher mean macro F1 score than ResNet50.

## 图 2 各类别召回率对比

**中文图注：** 两种模型在目标矿物、含钛干扰矿物、脉石/废石和金属光泽干扰四类上的测试集召回率。EfficientNet-B0 对金属光泽干扰矿物的平均召回率更高，而脉石/废石是两种模型最易识别的类别。柱形和误差线定义同图 1。

**English legend:** Class-wise test recall for the four-category task. EfficientNet-B0 improves recall for metallic hard negatives, while gangue is the most separable category for both models. Bars and error bars are defined as in Fig. 1.

## 图 3 最优 EfficientNet-B0 的混淆矩阵

**中文图注：** EfficientNet-B0（随机种子 20260728）在 1,284 张固定测试图像上的混淆矩阵。横轴为预测类别，纵轴为真实类别。含钛干扰矿物与脉石/废石、目标矿物之间的混淆是主要错误来源。

**English legend:** Confusion matrix of EfficientNet-B0 trained with seed 20260728 on the fixed test set of 1,284 images. Rows denote true classes and columns denote predicted classes. Confusion involving Ti-bearing negative minerals is the dominant error pattern.

## 图 4 主要错分方向

**中文图注：** EfficientNet-B0（随机种子 20260728）最常见的 6 类错分方向。“A -> B”表示真实类别 A 被预测为 B。含钛干扰矿物被误判为脉石/废石或目标矿物的次数最多。

**English legend:** Six most frequent error pairs for EfficientNet-B0 with seed 20260728. “A -> B” denotes images from true class A predicted as class B. Ti-bearing negative minerals are most often confused with gangue or target minerals.

## 图 5 Focal Loss 消融实验

**中文图注：** 在相同随机种子、数据划分和训练设置下，加权交叉熵与加权 Focal Loss（gamma=2.0）的比较。Focal Loss 提升了含钛干扰矿物的召回率，但降低了金属光泽干扰矿物召回率，最终 Macro F1 未超过交叉熵基线。

**English legend:** Controlled loss-function ablation under the same seed, split, and training configuration. Weighted focal loss (gamma=2.0) improves Ti-bearing-negative recall but reduces metallic-hard-negative recall and does not exceed the cross-entropy baseline in macro F1.

## 使用说明

- 所有图对应的可追溯源数据位于 `source_data/`。
- 可优先在论文中使用 SVG 或 PDF，在 Word 技术报告中使用 PNG。
- 图 1 和图 2 的误差线是 3 个随机种子的样本标准差；图 3 至图 5 是固定随机种子 20260728 的单次受控结果，应在图注中保持这一说明。
