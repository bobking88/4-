# 论文与技术报告图注（v1.1）

## 图 1 模型总体性能对比

**中文图注：** ResNet50、EfficientNet-B0 与角色感知 EfficientNet-B0 在固定测试集上的总体性能对比。柱形表示 3 个随机种子独立训练结果的均值，误差线表示样本标准差。角色感知模型的平均 Macro F1 与普通 EfficientNet-B0 接近，不将两者差异解释为显著优劣。

**English legend:** Overall fixed-test-set comparison among ResNet50, EfficientNet-B0, and role-aware EfficientNet-B0. Bars show the mean of three independent seeds and error bars indicate sample standard deviation. The role-aware model is close to the standard EfficientNet-B0 in macro F1 and is not claimed to be significantly superior.

## 图 2 各类别召回率对比

**中文图注：** 三种模型在目标矿物、含钛干扰矿物、脉石/废石和金属光泽干扰矿物四类上的测试集召回率。角色感知模型提高了含钛干扰与金属光泽干扰的召回，但目标矿物召回率下降，体现风险取舍。

**English legend:** Class-wise test recall for the four-category task. The role-aware model improves recall for Ti-bearing and metallic hard-negative minerals while reducing target-mineral recall, revealing a risk trade-off.

## 图 3 EfficientNet-B0 的混淆矩阵

**中文图注：** EfficientNet-B0（随机种子 20260728）在 1,284 张固定测试图像上的混淆矩阵。横轴为预测类别，纵轴为真实类别；含钛干扰矿物与脉石/废石、目标矿物之间的混淆是主要错误来源。

**English legend:** Confusion matrix of EfficientNet-B0 trained with seed 20260728 on 1,284 fixed test images. Rows denote true classes and columns denote predicted classes. Confusion involving Ti-bearing negative minerals is a dominant error pattern.

## 图 4 主要错分方向

**中文图注：** EfficientNet-B0（随机种子 20260728）最常见的 6 类错分方向。“A -> B”表示真实类别 A 被预测为 B。含钛干扰矿物被误判为脉石/废石或目标矿物的次数最多。

**English legend:** Six most frequent error pairs for EfficientNet-B0 with seed 20260728. “A -> B” denotes images from true class A predicted as class B.

## 图 5 Focal Loss 三随机种子消融

**中文图注：** 在相同数据划分和训练设置下，加权交叉熵与加权 Focal Loss（gamma=2.0）的三随机种子比较。Focal Loss 提高了含钛干扰矿物的平均召回率，但降低了金属光泽干扰矿物召回率，且未在总体与目标代理风险指标上形成一致改善。

**English legend:** Three-seed loss-function ablation under the same split and training configuration. Weighted focal loss improves mean Ti-bearing-negative recall but reduces metallic-hard-negative recall and does not provide a consistent improvement across overall and target-proxy risk metrics.

## 图 6 基线目标代理指标

**中文图注：** 普通 EfficientNet-B0 在目标代理/非目标代理折叠任务上的三随机种子指标。该类指标用于描述公开标本图像条件下的预选代理风险，不代表实际回收率、精矿品位或抛废率。

**English legend:** Target-proxy metrics for standard EfficientNet-B0 across three seeds. These are proxy decision metrics on public specimen images and do not represent recovery, concentrate grade, or rejection rate.

## 图 7 三种策略的目标代理风险比较

**中文图注：** 加权交叉熵、加权 Focal Loss 和角色感知困难负样本学习的目标代理 F1、目标漏选率以及两类干扰误入目标比例。角色感知方法降低了含钛干扰与金属光泽干扰误入目标的比例，但目标漏选率上升，体现为保守的风险取舍原型。

**English legend:** Target-proxy F1, target miss rate, and intrusion rates of two hard-negative roles for weighted cross-entropy, weighted focal loss, and role-aware hard-negative learning. The role-aware method lowers high-risk intrusion rates at the cost of a higher target miss rate, representing a conservative risk-trade-off prototype.

## 使用说明

- 所有图对应的可追溯源数据位于 `source_data/`。
- 论文优先使用 SVG 或 PDF，Word 技术报告使用 PNG。
- 图 1、图 2、图 5、图 6 和图 7 的误差线均为 3 个随机种子的样本标准差；图 3、图 4 是固定随机种子 20260728 的单次错误分析结果。
