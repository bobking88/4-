# 单专家样本外门控诊断训练协议

运行前记录。目的为检验门控监督来源，不是新的确认性性能实验。

- 专家使用 EfficientNet-B0 ImageNet 预训练初始化，不加载任何既有矿物任务检查点或短程试跑权重。
- 专家训练 4,176 张，停止子集 893 张；选最佳停止集 Macro F1，最多 30 轮，patience=8。
- 种子 20260920，batch_size=16，AdamW 学习率 1e-4、权重衰减 1e-4；沿用 HRGV 残差验证器与现有默认辅助损失权重，gate_regret 权重为 0。
- `--validation-only`；专家清单没有 test 行，也完全不包含 892 张 gate_unseen 图像。原始验证、测试清单不用于本次专家训练或模型选择。
- 门控 seen 对照从 expert_fit 内按重复组确定性选取，与 gate_unseen 逐矿物和角色图像数完全一致，各 892 张。
- 只有一个专家种子，不能据此作独立多种子或工业泛化结论。之后门控开发仍应记录选择数据的重复使用。

数据：`outputs/training/gate_supervision_manifests_v1`，清单哈希见其中 audit.json。正式探索输出：`outputs/training/gate_supervision_expert_seed20260920`。独立的 smoke_v1 目录仅验证执行路径，随机初始化、两批次结果不得用于性能对比。

命令参数见训练输出 environment.json；完整调用为项目训练 Python 执行 `scripts/train_hrgv_mineral_classifier.py --manifest outputs/training/gate_supervision_manifests_v1/expert.csv --dataset-root D:/成信工科研/人工智能选矿/数据集/mindat_manual_positive_v1 --output-dir outputs/training/gate_supervision_expert_seed20260920 --validation-only --device cuda --seed 20260920 --epochs 30 --patience 8 --batch-size 16 --num-workers 0`。
