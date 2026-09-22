# 专家训练中断与重跑记录

2026-09-21 检查发现，原会话 75214 已不存在；经只读进程查询，当前没有 python.exe 进程。原输出目录 `outputs/training/gate_supervision_expert_seed20260920` 的历史只到第 8 轮，没有最终 val_metrics.json 和 val_predictions.csv。不能将其记为正常完成或早停。中断原因未确认。

检查训练脚本发现 best_model.pt 仅包含模型及标签映射等信息，不包含优化器、调度器、随机数状态和数据加载器状态，因此不能严格从第 8 轮恢复。

处理：保留原目录及检查点，不覆盖、不拼接历史。在新目录 `outputs/training/gate_supervision_expert_seed20260920_retry1` 从 ImageNet 初始化重新执行原定训练，种子、清单、超参数、30 轮上限与 patience=8 均不变。该重跑不是增加一个独立随机种子，也不是根据指标选择额外配置。旧中断结果不得参与多种子平均。

只有正常结束、生成最终验证预测且核对清单隔离后，才能进入后续固定专家的 seen/unseen 门控对照。本轮不访问原测试图像。
