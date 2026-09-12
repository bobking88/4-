# RSG 后悔监督梯度隔离条件核验

日期：2026-09-07。此记录是实现与理论条件检查，不是训练性能实验。

## 条件与推导

设专家参数为 theta_e，门控参数为 phi，输入 z 包含共享特征与专家概率统计。
仅停止软目标及权重的梯度时，损失仍存在路径：

`L_reg -> g_phi(z) -> z -> theta_e`。

因此目标和权重的 detach 不能单独推出专家梯度为零。
当 z 也停止梯度、phi 与 theta_e 不共享时，上述路径被切断。
该结论只针对 L_reg 分支，不涉及其他训练损失的梯度。

## 自动微分对照

测试使用实际 HierarchicalRiskGatedVerificationNet、未预训练权重和固定种子 903。
在 eval 模式下，对相同模型权重、相同四张合成输入分别执行两次独立前向与反向：

| 完整门控输入停止梯度 | 软目标和权重停止梯度 | 主干及两个分类头 | 门控 |
|---|---|---|---|
| 否 | 是 | 每个模块均存在非零梯度 | 非零梯度 |
| 是 | 是 | 每个模块所有参数均无该分支梯度 | 非零梯度 |

测试不进行优化器更新，输入为合成张量，不读取训练、验证或测试集。
具体数值非零是此固定实例的反例证据；不是任意输入、任意参数下都非零的定理。

## 复现与结果

```powershell
python -m unittest tests.test_hrgv_network.HRGVModelTests.test_detached_targets_alone_do_not_isolate_regret_gradient tests.test_hrgv_network.HRGVModelTests.test_regret_gate_loss_is_isolated_from_backbone_and_experts_when_detached
```

实测：2 项测试通过，耗时 15.505 秒。
论文主稿 5.8 节和正式报告命题 8 已补全对应假设。
该核验支持理论与计算图一致性，不构成新的泛化定理、网络首创性证明或分类性能提升证据。
