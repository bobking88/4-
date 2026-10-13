# IC-RFA-B v1 最终代码审核与一次修复

## 审核身份与边界

独立只读审核范围：`548d2601d60fe23e11589030387c7a1c749a1f85`至`f933153387b39a4ff414e02ad2281249dc41d891`，获准实施计划Tasks1--7。审核者未改文件、未读真实特征或图片，未重跑一次计时。父实施者对发现逐项复现后，进行了单次TDD修复；未再次派发审核。

初次判定：无Critical，四项Important须修复；唯一Minor为Matplotlib生成SVG的行尾空白，不影响外观，保留。

## 重要发现与验证

| 问题 | RED证据 | 修复 | 验收测试 |
| --- | --- | --- | --- |
| 完成的worker packet可能再次优化后才发现结果已存在 | 预先写入result.pt后Mock fitter仍被调用；中断任务可再次进入 | 计算前检查结果；原子task_started.json消费锁，失败仍保留锁；同型保护也加到一次计时worker | test_completed_packet_is_rejected_before_fitting；test_interrupted_packet_claim_remains_consumed；test_consumed_timing_worker_never_restarts |
| 重放验证未检查最终执行失败 | RUNNING/FAILED/POWER_RELEASE_FAILED/TIMED_OUT状态未被拒绝 | 公共交付验证先检查成功终态、累计预算、所有job监督及电源释放；运行中的分析单独标记provisional，须活跃监督与已消费分析任务锁 | test_failed_terminal_delivery_never_opens_artifacts；test_provisional_analysis_requires_live_supervision |
| 数学依赖未完整登记 | 两个实际执行依赖不在SOURCE_FILES | 登记本地导入闭包34个源文件，包含温度/验证与指标实现；各文件hash和GitHub身份均须匹配 | test_executed_dependency_hash_changes_fail_closed |
| 诊断可空缺或被伪造后仍通过 | 空完整梯度/布局、删除或污染Jacobian、反转行顺序、伪造移除分支漂移均未被拒绝 | 检查完整参数布局、形状、dtype/设备、有限性、行/角色身份；同状态前向重算分支移除和objective，缺少必需诊断即失败 | test_diagnostic_schema_and_ablation_tampering_rejected |

一次RED运行21项测试出现11个失败，均覆盖上述缺口及子案例；一次修复后21项全部通过（4.518秒）。最终全部新测试60项通过（12.742秒），固定范围旧纯合成回归33项通过（1.727秒），合计93项。

此处诊断schema与前向重放验证不冒充再次计算全部参数梯度/Jacobian；保持仅推理重放，不优化。数据梯度数学正确性仍依赖固定方向有限差分、模型单测、原训练日志及未来独立数学核查。

## 保留的限制与判断

- 科学新颖性、独立定理证明与真实矿物效果不在代码审核判定内：这些是后续论文验证事项，而非代码通过后自动成立。
- 审核者不做真实数据、GitHub或接电实验：本次原生合成安全证据和GitHub核验由父实施者执行，不能称为审核者独立复测。
- 审核者未重新打开历史保护文件：父实施者已分别完成1817项保护文件、2028项旧v2证据及两处正式报告的哈希核验。
- 图的生成SVG标准行尾空白作为Minor保留；手写实现差异检查通过。
- 原生一次计时只在f933153运行。此次验证加固不改变网络/损失/网格，计时未重新执行；后续真实协议必须清楚说明预算估计的版本与不确定性。
- Task8、stop、原图和正式报告修改仍未获准。代码审核与合成成功没有越过真实实验授权门槛。

当前结论：四项Important已有RED→GREEN覆盖，93项范围内回归通过，可交付原生代码与合成验收；不是独立科学效果确认，也未合并主分支。
