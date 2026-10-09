# LC-RFA-B v1 有限真实开发实验呈交

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan（消费已批准规格，不重新设计网络）
- Origin Date: 2026-10-09
- Verification Status: UNVERIFIED（真实实验未开展；准备状态与hash已核验）
- Version Label: lc_role_development_submission_v1

## 待批准的唯一范围

请求批准既定Fold0的340个fit分组及340个已暴露stop分组，开展LC-RFA-B v1和固定强对照：**最多422次拟合、另4次温度拟合、累计6小时硬上限、单worker600秒、锁定后一次stop评价**。CPU/float64/单线程。预计完整排程约2.17小时，但不保证真实求解耗时。

不包括训练重放、其他外折、新图片、独立确认、正式报告修改、失败重试或自动扩网格。未收到对应真实训练授权前，不生成 `APPROVED_FOR_DEVELOPMENT` 协议，不调用fit/evaluate。

机器可读呈交文件：[lc_role_adaptation_v1_submission.json](../experiment_protocols/lc_role_adaptation_v1_submission.json)。其中请求授权不是已取得授权。

当前完整运行草案：[draft_protocol.json](../../outputs/training/lc_role_adaptation_v1/preflight/draft_protocol.json)，状态 `DRAFT_NOT_AUTHORIZED`；字节SHA-256为 `c7a79b5d6f4117109ca13af8711683930f7948cd6f45ff9bd5e57d11280b4ec5`，规范化语义hash为 `aedd39b332bc62c59d7129e2830f4a1a501ca8e145d675f73fdb3328d03e4b21`。

运行源码对应提交 `8f8bb1608a454df357050a783de73bd15b191332`；规格/计划/正式报告、冻结骨干/cache、原始三源表和软件环境均已绑定。结构图、合成验收及preflight/benchmark证据hash另外收入呈交文件。2026-10-09只读验收确认33项来源hash仍匹配，草案拒绝fit/evaluate/verify，未执行真实优化。

## 要验证的网络创新

不是“更换骨干”“首次使用低秩适配/多任务/有界激活”。本轮候选贡献是：**将矿物角色监督绑定到三条条件视觉适配分支，再以目标—干扰竞争概率路由，配合有限纠错容量与扰动界，检验是否优于同预算普通学习策略。**

| 假设 | 对照及证据 | 被否定或不足时的结论 |
| --- | --- | --- |
| H_ROUTE：角色路由有用 | R1与S1，尤其R1预算匹配S1；同3024参数、同辅助视图/损失 | 不能把普通适配/辅助监督收益归于角色路由 |
| H_SPECIALIZATION：分支不是退化副本 | R1/R0、S1/S0；前五步数据梯度、每对NLL、delta范数/cosine、同权重逐支禁用 | 不称形成了有效角色专门化；诊断仅描述，不能选stop超参数 |
| H_BOUNDED_BENEFIT：限幅有净作用 | R1/U1、理论A/B/C、纠错新增/损失与干扰侵入 | U1在NLL、正确目标、Ti/M假目标上Pareto支配时，不支持限幅净效益 |

固定预算S1仍可在fit中选择不同lambda/更新数；因此这是公平搜索下的**学习流程比较**，不能当成完全固定优化条件的因果路由效应。禁用分支也只是同权重诊断，不替代独立重训练对照。

令开发样本上的四类平均NLL为

$$
\widehat L(A)=\frac1n\sum_{i=1}^n[-\log q^A_{i,y_i}],\qquad
\Delta_{A\to R1}=\widehat L(A)-\widehat L(R1).
$$

对每个已登记种子，要求相对T0及最强合格控制均有 `Delta>=0.005`；相对T0净正确目标新增至少1个，Ti/M假目标各最多增加1个；R1的NLL还须低于预算匹配S1。缺组、可靠凸证书缺失、400步优化截断或理论违规均不得支持晋级。筛查口径与现有 `screen_development` 完全一致，不增设数据驱动阈值。

三个种子共享同一stop样本，不是三个独立临床/现场重复样本；这些门槛是开发筛查，不是显著性检验。对已重复使用的stop，即使全部通过，也只报告开发收益，不能写成独立泛化证明。

## 理论、机制与效果分开交付

1. 理论A/B/C：记录最大数值残差、违规行、证书大小、保护/宽松比例和目标不可纠正包络。命题B只约束**同权重主路径**，不扩展到无缩放辅助视图；预测不变不等于预测正确。
2. 机制：全部分支诊断、零范数未定义数量、数据梯度与总梯度分别保留。对照允许推翻角色机制解释。
3. 效果：所有最终组/种子的完整NLL、Accuracy/Macro F1、目标P/R/F1、各干扰假目标分母、Brier/ECE及新增/损失目标。不得只保留最好种子。
4. 复验：状态/来源hash和严格推理重放。默认verify不训练；优化器历史真实性须另行批准完整训练重放。

## 执行及停止规则

批准后才将授权原话与边界写入正式运行协议 `docs/experiment_protocols/lc_role_adaptation_v1.json`，提交并远端核验后执行。预处理及所有超参数/检查点选择只在fit中完成；369个CV轨迹，23个基础最终组拟合、27个匹配CV模板及3个匹配最终模板组成422上限。同协议的匹配CV和完全相同最终配置复用，不另算新搜索机会。

fit原子生成 `selection.lock.json` 并绑定全部状态hash后，才允许一次stop评价。错误/数值范围失败/超时立即终止当前批次，不执行下一阶段、不重试。400被选中保留截断标签；不能将优化不足悄悄改成“网络无效”或“网络有效”。

输出根固定为 `outputs/training/lc_role_adaptation_v1/development_fold_0/`，尚未创建。阶段日志、状态、全部逐图CSV/JSON和失败信息同步GitHub；原图/cache/权重仅本地与hash归档。正式报告不变。

## 授权后下一步

收到本范围的明确批准后，可以直接执行登记、fit、一次evaluate及默认推理verify，无需逐阶段再次确认。若出现预登记无法核验、输入/环境改变、失败或timeout，则停止并告知；不把既往ABMP授权或自动目标续跑消息当成本协议的真实训练批准。
