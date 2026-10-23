# MSB trace 发布与首发错误保留：文档工作包记录

日期：2026-10-23；基线 `05727007209422c488a9ecdd8107018f883087fa`。

人类通过 `call_b95bed45520f49c4841956cf3c512ad7` 批准“仅编写修复规格与计划”。按明确的成对文档要求起草以下两份待审文件，不将起草许可当成书面规格审核或开发许可：

- [最小修复规格](../superpowers/specs/2026-10-23-msb-trace-publication-retention-design.md)
- [单工作包实施计划](../superpowers/plans/2026-10-23-msb-trace-publication-retention.md)

## 已完成的文档决策

1. 保留通用 `_write_json`，只为 trace 定义五阶段真实 helper 与可全模拟文件适配层；序列化前置，关闭至多一次，WRITE/CLOSE 双异常按序保留，失败后不替换、不删除原件、不重试。
2. controller 的循环与 finally 共享首发错误锁存；已有失败时最终 trace 不再访问文件。第一 OS 原因仍未知，不承诺这项修复解决潜在 Windows 共享冲突。
3. 将 row/body/stop/trace/诊断回执自身的错误分开；新 controller_failure 只能证明失败，collector 明确拒收。内层释放证据缺失时仍写 null，不根据 finally 推断完整清理。
4. 未来仅三个指定文件、十四个固定新方法、collector-red/green/final 三阶段，每阶段最多三次且失败计入；保留原 253 全名，267 只是待验证目标。所有原生接口及本轮 trace I/O 全模拟，不扩底层监督或数据边界。

自检时补清了首次失败对象属性、finish 不再抛错遮盖 body 的语义，以及失败诊断写前/写后各最多一次的既有 `_remaining` 检查；不新增时钟源、周期采样或预算。规格与计划的接口和测试表相互对应。

## 本轮实际核对

仅运行文本/JSON/哈希/Git 检查，不导入项目。静态自检确认十四测试名唯一、九个执行步骤均未勾选、无占位文本；未来 synthetic 和 scratch 根不存在。

331 条组合保护记录在 README 索引更新前全部匹配，46 项登记源码和 14 份失败原件保持不变。归档后核对记录见 `outputs/msb_trace_publication_retention_planning_v1/protection_after.json`，只允许 README 索引变化。六张预存脏 PNG 不修改、不提交；两处正式报告仅核对哈希，不打开正文。

## 未完成与下一关

当前为 `DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。未改代码、未运行合成测试/复现/进程或计时探查/原生验收、未读取真实数据、未训练或模型前向、未派代理、未改正式报告。不存在新的科学效果证据。

请先审核书面规格和计划；后续建议仅另批原生会话内代码与有限合成验收。独立复查、新原生协议/验收、真实 MSB-RRE 机制实验仍分别另批，不复用旧 v2 许可或输出目录。工程修复不作为论文理论创新效果。
