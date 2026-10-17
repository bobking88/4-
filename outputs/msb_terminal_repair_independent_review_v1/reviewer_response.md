**发现**
以下行号均对应冻结 HEAD `a40567ab2f99fde695194e15fe726a6c3b9dd863`。三项均为静态控制流发现，未执行反例，未复现 Windows 故障。

1. **P2：DONE 后的退出接收分支遗漏冻结租约复查。**  
   位置：[scripts/msb_execution_safety.py:522](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/msb_execution_safety.py:522)。  
   静态触发：在租约 `L-1` 接受 DONE，随后进程于 `L` 才完成退出。退出后的 `_drain_control` 发现 `control.done` 即返回，只检查累计预算；`require_complete` 不检查时间，随后仍可设置 COMPLETE。因此，即使累计期限尚未到达，也可能接受租约等号时才退出的 worker，并继续下一 job。建议在成功接收退出的边界重新检查冻结租约，补充 DONE 与退出跨越租约等号的独立事件反例。

2. **P2：守卫验收器允许跳过测试后仍发布 PASSED。**  
   位置：[tests/run_msb_terminal_repair_acceptance.py:118](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/tests/run_msb_terminal_repair_acceptance.py:118)。  
   `wasSuccessful()` 不将 skipped 视为失败，而 `successful` 未检查 `result.skipped`。任一合同测试被 `skip` 或 `SkipTest` 跳过且无其他错误时，入口仍返回零并写 PASSED，违反“跳过不能记为成功”的验收要求。建议将非零 skipped 明确纳入失败条件。**现有最终日志为零跳过，本发现不否定该日志实际记录的 146 项结果。**

3. **P3：终态发布与失败记录同时失败时，抛出接口丢失首要原因。**  
   位置：[scripts/run_msb_role_mechanism.py:99](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/run_msb_role_mechanism.py:99)。  
   例如最后成功写入跨期触发 TIMED_OUT，接着失败记录写入抛出另一 OSError：首个异常只被转成局部 `state["error"]`，最终 RuntimeError 仅链接第二个异常，且 state 不返回。CLI 虽正确非零、也未谎报持久化撤销，但调用者失去原超时原因。建议抛出结果同时携带首要异常、失败状态和记录失败异常；现有测试只断言“not persisted”，未覆盖原因保留。

**范围与证据**
使用指定 checkout 的 `git -C`，读取冻结 base→HEAD 的两项生产源码及两项测试差异，以及 HEAD 中完整源码、测试和新验收器。另读了两轮执行规格、本轮计划、此前独立复查、指定三份开发/验收记录；必要时仅查看旧 guard 和 `terminal_status` 的静态定义。

证据读取包括最终完整日志、摘要、快照目录及记录身份、`source_identity.json`、`preservation_and_scope.json`，以及早期各次摘要和有效 RED 日志的失败条目。未重复母任务负责的文件/Git 哈希核验。

作者已有证据记录：最终 **146/146、零失败/错误/跳过、零未模拟受限调用**。测试差异保留原有测试名，新增安全模块20项、runner5项；43源码闭包仅两项变化及保护文件不变，属于已读记录声明，不冒充本次重新核验。

阶段记录一致：Task1 两次导入失败各0项，第三次27项/5失败后27绿；Task2 32项/6失败后32绿；Task3 42项/13失败、身份门45项/6失败后45绿。Task1 RED 三次额度已耗尽。

**T1–T4**
| 原问题 | 判断 | 理由 |
|---|---|---|
| T1 | PARTIAL | 单一成功发布、写后原预算检查及非零失败已接线；双重失败的原因保留仍有发现3。 |
| T2 | ADDRESSED | 父预持有 ownership，已知进程跨启动异常保留；未知创建保守报告存活/清理不确定。限静态及既有合成证据。 |
| T3 | ADDRESSED | 退出后增加有界排空；fixture 的退出不再由读取 DONE 触发，覆盖首次读取暂空窗口。 |
| T4 | PARTIAL | 精确在途 ACK、两种合法交错、同锁写入、锁外等待及 watchdog 独占读取已实现；最终租约边界仍有发现1。 |

**V2 与验收边界**
- v2、terminal-spec 身份与旧科学/repair-spec 身份均接入验证；v1及已列明文档、开发、旧复查授权被拒绝。完成状态还要求父端握手及退出确认。
- 授权检查不是对任意新消息 ID 的语义认证。本次 `call_a4c9ef81bcea447aa7350f7cd9aff51c` **不是实际运行许可**。
- 七阶段、每阶段最多三次独占尝试及固定九模块入口保留。torch/sklearn 必要 bootstrap、调用即失败的 torchvision 测试替身均已披露，不能解释为真实 CNN 验证或“没有任何原生库装载”。
- 身份集成测试放在 safety 模块并复用 RunnerTests，不构成功能缺陷；完整 runner 仍在 final。但排空上限测试主要验证拒绝 `max_frames=9`，未直接驱动八帧耗尽，是覆盖缺口。
- 保留 PowerShell/apply_patch 替代、既有 worktree 复用、scratch留存及旧快照重建来源限制；规格/计划的 `DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN` 是历史状态，不应回写覆盖。

**未判断项**
- 原生句柄继承、嵌套 Job、父硬退出、真实管道/EOF、线程调度、阻塞清理上界、电源连续性及真实文件 I/O：仅有 fake/静态证据。
- 断电、硬杀、解释器极窄异步中断窗口的完整清理：未建立通用保证。
- 真实数据内容、泄漏/污染、科学效果、性能、泛化、原创性及报告/图片质量：禁止读取或不属本次工程范围。
- 白名单外代码的全库兼容性，以及保护文件、源码闭包的实际归档身份：未扩大审查，由母任务另核。

**结论**
合成工程修复：**CHANGES_REQUIRED**。原生验证：**INSUFFICIENT_EVIDENCE**。

现有证据不足以直接放行真实实验。残余风险包括上述租约接收缺口、验收跳过门及未经验证的原生行为；修复、后续合成验证、原生无数据验收、新 v2 协议登记和真实运行均需相应独立授权。归档或合并不代表运行许可。

确认：未修改任何文件；未运行测试、读取真实数据、执行前向/拟合或原生验收；未派发其他代理。
