**问题**
以下 4 项均为静态控制流发现，未执行反例，也不代表已观察到原生 Windows 故障。

1. **[P2] 最后一次成功状态写入未受截止时间检查保护。**  
   位置：[run_msb_role_mechanism.py:347](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/run_msb_role_mechanism.py:347)。  
   第 339 行检查通过后，仍会更新历史、采样并再次执行包含 `fsync/replace` 的 `_json`，随后直接返回。若此时跨过原始 deadline，仍可留下 `MECHANISM_COMPLETE`、`delivery_verified=True`，CLI 返回成功。这不是获准的失败清理，而是超时成功导出。  
   **建议：**将最终成功写入统一纳入异常处理及写前、写后预算检查；超时必须改写为失败并撤销交付。补充“仅最后一次状态写入跨期”的假时钟回归。

2. **[P2] 启动后的异常仍有丢失进程对象的路径。**  
   位置：[msb_execution_safety.py:523](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/msb_execution_safety.py:523)。  
   已修复的 `_WorkerStartError` 只覆盖继承标志恢复失败。若 `Popen` 已返回，但日志上下文退出抛错，或在返回前的相关窗口发生 `KeyboardInterrupt`，且继承标志恢复成功，原异常不会携带 `spawned_process`。父端第 364 行的赋值尚未完成，第 404 行也无法恢复对象，因此不执行 kill/wait，却可能报告 `attempts=0`、`alive=False`、`cleanup_confirmed=True`。子端断管后可能自行退出，但这不能替代父端退出确认。  
   **建议：**任何启动后异常都保留进程所有权；增加日志退出异常和启动后中断的 fake 回归。

3. **[P2] 正常退出可能早于父端读完已排队的 ACK/DONE，被错误判失败。**  
   位置：[msb_execution_safety.py:393](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/msb_execution_safety.py:393)。  
   每轮只读一条消息，随后立即检查进程退出。合法 worker 可以已经发送 ACK、DONE 并以 0 退出，而父端只读到 ACK；此时 `done=False`，立即报监督中断。正常调度延迟即可触发，导致单次授权被失败消耗。现有 fake 在父端**读到 DONE 时才设置退出码**，见 [test_msb_execution_safety.py:179](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/tests/test_msb_execution_safety.py:179)，因此排除了该时序。  
   **建议：**在原预算及有限收尾边界内排空、验证退出进程的控制消息；缺失或非法 DONE 仍须失败。

4. **[P2] RENEW 与正常 DONE 之间缺少终结握手。**  
   位置：[msb_execution_safety.py:397](D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/msb_execution_safety.py:397)。  
   父端读到暂时无消息后，worker 可以发送当前序号的 DONE；父端同时发送下一序号 RENEW，并将 `acked=False`。下一轮合法 DONE 随即因序号及 ACK 状态不匹配被拒绝。子端 watchdog 与主线程的 ACK/DONE 写入也未统一串行化终结状态。  
   **建议：**定义明确的结束确认协议，协调续租与关闭；不能简单放宽序号检查。增加两个方向的确定性交错回归。

**优点**
- 精确 `projector_fit` 检查覆盖 train/validation 每行；摘要转换发生在严格验证之后，未发现 normalize、补值或推断放行。
- 三次 load 后、首个 fit packet 前执行共享联合校验；后续 worker 在本折输入加载及模型/优化器创建前重验证书、摘要哈希和联合身份。
- 父句柄创建身份、专属管道、Job 分配先于 GRANT，原始整数预算随绑定传递。未发现旧式“仅凭陈旧 RUNNING 放行”的直接路径。
- 允许范围内的 diff 确认科学模块未变，训练模块仅替换计时接口；43 项闭包已登记，80 任务、63+12 拟合及 615+12 核验边界保留。

**证据边界**
既有日志确实记载：v1 为 **9 失败、1 错误**，v2 为 **119/119**，v3 为 **121/121**，最终守卫计数为零未模拟调用。这些是作者既有验收记录，不是本轮重现，也不能覆盖上述未测试时序。

早期两次 runner 回归采样原生只读时钟的偏差已明确披露，没有被反向归入完全模拟；未插桩次数仍未知。另需保留来源限定：v1 的 `acceptance_harness_source.json` 明标为按已记录修改**重建**的源码快照，并非独立确认的当时原件。旧 34 源码及保护文件的完整哈希保持结论，本轮只核阅其记录，未重新读取全部对象验证。

**不予判断**
- 原生句柄继承、嵌套 Job、父硬退出、真实管道及阻塞清理上界：未获准执行原生验收，属于证据不足，不一概列为代码缺陷。
- 真实 prepared 内容、历史泄漏或实际污染：禁止读取，不能由静态缺口反推。
- 全矩阵运行成本、真实效果、泛化及原创性：本轮没有相应执行或研究。
- PNG/DOCX 内容及当前工作区二进制身份：未打开，不判断其质量或当前字节状态。
- 白名单之外的测试源码及全库兼容性：未扩大审查范围，不能由日志名称推定覆盖充分。

**建议与裁决**
代码裁决：**CHANGES_REQUIRED**。以上缺口应先修复，并另获批准补充针对性合成回归，当前不予 **READY_FOR_NEXT_REGISTRATION**。

原生监督裁决：**INSUFFICIENT_EVIDENCE**。这与代码缺陷分开；即使代码修复，原生无数据验收、新真实协议登记及真实运行仍各需单独批准，本审查不授权训练。

审查范围为 `8de3804ea8d7893ad21df72281614c9d559a7d4b..0bb95a48a8fc47286dfcc3bd864d6ecfaf02ef58`，共 5 个提交，源码引用均取指定 HEAD。全程独立只读：未运行测试、被审代码、导入、Python、前向、拟合、计时/电源/进程实验或网络操作；未读取真实数据、张量、权重或图片；未改文件、index、HEAD、分支，未创建 worktree 或派发代理。
