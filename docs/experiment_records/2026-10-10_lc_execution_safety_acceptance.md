# LC-RFA 执行安全修复与合成验收

日期：2026-10-10（Asia/Shanghai）。分支：`codex/theory-aware-report`。

## 1. 授权与研究状态

人类批准原话：“批准代码与合成验收（推荐）”。对应问题限定为临时防待机、退出释放和监督中断即失败；不得改变 LC-RFA-B 网络、损失、网格，不续跑失败目录，不开展真实训练，不修改正式报告。真实完整重跑仍须另行登记并批准。

本次只做执行层修复和合成验收。没有读取新矿物图片，没有真实 fit、stop 评价或训练重放，没有据此新增网络有效性结论。2026-10-10 的真实开发批次仍是 **TIMED_OUT / NOT_EVALUATED**；旧目录、旧协议和失败档案未改写。

修复基点：`1da47db851ef6cfff312416be214c56b35f6f4f1`。旧协议 SHA-256：`36254788f172af988124fe7058c7822dd96ac5be803d37d99b02ca754340e1db`。正式报告 SHA-256（主目录与工作树相同）：`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。

## 2. 修复内容

- 新增 `scripts/lc_execution_safety.py`：仅在保护区间请求 `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`，不要求屏幕常亮、不修改系统电源方案；同一线程退出时恢复原有执行状态，避免清除调用者原先的请求。
- 电源请求失败、时钟不可用或监督不连续时，不启动下一 worker；退出释放失败不能作为成功。嵌套 worker 会先检查外层阶段的连续性，覆盖任务之间的暂停。
- 使用 `GetTickCount64` 的含休眠经过时间和 `QueryUnbiasedInterruptTime` 的活动时间。固定安全阈值：相邻监督采样间隔大于30秒，或两种经过时间增量的差异绝对值大于2秒，或时钟倒退/非有限值，均记为 `SUPERVISION_INTERRUPTED`。阈值是执行安全参数，不是模型超参数，也不是对所有微小暂停的检测保证。
- 在检查 worker 是否退出之前检查监督连续性，即使 worker 已返回0也不能掩盖长暂停。超时、中断和异常均清理当前子进程并等待退出；记录实际 PID、退出码和是否仍存活，不把未退出进程伪记为已退出。
- 保留 `POWER_REQUEST_FAILED`、`POWER_RELEASE_FAILED`、`SUPERVISION_INTERRUPTED` 等终止状态；中断发生在写任务日志前也保留失败记录，不读取该 worker 的模型结果。
- 异常或未完成阶段留下的选择锁不能进入后续评价。未删除锁或失败证据，也没有自动重试、恢复或扩网格。

现有422次拟合上限、另4次温度拟合、21600秒累计预算和600秒单worker预算均未改变。预算不扣除休眠或暂停时间。网络、辅助损失、优化器、学习率、候选网格、随机种子、数据折和正式报告均未改动。

## 3. 合成验收

Windows API 外部边界用模拟对象替代；执行器的正常退出、崩溃、超时和父进程中断使用真实短命 Python 子进程。现有数据/训练测试只使用合成夹具。未调用真实防待机 API，也未主动让电脑休眠。

| 验收 | 实际结果 | 证据 |
| --- | --- | --- |
| 新模块首轮 RED | 10项因缺少模块失败 | 本地 `outputs/lc_execution_safety_qa/red_native.log` |
| 执行器完整 RED | 21项，7个失败、1个错误 | 本地 `red_runner_complete.log`；包含环境展开，故不公开上传 |
| 后置中断/异常锁 RED | 2项，1个失败、1个错误 | `outputs/lc_execution_safety_qa/red_post_worker.log` |
| API/时钟合成验收 | 12/12，0.003秒，exit 0 | `green_native_final.log` |
| 执行器合成回归 | 22/22，21.467秒，exit 0 | `green_runner_final.log` |
| 全项目回归 | **483/483，139.608秒，exit 0** | `full_suite.log` |

较上一轮464项增加19项；已有选择锁测试也增加异常阶段回归。针对性测试覆盖：请求失败不启动、原有状态恢复、请求后初始化失败释放、异常/KeyboardInterrupt释放、释放失败、长轮询缺口、短休眠差异、异常时钟、线程归属、嵌套阶段暂停、已退出worker、无下一worker、后置日志中断、异常选择锁和具体终止状态。

自审发现的“异常锁仍可读取”和“后置中断缺失任务日志”均先运行 RED 再修复并完成 GREEN。验收为作者自审，不是独立评审。没有保留待修复的已知阻断项。

## 4. 边界与后续

临时请求不能阻止用户手动休眠、合盖、强制终止进程或系统崩溃。Windows停止调度时，Python监督器本身也不能执行，因此不能宣称硬件层面的“600秒内绝对终止”。本修复保证的是：在获得调度、观察到超阈值暂停后，失败关闭并拒绝继续实验；并非实际机器待机测试已通过。强制结束监督器也无法保证Python的 `finally` 执行。

官方依据：[SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)说明线程请求和主动睡眠的边界；[Interrupt Time](https://learn.microsoft.com/en-us/windows/win32/sysinfo/interrupt-time)区分含休眠与活动时间；[PowerSetRequest](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-powersetrequest)另提示Modern Standby电池模式下电源请求存在时间限制。本实现采用前者，不把后者的限制直接等同为本API的保证。

新增安全模块已纳入未来默认源码快照。执行器源码改变意味着旧登记协议会被来源校验拒绝，这是预期行为；没有重写旧44项哈希来复用上次训练授权。

下一轮真实研究须先单独安排运行环境验收（连接电源、保持主机可调度，约定合盖/手动睡眠限制），然后生成新的源码与安全阈值登记、新输出目录和明确人类授权。不能从旧345条完整CV结果接着跑，也不能利用这次代码批准开始训练。新网络的创新效果仍须由完整同预算强对照实验验证，不能写入正式报告为已证实。
