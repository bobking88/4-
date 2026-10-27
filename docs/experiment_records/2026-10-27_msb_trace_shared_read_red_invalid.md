# MSB 共享读取修复：首轮无效 RED 与停止记录

日期：2026-10-27。开发授权 `call_4490c98e36aa4667a22d7065e677fb99`，认可文档提交 `a09d55a8088c6c71806d077efe66846321ef309c`。

**当前状态：RED_INVALID_STOPPED_TEST_CORRECTION_REQUIRED。Task1 未完成；未实施生产读取器，未运行顶层 GREEN/final，没有修复成功结论。**

## 实际工作与原件

仅修改了批准三文件中的 driver 测试与原验收器：添加固定 12 个测试方法、模拟文件内核和旧 observer fake 的读取注入；验收器迁移本次证据/scratch 路径并预设新读取工厂守卫。driver 本身零修改。两份源码是未通过验收的草稿，不是可运行原生协议或已验收实现。

顶层只运行一次 `collector-red`，路径 `outputs/msb_trace_shared_read_synthetic_v1/collector-red/attempt-001`。收集 136 项，14 个 failure entries、10 个 error entries，零 skip/expected failure/unexpected success/未模拟边界调用。failure/error 数包含 subtests，不能用它们推算失败方法个数。旧 driver 124 个完整方法名均保留，新类有且只有固定12项，但旧124并非全部通过。

保留[原始 summary](../../outputs/msb_trace_shared_read_synthetic_v1/collector-red/attempt-001/acceptance_summary.json)、[完整日志](../../outputs/msb_trace_shared_read_synthetic_v1/collector-red/attempt-001/synthetic_tests.log)、[当时源码快照](../../outputs/msb_trace_shared_read_synthetic_v1/collector-red/attempt-001/source_snapshot.json)；没有覆盖、删除、续跑或自动重试。日志中的嵌套 collector-green/单项 skip 来自旧验收器自检的临时 ROOT，不是本次顶层 GREEN。

## 无效原因及作者错误

1. 作者漏改了旧测试夹具的两处输出路径。旧 `HarnessGateTests.test_actual_main_never_passes_a_skipped_input` 与 `DirectLaunchBoundaryTests.test_synthetic_scope_covers_runtime_and_thread_adapters` 仍读 namespace-v3 路径，合计产生9个 error entries。
2. 新 R11 错把现有 `classify_native_cases` 的 `errors` 字段写为 `reasons`，产生第10个 error entry。
3. R12 只检查验收器源文包含工厂名，尚无生产工厂也能通过，不能证明实际原生读取工厂受守卫。这不满足规格要求的 R12 有效 RED。

因此，虽有缺接口、旧 Path.read_text 回退、ready 前置缺失等明确 failure，**整轮仍不能当作有效 RED**。早先关于“核心红灯已正确、仅五个子例表达问题”的口头判断有误，以本记录和完整日志为准。依规格的无效 RED 停止要求，不借此实现生产代码或追绿。

静态复核还发现，当前新增测试尚不能充分证明 R01 ABI/延迟绑定、R03/R04 多异常下解析次序、R06 所有期限边界、R09 到期窗口与死亡隔离、R10 实际 callback/stop 传播、R11 首因锁存/N4失败回执先读、R12 实际守卫。不能以“12项数量满足”替代行为覆盖，不能为了让草稿通过而缩减生产规格。

## 保护与证据限度

回溯比对上次已归档的537条保护记录（包含重复项，并采用上次 planning 后的 README 身份）：525条匹配，12条变化只对应本次获批的两个测试文件；其余源码、旧失败原件、六张预存脏图和主目录/工作树两份正式报告保持字节不变。详见[保护核对](../../outputs/msb_trace_shared_read_synthetic_v1/protection_check.json)。

本轮没有在首次运行前单独保存 development-before 保护快照，不能把上次 planning 记录冒称本轮前快照。入口 stdout/stderr 也没有分别落盘，只保留工具调用输出及原验收器自身的完整日志、summary、source snapshot；不事后重建伪装原件。Git Bash workspace helper 曾因 `NtCreateDirectoryObject 0xC0000022` 失败，随后仅手工建立同名 ignored 账本和 plan-path；不改任何执行门。

暂存检查发现原始 `synthetic_tests.log` 的第32、126行有 unittest 写出的行尾空格；为保持原件字节，不作格式化。以命令级 `core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol` 识别原件 CRLF 的 `git diff --cached --check` 因这两行返回1，不把它描述成全局格式检查通过；测试源码与手写记录的定向检查返回0，不涉及这项原始日志例外。合成器的 `previous_sha256` 指向更早历史源码，不等于本轮开发起点，因此本轮源码差异以 planning 保护清单为依据，另核对当前56项与本轮 RED source snapshot 相同。

默认 Git 文本过滤会改变三份原始输出的 CRLF；归档暂存这三份输出时仅采用命令级 `core.autocrlf=false`，逐一要求 index blob 等于 `git hash-object --no-filters`。不改 Git 全局/仓库配置或 `.gitattributes`，不重写本地原件。手写记录与测试源码继续使用仓库现有文本规则。

受限原生接口由验收器守卫；本次无原生验收、进程/计时探查、矿物原图/真实 prepared/fit/stop 读取或真实训练，无代理，无网络/公式/科学矩阵/正式报告变更。合成代码本身有导入，不能称本轮“零项目导入”。原 Windows WinError5 的唯一根因和任何网络理论收益均未验证。

## 下一步边界

先确认是否继续修正上述测试草稿：仍限原三文件、固定12方法与原规格全部行为；保留 attempt-001，最多使用剩余2次 RED 额度，取得有效 RED 后冻结测试再实施生产 driver，GREEN/final 仍各最多3次。不新增阶段、目录、模块、bootstrap例外或实际实验。建议先静态核对全部12项与旧夹具，避免把新的测试错误带入下一次有限运行。

本次归档只保存失败与未完成草稿，**不放行下一阶段**。独立复查、未来新原生协议/验收、真实机制强对照仍另批；这项工程修复不是报告或论文的理论创新，不写入科学优势结论。
