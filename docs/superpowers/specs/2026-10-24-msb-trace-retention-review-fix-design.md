# MSB Trace Retention Review Fix Design

日期：2026-10-24。状态：**DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN，待人类审核。**

## 1. 授权、事实与目标

本轮授权为 `call_21f6de490d944c7bb9b0116fc5c3dda9`，仅编写修复规格、配套实施计划和 GitHub 归档。基线为 `cd9dcb287cbf55a1a960e021130dffde9ba8bf9a`，受审源码仍为 `e2c9cc4`。不改代码、不导入项目、不运行测试、前向、拟合、计时或原生验收，不读真实数据，不派代理，不改正式报告。

依据：[独立复查与作者静态核对](../../experiment_records/2026-10-23_msb_trace_publication_retention_review.md)。R1/P1 是多行异常脱敏缺口；R2/P2 是停止再次抛错时先前原因未返回；R3/P2 是 N4 严格快照检查早于失败回执读取。三项是静态可达的条件缺陷，不是已发生的泄露或原生故障。既有 final=267 的日志原件有效但覆盖不全，当前未重跑。

目标仅为补齐诊断保留与失败关闭，不改变监督策略或网络。Windows 首次文件发布错误仍未知；即使本修复以后通过，也不能直接证明原生兼容、MSB-RRE 理论收益或论文创新有效。

## 2. 方案与范围

以下是作者在本次文档授权下提出的选择，尚不是人类对新设计的批准：

- 采用现有三文件局部修复：保守消息脱敏、调用方持有预算错误列表、失败回执先读并进入专用拒收记录。范围和现有接口最接近。
- 不采用新执行框架、重复停止/重新采样或重试发布：会改变监督时序及预算，超出本次问题。
- 不仅把正则改成 DOTALL 后就声称完整解决：敏感文本可能位于标识前，且不能填补 R2/R3 与联合异常覆盖。

未来代码仅允许修改：
`scripts/check_msb_native_supervision.py`、
`tests/test_check_msb_native_supervision.py`、
`tests/run_msb_native_supervision_synthetic_acceptance.py`。

不新建模块，不改底层 MSB/LC、模型/损失/公式/矩阵、NativeServices 的停止/线程实现、N4FaultGate、原生协议或正式报告。不重用旧原生许可或输出目录；若需要第四源码文件或扩大接口行为则停止另批。

## 3. R1：保守脱敏与原对象分离

保留 `_fault_error_record(error) -> dict` 的现有 type/message/frames 结构与最多 8 帧，不改数值 errno/winerror 白名单。

消息处理采用以下固定规则：
1. 只对异常消息作一次文本提取。若消息含换行/分行控制符，或匹配既有大小写不敏感标识 `nonce|handles?|environment|env` 的词边界，则整条 message 改为固定 `<redacted context>`，不保留标识前后片段。
2. 分行集合固定为 `\n, \r, \v, \f, \x1c, \x1d, \x1e, \x85, \u2028, \u2029`；不把字面量反斜杠+n误当真实换行。字面量测试与真实分行测试分别记录。
3. 其他单行消息沿用现有 64 位十六进制、0x 地址、整数替换及 256 字符截断。普通单行原因可读性不变。
4. trace、body、row.error、预算、停止、缓存 callback、receipt 及 CLI 的新增诊断都经过同一消息边界；不能把原异常 repr、args、环境或异常链直接输出。
5. 原异常对象仅保存在内存供身份/顺序断言，不因脱敏替换其对象，也不序列化任意属性。

这是针对上述标识和分行边界的修复，不宣称能识别任意未标识的秘密，也不扩成通用日志安全改造。新测试的敏感值全部为合成短文本，不能用真实环境内容作反例。

## 4. R2：停止前检错误由调用方持有

最小扩展接口为：
`_bounded_fault_stop(watch, binding, services, *, errors: list[BaseException] | None = None) -> tuple[FaultStopResult, list[BaseException]]`。

未传 errors 时保持既有调用者的二返回值/异常传播契约。传入空列表也必须复用该对象，不能用 `errors or []` 丢掉调用方引用。预算前检失败仍记录该原对象并把传给 stop_fault 的剩余量设为 0；stop_fault 仍至多调用一次。若停止自身抛错，仍向上抛原异常，已记录列表不能随未完成的 tuple 赋值丢失。仅在原先允许的成功路径做原后检，不增加时钟、预算、采样或重试。

run_controller 保留自己的 stop_errors 列表并通过上述参数传入。停止异常后只读取已存在的 `watch.stop_result`、`watch.error` 缓存，不调用 join、is_alive、事件等待、guard 或新的 query。缺失缓存记为 unavailable，不推断“已停止”；缓存与返回值冲突要分别保留，不能互相覆盖。

`ControllerTracePublicationError` 构造器末尾增加可选 `fault_stop_evidence=None`，原调用保持有效。该属性为固定键字典，保存原引用：returned_result、budget_errors、stop_exception、cached_result、cached_callback_error。record() 只导出白名单安全记录，现有 fault_stop.result/errors/exception/stopped 字段保留，附加 cached_result、cached_callback_error、固定为 true 的 cached_result_diagnostic_only。没有证据用 null，不凭空造 FaultStopResult；不能 deepcopy 或直接序列化含原异常的 evidence 字典。

有停止异常时 stopped 必须为 false，即便缓存的 stop_ok 为 true。缓存只用于诊断，不能作新的停止/释放确认。无 trace 失败时现有异常优先级不改：停止自身原异常仍先于 body 异常抛出。NativeObserver 不需要传新参数，其调用契约保持不动。

## 5. R3：先读失败回执，正常 N4 门不放宽

新增同文件 helper：
`_read_controller_failure(path: Path) -> dict | None`。

- 文件不存在才返回 None；不进行等待、重读或修复写入。
- 成功解析为 status=`FAILED_CONTROLLER_TRACE_PUBLICATION` 的字典时返回该诊断；它永远不参与正常成功判断。
- 部分 JSON、解码/读取失败、null/数组/错误形状均返回非空诊断哨兵，status=`INVALID_CONTROLLER_FAILURE`，仅带安全错误摘要，不带原始文本/字节。不得将这些情况当“无失败回执”。
- FileNotFoundError 作为不存在；其他读取错误失败关闭。不得删除、覆盖或补全回执。

`_load_case_record` 首先调用该 helper。结果非 None 时，立即构造专用失败 case record：保留当前 case_id/binding/outer/observation、controller_failure，附加 diagnostic_only=true；msb、parent_power、fault、cleanup_intent、worker_closed 均为 null，events=[]，不读取其他 JSON、不尝试拼接 N4 成功快照。该回执自己的 inner_summary 仍保留既有诊断，原始文件仍留在磁盘。

只有结果为 None 时才走原正常路径，原 N4 parent_fault/snapshot/前缀一致性检查全部原样保留，正常 record 不新增 diagnostic_only 字段。`_check_case` 的第一条非空 controller_failure 拒收规则不放宽。collector 现有“先 append record，再 check”接线不改：诊断能进入汇总，但该例仍失败且后续例不启动。缺快照本身仍不能通过。

## 6. 联合异常及部分回执

补一条真实 run_controller 的合成接线：run_job 已返回 row，父 scope 退出再抛 body/release 异常，N4 停止自身再抛异常且缓存 callback，最终 trace 失败，独立失败回执写入部分内容后再次失败。所有对象按角色分别存在；row 不因后续异常消失，trace 首错不改，receipt_error 不覆盖其他原因，CLI 返回 1 且不输出原敏感文本。

另测停止预算前检异常与停止自身异常联合存在；该情形的后续回执预算检查必须继续耗尽，不得靠“下一次时钟恢复”虚构剩余预算。无回执写入和已有部分回执是两个不同情形。

部分回执测试必须执行既有实际 `_write_json(..., exclusive=True)` 的函数体，只在目标 receipt 的文件流边界使用内存适配；写出合成前缀后抛错，或完成写入后 close 抛错。不能仅把 _write_json 替换为打开前就抛错。不改通用 helper，不重试/续写/unlink。内存假文件中的前缀须保持，collector 的实际新读取 helper 将该部分内容判为非空失败诊断。

## 7. 未来有限合成边界

未来仍为原生会话内实施，不派代理；进程、线程 start/join/is_alive、时钟/睡眠、电源、网络、runtime、trace I/O 和新增 receipt 故障 I/O 全模拟。保留原 scratch 合成张量契约，不新增真实张量/缓存/图片/模型读取或科学前向/拟合。

预留新输出根 `outputs/msb_trace_retention_review_fix_synthetic_v1/`，scratch 根 `.superpowers/sdd/2026-10-24-msb-trace-retention-review-fix/test-tmp`。当前不创建。使用原 collector-red、collector-green、final 三阶段，每阶段最多 3 次独占尝试，失败也计数；保留旧原件，不自动重试。保持十个 final 模块、56 路径保护、数值 bootstrap、所有禁止接口守卫与 _acceptance_passed 不变，验收器只适配新 output/scratch 路径。

保留原 267 个完整测试名，计划新增固定 10 名（见配套计划），driver 未来为 116、final 未来为 277。数字只是覆盖期望，不是结果；不能只核对数量。RED 须在 R1/R2/R3 核心行为得到断言失败而非导入/TypeError/守卫失败，兼容性测试可以原本通过。GREEN/final 须零失败/错误/跳过/预期失败/意外成功/未模拟调用。

## 8. 接收、反目标与交接

接收需要同时证明 R1 的真实分行边界、R2 的原对象和一次停止、R3 的 N4 诊断先读及严格正常门、实际 helper 的部分回执、联合异常和完整旧名保留。状态只能到作者合成验收完成，不能自动升级为独立 APPROVE、原生通过或科学收益。

本轮只写文档。独立复查不在本轮授权内；开发/合成、新原生登记/执行及真实机制实验分别另批。下一次科学工作仍应回到既定 MSB-RRE 理论命题、强对照与失败标准，而非将这些工程修复当作网络创新或论文阳性结果。
