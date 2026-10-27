# MSB Trace Shared Read 最小修复规格

日期：2026-10-27。状态：**DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN**。本轮授权仅为 `call_2e34a6fa4f8f4207ad5cf4476536803a` 的规格、计划与 GitHub 归档；以下实现和验收要求是待审核方案，不是运行许可。

## 1. 依据与目标

文档基线为 `cf2e62e89294582ee49f63e9d6f3226e2ce336df`，执行源码仍固定为已复查的 `4312036b33c05a9b8c29f4849860562eb7c68b87`。详见[离线诊断](../../experiment_records/2026-10-27_msb_native_supervision_v3_offline_diagnosis.md)。

已确认 controller 在 trace 替换阶段发生 REPLACE / PermissionError / errno=13 / winerror=5；观察器即使尚无 worker_ready 也打开 trace，普通读取没有显式删除共享契约。**观察器持有不兼容读句柄是有依据但尚未实证的优先假设，不是已确定的唯一根因。** ACL、属性、外部过滤器等尚未排除。

目标是明确这一内部并发读写接口：观察器只读且非继承地打开 trace，允许发布方替换目标，及时关闭句柄；ready 条件未满足时不打开 trace。不得降低身份、证据、累计期限或失败关闭门槛，不重写监督器。

## 2. 方案与取舍

| 方案 | 取舍与本规格裁定 |
|---|---|
| 显式 Windows 共享读取，保留原发布器 | 推荐候选，改动局限于读取端，可对句柄生命周期作确定性模拟；不能保证解决所有访问拒绝原因 |
| 不可变版本化快照 | 可减少对同一路径的替换依赖，但改变发布/发现协议，本轮不采用 |
| 仅延后读 trace 或增加重试 | ready 前置只减少无效读取，不能单独解决共享契约；不增加重试，不吞掉 PermissionError |

上述推荐是作者提出的实现选择，尚待用户审核。FILE_SHARE_DELETE 是允许其他打开者请求相容的删除/重命名访问，不是给本读句柄授予 DELETE 权限，更不是绕过 ACL。[CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)、[MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)

## 3. 精确修改范围与不变量

未来仅修改三文件：

- `scripts/check_msb_native_supervision.py`：新增 trace 专用只读适配、有限错误记录扩展及观察器前置检查。
- `tests/test_check_msb_native_supervision.py`：新建固定 12 个测试方法；必要旧夹具只注入 fake 读取器，保留旧方法名和行为断言。
- `tests/run_msb_native_supervision_synthetic_acceptance.py`：新证据/scratch 两处路径及新原生读取边界的模拟守卫。

原 46 项执行来源和 56 项历史保护路径集合不变；其余 43/53 项源码不改。网络、损失、公式、科学实验矩阵和正式报告不改。trace 写入器、首发失败锁存、独立失败回执、N4 失败回执先读与快照约束不改。通用 `_read_json_if_present` 不整体替换。

保留 `msb_parent_bound_v2`、`DIRECT_CPYTHON311_VENV_V1`、N1–N4、480 秒累计接收上限、case limits `[90,45,75,45]`、接电、单次及失败即停。无新依赖、无权限修改、无删除 pending、无兼容回退或静默降级。v3 原生输出和原许可均已消费，不因本修复恢复可运行状态；未来新命名空间/协议/独立输出另行登记，不在本工作包生成。

## 4. 读取接口与生命周期

在 driver 内新增私有接口，不拆出新模块：

- `_trace_reader_kernel()`：延迟绑定 CreateFileW、ReadFile、CloseHandle 的 Windows ABI；仅实际读取入口需要时调用，不在模块导入时创建 DLL 或句柄。HANDLE 使用指针宽度，大小/计数使用 DWORD，返回 BOOL；明确 argtypes/restype 和 last-error 语义。
- `_TraceReadFileIO(kernel=None)`：可注入 fake kernel；提供 `open_shared_read(path)`、`read_chunk(handle, size) -> bytes`、`close(handle)`。省略 kernel 才调用上述原生工厂。
- `_read_trace_json_if_present(path, *, check, file_io=None)`：执行受限读取并返回解析后的 JSON 或缺失时的 None。
- `NativeServices.read_trace(path, binding)`：通过 `check=lambda: _remaining(binding, self)` 调用该 helper；观察器只能传入当前已验证 case 下的 `controller/trace.json`。

打开参数固定：GENERIC_READ=`0x80000000`，share=`FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE = 7`，OPEN_EXISTING=`3`，FILE_ATTRIBUTE_NORMAL=`0x80`，security attributes 为 NULL，template handle 为 NULL。不请求写、DELETE 或继承权限。不得先 exists 再退回普通 Path.open/read_text。

每次成功打开获得一个自有读句柄，4096 字节分块读至正常 EOF；ReadFile 失败必须立即保存原始错误，不让后续 API 覆盖 last-error。OPEN 失败/无效句柄不关闭；有效句柄在 finally 中**恰好一次**尝试 CloseHandle，失败也不重试。CloseHandle 返回失败必须转为异常，不能当作关闭成功。

使用同一个已继承累计期限：打开前、每块读取前后、关闭后及返回前检查；不重置计时，不新增等待或宽限。即使读取或期限检查失败也尝试关闭自有句柄，随后保留首因和关闭/末端期限错误。同步系统调用本身不能据这些检查宣称可在任意阻塞下硬实时中断。

先完成字节读取和句柄关闭，再严格 UTF-8 解码和 JSON 解析。只有 OPEN 的 WinError 2/3 表示暂缺文件并返回 None；WinError 5/32、读错误、关闭错误、Unicode 错误及期限错误不得转换为缺失。成功关闭且期限有效后的 JSONDecodeError 可沿用既有观察器的“不完整 JSON 等待下一轮”语义；带有关闭/期限失败时不能被这一路径吞掉。没有新重试循环。

一个已经打开的句柄可读到替换前的完整旧快照；下一轮重新打开可看到新快照。这不允许把旧快照伪造成 ACK，也不承诺其他持有者、ACL 或外部过滤器一定允许替换。[os.replace](https://docs.python.org/3.11/library/os.html#os.replace)

## 5. 错误保留与 ready 前置

新增 `TraceReadError(errors)`，errors 为按发生顺序保存的 `(phase, original_exception)`，phase 限于 OPEN、READ、CHECK、CLOSE。首个异常始终是首因，后续清理/期限错误不覆盖它；原始对象保留在内存，公开记录继续脱敏，不输出原始路径、nonce、句柄或多行消息。

`_fault_error_record` 仅对该新类型增加有界的 `trace_read_errors` 字段，每项含 phase、沿用原脱敏的 type/message/frames，以及确为整数的 errno/winerror。不递归抓取任意异常属性，不更改既有普通错误记录。现有 callback、FaultStopResult、ObserverTerminationError 发布链应自然保留这份结构；不得因观察器停止异常丢掉读取首因。解码/JSON 在关闭之后失败时保留原异常类型，不包装成可忽略的 I/O 缺失。

`NativeObserver.sample` 在原锁及原 _remaining 内按下列顺序执行：

1. 读取 worker_ready；缺失即返回，不打开 trace。
2. 读取 controller_ready；缺失即返回，不打开 trace。
3. 检查既有 case、working、job_assigned、controller binding 条件；缺字段/类型不合法仍失败关闭，不以“未就绪”掩盖。所需身份字段仍须通过既有严格校验。
4. 经 `services.read_trace` 读 trace；暂缺则返回。解析失败仅按第 4 节规定处理。
5. 保留原 assigned、sequence=1 ACK、spawn 身份、活性、5000ms ACK 窗口和只查询句柄校验，再发布原 collector_ack。

不得仅因 ready 文件存在就确认身份；不得新增 ACK 默认值。观察器 RLock 仍只用于自身同步，不声称它协调跨进程发布。N4 的内存快照/杀父路径不改；`_load_case_record` 的终结后读取与失败回执先读不改，不引入新的活跃 trace 读取点。

## 6. 有限合成验收方案，尚未获运行许可

未来固定证据根 `outputs/msb_trace_shared_read_synthetic_v1`，scratch `.superpowers/sdd/2026-10-27-msb-trace-shared-read/test-tmp`；本轮不创建。collector-red、collector-green、final 各最多 3 次独占尝试，含失败，保留全部旧原件，不自动重试或换未批准目录。

保留原 driver 124 和全套 285 个完整测试名；新增前缀 `test_check_msb_native_supervision.TraceSharedReadTests.` 的以下 12 个方法。未来目标为 driver 136、final 297，**不是已有运行结果**。不得仅靠数量认定覆盖。

| ID | 方法名 | 必须验证的行为 |
|---|---|---|
| R01 | test_windows_read_flags_and_handle_abi | 实际适配器配 fake kernel，读权限/共享7/非继承/OPEN_EXISTING及指针宽 ABI；导入不绑定 DLL |
| R02 | test_held_reader_sharing_controls_replacement | 确定性共享模型：持有不含 DELETE 共享的句柄拒绝替换，含 DELETE 时替换可成功，原句柄仍读旧快照；释放后可重开读新快照 |
| R03 | test_missing_access_errors_and_json_polling | 仅 Win2/3 返回 None；Win5/32 原因保留；成功关闭后不完整 JSON 可按旧轮询处理，不能吞关闭/期限异常 |
| R04 | test_read_and_decode_failures_close_once | 多块/零字节 EOF、ReadFile 失败和 UTF-8 失败；读失败仍关闭一次，解码发生在关闭后，无效句柄不关闭 |
| R05 | test_close_failure_preserves_ordered_original_causes | CLOSE 单异常及 READ+CHECK/CLOSE 联合异常，原对象及有序首因不变，关闭不重试，末端预算错误也保留 |
| R06 | test_inherited_deadline_checks_surround_owned_handle | 打开前、块间、关闭后及返回前的到期/等于期限均拒绝；自有句柄仍一次关闭，不新计时/等待 |
| R07 | test_ready_absence_does_not_open_trace | worker_ready 缺失、controller_ready 缺失均零 trace 打开、零身份查询、零 ACK |
| R08 | test_invalid_ready_binding_fails_before_trace | 错 case/working/job/binding及缺字段在 trace 打开/进程查询前拒绝，不把错误当等待 |
| R09 | test_observer_uses_shared_reader_and_retains_ack_checks | 实际 sample 调用新读取接口；禁 trace 的 Path.read_text 回退；正常 ACK 主链与旧身份/活性/窗口拒绝断言保留 |
| R10 | test_callback_records_read_close_and_stop_causes | 实际回调错误记录与停止错误联合，脱敏记录保留 phase、errno/winerror与有序首因，无原始多行/nonce/路径泄露 |
| R11 | test_trace_writer_and_n4_failure_precedence_unchanged | 相同事件快照发布及真实 helper 的首因锁存行为不变；N4 先读失败回执，不以快照替代失败 |
| R12 | test_harness_guards_new_reader_native_boundary | 工厂未注入即被守卫拒绝；显式 fake kernel 可测试实际适配器；原严格通过谓词/十模块/历史保护与两处新路径保持 |

R02 是 Windows 规则的条件模型，**不是已在本机复现原故障**。R01 的 fake ABI 断言也不证明实际 Windows 调用兼容。生产 helper 和 sample 必须真实执行于未来合成测试，不能用“被测函数直接返回成功”的替身自证；仅模拟原生边界和 scratch 元数据。进程、线程启动/join、时钟、电源、网络、runtime、Windows 文件 API 全部模拟。旧 scratch 合成张量例外不扩大，不读真实 prepared/fit/stop 或原图。

新增断言/helper、旧夹具适配和 harness 在有效 RED 后冻结；RED 必须是明确缺行为断言，不得以导入失败、AttributeError、未模拟边界调用或错误收集数冒充。GREEN/final 零失败、错误、跳过、预期失败、意外成功和未模拟调用，源码身份相同。若需改变冻结断言或扩范围，先停并记录，另行审核。

## 7. 验收与后续门槛

实施计划仅一个不可拆分的三文件工作包。提交前核对来源/旧失败原件/六张预存脏图/两处正式报告保护，范围超出即停。完整合成日志、所有失败尝试、来源快照和作者裁定均保留；通过只能标为作者合成通过、独立复查待批。

未来依次仍需独立批准开发与合成、只读独立复查、有效新原生协议与单次运行、真实 fit-only 机制实验。此次文档或未来工程绿灯均不证明神经网络理论收益。真正论文创新仍须用固定强对照检验间隔预算的纠错/损伤条件和角色证据贡献，不能把此工程修复写成论文方法创新或预写阳性结果。
