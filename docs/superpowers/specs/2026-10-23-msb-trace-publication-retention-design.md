# MSB trace 发布与首发错误保留：最小修复规格

日期：2026-10-23。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`，书面规格及配套计划待人类审核。

## 1. 意图、证据与当前授权

人类授权 `call_b95bed45520f49c4841956cf3c512ad7` 仅允许编写规格、实施计划并归档 GitHub。基线为 `05727007209422c488a9ecdd8107018f883087fa` 的离线诊断。本文件不是开发、测试、原生验收或真实实验许可；按本次明确要求同时提供配套计划草案，不将两份文件的起草视为已获书面规格审核。

目标是修复两个已证实的工程缺口：固定 trace 临时文件在首发失败后被 finally 再次创建；内层已经捕获的 `row.error` 又因最终 trace 失败而无法落盘。**不承诺修复第一次 Windows 替换失败，不声称其确为共享冲突。** 网络创新、理论公式和科学效果不在本修复范围内。

## 2. 方案取舍与推荐

| 方案 | 取舍 | 裁定 |
| --- | --- | --- |
| trace 专用分阶段写入，加首次失败锁存和独立失败回执 | 保留现有替换策略，增加可定位性，其他 JSON 发布不变 | 本规格采用 |
| 全部 JSON 改唯一临时名并重试替换 | 扩大影响面；唯一名称不能解决目的文件共享问题，重试改变既定失败规则 | 不采用 |
| 改成追加日志/不可变快照，并重写观察器读协议 | 可重新设计读写关系，但涉及新协议及更广运行验证 | 本轮不采用，另题评估 |

本工作包限定三个现有文件：`scripts/check_msb_native_supervision.py`、`tests/test_check_msb_native_supervision.py`、`tests/run_msb_native_supervision_synthetic_acceptance.py`。不新增模块、不修改底层 MSB/LC 执行器、collector 观察器停止语义、原生协议或 CLI 授权门。

## 3. trace 专用真实 helper

保留通用 `_write_json` 的现有实现与调用契约；新增 `_write_trace_json(path: Path, events: list, *, file_io=None) -> None`，只由 `NativeServices.persist_trace` 接入。这样不会改变根状态、ACK 和其他回执的发布行为。

- `_TraceFileIO.open_exclusive(path)` 返回文本流，原生默认仍用 `Path.open('x', encoding='utf-8')`；`replace(source, target)` 仍用 `Path.replace`。测试必须使用该实际 helper 的函数体，但将此适配层完全替换为内存假文件系统，不能实例化 `NativeServices` 或调用真实 Path I/O。
- 先以原有 `ensure_ascii=True`、`allow_nan=False` 规则生成一次完整 JSON 文本和末尾换行，再打开同一目录的固定 `<name>.pending`。正常字节格式与旧 trace 相同，不改成直写目的文件。
- 固定阶段为 `SERIALIZE`、`CREATE`、`WRITE`、`CLOSE`、`REPLACE`。写返回长度必须等于文本长度，短写按 WRITE 失败处理。
- 流一旦获得，关闭至多一次；WRITE 失败仍尝试这一次 CLOSE。WRITE 与 CLOSE 双异常都保留并按发生顺序记录，WRITE 仍为首发。CREATE 失败没有获得流时不得假装关闭或拥有既有文件。
- 只在 WRITE 和 CLOSE 均成功后调用 REPLACE 一次。任阶段失败后不重试、不 unlink、不覆盖既有 `.pending`、不额外查询存在性；保留完整或部分原件，文件是否已创建不得超出返回证据推断。
- 抛出 `TracePublicationError(RuntimeError)`，构造参数及属性 `errors: tuple[tuple[str, BaseException], ...]` 保有有序阶段与原异常对象，`first_error` 返回第一原异常；`__str__` 固定为不含原消息的安全摘要。`record()` 返回 errors（phase 加脱敏错误记录）及 first_phase。记录中 error type/message/frames 复用 `_fault_error_record` 的现有脱敏，再补 OSError 的白名单数值 `errno`（0--4095）和 `winerror`（0--65535）；非整数、bool、越界或不存在均记 null。

## 4. 首次失败锁存

新增无参构造的 `TracePublicationState()`，每个 controller 一个实例，由 `TraceAPI` 与 controller finally 共用，不跨进程共享、不新增锁或线程。`TraceAPI.__init__` 只增加末尾可选 `publication=None`，未传时创建局部状态；run_controller 必须显式传入其共享实例。

- `persist(path, events, *, services) -> None`：健康时调用原服务一次，增加尝试计数；成功计数仅在服务返回后递增。首次异常保存对象，状态变为 FAILED，再抛安全类型错误让原 MSB 清理路径运行。
- 服务若抛出非类型化异常，阶段只记 `ADAPTER_UNKNOWN`，不得伪造 CREATE/REPLACE 原因。
- `first_failure` 为 `TracePublicationError | None`；非类型化服务异常须封装并保留其原对象。首次失败后，后续 persist 仅重抛同一 first_failure，不得访问文件或服务；`finish(path, events, *, services) -> None` 在 FAILED 时只标记最终发布因先前失败跳过，不能再创建同一 `.pending`。健康时 finish 仅尝试一次最终发布，捕获该次 TracePublicationError 入同一锁存而不再次抛出，由 controller 随后统一裁定；不得在 finally 提前覆盖 body 异常。
- `record() -> dict` 为纯内存读取，包含 state、attempts、successful_publications、first_failure、final_attempted、final_skipped_after_error。它不采样时钟、不查询文件/进程、不重新序列化 trace。
- `TraceAPI.pause` 使用共享状态的 persist；不改 terminate_job、kill_process、wait/句柄清理、租期和原始事件顺序。N4 硬退出仍不制造 finally 已运行的证据。

## 5. 内层执行与最终发布错误分离

`run_controller` 保留原 `row`、body 异常、原一次 fault-stop 的结果/异常、parent 现有缓存证据及 trace 状态。最终 trace 异常不得覆盖这些对象，也不得跳过原清理。parent 证据只读已有缓存；不新增进程、线程、guard、时钟源或周期采样。新增失败回执 I/O 只允许复用既有 `_remaining(binding, services)` 在写前及写后各检查一次，仍属原累计期限；写前失败则跳过写入和写后检查。不创建或重置预算，不做额外计时探查。

有 trace 失败时：

1. 不创建正常 `controller_result.json`，不把诊断回执当正常执行完成。
2. 汇总为 `ControllerTracePublicationError`。其 `record()` 的固定字段为 status、case_id、binding、inner_summary、parent_power、body_errors、fault_stop、trace_publication、receipt_publication。status 固定 `FAILED_CONTROLLER_TRACE_PUBLICATION`。
3. `inner_summary` 仅保留已获得 row 的 status、脱敏 error、exit_code、cleanup_confirmed、exit_confirmed、end_handshake_verified、lease_sequence；row 缺失则为 null。parent_power 仅复制现有 guard 缓存的既有电源证据字段，不获新 guard，不由“执行过 finally”推断释放。
4. 在原剩余预算允许时，将该安全 record 独占写入新的 `controller/controller_failure.json` 一次。使用原 `_write_json(..., exclusive=True)`，不用 trace 临时路径。预算耗尽则不写，标记 `SKIPPED_BUDGET_EXHAUSTED`；写入错误独立留在 receipt_publication。不得重试或续写旧目录。
5. 新异常的 CLI 分支仅输出脱敏 record，并返回 1；避免隐式异常链将原始异常消息重新输出。不根据回执是否成功写入改变失败性质；若输出通道也失败，保留非零失败，不假造送达。

`_load_case_record` 附带读取可选 `controller_failure`，使 collector 保存可见的失败回执；`_check_case` 对非 null 的 controller_failure 明确拒收，不能因伪造 COMPLETE 外层或旧 trace 而通过。正常及既定 N2/N3 故障场景的原接受规则、正常回执字段与退出码不变。只有新 trace 发布失败进入上述新错误路径；其他既有错误语义不扩写。

## 6. 合成验证边界

未来开发仅允许原生会话内实施，不派代理；进程、线程 start/join、时钟、睡眠、电源、网络、runtime 查询和本轮 trace 文件适配层全部模拟。不读真实 prepared/fit/stop、原图或模型权重，不真实训练或模型前向；原允许的 scratch 合成张量测试边界不得扩大。

新固定根预留 `outputs/msb_trace_publication_retention_synthetic_v1/`，scratch 预留 `.superpowers/sdd/2026-10-23-msb-trace-publication-retention/test-tmp`，当前不创建。沿用 `collector-red`、`collector-green`、`final` 三阶段，分别最多 3 次独占尝试，失败尝试计数，不覆盖或续跑旧目录。阶段预算不是自动重试许可，遇越界或超出三次即停止。

保持既有十个 final 模块、56 路径保护、原三文件可变白名单和所有禁止接口守卫；只有 output/scratch 与本次必要测试名/身份期望适配。不得修改 `_acceptance_passed` 或放宽通过标准。拟新增十四个完整测试方法名，原 253 名必须保留，目标 267 只是未来验收期望，不是本轮结果。精确清单见配套计划。

有效 RED 必须显示缺少本规格行为的断言失败，不能把导入错误、守卫逃逸或未模拟接口调用算成有效 RED。GREEN/final 必须零失败、错误、跳过、预期失败、意外成功和未模拟边界调用；final 还须验证原 253 完整名包含关系及十四个新名，不以总数代替。

## 7. 接收与停止标准

验收应证明：实际 helper 五阶段和双异常有证据；首次失败后零第二次 I/O；旧 `.pending` 不变；内层/body/trace/receipt 原因分离；已有 cleanup/期限和失败关闭不变；失败诊断不能让 case 通过；正常场景保持契约。模拟时必须包括目的替换抛共享错误，但其通过不等于确认本机首发共享冲突或原生兼容。

出现第四文件需求、新执行器、扩大依赖/bootstrap、真实接口调用、漏失首因或能力信息外泄时停止并记录，不擅自扩范围。开发合成通过后只能归档作者结果；独立复查、任何新原生协议/运行和真实科学实验分别另批。旧 v2 许可/目录仍已消费，当前科学状态仍为未验证。
