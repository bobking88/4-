# MSB Trace Publication Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task only after separate human approval. Native inline execution is proposed; no agents are authorized. All unchecked steps are future work, not permission to execute now.

**Goal:** 保留首次 trace 发布失败及内层结果，防止 finally 重复创建同一临时文件遮盖原因。

**Architecture:** trace 专用真实 helper 提供分阶段异常，controller 共享首次失败锁存。失败时另存安全诊断、非零退出，collector 只能记录而不能据此接收。通用 JSON、底层监督、网络及原生协议不变。

**Tech Stack:** 既有 Python 3.11、unittest/mock、原三阶段合成验收器；不加依赖。

**Spec:** `docs/superpowers/specs/2026-10-23-msb-trace-publication-retention-design.md`，当前待审核；本计划为同一次仅文档授权下的配套草案。

## Global Constraints

- 当前 `DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`；人类仅批准文档，不批准以下步骤执行。
- 未来只改 driver、driver 测试和原合成验收器三个指定文件；不新增模块或改原生协议。
- 原生会话内实施，不派代理；受限进程/线程/时钟/电源/网络/runtime 及 trace I/O 全模拟。
- 不读真实 prepared/fit/stop、原图、模型权重，不真实训练或前向，不改网络、损失、公式、科学实验矩阵或正式报告。
- 固定新 output/scratch，三阶段各最多 3 次独占尝试，保留所有失败原件；不得扩模块或放宽 `_acceptance_passed`。
- 保留旧 253 完整名，加固定 14 名，未来 final 目标 267；当前不作通过声明。
- 原累计期限、一次停止、清理和严格身份门不变；任何原生或真实科学运行另批。

## Review Focus

1. 已有 pending 的 CREATE 失败不能删除/覆盖不归本次所有的证据：测试 T01/T04。
2. WRITE 与 CLOSE 同时失败，首发顺序和两原对象不能丢失：T02/T03。
3. body/row、最终 trace 和失败回执本身同时异常，不得互相遮盖：T08/T09。
4. 预算耗尽及清理失败时，诊断输出不能新增监督动作或假造完整释放：T11。
5. 能力信息出现在异常文本/行内数字、伪造正常外层加失败回执时，不能泄露或误通过：T10/T12。

## Task 1: 三文件最小实施与有限合成验收

**Files:**
- Modify: `scripts/check_msb_native_supervision.py`，trace helper/状态、run_controller、失败回执读拒收和新错误 CLI 分支。
- Test: `tests/test_check_msb_native_supervision.py`，新增十四方法，旧测试不得删改名。
- Modify: `tests/run_msb_native_supervision_synthetic_acceptance.py`，仅本次路径及必要固定验收身份适配，不改 guard/模块/通过谓词。

**Interfaces:**
- `_TraceFileIO.open_exclusive(path)` 与 `replace(source, target)`：唯一 trace OS 文件适配边界，原生默认与原 Path 行为一致，测试使用内存 fake。
- `_write_trace_json(path: Path, events: list, *, file_io=None) -> None`：实际 helper；五阶段错误进入 `TracePublicationError`，保留有序原异常对象；WRITE 失败仍一次 CLOSE，不再 REPLACE。
- `TracePublicationState.persist(path, events, *, services) -> None`、`finish(path, events, *, services) -> None`、`record() -> dict`：同一 controller 共享，首次错误锁存后无第二次服务/I/O。
- `TracePublicationError(errors: tuple[tuple[str, BaseException], ...])` 的 errors/first_error 和 `record() -> dict`，以及 `ControllerTracePublicationError.record() -> dict`：按规格脱敏及字段白名单；保留内存原异常对象，字符串摘要不得带原消息。TracePublicationState.first_failure 为前者或 null，finish 捕获入锁存而不再抛出；TraceAPI 构造末尾可选 publication=None，由 controller 显式传共享实例。
- `NativeServices.persist_trace` 接实际 helper；新的控制器测试通过这个原方法与 fake 文件适配层接线，不能继续只用直写替身证明覆盖。
- 正常 controller_result 契约不变；新 controller_failure 仅表示失败，collector 明确拒收。

### 固定测试清单

以下方法置于 `TracePublicationRetentionTests` 类，除表列参数子情形外不增加或替换完整方法名。完整名以前缀 `test_check_msb_native_supervision.TracePublicationRetentionTests.` 拼接。

| ID | 新方法名 | 必须固定的断言 |
| --- | --- | --- |
| T01 | test_trace_helper_records_create_failure_and_preserves_existing_pending | 原异常对象相同；阶段 CREATE；既有 pending 字节不变；零 write/close/replace |
| T02 | test_trace_helper_records_write_close_replace_and_short_write_failures | 分别注入 WRITE/CLOSE/REPLACE/短写；首发阶段正确；关闭次数至多 1；失败后不替换 |
| T03 | test_trace_helper_retains_write_and_close_errors_in_order | errors 的原对象顺序为 write、close；首发仍 WRITE；零 replace |
| T04 | test_trace_failure_latch_prevents_second_io_and_preserves_pending | 首次失败后调用 persist/finish；服务及 I/O 计数不增加；final_skipped_after_error=True |
| T05 | test_native_persist_trace_uses_real_helper_with_only_fake_file_io | 调用原 NativeServices.persist_trace 方法而不构造原生服务；命中实际 helper；所有文件操作属于 fake |
| T06 | test_trace_success_preserves_serialize_write_close_replace_order | 合成五事件文本与原格式相同；CREATE→WRITE→CLOSE→REPLACE；最终路径只出现完整文本 |
| T07 | test_controller_first_trace_failure_survives_inner_row_conversion | parent 转换成 row.error 后失败回执仍保存原 trace 阶段/对象；finally 不再打开 pending；无正常回执 |
| T08 | test_controller_final_trace_failure_keeps_inner_row_and_body_errors | 正常 row＋最终 trace 失败、body＋最终 trace 失败两情形；所有已取得原因按角色分别存在 |
| T09 | test_controller_failure_receipt_error_is_distinct_and_fail_closed | 独立回执再失败；原错误仍在，receipt_error 独立；只尝试一次；CLI 非零且输出脱敏摘要 |
| T10 | test_collector_retains_controller_failure_but_never_qualifies_case | load 后含 failure；即使伪造 outer COMPLETE，check 仍拒收；缺失败文件时旧契约不变 |
| T11 | test_controller_trace_failure_keeps_existing_cleanup_and_deadline | cleanup/fault-stop 原次数不增加；预算耗尽不写回执；仅原 _remaining 写前/后各最多一次，无新 guard/预算/时钟源或周期采样；缺内层证据仍 null |
| T12 | test_controller_failure_record_redacts_capabilities_and_whitelists_os_codes | nonce/handles/env 原文不输出；原错误对象内存仍在；errno/winerror 边界、bool、越界/null 按规格 |
| T13 | test_trace_serialization_failure_opens_no_file | 合成 NaN 或不可序列化值导致 SERIALIZE；零 open/write/close/replace，不删原件 |
| T14 | test_normal_and_expected_fault_controller_contracts_remain_unchanged | 原 N1/N2/N3 正常或预期故障契约、退出码及 normal record 不变；新失败回执不伪造 N4 完成 |

### 执行步骤，当前全部待批准

- [ ] **Step 1: 冻结实施起点与保护。** 审核批准后先核验届时精确文档提交、空 index、三文件白名单和历史 56 路径；保留现有六张脏 PNG 和正式报告哈希。新根不存在才可继续；不触碰旧 v2 原件或协议。
- [ ] **Step 2: 写十四个反例。** 只在上述类新增方法；新接口缺失先用断言给出明确缺行为信息，不用 NameError/import error 代替 RED。先写 fake 文件适配层和有序异常断言，再写控制器/collector/CLI 联合反例；不得导入原生服务构造器。
- [ ] **Step 3: 准备原验收器。** 将 output/scratch 指向规格固定新根；维持三 phase、十模块、56 路径和 guard。确认测试项目仍在 guard 下导入，数值 bootstrap 不扩容；不调用原生API检查环境。
- [ ] **Step 4: 执行 collector-red 一次并读原件。** 仅在另获开发/合成批准后，执行下列 RED 命令；必须有对应本规格的断言失败，零错误、跳过或未模拟调用。原始失败日志即使空白亦保留，不重写；异常验收器失败仅在同阶段三次总限内处理，不自动改标准。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-red
```

- [ ] **Step 5: 最小实施。** 依规格接入 trace helper、共享失败锁存、独立 controller_failure 与安全 CLI；原 `_write_json`、底层执行器和无新 trace 错误时的路径保持不动。不得靠删除 pending、改成 w、重试替换或放宽租期通过测试。
- [ ] **Step 6: 执行 collector-green 并核对十四新名。** 执行下列 GREEN 命令，要求零失败/错误/跳过/预期失败/意外成功/未模拟调用；核对真实 helper 的 fake 适配调用顺序，不能只看总数。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-green
```

- [ ] **Step 7: 执行 final，核对完整名称与全部原件。** 执行下列命令。未来期望 final=267；从原253日志和新14清单核对集合关系，不能只用267替代。检查全部失败尝试、snapshot、受限调用记录及三阶段最多3次计数；不额外运行 pytest、单测、计时或原生探查。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase final
```

- [ ] **Step 8: 作者静态复核与保护核对。** 查五个 Review Focus 的测试名和断言、错误脱敏、保留顺序、正常契约以及无额外监督动作。再次 hash 核对原件、底层源码、网络、矩阵、两处报告和六张脏图。发现第四文件或新原生需求即停，不能自行扩范围。
- [ ] **Step 9: 按精确白名单提交及 GitHub 实际内容回读。** 仅归档三个修改源码和新合成/保护/作者记录。全部成功才可标 AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING；不得标原生通过、科学有效或独立 APPROVE。独立复查及新原生/真实实验必须另批，不续跑旧 v2。

## 当前文档自检与交接

三文件范围、状态、接口、十四测试名、阶段数/尝试上限、数据和原生边界已逐条对照；没有已执行步骤或缺省运行许可。新文件路径仅为未来预留，当前未创建 synthetic 或 scratch 根。只读文档自检不等于合成验收或独立工程复查。

请人类审核本规格与计划。建议以后选择原生会话内实施这一单工作包，避免扩展并发或再次改底层监督；当前不开始实施，也不派发独立复查。
