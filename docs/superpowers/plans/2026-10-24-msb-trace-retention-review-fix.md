# MSB Trace Retention Review Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans only after separate human approval. Inline execution is proposed; no agents are authorized. All unchecked steps are future work, not permission to execute now.

**Goal:** 修复 R1--R3 并补联合异常/部分回执覆盖，不改变原监督策略或科学网络。

**Architecture:** 在原三文件内保守脱敏；预算错误列表由调用方持有，原停止异常及缓存证据按角色保留；失败回执先读进入只能拒收的诊断路径。正常 N4 严格校验、通用 JSON、底层执行器和模型不改。

**Tech Stack:** 既有 Python 3.11、unittest/mock、原三阶段合成验收器；无新依赖。

**Spec:** [配套规格](../specs/2026-10-24-msb-trace-retention-review-fix-design.md)。本计划为 `call_21f6de490d944c7bb9b0116fc5c3dda9` 的仅文档草案，与规格一并待审核；不是已批准实施。

## Global Constraints

- 当前 DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN；基线 cd9dcb2，受审源码 e2c9cc4；本轮无任何代码/测试/数据/原生/报告执行许可。
- 未来只改 driver、driver 测试和原合成验收器三个指定文件；无第四源码文件、无依赖/bootstrap 扩容。
- 原生会话内实施，不派代理；受限原生接口及 trace/receipt 故障 I/O 全模拟。
- 不读真实 prepared/fit/stop、原图、权重或缓存，不真实训练或科学前向，不改网络、损失、公式、矩阵或正式报告。
- 原累计期限、一次停止、句柄/guard 释放、严格身份及失败关闭不变；缓存仅作诊断。
- 新 output/scratch；三阶段各最多 3 次独占尝试，失败计数，保留全部原件，不自动重试。
- 保留十模块、56 路径、原 267 完整名和全部守卫；新增固定 10 名，未来 driver=116、final=277，不以数量代替断言。
- 旧原生 v2 许可/目录已消费；任何新原生或真实科学运行另批。

## Review Focus

1. 敏感值在标识之前/之后及各种真实分行后，输出不能残留，原对象不能改：Q01/Q02。
2. 空 errors 列表的同一性、预算前检与 stop_fault 双异常，不能因 tuple 未赋值而丢原因：Q03/Q04。
3. 缓存停止结果看似成功、callback 冲突或缓存缺失时，不能伪造停止成功或增加查询：Q04/Q05。
4. N4 缺快照且有失败/部分回执，诊断仍须进入汇总而非先抛快照错误：Q06/Q07。
5. row 已取得后 release/stop/trace/receipt 联合异常，及实际 helper 部分写入，不能互相覆盖或重试：Q08/Q09；旧正常门由 Q10 固定。

---

## Task 1: 三文件最小修复与有限合成验收

**Files:**
- Modify: `scripts/check_msb_native_supervision.py`，现有约 89--96、347--370、395--404、576--629、1152--1166 行。
- Test: `tests/test_check_msb_native_supervision.py`，新增 `TraceRetentionReviewFixTests`；旧类/方法不删、不改名。
- Modify: `tests/run_msb_native_supervision_synthetic_acceptance.py`，只改 output/scratch 字面路径；通过谓词/守卫/模块不改。

**Interfaces:**
- `_fault_error_record(error) -> dict`：规格第 3 节的整消息替换；原类型/帧/数值白名单不变。
- `_bounded_fault_stop(watch, binding, services, *, errors: list[BaseException] | None = None) -> tuple[FaultStopResult, list[BaseException]]`：可选调用方列表，未传时保持旧契约；不额外停止/采样。
- `ControllerTracePublicationError(case_id, binding, *, inner_row, parent_power, body_errors, fault_stop, trace_publication, fault_stop_evidence=None)`：末尾可选原对象证据，记录只导出安全白名单；cached_result 不能推断 stopped。
- `_read_controller_failure(path: Path) -> dict | None`：缺失为 None，其余失败非空；不带部分原文、不重读。
- `_load_case_record(case_dir, binding, outer, observation) -> dict`：先读失败；非空立即产 diagnostic_only 拒收记录，否则原 N4 路径不变。
- `_check_case(record, case_id, binding)` 的现有首条拒收不放宽；collector append/check 接线不改。

### 固定反例清单

完整名前缀固定为 `test_check_msb_native_supervision.TraceRetentionReviewFixTests.`。十个方法不得用总数相同的替换名充数；子情形不增加完整方法数。

| ID | 方法名 | 必须观察的断言 |
| --- | --- | --- |
| Q01 | test_error_redaction_drops_multiline_context_without_changing_original | 所有指定真实分行、大小写标识、敏感值位于标识前后均整消息替换；字面量反斜杠+n另测；普通单行仍沿用数值/截断；原对象不变 |
| Q02 | test_controller_diagnostics_and_cli_redact_all_multiline_error_roles | body/row/trace/预算/stop/cached callback/receipt 各角色的短合成敏感值不进入 record 或 CLI；原异常仍按角色保留、CLI=1 |
| Q03 | test_bounded_fault_stop_retains_caller_errors_when_stop_raises | 真实 helper；errors 空列表身份不变；预算前检原对象先记录、stop_fault 收到 0 且仅一次、抛原停止异常、无后检 |
| Q04 | test_n4_trace_failure_retains_budget_stop_callback_and_cached_snapshot | 真实 controller N4；前检耗尽+stop 异常+缓存 callback/snapshot+trace 失败；全部原引用保留；stopped=false，回执预算仍耗尽、不写、无再 join/query |
| Q05 | test_stop_capture_missing_or_conflicting_cache_never_promotes_success | 缺失/冲突缓存分别保存且不推断；停止抛错时原异常优先级不变；未传 errors 的旧返回/异常契约不变 |
| Q06 | test_n4_failure_receipt_precedes_snapshot_checks_and_is_rejected | 实际 loader 先读失败；无/不匹配快照均返回 diagnostic_only record，collector 先保存再拒收，后续例不启动；非空 failure 绝不能通过 |
| Q07 | test_partial_or_invalid_controller_receipt_remains_nonempty_failure | 实际读取 helper 处理部分/坏 JSON/null/数组/错误形状/读取错误；哨兵非空、安全、不回显原文；仅 FileNotFound 为 None；原件不改 |
| Q08 | test_actual_receipt_writer_preserves_partial_bytes_without_retry | 实际 _write_json 函数体、仅 receipt 流模拟；部分 write 后抛错或 close 抛错；一次独占打开、无补写/删除/重试；原前缀保留、receipt_error 独立 |
| Q09 | test_obtained_row_survives_scope_stop_trace_and_receipt_failures | row 返回后 __exit__ 抛错，N4 stop+callback、final trace、部分 receipt 再错；原 row/各原异常保留，trace 首错不改，CLI=1、无正常回执 |
| Q10 | test_absent_failure_keeps_legacy_case_and_stop_contracts | failure 缺失时 N1--N3 原契约、有效 N4 已存快照/前缀规则保持；无效 N4 仍拒收；正常 record 不含 diagnostic_only，不能模拟出“正常 N4 controller 返回” |

### 步骤，当前均未授权执行

- [ ] **Step 1: 冻结起点。** 开发批准后核验届时精确文档提交、空 index、三文件白名单和历史 56 路径；新 output/scratch 均不存在才继续。保留六张预存脏 PNG、两处报告哈希、旧原生及 267 日志原件。

- [ ] **Step 2: 写固定十个反例。** 使用已存 fixture，新增类；核心 Q01/Q03/Q04/Q06 要能在旧源码上给出缺行为断言。缺新函数/参数先用 callable/签名断言报告；旧 loader 在 Q06 抛快照异常须转为明确的缺诊断行为断言失败，不能让它成为测试 error。不能依赖 TypeError、NameError 或导入失败作为 RED。Q10 兼容反例可以在旧源码通过。

- [ ] **Step 3: 适配原验收器两处路径。** output 为 `outputs/msb_trace_retention_review_fix_synthetic_v1/`，scratch 为 `.superpowers/sdd/2026-10-24-msb-trace-retention-review-fix/test-tmp`。禁止改 _acceptance_passed、十模块/56 路径/bootstrap/守卫。冻结 RED 后测试清单；必要断言问题只能在阶段总次数内记录，不自动扩大范围。

- [ ] **Step 4: collector-red。** 仅在另获开发/合成批准后执行下列入口一次。未来 driver 应为 116；核心 R1/R2/R3 均有对应有效断言失败，旧 106 完整名保留；零错误/跳过/未模拟调用。若不是有效 RED，保存原件并按上限处理，不能宣称验证了修复。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-red
```

- [ ] **Step 5: 按规格最小实现。** 先改消息边界，再加调用方 errors 引用并保留停止原对象/缓存，最后先读失败回执走拒收支路。只改三个文件；不修改通用 _write_json、原停止线程/监督实现或原 N4 成功规则。不能以删除 pending、重试 I/O、重复 stop、重新采样或放宽期限过关。

- [ ] **Step 6: collector-green。** 运行下列入口；未来 116 项零失败/错误/跳过/预期失败/意外成功/未模拟调用。逐项核对 Q01--Q10 的真实 helper、原对象同一性、调用计数和部分字节，不用绿色总数替代。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-green
```

- [ ] **Step 7: final。** 仅运行下列原十模块入口；未来 277 项。以旧 final 日志全名集合加固定十新名精确比较，保留所有失败尝试、entry stdout/stderr、summary/snapshot/log。原 scratch 合成张量边界不扩，不另跑 pytest、单测或原生探查。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase final
```

- [ ] **Step 8: 作者静态核对。** 按五个 Review Focus 查验接线；核对三阶段次数、3 文件、旧原件/源闭包/报告/图片哈希。只有源码契约与完整证据同时满足，才能到 AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING；不标独立/原生/科学通过。

- [ ] **Step 9: 精确提交和远端回读。** 仅提交批准的代码及合成/作者证据，不带预存脏图或报告。GitHub 实际文件内容、commit/tree/ref 均匹配才报告同步完成。发现第四文件、未模拟接口、秘密外泄、原因丢失或超出尝试上限即停。独立复查及任何实际运行分别另批。

## 文档自检与交接

本轮只做源码/文本/哈希与既有记录核对：R1 对应 Q01/Q02，R2 对应 Q03--Q05，R3 对应 Q06/Q07，复合覆盖对应 Q08/Q09，兼容门对应 Q10。接口默认值、原列表身份、字段及失败状态在规格/计划一致；本轮没有执行任何上述步骤，没有创建预留 synthetic/scratch 根。

由于人类明确禁止代理，本轮不派 builder/reviewer；这份自检不是独立复查或合成验收。请审核规格和计划；建议以后仍在原生会话内实施这一单工作包，但当前只交付文档，不开始开发。工程门解除后仍须按科学协议检验 MSB-RRE 的网络机制、强对照与负结果，不把异常处理修改计为理论创新。
