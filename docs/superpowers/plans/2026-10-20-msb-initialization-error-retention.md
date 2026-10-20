# MSB初始化异常保留修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 当前仅起草文档，不实施、不运行、不派代理。

**Goal:** 修复已确认R4/P2初始化检查与guard释放双异常的结构化原因丢失，不改变原验收或科学合同。
**Architecture:** 将既有body try上移至state建立后，复用现有body/外层release/末端错误聚合与发布器。只更改捕获范围，测试注入实际合成RUNNING写后的故障；不重写guard或状态机。
**Tech Stack:** 既有Python、unittest、mock、合成JSON/原guard+fake API；不安装依赖。
**Spec:** [最小修复规格](../specs/2026-10-20-msb-initialization-error-retention-design.md)。
**Baseline:** `11f6ab75ea5482dd72c5f6c90aaba5e72c3af1f7`，R4 PARTIAL/UNFIXED；仅文档授权`call_754f23a7456b4c5cbe8e744ba0dd6114`。

## Global Constraints

- 当前只提交规格、计划、作者文档记录及README；写两份文档的直接授权不等于认可尚未存在的文件或开发许可。
- 未来仅改原三验收文件，原43执行源码/10测试、网络/损失/公式/矩阵及正式报告不改；全旧230名称保留。
- 新根`outputs/msb_initialization_error_retention_synthetic_v1`；scratch `.superpowers/sdd/2026-10-20-msb-initialization-error-retention/test-tmp`。
- 固定三阶段collector-red、collector-green、final；每阶段最多3次独占attempt含失败，保留全部原件。其余四phase不运行，不扩模块/网格，不追加任务完成探查。
- 保持原480秒/30秒gap/2秒漂移、N1--N4和7200秒科学边界；不重做原生/电源/空闲/进程验收或真实实验。
- 复用所有fake/guard及torch/sklearn bootstrap，保留monotonic守卫、CLI初始时钟注入；不读真实prepared/fit/stop、图片、缓存或权重。
- 当前不创建新合成根/scratch/原生协议/原生输出，不运行测试、前向、拟合或计时；未来开发和验收须先获得新的明确批准。
- 建议后续由当前会话原生实施这一项任务；不自动调用代理或独立复查，执行方式随下一次明确批准确认。

## Review Focus

1. 首次RUNNING真的已写入后check失败，不能被“写入前失败”夹具代替；由Step1第1、2、4反例锁定。
2. body与release两种原始对象同时存在，再叠加一次失败持久化；Step1第2反例锁定对象identity与发布次数。
3. state已建立但初始化文件写失败，不能丢失前因或伪造RUNNING持久化；Step1第3反例锁定。
4. state建立前的拒绝不能为留失败回执创建目录；Step1第5及既有许可拒绝子例锁定。
5. CLI初始monotonic不能重现历史未守卫真实调用；Step1第4和最终严格门锁定，不能用旧summary的0作完全模拟证明。

---

### Task 1: 初始化捕获边界及有限合成接收

**Files:**
- Modify: `scripts/check_msb_native_supervision.py`，仅NON_NATIVE_IDS及run_acceptance内body try起点。
- Test: `tests/test_check_msb_native_supervision.py`，新增固定五方法；同步隔离验收门的根路径，既有拒绝方法追加两个已知ID子例。
- Modify: `tests/run_msb_native_supervision_synthetic_acceptance.py`，仅新输出/scratch路径，不改PHASES、模块、守卫或bootstrap。
- Create later: 新合成根下各attempt原始summary/log/source_snapshot、final来源/保护清单及作者验收记录；只有获批后才创建。

**Interfaces:**
- Consumes: 原`FakeCollectorServices`、`CollectorTests.protocol_file(tmp)`、`CollectorEnvironment`；原实际合成write_json、guard/check及publish_acceptance合同。
- Produces: 原签名run_acceptance，返回失败errors为body在前、collector_scope_or_release在后；原AcceptancePublicationError接口与原始对象顺序。
- 不增加生产helper/public字段、运行许可、状态或新阶段。

- [ ] **Step 1: 写五个具体反例并登记新根。** 按规格第4节逐一写下列方法；只为真实调用加局部观察/注入包装，不替换错误聚合结果。更新两根路径及同文件隔离夹具预期路径；加入本文两个非原生ID拒绝子例。保护全部旧名称，先不改生产捕获范围。

| 方法名 | 精确注入及关键断言 |
|---|---|
| test_initial_running_check_failure_and_release_preserve_both_causes | 原write_json返回且读回RUNNING/active_case=None后，分别设ac=False、api.now+=31000、api.now=binding.start_tick_ms+480000；release_error=True。errors前两phase为body/release，首项等于实际检查异常type/message，status失败、delivery=False、case_records=[]、launch/observe为0、released=False。 |
| test_initial_running_triple_failure_preserves_exception_objects | 同一AC窗口+释放失败；仅FAILED收据写时抛已知OSError。断言primary_errors前两项is捕获的check/release对象、primary_error is首项、persistence_error is注入对象、失败持久化尝试1次、failed_state非PASS。 |
| test_initial_write_failure_and_release_preserve_both_causes | registered_protocol及首次RUNNING两路径写入前抛独立已知OSError+释放失败；原write错误排在release之前，无case，不伪造磁盘RUNNING存在。 |
| test_initial_running_failure_cli_is_nonzero_and_clock_injected | main真实路径、NativeServices工厂与monotonic明确fake；实际原guard+fake环境；双失败code=1，stdout JSON保留两因，入口时钟call_count=1，无launch。 |
| test_pre_state_rejection_preserves_no_output_contract | registration_ok=False、sources_ok=False及fake guard进入失败各子例；保持原异常、output不存在、write/launch/observe=0。 |

返回型测试按type/message断言；抛出型测试按对象is断言，捕获后必须原样raise。失败收据写入注入不重复触发初始化故障。正常四case正例和既有联合错误测试保留。

- [ ] **Step 2: 运行collector-red取得有效红灯。** 仅在新开发批准后运行下方入口；预期新双/三异常保留断言失败，旧73项通过。模块导入、夹具或意外真实调用失败计入尝试额度，但不冒充有效功能RED；最多3次，原件不覆盖。发现需要扩写旧43源码/新模块/原生权限即停。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' '.\tests\run_msb_native_supervision_synthetic_acceptance.py' --phase collector-red
```

- [ ] **Step 3: 最小实施捕获上移。** 在state字典成功建立后开始现有body try，包住registered_protocol写入、首次RUNNING写入/check及原case循环。保留原except/body对象、外层release对象、postflight、publish_acceptance及primary_errors重包顺序；不挖掘异常链、不新增状态/重试。NON_NATIVE_IDS仅追加规格中的两个已知文档/只读ID。

- [ ] **Step 4: 运行collector-green与固定final。** 同一入口分别传下方phase，每phase最多3次，失败占额度；不运行contract/worker阶段、不unittest discover。保留所有旧230全名及固定五新名；预计驱动78项、final235项，以真实日志而非预填数为准，最终0非通过项和0受限未模拟调用。直接快照必须是实际运行前读取的三文件，不能事后伪造；旧身份/历史时钟缺口保留。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' '.\tests\run_msb_native_supervision_synthetic_acceptance.py' --phase collector-green
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' '.\tests\run_msb_native_supervision_synthetic_acceptance.py' --phase final
```

- [ ] **Step 5: 作者静态接收并归档。** 核对源码/测试/历史证据/正式报告/六张脏PNG保护哈希、实际五新方法与原230名、三个phase尝试上限、目录独占、直接快照及所有失败原件；仅作者接收。代码、原始合成证据、来源身份、作者说明及README用显式白名单提交既有分支，GitHub独立读回实际内容；不自动派发独立审查、不合并/清理worktree。有限失败或证据不足就保留REQUEST_CHANGES，禁止扩额度重试。真实及原生运行仍分别另批。

## 文档自审与交接

规格各节由此单任务覆盖；五项Review Focus逐一对应Step1，签名/错误phase/目录/235预计计数一致。计划没有未决占位符；当前五步均未开始。
本轮用户明确同时授权规格和计划起草，因此两份待审文档一并交付；不是将起草许可当开发许可。请先审核书面文件并另批原生实施及有限合成验收，当前不执行以上命令。
本任务只改善错误证据完整性，未验证MSB-RRE角色证据、间隔预算或强对照收益，不将工程进展写成报告/论文理论创新结果。
