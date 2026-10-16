# MSB Execution Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after separate approval. Native execution is the preserved user preference. Steps use checkbox syntax; all implementation steps below remain unstarted.

**Goal:** 修复活跃父监督与继承累计期限、严格 fit-only 语义、拟合前联合三折门，保持 MSB-RRE 的网络理论和实验矩阵不变。

**Architecture:** 新增 MSB 专属安全模块，以父进程句柄、匿名控制管道、Job Object 及固定入口时钟约束 worker。数据层输出受限身份摘要，由父与子共用联合校验器，在原三次 load 和第一次 fit 之间建立验收门。

**Tech Stack:** 既有 Windows / CPython 3.11 训练环境、stdlib ctypes / subprocess / unittest.mock、现有 PyTorch CPU float64；不安装依赖。

**Spec:** [修复规格](../specs/2026-10-16-msb-execution-contract-repair-design.md)。科学合同另见 [MSB 原规格](../specs/2026-10-14_margin_scaled_bilinear_role_evidence_design.md)。基线提交 `3cc8856dbe1778fb4c1c9e1c7803c05227f32ffc`。

**Current Authorization:** `call_1690e5bbad7a415f8880b0ff473b9521` 只批准两份文档及 GitHub 归档。本计划不是开发许可，下面的命令本轮不执行；任务均未开始。未来实施须经用户审核并另批“原生代码与合成验收”。

## Global Constraints

- 不读取真实 prepared / fit / stop、原图、数据集或缓存；测试只在 TemporaryDirectory 创建 synthetic-* fixture。
- 不实际运行 Popen、计时、休眠、空闲、电源或 Job Object 验收；原生 API 通过注入 fake 测试；不安装依赖。
- 网络、损失、公式、种子、优化器、主点、强对照求解器与晋级门不变；不重绘结构图、不改正式报告。
- 原科学任务数量固定 80：3 load + 63 neural + 12 convex + analysis + final-verify；0 新 PCA/T0；75 次拟合上限不变。
- 340 个既有 fit 分组，[66,130,97,47]；输入 h64 / e23 / log_anchor4，标签顺序 0/1/2/3 不变。
- 固定累计 7200 秒，入口 start + 7_200_000 ms，所有 worker 继承；清理不授予额外计算预算。615 + 12 最终核验计数不变。
- 旧 34 个执行源码、原科学规格、83 项历史证据、六张用户 PNG 与两处正式 DOCX 均不改；失败目录不重试、不续跑。
- 新模块纳入未来 43 项闭包；本轮 documentation-only ID 不能转为真实许可；不创建真实协议。

## Review Focus

1. 文件 RUNNING 合法而原父进程已退出，或 PID 被复用：不能消费任务/读输入。Task 2 / 3 / 4 覆盖。
2. 准备耗尽预算、child guard 新建、deadline 恰好等于 now、最终导出时过期：同一累计期限且不成功交付。Task 2 / 4 覆盖。
3. Job 分配失败、控制管道丢失/重放、线程卡住或 kill/wait 未确认：不降级 launcher，不伪造释放/成功。Task 3 / 4 覆盖。
4. train 记录缺 scope / 其他 scope，但另一折 validation 同 ID 是 fit：必须在任何优化前失败。Task 1 / 4 覆盖。
5. 单折均合法，但三折 OOF 重复、标签/scope 不一致，或通过后摘要被替换：拟合计数为零，或后续 job 在加载本折前终止。Task 1 / 4 覆盖。

## Files And Interfaces

| 路径 | 职责 |
|---|---|
| scripts/msb_execution_safety.py（新增） | 预算、原生 API 薄适配、管道/Job 生命周期、父监督、子 scope |
| scripts/msb_role_data.py | 严格 scope、身份摘要及联合校验器 |
| scripts/run_msb_role_mechanism.py | 监督集成、load JSON、preflight 门、43 项闭包与协议合同 |
| scripts/msb_role_training.py | 只替换累计检查的导入，不改训练算法 |
| scripts/analyze_msb_role_mechanism.py | 活跃 scope / preflight 绑定，保留最终验收 |
| tests/test_msb_execution_safety.py（新增） | fake 时钟/原生句柄/管道/进程回归 |
| tests/test_msb_role_data.py、test_run_msb_role_mechanism.py、test_msb_role_training.py、test_analyze_msb_role_mechanism.py | 合成负例、任务顺序与计数、既有用例兼容 |
| tests/msb_synthetic_helpers.py | 仅在需要时增加纯合成身份/控制 fixture，不加真实路径 fallback |

未来证据使用独立 `outputs/msb_execution_repair_synthetic_v1/` 及 `docs/experiment_records/2026-10-16_msb_execution_contract_repair_acceptance.md`；这些文件本轮不生成。开发 ledger 用新的 `.superpowers/sdd/2026-10-16-msb-execution-contract-repair/`，不把旧 MSB 实施 ledger 当本计划的 ledger。

---

## Task 1: Strict Scope And Shared Joint Identity Validator

**Files:** Modify scripts/msb_role_data.py；Test tests/test_msb_role_data.py；必要时 Modify tests/msb_synthetic_helpers.py。

**Interfaces:**
- 保留 `validate_batch(batch: dict)->int`、`validate_prepared_fold(data: dict, fold: int)->dict` 和 `validate_fold_collection(folds: list[dict])->dict`。
- 新增 `prepared_identity_summary(data: dict, fold: int, entry: dict)->dict`：规格第 6.1 节的无张量摘要。
- 新增 `validate_identity_collection(summaries: list[dict])->dict`：返回 groups / counts / canonical_identity_sha256；validate_fold_collection 委托它，不比较不同折 h/e。

- [ ] **Step 1: 写失败用例。** 增加 `test_every_train_and_validation_row_requires_exact_fit_scope`，对 train/validation 分别注入 missing、None、空串、projector_stop、other、大小写和尾随空格；均断言 ValueError，原 fixture 通过。
- [ ] **Step 2: 运行该测试确认红灯。** 以下均为将来命令，工作目录为本 worktree：

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -m unittest discover -s tests -p test_msb_role_data.py -v
```

预期新增负例失败；记录失败内容，不先修改生产代码消除红灯。

- [ ] **Step 3: 最小实现严格语义与摘要接口。** scope 只精确比较，不填补；摘要字段集合、整数类型和规范身份 SHA 按规格；联合检查复用单一算法。
- [ ] **Step 4: 加入联合反例并验绿。** `test_joint_summary_rejects_locally_valid_oof_duplicate`：用另一合法折副本替换 fold1 的数据并改 inner_fold，使每折单验通过但 OOF 重复；`test_cross_fold_training_identity_and_scope_conflict`；`test_summary_schema_rejects_bool_indices_and_unknown_fields`；`test_fold_specific_features_are_not_required_equal`。断言 groups=340、counts=[66,130,97,47]，摘要无 torch.Tensor，不以输入摘要相同代替语义验收。再运行上面模块，全部通过。
- [ ] **Step 5: 审查并明确路径提交。** 仅 add 本任务文件，提交 `fix: enforce exact MSB fit scope and shared fold identities`；不 stage 用户 PNG 或无关文件。

## Task 2: Inherited Budget And Worker Capability Contracts

**Files:** Create scripts/msb_execution_safety.py；Create tests/test_msb_execution_safety.py。

**Interfaces:**
- `RunBudget(start_tick_ms: int, deadline_tick_ms: int)`（frozen）；`remaining_seconds(now_tick_ms: int)->float`、`elapsed_seconds(now_tick_ms: int)->float`，到期抛 TIMED_OUT。
- `NativeMSBAPI` 薄适配方法固定为 `tick_ms()->int`、`process_identity(handle)->tuple[int,int]`、`process_alive(handle)->bool`、`read_message(handle, max_bytes=8192)->dict|None`、`write_message(handle, message: dict)->None`；None 表示本次无消息，断管是异常，不混同。
- `MSBWorkerScope(binding: dict, *, api, awake_guard)` context manager：`check()->float`、`evidence()->dict`、`begin_work()->None`、`begin_publish(worker_power: dict)->None`。binding 只从握手构造，必须带 run/job/protocol/packet 身份、父句柄身份、原预算、已验 GRANT 和序号；不是给 JSON 直接加 verified=True 即放行。BINDING/WORKING/PUBLISHING/CLOSED 次序按规格；发布阶段继续监督但不调用已释放 awake guard。
- `msb_stage_elapsed_seconds()->float`：要求活跃 MSB scope；无 context 则拒绝，不回零。测试训练函数时显式 patch 此接口或绑定 fake scope，不加“synthetic ID 即跳过监督”的生产后门。

- [ ] **Step 1: 先写预算与权限失败测试。** `test_exact_deadline_and_noninteger_budget_rejected`、`test_new_child_guard_does_not_restart_budget`、`test_missing_parent_capability_cannot_be_recovered_from_packet`、`test_reused_pid_with_different_creation_time_rejected`、`test_renewal_cannot_extend_original_deadline_or_replay_sequence`；用固定 int tick，无真实采样和 sleep。
- [ ] **Step 2: 验红。** 将 Task1 命令中的 pattern 改为 `test_msb_execution_safety.py`；预期缺接口/断言失败，保存日志。
- [ ] **Step 3: 实现纯合同与消息边界。** start/deadline 差精确 7_200_000；lease 30_000 ms、renew 间隔上限 15_000 ms、deadline 不能延长。JSON 编解码拒绝重复键、非法 kind/字段、NaN/Infinity、超 8192 bytes、截断和倒序；不可使用 pickle。父进程创建时间只作身份，不能拿 UTC 计算预算。
- [ ] **Step 4: 验绿并补边界。** 增加 `test_parent_death_disconnect_and_expired_lease_revoke_scope`、`test_unbound_context_fails_instead_of_zero_elapsed`、`test_invalid_framing_and_duplicate_json_keys_fail_closed`、`test_publish_keeps_deadline_after_awake_release_without_reacquire`；fake awake guard 同时证明 owner/release 语义未被绕过。
- [ ] **Step 5: 提交。** `feat: define parent-bound MSB budget and worker scope`，只 add 两个新文件。

## Task 3: Native Adapter And MSB-Only Supervised Launcher

**Files:** Modify scripts/msb_execution_safety.py、tests/test_msb_execution_safety.py。

**Interfaces:**
- `MSBParentSupervisor(budget: RunBudget, run_id: str, protocol_digest: str, *, api=None)` context manager；`run_job(job: dict, protocol: dict)->dict` 保留旧结果字段并添加规格第 4.4 节证据。
- 扩展 NativeMSBAPI：`create_channel()->dict`、`start_worker(job, child_handles)->object`、`create_kill_job()->int`、`assign_job(job_handle, process)->None`、`terminate_job(job_handle)->None`、`wait_process(process, timeout_seconds: float)->bool`、`close_handle(handle)->None`。handle 为不透明整数，process 对象只在适配层使用。
- `open_worker_scope(packet_path: Path, handles: dict, *, api=None, awake_guard)->MSBWorkerScope`：只处理元数据/握手；工作体以后调用，先成功绑定再 task_started。

- [ ] **Step 1: 写次序/失败回归。** `test_no_grant_before_job_assignment`、`test_parent_exit_between_publish_and_launch_denies_manual_worker`、`test_parent_exit_before_assignment_never_enters_body`、`test_job_assignment_failure_kills_without_fallback`、`test_lost_ack_does_not_renew_or_launch_next_job`。fake 事件序列断言 GRANT 之前没有 body/data-open。
- [ ] **Step 2: 验红。** 运行 safety 模块，确认新 launcher 未实现的红灯，不进行原生进程探查。
- [ ] **Step 3: 实现单个受监督 launcher。** Popen 使用隐藏窗口、close_fds 和显式 handle_list；Job 句柄仅父持有，禁止 breakaway。启动后立即复原可继承标志并关闭父持有的子端副本；异常路径也关闭。子 bootstrap 等待绑定，父 assign 成功后才 GRANT；GetProcessId/GetProcessTimes/WaitForSingleObject 均经 API 适配，不以 PID 列表轮询替代句柄身份。
- [ ] **Step 4: 实现租约与清理，并验绿。** 单一读线程管理子控制通道，父主循环在接电/连续性检查后发送有序 RENEW；poll 目标 0.05 秒，子 watchdog 撤销后不可写成功结果。测试 `test_no_renewal_from_unconditional_background_thread`、`test_job_handle_not_inherited_and_all_channel_ends_closed`、`test_kill_wait_or_release_uncertain_cannot_complete`、`test_cleanup_after_deadline_cannot_grant_compute_time`；有疑点即失败，不加 fallback。
- [ ] **Step 5: 提交。** `feat: add fail-closed MSB parent supervision adapter`。该提交只代表实现及 fake 验收，不宣称 WinAPI 已本机实测；旧 lc_execution_safety.py / run_lc_role_adaptation.py 原样。

## Task 4: Preflight Gate And End-To-End Scope Integration

**Files:** Modify scripts/run_msb_role_mechanism.py、msb_role_training.py、analyze_msb_role_mechanism.py；Modify tests/test_run_msb_role_mechanism.py、test_msb_role_training.py、test_analyze_msb_role_mechanism.py。

**Interfaces:**
- 保留公共 `run_stage(root, protocol_path, output, stage)`；其第一项动作取得原始 tick 并构造固定预算；MSB 使用新 supervisor，旧 launcher 不再被 MSB 直接调用。
- 内部 `worker(packet_path, *, handles)` 必须实时握手；load 导出 JSON 摘要，analysis/final 也需相同 scope。公共 CLI 没有单独 verify / replay / resume。
- 新增 runner `write_joint_preflight(output: Path, protocol: dict, state: dict, supervisor)->dict`：仅在三 load 后联合校验/写证书。
- 新增 runner `require_joint_preflight(output: Path, protocol: dict, packet: dict, scope: MSBWorkerScope)->dict`：重新校验三 load 摘要及证书，在本折 prepared 加载前调用。
- `supervised_analysis(root, output, protocol, scope, kind)` 只接受活跃 MSBWorkerScope，原裸 WindowsAwakeGuard 不再足够；绑定 ContextVar 仍 once-only。训练只改时间检查导入到新模块。

- [ ] **Step 1: 先复现两项父门缺陷。** 写 `test_locally_valid_bad_threefold_preflight_has_zero_fit_calls` 和 `test_nonfit_training_record_fails_before_any_optimizer`；用 Task1 合成 fixture，patch load_prepared_fold 返回这些 fixture，fit_trajectory / fit_convex_fold / Adam 均为 fail-if-called。反例不得通过 mock 掩盖真实联合验证逻辑。
- [ ] **Step 2: 验红。** 使用 Task1 命令依次运行 runner/data 模块，保存旧代码在新用例上的失败证据，不跑原生 launcher 或真实 prepared。
- [ ] **Step 3: 集成先验收后拟合。** load result.json 由父按 hash 读回；三 load 全通过才写 exclusive joint_preflight.json 与 PASSED，然后生成 fit packet。证书字段/规范 SHA 按规格，无第 81 job。真实 prepared 仍由每个授权 worker 原路径/原 SHA 读取，没有父缓存和自动重建。
- [ ] **Step 4: 集成活跃期限到所有路径。** 源码闭包新增安全模块为 43 项；协议 schema 加 execution_contract / repair_spec_sha256，保留原科学规格 SHA；拒绝本轮文档许可及所有旧/代码许可。训练 timer 导入替换，analysis/final 以 scope.check 回调；结果写前/后、最终状态提交前也检查同一预算，不创建新 guard 原点。
- [ ] **Step 5: 验绿并扩展强反例。** `test_preflight_artifact_tamper_denied_before_fold_load`、`test_forged_passed_state_without_live_parent_denied`、`test_analysis_and_final_export_share_original_deadline`、`test_metadata_registration_time_is_not_refunded`、`test_preflight_scope_does_not_allow_raw_or_stop_fallback`。fake 完整 80 job 断言任务序列不变、三 load 后才出现首 fit、拟合元数据仍 63+12、预算单调减少、final budget 615+12，不将元数据计数说成实际训练。
- [ ] **Step 6: 回归现有测试的作用域。** 原 runner 15 项不因新参数消失；训练与分析单元测试显式使用 fake MSB scope/patch timer；不以全局跳过权限检查取得绿灯。运行四个被改模块及 safety 模块，全部通过。
- [ ] **Step 7: 提交。** `fix: gate MSB fitting on live scope and joint preflight`；逐文件 add，确认 model / controls / audit / figure、旧 34 源码和旧日志无 diff。

## Task 5: Scoped Synthetic Acceptance And Evidence Handoff

**Files:** Future Create outputs/msb_execution_repair_synthetic_v1/{synthetic_tests.log,source_identity.json,contract_summary.json,protected_artifact_preservation.json}；Future Create docs/experiment_records/2026-10-16_msb_execution_contract_repair_acceptance.md；Modify README.md 的进度段。

**Interfaces:** 无新科学接口。证据对象必须将 `scope=SYNTHETIC_CONTRACT_ONLY`、`native_os_exercised=False`、`real_data_opened=False` 与实际测试计数/源码身份分开记录，不能写入真实运行授权。

- [ ] **Step 1: 复核执行白名单。** 确认前四任务 red/green 日志可追溯；新测试模块不得直接 Popen/WinDLL/睡眠，既有 fixture 不读取项目路径；采用当前固定解释器，不扫描真实数据。
- [ ] **Step 2: 运行一次限定回归。** 下面命令是批准开发后才执行的命令；包括八个既有 MSB 模块加一个 safety 模块：

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -m unittest discover -s tests -p 'test_msb*.py' -v
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -m unittest discover -s tests -p 'test_run_msb_role_mechanism.py' -v
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -m unittest discover -s tests -p 'test_analyze_msb_role_mechanism.py' -v
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' -m unittest discover -s tests -p 'test_generate_msb_role_figure.py' -v
```

预期所有实际发现的用例通过，记录三个修复的反例结果与总数；第一条必须先静态列明匹配文件仅为五个既有 test_msb_role_* 加新 safety 模块，如出现额外模块则改用逐一明确模块命令而非扩大白名单。旧 83 项历史日志不覆盖，新增数量以本次实际结果为准。

- [ ] **Step 3: 登记保护证据与限制。** 核对旧 34 及未改科学源码、科学规格、用户 PNG、两处正式报告哈希；生成新 43 源码身份。contract_summary 记录 preflight 先于 fit、mock 三折失败拟合为零、原始预算不重置、所有失败不交付；明确原生 API/真实协议仍未验收。
- [ ] **Step 4: 自审并归档。** 静态 diff/链接/空白检查；只提交候选源码、对应测试及本次合成证据，按授权同步 GitHub，读回提交及文件；不创建 PR、不改旧证据、不误带 PNG/pt。
- [ ] **Step 5: 停在许可闸门。** 交付实际测试结论和未覆盖项。独立复查、原生无数据进程验收、新真实协议及真实运行均须另批；不因合成通过自动执行原计划 Task9。

## Document Self-Review And Execution Handoff

本轮已按规格逐项映射：E1 -> Tasks2/3/4，E2 -> Task1/4，E3 -> Task1/4，科学保护及证据限度 -> Task5。五项 Review Focus 均有具名反例，字段/函数名称在前后任务一致；新增模块导致 43 项闭包已明确，不伪称 42 项身份不变。

当前全部 checkbox 保持未开始。用户审核书面规格与计划后，再决定是否批准原生代码与合成验收；保留原生实施偏好，不自动派新代理或开始真实实验。修复的目的仍是为 MSB-RRE 理论机制提供可信验证条件，而不是用工程工作替代网络创新证据。
