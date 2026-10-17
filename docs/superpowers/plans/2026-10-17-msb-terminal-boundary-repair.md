# MSB Terminal Boundary Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after separate human approval. Native implementation is the preserved method; do not dispatch agents automatically. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复四项 P2 发布/清理/终结时序问题，为既定 MSB-RRE 强对照实验恢复可信的执行准备条件，不改研究假说。

**Architecture:** 在现有两项执行源码内增加启动所有权记录与显式终结状态机。父端单一成功发布，退出后有界排空；子端 watchdog 独占读取，所有控制写入和终结状态在同一锁下串行化。

**Tech Stack:** 现有 Python、unittest/mock、PowerShell、Git；原生 WinAPI 薄适配只用 fake，不增加依赖。

**Spec:** [2026-10-17-msb-terminal-boundary-repair-design.md](../specs/2026-10-17-msb-terminal-boundary-repair-design.md)

日期：2026-10-17。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档授权：`call_a697651b584d4fb8b2116607dcc82b52`；基线 `82ab7bfa935f9cb07919e1492d47db59e6b684f9`。
本轮按用户同时批准的“规格与实施计划”一并起草；书面文件仍待审核，不能把文档许可当成开发许可。

## Global Constraints

- 本轮不改代码、不跑测试或原生验收、不读真实数据、不训练、不改正式报告。
- 未来开发仅改 scripts/msb_execution_safety.py 与 scripts/run_msb_role_mechanism.py 两项执行源码；43 个执行源码闭包不增减；旧 34 源码和四个科学模块字节保护。
- 三个独立 rank4 分支；1083 当前参数及 1083 固定参照；h64/e23/log_anchor4；`R_ref=2*abs(m)+0.25`、`d=R_ref*tanh(s)`、损失/优化器/初始化不变。
- 340 个既有已暴露 fit 分组及 [66,130,97,47]；三份原登记 prepared；不读 stop/原图；0 次新 PCA/T0。
- 11 单元、3 折、三种子 20261002/20261003/20261004；63+12=75 拟合；400 步上限、100 步主点；80 job；615+12 最终核验；七参照晋级门不变。
- 原始累计预算 7200 秒；相等即过期；不新建时钟原点、不因握手/收尾延长租约或预算；失败清理 wait 至多 5 秒，不能用于计算。
- 控制帧单条最多 8192 bytes；新增 kind 仅 CLOSE_REQUEST / CLOSE_GRANT；单次排空最多 8 帧；目标轮询 0.05 秒；原最长租约 30 秒/续租间隔 15 秒不变。
- 旧 83/121 项证据、失败目录、旧验收器、六张用户 PNG 和两处正式报告不覆盖、不暂存、不改写。
- 以后每次合成运行均保存直接捕获的源码快照与身份；不以重建快照代替当时原件。不预报新通过总数，不进行真实实验重跑。
- 开发、独立复查、原生无数据验收、新真实协议登记与真实运行分别另批；不自动重复六分钟空闲或计时探查。

## File Structure

| 后续文件 | 职责 |
|---|---|
| Modify scripts/msb_execution_safety.py | WorkerLaunchOwnership、MSBParentControl、_finish、_drain_control、启动异常回收 |
| Modify scripts/run_msb_role_mechanism.py | _publish_terminal_state、终结证据集成、v2 执行身份及文档许可拒绝 |
| Modify tests/test_msb_execution_safety.py | 确定性双端交错、退出独立事件、启动异常 fake |
| Modify tests/test_run_msb_role_mechanism.py | 最后发布跨期、失败写入、版本/许可及全流程 fake |
| Create tests/run_msb_terminal_repair_acceptance.py | 固定阶段/模块守卫、独占合成输出、运行前源码快照 |
| Create outputs/msb_terminal_repair_synthetic_v1/ | 后续获准开发的 red/green/final 原始证据，当前不得创建验收结果 |
| Future Create docs/experiment_records/2026-10-17_msb_terminal_boundary_repair_acceptance.md | 实际验收结论及限制，只有实施后填写，不提前宣称通过 |

上述源码/测试条目仅为未来允许范围，本轮仅提交本计划、规格、文档阶段记录与 README。

## Review Focus

1. 只有最后成功写入跨过 deadline 或失败写入也失败：不能返回成功或谎报撤销已落盘；Task1。
2. Popen 返回后日志退出/中断/恢复标志相继抛错：保留对象与两类错误；不知道是否创建必须报清理不确定；Task2。
3. 子发送 DONE 并退出发生在父第一次读暂空之后：退出不得依赖父读消息触发，也不能因此误判合法结束；Task3。
4. CLOSE_REQUEST 与在途 RENEW 两个方向交错、以及请求引用再老序号：前两者成功，后者严格拒绝，不削弱重放校验；Task3。
5. 收尾失去父活性、租约/总期限到达、截断帧/排空超界或 watchdog 不退出：失败且无下一 job，不加等待额度；Tasks3/4。

---

## Task 1: Guarded Evidence Entry And One Terminal Publisher

**Files:** Create tests/run_msb_terminal_repair_acceptance.py；Modify tests/test_run_msb_role_mechanism.py、scripts/run_msb_role_mechanism.py。

**Interfaces:**
- `_publish_terminal_state(output: Path, state: dict, *, budget: RunBudget, api) -> dict`；仅接受已整理的终态候选，检查/原子写/再检查，不在成功尾部重复写。
- 新守卫入口 `main(argv: list[str] | None = None) -> int`，只接受 `--phase`。phase->modules 固定：task1-red/green 为 test_run_msb_role_mechanism；task2-red/green、task3-red/green 为 test_msb_execution_safety；final 为原九模块固定列表。
- 输出固定 outputs/msb_terminal_repair_synthetic_v1/<phase>/；目录或回执存在即拒绝覆盖。不得提供任意输出/模块/真实 stage 参数。
- 各阶段先以获批 scratch 创建临时目录；保存验收器和被测源码直接原文/哈希，再开始测试。标准运行库及既有 torch 的必要装载单列为 bootstrap；被测模块导入前先禁止真实 Popen/WinDLL/os._exit，安全模块仅加载定义后立即禁止 NativeMSBAPI 构造/tick，torch.load 白名单先于测试导入；不执行 import-time 原生探查。

- [ ] **Step 1: 建立守卫和发布反例。** 保留现有 simulate 的假 NativeMSBAPI 全局注入。新增 `test_only_final_success_write_crosses_deadline_is_revoked`：fake _json 在“终态 history 已整理”的成功写入中将 tick 推至 deadline，断言返回 TIMED_OUT、delivery_verified=False、CLI 非零且无后续成功写；不得 mock 掉被测终态处理。新增 `test_terminal_write_error_cannot_return_complete`、`test_failure_record_write_error_is_not_claimed_persisted`、`test_success_has_no_unchecked_tail_write`。
- [ ] **Step 2: 验红。** 执行下述 task1-red 命令。预期新跨期反例揭示当前公共尾部漏洞；原有 runner 合同不应失效。守卫自身错误不算有效红灯，保留失败后先停下处理，不用真实接口探查补救。
- [ ] **Step 3: 实现单一发布。** 整理 history/power/elapsed 后统一交给 _publish_terminal_state；把最终成功写及后检查放在异常撤销内，移除公共成功尾写。失败记录只用于失败清理，不能从旧 verified 标志恢复完成；写不出失败证据则非零并说明未持久化，不伪报原子撤销。
- [ ] **Step 4: 验绿。** task1-green；精确 deadline 与 deadline+1 都拒绝，deadline-1 且最终写后仍未过期可成功；成功返回前最后一个成功写有前后检查。
- [ ] **Step 5: 提交。** 仅逐文件暂存该任务实现/测试/新阶段证据；建议消息 `fix: guard the last MSB terminal publication`。旧日志不动，尚不宣称全部四项已修复。

未来批准后才执行：

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase task1-red
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase task1-green
```

## Task 2: Own A Launched Process Across Every Startup Exception

**Files:** Modify scripts/msb_execution_safety.py、tests/test_msb_execution_safety.py；沿用 Task1 守卫入口。

**Interfaces:**
- `WorkerLaunchOwnership()`：attempt_started=False、process=None、creation_uncertain=False；父端持有唯一实例。
- `WorkerLaunchOwnership.note_attempt() -> None` 只允许一次；`note_process(process: object) -> None` 记录同一已创建对象，拒绝被替换；错误处理不能清空。
- `NativeMSBAPI.start_worker(job: dict, child_handles: dict, *, ownership: WorkerLaunchOwnership) -> object`；所有 fake 同签名。
- _WorkerStartError 携带 ownership/可恢复 process、主异常及恢复异常；父 run_job 即使赋值未完成也能从同一记录进入清理。

- [ ] **Step 1: 写最小异常回归。** `test_log_exit_error_after_spawn_still_kills_and_waits`、`test_keyboard_interrupt_after_spawn_retains_ownership`、`test_restore_error_does_not_hide_primary_startup_error`：Popen fake 先返回对象，日志退出或启动后检查再抛错；断言 attempts=1、kill/Job terminate 与 wait 确认同一对象、句柄均被尝试释放、无第二次 Popen。只模拟日志上下文，不打开真实子进程。
- [ ] **Step 2: 加入不确定性反例并验红。** `test_attempt_without_recoverable_process_is_uncertain_not_clean` 断言 alive=True、cleanup_confirmed=False、creation_uncertain=True、SUPERVISION_INTERRUPTED；`test_failure_before_popen_has_zero_attempts` 断言日志打开前失败不冒称已创建。运行 task2-red，既有恢复异常回归保留。
- [ ] **Step 3: 实现所有权闭环。** 父端先创建 ownership，再调用适配器；Popen 前 note_attempt，获得对象后立即 note_process。整个启动/日志退出/恢复段捕获 BaseException，保留原错误及次级错误；父 finally 从记录恢复对象并 kill/wait，未知创建情况保守失败，不扫描 PID 或 fallback 启动。
- [ ] **Step 4: 验绿。** task2-green；同时核对 wait 失败、句柄关闭异常、Job 未分配/已分配两条清理路径都不 COMPLETE，后续 job 数为零。
- [ ] **Step 5: 提交。** 建议 `fix: retain MSB launch ownership through startup exceptions`，不更改旧 launcher 或网络。

未来命令与预期：`--phase task2-red` 返回新反例失败；`--phase task2-green` 返回该模块全部通过且未模拟边界调用为零。完整命令前缀与 Task1 相同。

## Task 3: Serialize The Closing Handshake And Drain Exited Workers

**Files:** Modify scripts/msb_execution_safety.py、tests/test_msb_execution_safety.py、scripts/run_msb_role_mechanism.py、tests/test_run_msb_role_mechanism.py。

**Interfaces:**
- `MSBParentControl(binding, *, lease_until_tick_ms: int, granted_tick_ms: int)`：在初始 GRANT 已确定后创建；issued=(1,lease)、last_acked=None、outstanding=True、phase=RUNNING、done=False；初始续租原点用 granted_tick_ms，不能从被 deadline 截短的 lease 倒推。
- `MSBParentControl.accept(message: dict, now_tick_ms: int) -> list[dict]`：验证绑定及有序 ACK/CLOSE_REQUEST/DONE，更新状态并返回至多一个 CLOSE_GRANT。
- `MSBParentControl.renew(now_tick_ms: int) -> list[dict]`：仅 RUNNING、当前已 ACK 且距上次发出 GRANT/RENEW 达到原续租间隔才返回一次 RENEW；任何 closing 阶段返回空，不延长固定 deadline。
- `MSBParentControl.require_complete(exit_code: int) -> None`：仅合法 DONE、当前 ACK、CLOSE_GRANTED 已完成且 exit_code==0 才通过。
- `MSBParentSupervisor._drain_control(channel: dict, control: MSBParentControl, binding, *, max_frames: int = 8) -> int`：最多 8 帧；每帧前后 check；即时无完整消息则返回，DONE 后不再读。
- `MSBWorkerScope._finish() -> None`：由正常 __exit__ 调用一次；进入 CLOSING、请求、等待唯一 reader 通知、串行 DONE、CLOSED；异常不走成功收尾。_pump 增加 CLOSE_GRANT 处理，仍独占读端。

- [ ] **Step 1: 建立独立事件 driver。** fake 双向队列、fake tick、process exit 事件分开；驱动器不得在 read_message(DONE) 内设 returncode。不 mock accept/renew/_finish 等被测核心，不真实创建线程/进程或调用 sleep；watchdog 调度由 fake API 的事件队列触发。
- [ ] **Step 2: 写两向交错和退出回归。** `test_request_before_inflight_renewal_waits_exact_ack`：请求 q、在途续租 q+1，缺 ACK 不给 CLOSE_GRANT；补 ACK 后 grant/done 必须为 q+1。`test_renewal_ack_before_close_request_finishes`：ACK q+1 后请求 q+1 正常结束。`test_exit_after_grant_drains_queued_done_before_failure`：父第一次读取暂空，子发送 DONE 且 exit=0 独立发生，父确认退出后再次排空，验证 COMPLETE。由于新协议不再允许无 CLOSE_GRANT 的旧 ACK/DONE 序列，不以放行旧协议恢复测试。
- [ ] **Step 3: 写严格拒绝及锁合同回归并验红。** `test_old_or_duplicate_close_request_is_rejected`、`test_done_before_close_grant_is_rejected`、`test_close_grant_wrong_pair_or_binding_is_rejected`、`test_closing_without_inflight_ack_cannot_complete`、`test_close_deadline_disconnect_and_watchdog_failure_stop_next_job`、`test_exit_drain_bound_or_truncated_frame_cannot_complete`；另以 `test_ack_request_done_share_state_lock_and_wait_releases_it` 的 fake锁/写端检查三种整帧写入均持有同一状态锁，等待事件时锁已释放且 fake watchdog 可推进。stage 为 task3-red。断言再老序号/错 lease、缺 ACK、非零退出、截断/超过8帧、deadline等号均失败，不通过扩大等待取得绿灯；fake调度不是原生线程调度证明。
- [ ] **Step 4: 实现纯状态机与两端接线。** 父记 last_acked/唯一 outstanding；请求冻结未来续租，精确在途 ACK 后 grant；主循环及观察退出后调用有界排空。子同一 RLock 保护 phase、序号、lease及所有 ACK/REQUEST/DONE 写入；等待事件时释放锁，watchdog 继续处理在途 RENEW，CLOSE_GRANT 后不得再接受续租。
- [ ] **Step 5: 固定结束证据。** 正常 COMPLETE 需 end_handshake_verified=True、exit_confirmed=True，外加既有 parent_binding_verified/job_assigned/释放证据。run_stage 核对这两项；任一未知均失败。更新原 fake 终结 fixture 为显式握手与独立 exit，保留原反例目的及负例，不减少校验来迁就旧 fake。
- [ ] **Step 6: 升级未来执行身份。** runner synthetic_protocol 和 validate_protocol 采用 msb_parent_bound_v2，加 terminal_spec_sha256 的 LF 规范哈希验证，保留旧 repair_spec_sha256 和科学规格身份；拒绝 v1 及当前文档许可冒充真实授权。仅在临时 synthetic dict 中验协议，不读取/创建真实协议文件。真实协议仍未登记。
- [ ] **Step 7: 验绿。** task3-green safety 模块通过；其 runner 集成回归在 Task4 final 验收。若新合法事件路径失败，保留当前证据，不用随机重试/换时序隐去。
- [ ] **Step 8: 提交。** 建议 `fix: coordinate MSB close grants and bounded exit draining`；只改两项执行源码，43项闭包不增减。数据联合门、timer接线、损失与矩阵原样。

## Task 4: Guarded Nine-Module Acceptance And Handoff

**Files:** 沿用新验收器；Future Create outputs/msb_terminal_repair_synthetic_v1/final/ 中实际日志/快照；Future Create 对应验收记录；Modify README 的进度段。

**Interfaces:** 不新增科学接口。证据包含 scope=SYNTHETIC_CONTRACT_ONLY、native_os_exercised=False、real_data_opened=False、real_fits=0，测试实际计数、受限调用计数、源码与保护身份、T1--T4 逐项结论分别登记。

- [ ] **Step 1: 静态白名单与保护核对。** 两项执行源码之外不得出现源码 diff；43项闭包数量不变而这两项哈希更新；旧34、四科学模块、data/training/analysis、科学/上轮修复规格、83/121日志、六PNG和两正式报告保持身份。
- [ ] **Step 2: 验证未来许可与整合反例。** runner 新测 `test_v1_or_documentation_approval_opens_no_real_input`、`test_missing_close_evidence_stops_next_job`；fake80job仍按原顺序，三load联合验收先于fit，拟合元数据63+12、final预算615+12不变。无真实输入且不把mock计数写成实际训练。
- [ ] **Step 3: 运行限定 final。** 使用下述命令一次运行固定九模块；不执行旧验收器、不重跑其输出目录、不用广域发现。原121项合同语义+新增测试均应通过，实际数量以后填，跳过/禁止调用/白名单越界不能记为成功。
- [ ] **Step 4: 完整记录原件与限制。** 每阶段日志、当时验收器直接快照和哈希、两项源码及43闭包身份、保护文件核对；披露任何守卫误配置/未模拟调用，不覆盖历史。保留 v1 原验收器重建来源的旧限制，不重新标成当时原件。
- [ ] **Step 5: 自审并归档 GitHub。** diff/链接/空白检查；逐文件暂存，不加真实pt/权重/用户PNG；推送既有分支并核对远端身份。只归档实际合成结果，不创建 PR 或修改正式报告。
- [ ] **Step 6: 停在许可边界。** 后续独立复查、原生无数据验收、新真实协议及真实运行另批。即使全绿，也不宣称网络有效、理论原创性成立或具备工业分选能力。

未来批准后才执行：

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase final
```

固定九模块：test_msb_execution_safety、test_msb_role_data、test_msb_role_model、test_msb_role_controls、test_msb_role_audit、test_msb_role_training、test_run_msb_role_mechanism、test_analyze_msb_role_mechanism、test_generate_msb_role_figure。

## Self-Review And Handoff

覆盖：T1->Task1；T2->Task2；T3/T4->Task3；未来v2身份/43闭包、许可及全流程->Tasks3/4；历史保护与证据限度->Task4。五项 Review Focus 各有具名反例，不以“以后测边界”替代任务。

接口统一：WorkerLaunchOwnership由父创建并传至 start_worker；MSBParentControl 含完整 binding 与唯一在途续租；_drain_control 只消费控制帧；_finish 不与 watchdog 抢读；_publish_terminal_state 是唯一成功尾部。失败清理不兑换为新预算，状态机新增消息不增加 job/fit/forward。

本计划是待审核文档，不是已执行清单；所有 checkbox 保持未开始。保留原生实施偏好，建议下一步只在用户审核后批准 Tasks1--4 的代码与合成验收，不自动派独立代理或开展真实实验。理论创新目标仍未完成，不能把这份执行计划当作论文方法验证结果。
