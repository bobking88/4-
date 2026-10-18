# MSB Completion Acceptance Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for future separately approved native implementation. Do not dispatch agents automatically. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复最终退出租约接收、严格合成验收门及双重失败诊断，并补排空循环边界证据；不更换科学假说。

**Architecture:** 在原父控制接口显式接收当前tick；现有验收器用唯一严格结果门及独立固定输出根；原发布器携带两个异常及失败状态。不新建生产模块或通用IPC框架。

**Tech Stack:** 既有Python、unittest/mock、PowerShell及Git，无新依赖，原生接口全部fake。

**Spec:** [2026-10-17-msb-completion-acceptance-repair-design.md](../specs/2026-10-17-msb-completion-acceptance-repair-design.md)

起草：2026-10-17；静态校核与归档：2026-10-18。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档授权call_aa6d081524a94817b20cceb8fbd25053，基线655cf76a1c164052c87172b776bb0ed1aea21c96。
按用户明确批准一并起草规格与计划；两份均待审核，当前不实施任何步骤。

## Global Constraints

- 当前只写文档并归档；不改代码、运行测试/前向/拟合/PCA/T0/计时/原生验收，不读真实数据，不派代理，不改正式报告。
- 未来生产源码仅scripts/msb_execution_safety.py及scripts/run_msb_role_mechanism.py；测试仅两既有模块及tests/run_msb_terminal_repair_acceptance.py；43闭包不增减，旧34、其余41及科学模块/数据门不变。
- h64/e23/log_anchor4、三rank4、1083当前及1083固定参照、R_ref=2*abs(m)+0.25、原损失/初始化/优化器不变。
- 340个既有已暴露fit分组及[66,130,97,47]、三份prepared；11单元/3折/三种子20261002/20261003/20261004；63+12拟合、80 job、400步上限、100步主点、615+12核验、七参照晋级门不变。
- 原累计7200秒、等号过期；租约30000ms/续租15000ms、帧8192bytes、排空8帧、清理wait最多5秒，不兑换计算额度。
- 未来新输出根outputs/msb_completion_repair_synthetic_v1；七phase为gate-red/green、lease-red/green、publish-red/green、final，各最多3次独占attempt，导入/守卫失败也计数。旧全部根/日志/失败记录/报告/六PNG均不覆盖、不暂存。
- 原生实施偏好保留；独立复查、原生无数据验收、新真实协议与运行仍另批。当前不为工程修复增加网络创新主张。

## File Structure

| 未来文件 | 最小改动 |
|---|---|
| tests/run_msb_terminal_repair_acceptance.py | 严格门、七固定新phase/独立根、实际非通过计数；守卫不放宽 |
| tests/test_run_msb_role_mechanism.py | 门函数/main接线、双重异常及非真实授权回归 |
| scripts/msb_execution_safety.py | require_complete显式新tick及父最终接收接线 |
| tests/test_msb_execution_safety.py | 独立退出跨租约、显式新参数、实际8帧计数及缺DONE拒绝 |
| scripts/run_msb_role_mechanism.py | MSBTerminalPublicationError及首要异常保留；既有非真实ID拒绝更新 |
| outputs/msb_completion_repair_synthetic_v1 | 未来获准运行后才创建直接原始证据，本轮不得创建 |
| docs/experiment_records/2026-10-17_msb_completion_acceptance_repair_acceptance.md | 未来实际验收记录，不提前生成成功结论 |

## Review Focus

1. DONE于L-1已接受，poll/父guard检查推进到L或L+1：最终接收仍拒绝，不能仅测总期限；Task2。
2. helper正确但main仍用旧表达式、全skip/空suite/expectedFailure：不得发布PASSED，外层不skip；Task1。
3. 成功写后过期、记录写再失败：两个异常和TIMED_OUT不能因第二次OSError消失；Task3。
4. 排空有第9项、8项读完仍无DONE：循环不越界，真实状态机也不能成功；不能伪造8个合法ACK；Task2。
5. 无新增真实许可、旧v1或文档/复查许可、旧目录及直接快照不完整：不读真实输入/不覆盖历史，不能把合成成功当真实许可；Tasks3/4。

## Task 1: Strict Gate And Exclusive Guarded Evidence Entry

**Files:** Modify tests/run_msb_terminal_repair_acceptance.py、tests/test_run_msb_role_mechanism.py。

**Interfaces:** `_acceptance_passed(result, *, forbidden_calls: list, harness_error: str | None) -> bool`；`main(argv=None) -> int`仍仅--phase。gate/publish对应runner模块，lease对应safety模块，final为原九模块。ROOT解析方式、五项SNAPSHOTS及原生守卫保持。

- [ ] **Step 1: 建立新证据入口及无语义修复接缝。** 修改固定输出根/七phase；每phase最多3次独占attempt，先直接快照再导入项目。将旧successful表达式原样抽为门函数、main调用它，暂不增加skip判定。保留数值bootstrap和失败即报错torchvision替身，不创建新依赖。
- [ ] **Step 2: 写严格门反例。** 新增test_acceptance_gate_rejects_skipped_expected_or_empty_result，构造真实unittest.TestResult承载合成计数，覆盖None、testsRun=0、单个skip、全skip、expectedFailure、unexpectedSuccess及纯通过；仅纯通过可真。覆盖forbidden_calls非空及harness_error非None拒绝。外层测试不得skip。
- [ ] **Step 3: 写main接线反例。** test_acceptance_main_skip_result_is_failed_and_nonzero：在专用临时ROOT下提供五项合成源码占位文件，替换测试加载/执行结果为skipped=1且wasSuccessful为真的TestResult，调用实际main，断言回执FAILED、skipped=1且返回1。保留实际门与回执写入，不mock被测判定。os.chdir/tempfile.tempdir须在fixture退出恢复，内部回执留在测试scratch，不消耗项目实际phase额度；无真实项目模块测试被递归执行。
- [ ] **Step 4: 验红。** 未来执行--phase gate-red。应看到旧逻辑接受skip的有效反例失败；纯通过、旧runner合同保持。导入失败不算有效RED，保留并计尝试；不因失败放开原生接口。
- [ ] **Step 5: 最小修复并验绿。** 门必须同时要求wasSuccessful、testsRun>0、三个非通过集合为空、无未模拟调用及无harness_error；main只用该结果，增加expected_failures/unexpected_successes实际字段。--phase gate-green通过，零skip/预期失败/意外成功/受限调用，不提前声明总数。
- [ ] **Step 6: 提交。** 逐文件暂存Task1及直接红绿证据，建议test: require strictly passed MSB acceptance contracts。原七phase旧根完全不动。

## Task 2: Fresh Lease At Exit Acceptance And Bounded Drain Coverage

**Files:** Modify scripts/msb_execution_safety.py、tests/test_msb_execution_safety.py；沿用Task1新入口。

**Interfaces:** `MSBParentControl.require_complete(exit_code: int, *, now_tick_ms: int) -> None`；`_drain_control(..., max_frames=8) -> int`签名不变。父调用前完成既有检查，最后采样新tick传入require_complete，紧接着COMPLETE，不在二者间插入原生/I/O操作。

- [ ] **Step 1: 写独立退出事件反例。** test_done_before_lease_but_exit_at_lease_is_rejected：真实父监督与控制状态机，fake API使DONE在L-1被接受，process.poll独立把tick推到L再返回0，累计期限仍未到；断言SUPERVISION_INTERRUPTED，不以parent_binding_verified或exit_confirmed真覆盖失败。read(DONE)不得触发退出。另测L+1失败、L-1成功。
- [ ] **Step 2: 覆盖最后检查的采样位置。** test_exit_acceptance_samples_after_guard_check令最后guard检查耗时至L，断言拒绝；有效初始tick不能复用为最终now。保留deadline等号、时钟倒退和无ACK/缺DONE/非零退出反例。
- [ ] **Step 3: 补实际八帧计数。** test_drain_consumes_eight_and_preserves_ninth使用仅计数、不完成、不发帧的ProbeControl及9项队列，调用真实_drain_control，断言read_count==accept_count==8、返回8、队列仍1项且done=False。此probe仅测循环上界，不冒充真实协议。test_incomplete_real_close_cannot_complete另用真实MSBParentControl验证无DONE拒绝；保留原max_frames=9/截断负例。
- [ ] **Step 4: 验红。** --phase lease-red；至少独立退出等号/最后guard耗时反例在旧实现中暴露误COMPLETE。补覆盖测试可原本通过，如实记录，不制造假红灯。旧45项断言目的保留。
- [ ] **Step 5: 最小修复并验绿。** require_complete首先复用_check(now_tick_ms)再验结构，父按接口接线；旧单元调用补显式有效tick，不删旧断言。--phase lease-green通过；配合final中原runner失败即停测试证明不会派下一job，回收仍走原路径。
- [ ] **Step 6: 提交。** 建议fix: check frozen MSB lease at final exit acceptance。不得延租、延总期限或新增控制消息。

## Task 3: Preserve Both Publication Failures And Non-Real Authorization

**Files:** Modify scripts/run_msb_role_mechanism.py、tests/test_run_msb_role_mechanism.py。

**Interfaces:** `MSBTerminalPublicationError(RuntimeError).__init__(failed_state: dict, *, primary_error: BaseException | None, persistence_error: BaseException)`；保留_publish_terminal_state签名及唯一成功发布路径。

- [ ] **Step 1: 写双异常反例。** test_terminal_double_write_failure_preserves_both_errors用两个不同OSError对象依次触发成功写/失败记录写，断言新异常.primary_error及.persistence_error分别为原对象、.__cause__为第二对象，failed_state保留FAILED及两个False标志，str包含not persisted及首要原因。
- [ ] **Step 2: 写超时与已有失败反例。** test_timeout_then_failure_record_error_keeps_timed_out使最终写后预算到7200000，再让记录写失败；断言primary含TIMED_OUT、failed_state.status==TIMED_OUT，不能变成假成功。test_existing_failed_state_text_survives_record_error允许primary_error=None但原error文本不丢。test_cli_cannot_return_zero_after_publication_error在无原生进程的main调用中使run_stage传播该异常，不得返回0或打印成功。
- [ ] **Step 3: 固定许可边界。** 扩充既有文档/复查/代码许可拒绝回归，至少包括call_a4c9ef81bcea447aa7350f7cd9aff51c、call_aa6d081524a94817b20cceb8fbd25053及未来实施批准ID；使用临时合成metadata，断言load_prepared_fold未调用。不创建真实协议或把拒绝列表当语义认证。
- [ ] **Step 4: 验红。** --phase publish-red。应见首要异常/失败状态未保留的有效失败；旧唯一写入、非零及不谎报持久化测试保留。无真实输入或额外I/O重试。
- [ ] **Step 5: 最小修复并验绿。** 首要异常保存于except之外的变量；记录写失败抛MSBTerminalPublicationError并链接第二异常，保留浅层failed_state与原error。已有失败输入不伪造对象；消息保留not persisted。更新已知非真实ID拒绝但不改v2/规格/矩阵。--phase publish-green通过。
- [ ] **Step 6: 提交。** 建议fix: retain both MSB terminal publication failures。未落盘就明确未持久化，不追加写入或补成功。

## Task 4: Final Nine-Module Acceptance, Identity And Handoff

**Files:** 沿用新入口；未来创建新根/final实际日志和验收记录，更新README进度。所有源码/测试以实际运行前直接快照为准。

- [ ] **Step 1: 静态范围与保护核对。** 仅两生产文件/两测试/一个验收器允许代码diff；43闭包/旧34/科学模块/规格及83/121/146历史树不变。按旧记录原哈希核对六PNG及两报告等10项保护文件；不用读取真实科学输入来补证。
- [ ] **Step 2: 运行固定final一次。** --phase final运行原九模块，不广域发现；实际全部通过且零skip/expectedFailure/unexpectedSuccess/受限未模拟调用才PASSED。每次尝试保留源快照和原始日志，导入失败也计次，最多3次；不能续跑旧根。
- [ ] **Step 3: 记录科学与原生限度。** 汇总实际计数、real_data_opened=False、real_fits=0、native_os_exercised=False，临时合成读回与bootstrap单列。保留R1-R3/G1逐项证据，G1不冒充合法8帧序列，零结果不能记通过。
- [ ] **Step 4: 作者静态自审与归档。** 逐项比较规格/日志，核对链接/空白；原日志格式不为清洁diff而改写。仅暂存授权文件，非强制归档既有GitHub分支并读回提交/文件身份，不建PR/合并。
- [ ] **Step 5: 停在授权边界。** 独立只读复查、原生无数据验收、新真实协议和真实运行分别另批；作者自审不是独立批准，工程全绿不是网络有效/创新/工业能力证据。

## Future Commands, Not Authorized Now

以下仅供未来书面文件获认可且代码/合成验收另批后执行；本轮不运行：

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase gate-red
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase gate-green
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase lease-red
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase lease-green
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase publish-red
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase publish-green
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_terminal_repair_acceptance.py --phase final
```

固定九模块：test_msb_execution_safety、test_msb_role_data、test_msb_role_model、test_msb_role_controls、test_msb_role_audit、test_msb_role_training、test_run_msb_role_mechanism、test_analyze_msb_role_mechanism、test_generate_msb_role_figure。

## Author Document Self-Review And Research Handoff

覆盖对应：R2->Task1；R1/G1->Task2；R3及许可->Task3；历史保护、v2身份/固定全流程->Task4。五项Review Focus均有具名反例；接口、phase、输出根及次数与规格一致，不留占位步骤，不预报新增通过总数。
原生实施方式保留，当前所有checkbox未开始。文档审核不等于开发/合成验收许可；没有本轮代理自审。
研究目标仍未完成：后续要用既定MSB-RRE强对照检验角色非线性交互、间隔预算与净纠错，而不是把这份工程计划当论文理论成果。C路线、真实数据质量/外部泛化及正式报告修改不在本轮范围。
