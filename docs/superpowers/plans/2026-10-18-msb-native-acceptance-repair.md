# MSB原生验收驱动最小修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 当前仅为文档交付；不实施、不运行、不派代理。

**Goal:** 关闭独立复查的R1--R5五项缺陷，恢复申请限定原生验收前的工程条件，不把工程通过作为网络创新结果。

**Architecture:** 保留原生产MSB/LC/WindowsAwakeGuard；只修新增驱动的执行来源、N4故障证据、零等待和末端失败发布。原生接口的开发验收全部fake，纯验证与真实原状态机仍实际执行。

**Tech Stack:** 现有CPython/stdlib/unittest与已安装torch/sklearn；无新增依赖、不调整电源方案。

**Spec:** [MSB原生验收驱动五项缺陷的最小修复规格](../specs/2026-10-18-msb-native-acceptance-repair-design.md)。

## Global Constraints

- 文档授权call_03a86c6e5b064586b7bb980fa04b2a53同时允许规格、计划及归档，不允许开发或运行；实际实施另批。
- 基线92ff34eafdfdef7a7df15b4e4528f1b2357483bd，受审源码c092f1cae59fc4187670bf4bff655375309d83b0。REQUEST_CHANGES及五项UNFIXED保持到真实修复验证，不改历史记录。
- 未来只改scripts/check_msb_native_supervision.py、tests/test_check_msb_native_supervision.py、tests/run_msb_native_supervision_synthetic_acceptance.py三文件。旧43源码、10测试/验收器、全部旧证据不变。
- 共同RunBudget精确7200000ms、额外480秒累计窗口、N1--N4顺序、90/45/75/45秒场景上限、每例一次不重试、45秒工作/exit23/DONE后35秒迟退/父exit31均不变。
- 原30秒poll-gap、2秒双时钟差、30秒租约、15秒续租、5秒cleanup不改。N2/N4 ready-ACK上限5秒，N4退出观察上限5秒，等号拒绝；不做主动休眠、键鼠空闲或电源方案调整。
- 新合成证据根outputs/msb_native_acceptance_repair_synthetic_v1，scratch为.superpowers/sdd/2026-10-18-msb-native-acceptance-repair/test-tmp；文档阶段两者均不创建。
- 固定七phase：contract-red/green、worker-red/green、collector-red/green、final，每phase最多3次独占attempt含失败；前六只运行test_check_msb_native_supervision，final固定原10模块，原202名称保留，不预填新增通过数。
- 不读真实prepared/fit/stop、原图/缓存/权重；不真实训练/前向/PCA/T0；网络、公式、损失、实验矩阵和正式报告不变。
- 合成阶段原生进程/时钟/电源/线程启动/网络接口均fake或禁止；沿用既定torch/sklearn必要数值bootstrap和专用scratch round-trip边界，不扩大白名单。
- 真实原生协议、原生输出及MSB机制实验均不创建/运行；独立复查、登记及各次运行另批，不自动派代理。

## Review Focus

1. R2合法root但实际模块来自另一checkout或先加载后混入：Task1覆盖驱动/模块/spec来源与三入口接线，拒绝发生在原生构造和启动之前。
2. R1清理意图与故障提交并发、日志/锁失败导致陈旧无清理证据：Task2受控调度覆盖，不能以缺记录接收，更不能吞掉真实清理。
3. R1子退出先于父退出观测但晚于故障是合法顺序；同毫秒无法确认和5秒等号不合法：Task2分别正负覆盖，不夹断负差。
4. R3最后case之后及根guard释放后发生AC/采样/写入延迟：Task3覆盖写前写后与classifier缺证据，无新guard起点或重置预算。
5. R4业务失败、释放失败和失败回执写失败重叠；R5零工作自然跨tick：Tasks3/1分别保留全部异常并非零、0等待体不计工作。

## Files And Interfaces

Modify only the three files above; author records/evidence/README are additional deliverables, not runtime dependencies. Do not create a new general execution framework.
Existing public run_acceptance/run_controller/run_no_data_worker signatures remain except NativeServices constructor gains required root; all callers in this driver must be updated together.
Pure origin snapshot keys: driver, module:<name>:file and module:<name>:spec when present. Tests fake observed origins, not the validator's answer.

### Task 1: 执行来源绑定与零工作语义（R2、R5）

**Files:** Modify driver、test module及harness的新证据根/scratch接线，不改旧模块。
**Consumes:** 已归档SOURCE_FILES、真实__file__/sys.modules观测、原能力/协议校验、_wait_work。
**Produces:** execution_origins()、validate_execution_origins(root, origins)、NativeServices(root)，以及0毫秒直接返回0的_wait_work；供Tasks2/3使用。

- [ ] **Step 1: 先写具体反例。** 新test_origin_rejects_other_checkout_and_import_alias：在两个临时假源码树A/B设置driver=A、root=B，断言ValueError；另把已加载msb_execution_safety的file或spec指向B，同样拒绝。test_origin_accepts_same_resolved_root_only断言返回root/driver/module_origins与观测相同。test_zero_wait_does_not_turn_tick_overhead_into_work断言`_wait_work(0, packet, services)==0`、pause调用0，即使fake每次tick自增；负数/布尔/非整数ValueError，正时长原用例不删。
- [ ] **Step 2: 登记新七phase根并执行contract-red。** 使用下方固定入口，记录直接快照和实际失败；不以模块导入/夹具错误充当功能RED。每phase最多3次，失败占额度，额度耗尽停止。
- [ ] **Step 3: 最小实现规格第2、3节。** 公开许可仍先拒绝；collector/controller在固定协议读前、原生构造前验证来源；私有worker先能力校验、再只读验收packet核对root、再构造服务。每次项目导入后检查新增来源；命令使用已验证driver。0等待不采样差计工作，周边监督保留。
- [ ] **Step 4: 补真实接线与默认拒绝验收，运行contract-green。** test_three_entrypoints_reject_wrong_origins_before_native断言三入口错源时factory/launch次数0；相同来源正例走真实验证器及原scope/supervisor（native仍fake）。旧许可和本轮文档许可仍拒绝。不得删除45项旧测试；其临时合法夹具只替换来源观测，不mock验证结果。
- [ ] **Step 5: 记录并提交Task1。** 保留红绿原件、源码直接快照和旧53项身份；记录正确来源与注册哈希为不同必要条件。此时N4/末端修复尚未完成，不宣称总体通过。

### Task 2: N4活性、故障顺序及清理意图（R1）

**Files:** Modify driver及test module；harness仅在固定守卫所需时调整，不扩模块或原生能力。
**Consumes:** Task1已验证root/来源，原retained-query身份、TraceAPI、parent_fault_after_ready及_check_case。
**Produces:** N4FaultGate、collector的alive_at_ack、故障前alive_before_fault/活性tick、完整trace快照及新增N4接收条件。

- [ ] **Step 1: 写worker-red反例。** test_n4_dead_at_ack_is_rejected断言无collector_ack文件/无exit31；test_n4_rejects_pre_fault_exit_and_cleanup分别注入exit_tick<=fault.tick、terminate_job/kill_process及cleanup意图，断言accepted=False；test_n4_allows_child_observed_before_parent在fault=1104、child=1110、parent_observation=1120时断言限定接收。test_n4_fault_exit_same_tick_and_five_second_equality_rejected分别取child=fault、child=fault+5000，均False。
- [ ] **Step 2: 增加同一gate的受控调度反例并运行worker-red。** test_n4_cleanup_fault_race_never_qualifies覆盖cleanup先登记、fault先进入、query失败、trace写失败及锁1秒超时。断言非法路径不exit31；真实cleanup替身按原调用一次、不能因记录异常被跳过；未确认状态不获得通过。所有线程/锁时钟调度为fake，不启原生线程。
- [ ] **Step 3: 实现规格第4节。** collector先身份及alive再ACK；controller再次身份/活性确认。N4FaultGate.note_cleanup在转发清理前记录意图，commit_fault只在无意图/无异常/完整快照且活性期限仍有效时执行当前父exit31；采用同一gate排序及有界锁，不另写生产状态机。trace/fs异常不得吞掉清理，也不得以陈旧快照继续故障。
- [ ] **Step 4: 运行worker-green并验证原接线。** 将新增活性/故障快照字段接入实际collector/故障函数/_check_case，测试至少有正例真正走这些函数，不只改预制字典。保留N1真实renew/close、N2无DONE/EOF109、N3冻结租约及真实MSB类+fake native的旧覆盖。确认未复制Job/channel句柄，硬退出未显式观测release保持未知。
- [ ] **Step 5: 归档Task2。** 保存原始事件、fake调度顺序及全部失败；报告“有限故障证据”，不写成排除任意自发退出/任意OS竞争。若gate实现需要更改旧生产43项、扩大native能力或取消有界清理，停止并报告，不自行扩范围。

### Task 3: 末端连续性与所有失败路径的终结（R3、R4）

**Files:** Modify driver、test module及harness必要接线，仍仅三文件。
**Consumes:** Task1来源验证、Task2完整case记录、原根guard.last、原NativeEnvironmentAPI.sample/30秒/2秒合同、原publish_acceptance与AcceptancePublicationError。
**Produces:** NativeServices.postflight_sample()、FinalContinuity(binding,last_sample,*,services).check()及完整终态发布；最终作者合成证据。

- [ ] **Step 1: 写collector-red反例。** test_postflight_ac_and_gap_failure_never_publish_pass分别在最后case后、源码读取后、guard释放后及PASS写入后注入断电/未知AC或31秒gap；断言非PASS、delivery_verified=False、无后续launch。test_classifier_requires_final_continuity删除outer_evidence.final_continuity，断言accepted=False。边界fake30秒/2秒可通过原限值，超过限值及480秒等号拒绝，不能另开guard重置last。
- [ ] **Step 2: 写释放/持久化联合反例并运行collector-red。** test_guard_release_failure_finalizes_and_cli_nonzero：四例完成后__exit__抛POWER_RELEASE_FAILED，断言终态FAILED_NATIVE_SUPERVISION、release未伪造及main非零。test_body_release_and_write_failure_keep_all_errors：body失败+release失败+失败持久化失败，断言AcceptancePublicationError保存primary_errors两项、首项primary_error兼容别名和persistence_error，非零传播；不残留“已确认交付”的内存PASS。尚未创建RUNNING的登记失败仍不创建输出目录。
- [ ] **Step 3: 实现规格第5、6节。** 外层try涵盖根guard退出，body/release/continuity原因分开保留；active guard在潜在阻塞操作前后及退出前check。释放后FinalContinuity从最后原sample接续，经同一个NativeEnvironmentAPI采样，不重复电源请求。publish_acceptance对PASS要求final_check并写前写后调用；失败发布无需AC恢复，二次写失败完整抛出，无递归写入或续跑。
- [ ] **Step 4: 运行collector-green及固定final阶段。** 包括test_final_check_is_required_and_runs_before_and_after_pass_write、test_missing_or_failed_release_never_counts_as_observed_release、test_guard_is_not_reentered_for_postflight。final固定原10模块，原202名称保留；实际新增数/0非通过项/0未模拟native调用才接受。七phase各最多3次，含任何失败；不unittest discover、不改旧日志/跳过测试或扩bootstrap。若无法在批准的固定额度内闭合，则保留REQUEST_CHANGES并停。
- [ ] **Step 5: 静态保护核对并归档。** 新直接快照匹配三修改文件，旧43+10/历史证据/正式报告/六张用户脏PNG不变；源码身份、全部红绿日志、作者判断和README同步既有分支并独立远端读回。仅AUTHOR_REPAIR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING，未进行原生或真实实验；独立复查须另批，不自动派发。

## 未来命令与判定

以下命令仅在另批开发后执行，本轮不执行：

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_native_supervision_synthetic_acceptance.py --phase contract-red
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_native_supervision_synthetic_acceptance.py --phase contract-green
```

worker-red/green、collector-red/green、final依次使用同一入口与固定新根。仅实际原始日志、返回码、严格门和源码快照证明合成结果，不从计划预填PASS。拒绝空测试、skip、expectedFailure、unexpectedSuccess及任何守卫调用。
本计划所有复选框未勾选；R1--R5分别对应Tasks2、1、3、3、1。Review Focus五项均有明确归属及反例。作者自审不派代理、不运行测试，不把读源码或文档lint叫合成验收。

## 执行交接与科学目标

文档经人类审核后，建议由当前会话原生方式实施Tasks1--3，全部接口fake的合成验收也需明确批准；未获批准前不执行任何步骤。
后续独立复查、原生登记、一次原生运行、真实MSB协议和机制实验各须另批；本计划不预授权这些动作，不擅自清理证据/工作树或合并分支。
此修复不是论文网络创新。工程条件满足后才回到MSB-RRE的角色非线性证据、间隔预算跨界及强对照独立收益验证；科学规格/数据暴露边界/晋级门均不变。只有真实机制证据达到既定门槛，才能据实强化报告和论文贡献；否则保留NO_ROLE_PROMOTION并报告负结果。
