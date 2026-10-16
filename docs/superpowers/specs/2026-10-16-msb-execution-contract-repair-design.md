# MSB-RRE v1 执行与数据合同修复规格

日期：2026-10-16。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`，待用户审核书面规格及配套计划。
授权：`call_1690e5bbad7a415f8880b0ff473b9521`，用户批准“修复规格与实施计划”，不含开发或验收运行。
基线提交：`3cc8856dbe1778fb4c1c9e1c7803c05227f32ffc`。

## 1. 目的、依据与范围

本规格解决[独立复查](../../experiment_records/2026-10-16_msb_role_independent_review_v2.md)的三项工程问题，使未来实验能检验既定网络理论，而不是改变研究假说。工程修复不是新的算法贡献，也不证明角色分支有效。

| 编号 | 原问题 | 修复后的可验收条件 |
|---|---|---|
| E1 / P1 | 陈旧磁盘 RUNNING / 未消费 packet 可脱离原父监督启动，子 guard 重新计时 | 无活跃原父进程及其专属通信能力即不能进入工作体；全程继承入口的同一截止时间 |
| E2 / P2 | 仅拒绝 projector_stop，缺失或其他 scope 可进入优化 | train / validation 每条记录必须具有精确字符串 projector_fit，且跨折一致 |
| E3 / P2 | 联合三折 OOF 检查晚于 75 次拟合 | 三次既有 load 完成后先联合验收；不通过时优化器、神经拟合和凸拟合计数均为零 |

本轮只写本规格、实施计划及归档说明。不得改源码、测试、协议、既有证据、图、正式报告；不得执行测试、前向、拟合、计时、电源/空闲验收；不得读取真实 prepared / fit / stop、原图、训练缓存。只读源码、既有文本证据、文件身份和官方接口文档不属于真实数据访问。

## 2. 科学合同与历史保护

以下值和含义在未来修复中保持不变：

- 网络：三个独立 rank4 分支，1083 当前参数及 1083 固定参照；输入 h64 / e23 / log_anchor4；标签 T / Ti / G / M 对应 0 / 1 / 2 / 3。
- 公式：`R_ref = 2*abs(m)+0.25`，`d = R_ref*tanh(s)`；固定参照、损失各项、优化器和初始化不变。
- 11 单元、3 折、种子 20261002 / 20261003 / 20261004；63 条神经轨迹、12 次凸拟合，共 75 次拟合；每条至多 400 步，固定主点 100。
- 80 个任务：3 load + 75 fit + analysis + final-verify。联合验收不是第 81 个任务。
- 既有 340 个已暴露 fit 分组，类别数量 [66,130,97,47]；仅允许原有三份登记 prepared 路径；0 次新 PCA / T0。
- 累计上限 7200 秒，包括入口准备、登记核验、加载、联合验收、训练、诊断、导出和最终推理验收；不因进入子进程或新 job 重新起算。
- 最终验证 615 次 validation forward + 12 次 train certificate check；不增加模型前向来完成数据身份验收。
- 主点三种子、七参照及负类约束的晋级门不变；不通过仍为 NO_ROLE_PROMOTION，不以 400 步补救。
- 不修改旧 34 个执行源码、已接受的 MSB 科学规格及 83 项历史日志；六张用户 PNG 和两处正式版 DOCX 原样保护。旧失败目录不续跑、不覆盖。

原[科学规格](2026-10-14_margin_scaled_bilinear_role_evidence_design.md)与[原计划](../plans/2026-10-14-margin-scaled-role-evidence-implementation.md)保留历史身份；本规格仅补充执行合同，不能偷偷修订其中的数学或矩阵。

## 3. 方案比较与选择

1. 仅添加 PID 和磁盘 deadline：改动小，但不能可靠区分 PID 复用，磁盘标记也不代表父进程在监督。不采用。
2. 继续使用旧 launcher，再由 worker 检查 OpenProcess：能发现父进程死亡，但启动授权与持续活性仍缺少专属通道；单次检查也有检查后死亡窗口。不采用。
3. 新增 MSB 专属监督适配层：保留旧工具不动，以继承的父进程句柄、匿名管道握手、固定累计截止时间及 Job Object 组成闭环。采用；增加一个执行模块和对应模拟测试，不增加科学任务。

这是 Windows 本机执行合同，不是对同一用户主动注入/篡改进程的安全沙箱。操作系统完全停顿、断电及不可中断的系统调用不能宣称零延迟清理；无法确认清理或监督证据时一律失败，不得交付成功状态。

## 4. E1：入口时钟、活跃父监督与终止边界

### 4.1 固定时钟合同

新增 `scripts/msb_execution_safety.py`，无 torch / 数据加载依赖。入口 `run_stage` 第一项执行动作取得 Windows `GetTickCount64` 整数毫秒 `start_tick_ms`，早于读协议及登记核验；固定：

```text
deadline_tick_ms = start_tick_ms + 7_200_000
remaining_ms = deadline_tick_ms - current_tick_ms
elapsed_seconds = (current_tick_ms - start_tick_ms) / 1000
```

`start_tick_ms`、`deadline_tick_ms` 必须是非负 int，拒绝 bool、float、NaN、倒退、错误差值和被延长的 deadline；`current_tick_ms >= deadline_tick_ms` 即 TIMED_OUT，包括恰好相等。使用同一系统启动时钟域，不依靠 UTC 或假定不同进程的相对计时原点相等。保留既有 active/wall 连续性与接电检查，不能以该时钟代替休眠中断检测。

父进程在内存中持有不可变 `RunBudget`，每次派发检查它；子进程只能接受实时握手给出的同一对象值。packet / state 中的副本仅供交叉核对，不是授权源。子 guard 仅负责防待机与连续性，其创建时间不用于剩余预算。登记子命令也用剩余预算约束超时；任何准备耗时在后续入口检查中扣除。

### 4.2 启动能力与握手顺序

父进程持有真实自身进程句柄，权限限于 SYNCHRONIZE 和 PROCESS_QUERY_LIMITED_INFORMATION，记录 PID 与 GetProcessTimes 创建时间。每次 job 创建两条匿名管道及 256-bit 随机 nonce；父端、子端所有权明确，Job Object 句柄绝不继承。nonce 只通过内部启动参数交给该 worker，日志只存摘要，不写进可恢复授权文件；它不能替代继承管道能力，也不抵抗同一用户主动检查/注入进程。

使用 Popen / CREATE_NO_WINDOW / close_fds=True 和 STARTUPINFO 的显式 handle_list，只继承子端管道与父进程句柄。句柄数值不能从 JSON 恢复授权；普通 `--worker-packet file.json` 缺少实时通道即拒绝。保留该参数用于内部派发，额外内部句柄参数不可与公共 stage 混用。

固定顺序：

1. 父进程原子发布 RUNNING、active_job 和 packet 摘要，但不因此授予输入读取。
2. 创建子进程；子 bootstrap 只处理启动元数据，等待握手，不读 prepared、不创建模型/优化器、不写 task_started。
3. 父进程将该 worker 加入自身唯一持有句柄的匿名 Job Object，启用 KILL_ON_JOB_CLOSE，禁用 breakaway；失败立即 kill/wait，不退回旧 launcher。
4. 子端通过继承句柄核对父 PID、创建时间和未终止状态，发送 HELLO（本次 nonce、worker PID、run_id、job_id、packet / protocol 摘要）。
5. 父进程核对自己创建的 worker PID、当前 active_job、未消费状态、同一预算和已完成 Job 分配，发送有序 GRANT；包含原始 start / deadline、完整绑定身份及 lease 序号。
6. 子端核对 GRANT 与磁盘元数据全部一致，确认尚有预算，才独占写 task_started 并进入工作体。没有通道、父死亡、身份不符、超时、重复 GRANT 均不能以磁盘 RUNNING 放行。

父亲在步骤 1 后、步骤 2 前死亡：文件不是能力，之后手动调用 worker 不得读输入。步骤 2 至 3 之间死亡：bootstrap 没有 GRANT，管道断开/父句柄 signaled 后退出。步骤 3 以后父死亡：最后 Job 句柄关闭的 OS 行为与子端父活性检查共同约束，不能仅依赖 finally。

### 4.3 持续活性与父子职责

控制消息为 4-byte unsigned big-endian 长度前缀加 UTF-8 JSON，payload 单条最多 8192 bytes；不使用 pickle，不传张量、标签或样本记录。只允许 HELLO / GRANT / RENEW / ACK / REVOKE / DONE 及固定字段；拒绝超长、截断、重放、倒序、重复键及非有限值。一个线程独占每个读端，写端串行，避免并发读取混乱。bootstrap 等待 GRANT 至多 30 秒，且父活性/断管检查持续有效；超时不能继续等到获得“新”预算。

父轮询目标间隔不超过 0.05 秒；每至多 15 秒发送新 RENEW 并要求 ACK。lease 最长 30 秒，且不得晚于原始 deadline；重复旧序号不能续命。子 scope 保存经过核对的最近租约，同时监测父句柄和管道断开。子 scope.check 在每次输入打开前、更新/检查点、凸求解回调、analysis / final-verify、结果写入前后调用；绑定 ContextVar 的 `msb_stage_elapsed_seconds()` 不再借用新子 guard 的局部时间。

子端轻量 watchdog 只处理控制通道、父活性及期限，不操作 torch、图像或电源请求；发现父死亡、租约失效、原始期限到达时撤销工作许可。正常检查点抛异常走清理；主工作体阻塞时 watchdog 请求非成功退出，父端负责 kill/wait。watchdog 不能单独证明在 GIL 阻塞或 OS 停顿下的实时终止上界，不能将模拟日志写成该证明。

worker 生命周期为 BINDING -> WORKING -> PUBLISHING -> CLOSED。握手后进入既有 awake guard，`scope.begin_work()` 才允许科学工作；工作结束先退出 awake guard，`scope.begin_publish(worker_power)` 必须验证 acquired/released 均为真，才允许导出成功结果。PUBLISHING 仍保留父通信、watchdog 与原始预算，只是不再调用已经释放的 awake_guard.check，也不重新申请防待机。写入前后 scope.check 均有效，之后关闭通道；任何过期结果由父端拒收，不能因先写出 COMPLETE 就晋级。异常清理不进入正常 PUBLISHING。

所有耗时操作仍受父进程同一截止时间监督；父进程不得从后台线程无条件自动续租，而应在主监督循环确认子状态、时钟连续性及接电后续租。任一方丢失通道、活性或清理证据即失败，不重连、不重试。通道无响应时不继续派发。

截止时间之后只能进行撤销、终止、wait、释放和失败记录，不再读科学输入、计算或成功导出。清理 wait 最多 5 秒属于失败收尾，不授予额外计算时间；记录 cleanup_seconds，不将其隐藏为 7200 秒内完成。若无法确认 worker 已退出，记录 alive/cleanup 不确定并保持 delivery_verified=False，停止后续任务。

### 4.4 文件接口及证据

新模块导出 `RunBudget(start_tick_ms, deadline_tick_ms)`、`MSBParentSupervisor`、`MSBWorkerScope`、`msb_stage_elapsed_seconds()`；父类拥有 `run_job(job, protocol)->dict`，取代 MSB 对旧 run_supervised_job 的调用。旧 launcher 文件不改。

结果保持原有 status / attempts / alive / worker_power 等检查，新增 `parent_binding_verified`、`job_assigned`、`lease_sequence`、start / deadline、结束 tick、撤销/清理证据；nonce 只记录摘要，句柄数值不当成可复用凭据。任何失败不得伪造释放成功；硬杀后只可记录 OS 终止证据，不声称执行了 worker 的 finally。

未来协议增加 `execution_contract="msb_parent_bound_v1"` 与 `repair_spec_sha256`；旧科学规格身份不变。执行闭包由 42 扩为 43 个源码（新增安全模块）；旧 34 个源码逐项不变，不能沿用 42 项旧身份冒充修复后登记。此次授权 ID 必须被列为 documentation-only，不能转写成真实运行许可。

## 5. E2：严格 fit-only 语义

`validate_batch` 对 train 和 validation 的每条记录要求 `type(tc_subset) is str` 且 `tc_subset == "projector_fit"`。缺失、None、空串、大小写/空格变化、projector_stop、其他 split 均拒绝；不猜测、不剥空格、不自动填值、不根据 image_id 推断 scope。

跨折身份统一为 `(global_index, image_id, split_group_id, four_class_id, tc_subset)`。保留原 label / tensor / frozen state / 顺序 / 分组 / SHA 校验，four_class_id 仅沿用已有 0..3 规范化；新 scope 字段不宽松转型。每个出现位置都需检查，不能只验其作为 OOF 验证样本的那次。

摘要与日志记录错误字段、折和 train/validation 位置，但不在 GitHub 错误日志中倾倒真实样本全文。若未来真实文件不符合合同，停止并登记；修复方案不授权改 prepared、重新 PCA/T0、删除行或补 scope。

## 6. E3：三次 load 内完成拟合前联合验收

### 6.1 单折摘要与共享校验器

新增数据接口：

```python
prepared_identity_summary(data: dict, fold: int, entry: dict) -> dict
validate_identity_collection(summaries: list[dict]) -> dict
```

每个既有 load worker 仍只打开自己那份已登记 prepared，先验证字节 SHA、单折、严格 scope 和有序 ID，再输出 `result.json`，不再输出只含 validated=True 的 pt。摘要包含 schema、fold、登记 entry 摘要、train_indices / validation_indices、对应两部分规范化四字段记录、counts 和 train/validation 有序 ID 摘要；没有 h/e/log_anchor、预测或模型参数。

摘要精确字段为 `schema="msb_fold_identity_v1"`、`fold`、`prepared_entry`、`entry_sha256`、`train_indices`、`validation_indices`、`train_records`、`validation_records`、`train_ids_digest`、`validation_ids_digest`、`counts`。prepared_entry 保留协议原五字段；entry_sha256 是其规范 JSON 摘要。records 每条仅 image_id / split_group_id / four_class_id / tc_subset，类标签规范为字符串 "0".."3"；两个 ids_digest 与 entry 对应登记值一致。保留原 result 外层 status/kind/worker_power/监督证据，身份摘要放入 `identity_summary`。

摘要允许精确字段集合；fold / global index 是 int 而非 bool。长度合计 340；两部分索引不交、恰好覆盖 0..339；单折 image/group 不交；类别总数必须为 [66,130,97,47]。validate_identity_collection 检查三折完整、有序、每个 index 恰好一次 OOF、340 个唯一 image/group，以及每个 index 在各折 train/validation 中身份、标签与 scope 一致。canonical_identity_sha256 对按 global_index 升序排列的 `[index,image_id,split_group_id,four_class_id,tc_subset]` 列表取规范 JSON SHA256。

validate_fold_collection 继续验证原始张量合同，但委托同一联合身份校验器，避免拟合前和最终验收维护两套语义。不要求不同折的 h/e 数值相同（其既有折内预处理可不同），不读取数据集、stop 或原图追补信息。

### 6.2 父门与子门

三个 load 成功且 worker 已退出、power 与父绑定证据齐全后，父进程按已登记 artifact hash 读取三个 JSON 摘要，执行联合校验。全部成功才独占写 `joint_preflight.json`，并在 RUNNING state 中将 `preflight_status` 从 PENDING 改为 PASSED。

证书精确字段为 `schema="msb_joint_preflight_v1"`、`run_id`、`protocol_digest`、`start_tick_ms`、`deadline_tick_ms`、`prepared_entries`、`load_artifacts`、`groups=340`、`counts=[66,130,97,47]`、`canonical_identity_sha256`。load_artifacts 是按 fold 排序的三项 `{fold,relative_path,sha256}`，只指向本输出内 load result.json。规范 JSON 使用排序键、紧凑分隔、ensure_ascii=False、allow_nan=False 后取身份 SHA256；证书写出后 packet 的 `preflight_sha256` 另取整个证书文件的字节 SHA256。不包含可变 lease 序号或可重用授权秘密。

第一个 fit packet 只能在该证书写入并登记摘要之后生成，后续 75 个 fit packet 都携带相同 preflight hash。worker 不能仅相信 PASSED：在有效活跃 scope 下核对三个 load artifact hash，重做无模型前向的联合摘要校验，并核对证书后才加载本折 prepared、创建模型或优化器。analysis / final-verify 也必须核对同一证书；最终原有联合检查保留为纵深校验。

任一 load、摘要或联合证书失败：终态失败，actual_neural_fits=0、actual_convex_fits=0、delivery_verified=False；最多完成三次 load，不生成 fit packet、不尝试修正数据、不进入 analysis。单次授权被失败消耗，不用新目录自动补跑。

### 6.3 预算与科学不变性

三次 load 数量与读取路径不变。增加的父端及每个后续 worker 的 JSON 身份校验占用同一累计预算，既不是新 fit，也不是新模型 forward；不能以此延长 7200 秒。fit 仍每次按原登记 SHA 重验本折 prepared 后使用，不允许因“已经 preflight”跳过哈希。禁止缓存真实张量到父进程或创建新的可恢复数据缓存。

## 7. 后续实现边界与失败状态

| 文件 | 后续允许的最小职责变化 |
|---|---|
| scripts/msb_execution_safety.py（新增） | 原生 API 薄适配、预算/通道/lease、父监督/Job 清理、子 scope |
| scripts/run_msb_role_mechanism.py | 新监督层、元数据握手、load JSON、联合门、协议与 43 项闭包 |
| scripts/msb_role_data.py | 严格 scope、摘要 schema、共享三折身份校验 |
| scripts/msb_role_training.py | 仅将原 stage_elapsed_seconds 检查绑定到累计 scope，不改科学算式及训练步序 |
| scripts/analyze_msb_role_mechanism.py | 要求 MSBWorkerScope 和相同 preflight，沿用最终检查 |
| tests/test_msb_execution_safety.py（新增）及既有对应测试/helper | 全部假时钟/假句柄/合成 fixture，不实际启动 OS 子进程 |

msb_role_model / controls / audit / generate_msb_role_figure 原样；结构图不需要重绘，因为网络未变。计时、损失、权重、凸求解器和晋级门均不得借修复调整。

监督失败沿用 SUPERVISION_INTERRUPTED；预算用 TIMED_OUT；释放缺失用 POWER_RELEASE_FAILED；数据语义/联合校验用 FAILED 并给出 scope/preflight 原因。终态一律 delivery_verified=False。孤儿或硬杀情形可能没有母进程终态文件，残留 RUNNING 永远不是成功或可恢复凭据。

## 8. 验收要求、限度及后续闸门

开发若获准，仅模拟原生 API、时钟、句柄和进程；测试使用 tests/msb_synthetic_helpers.py 新建临时合成数据，禁止项目真实路径和真实 Popen/WinAPI。需要覆盖 stale RUNNING、父死亡、PID 复用、无管道、延长 deadline、租约重放、Job 分配失败、清理失败、analysis/final 超时、所有 scope 反例，以及“每折单独合法但联合不合法时拟合计数为零”。

保留原有 83 项测试语义，新增用例计数按实际输出登记，不能预报总通过数；必须保留 red/green 证据。全模拟 80 job / 75 fit 元数据不等于真实训练。原生 WinAPI 路径未被模拟测试实测，尤其句柄继承、嵌套 Job 限制、父硬退出与阻塞清理需独立批准的原生无数据验收后才能形成原生结论；本规格不授权该验收，也不自动重复六分钟空闲实验。

完成本轮文件不改变 REVIEW_COMPLETED_CHANGES_REQUIRED。顺序为：用户审核两份文件 -> 另批原生代码与合成验收 -> 必要的独立工程复查/原生验收另批 -> 新真实协议另行登记及运行批准。未验证网络真实增益前，不进入正式报告成果栏。

## 9. 接口依据（不是算法查新）

2026-10-16仅核对官方 API 语义，不做矿物/理论文献查新，不依据它们宣称研究原创性。

- 显式句柄继承的依据见 [Python 3.11 subprocess 文档](https://docs.python.org/3.11/library/subprocess.html#subprocess.STARTUPINFO)；临时可继承句柄必须在启动后立即复原，不能让无关并发进程继承。
- Job 关闭及子树管理语义见 [Microsoft Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)；分配失败/已有 Job 的限制见 [AssignProcessToJobObject](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject)。
- 系统启动毫秒计时依据见 [GetTickCount64](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-gettickcount64)；其为离散系统计时，不声称数学意义零误差 deadline。父创建时间依据见 [GetProcessTimes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes)。

上述机制是依据文档形成的工程设计选择，不是已运行的本机验收结果。
