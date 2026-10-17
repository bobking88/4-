# MSB-RRE v1 终结握手与发布边界修复规格

日期：2026-10-17。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
授权：`call_a697651b584d4fb8b2116607dcc82b52`，用户批准“仅编写修复规格与计划”，不含实现、测试或实验。
当前归档基线：`82ab7bfa935f9cb07919e1492d47db59e6b684f9`；被审实现：`0bb95a48a8fc47286dfcc3bd864d6ecfaf02ef58`。

## 1. 目的与问题定位

依据[独立复查及接收判断](../../experiment_records/2026-10-16_msb_execution_contract_repair_independent_review.md)，修复四项 P2。它们是静态控制流发现，不是已复现的原生 Windows 故障。本规格不是新的数学/算法创新。

| ID | 静态根因 | 后续验收目标 |
|---|---|---|
| T1 | run_stage 的受保护成功写入之后，公共尾部再次写状态并直接返回 | 只有一条成功发布路径，最后一次写入也在预算检查及异常撤销内 |
| T2 | start_worker 只将继承标志恢复异常包装成携带进程的异常 | 所有已获进程对象的启动后异常均保留所有权；无法确认是否创建时明确不确定，不能报零尝试且已清理 |
| T3 | 每轮只读一帧后立即以退出码判断缺少 DONE；fake 将退出绑定到父读 DONE | 退出后先处理有限的已排队合法终结消息；进程退出与父读取消息在 fixture 中独立 |
| T4 | RENEW 改变当前序号/ACK 状态，可与旧序号 DONE 竞争；子主线程和 watchdog 的终结状态未统一 | 显式 CLOSE_REQUEST / CLOSE_GRANT / DONE 握手，固定最终租约，并串行化子端状态与所有控制写入 |

本轮仅写本规格、配套计划、作者自审/授权记录及 README 进度并归档 GitHub。不改源码、测试、协议、既有日志/证据、结构图、正式报告；不读真实 prepared/fit/stop、原图、缓存或权重；不运行测试、前向、拟合、计时或原生电源/进程验收。不得新派实现或复查代理。

## 2. 继承合同与非目标

本规格补充而不覆盖[上轮执行合同规格](2026-10-16-msb-execution-contract-repair-design.md)及[科学规格](2026-10-14_margin_scaled_bilinear_role_evidence_design.md)。已修复的严格 fit-only 与拟合前三折联合门保持原样。

- 三个独立 rank4 分支；1083 当前参数及 1083 固定参照；h64 / e23 / log_anchor4；T/Ti/G/M 为 0/1/2/3。
- `R_ref=2*abs(m)+0.25`、`d=R_ref*tanh(s)`、损失、初始化、优化器、11 单元不变。
- 既有 340 个已暴露 fit 分组及 [66,130,97,47]；三份原登记 prepared；不读 stop 或原图；0 次新 PCA/T0。
- 3 折、种子 20261002/20261003/20261004；63 条神经轨迹 + 12 次凸拟合 = 75；每条最多 400 步，主点 100。
- 80 job = 3 load + 75 fit + analysis + final-verify；最终 615 次验证前向 + 12 次训练侧证书核验；原七参照晋级门和 NO_ROLE_PROMOTION 不变。
- 原始累计预算 7200 秒；相等即过期；不新建时钟原点、不因握手/收尾延长租约或预算；失败清理 wait 至多 5 秒，不能用于计算。
- 43 个执行源码闭包不增减；旧 34 源码、四个科学模块、已接受规格、83/121 项历史证据、六张用户 PNG 和两处正式报告均保护。旧失败/合成目录不续跑、不覆盖。

不改 CNN 或绘制新结构图，不增加网络性能/原创性主张，不把工程状态机公式写成论文理论贡献。原生不可中断系统调用、断电、硬杀及解释器异步中断极窄窗口的完整清理仍需分开评估；本规格不承诺通用硬实时或崩溃一致性。

## 3. 方案选择

| 方案 | 取舍 |
|---|---|
| 增大等待、允许 DONE 使用任意较旧序号 | 改动少，但会掩盖重放/租约缺口，不采用 |
| 显式终结握手 + 有界退出排空 + 启动所有权记录 + 单一成功发布 | 采用；只改 MSB 执行层和对应 fake 验收，可逐条检验四个根因 |
| 换用新 IPC/任务框架或重写旧 launcher | 超出修复需要，牵动旧 34 源码及部署环境，不采用 |

下面是待审核的接口决策，不是已经实现或验证的行为。未来实现使用原生方式，不自动派发代理。

## 4. T1：单一成功发布与失败撤销

保留 `run_stage(root, protocol_path, output, stage)` 公共签名。新增私有接口：

`_publish_terminal_state(output: Path, state: dict, *, budget: RunBudget, api) -> dict`

它在父监督退出、全部 worker 退出及电源释放证据已核对之后调用。成功候选的 history、power、elapsed 等全部元数据在最后成功写入之前整理好；删除现有“try 内写成功，公共尾部再写一次”的结构。

顺序必须是：检查原预算 -> 原子写候选终态 -> 再检查同一预算 -> 返回。成功分支在该次写后检查与返回之间不得再写 stage_state 或其他成功交付文件。CLI 不能以旧 delivery_verified 标志覆盖失败状态。

任一检查/成功写入异常，包括仅最后写入的 fsync/replace 耗时跨期：候选立即撤销；返回/抛出的非成功结果必须使 `delivery_verified=False`，预算问题为 TIMED_OUT，其他问题沿用 terminal_status。失败记录独立作为清理写入，允许晚于截止时间，但不得恢复成功。

失败记录也写不出时，保留原异常与发布失败信息并使 CLI 非零，不报告“撤销已持久化”。磁盘候选可能残留或写入失败，不能单凭旧 COMPLETE 文件判定交付；既有独占目录和不续跑合同继续有效。该限制不是放行超时成功，也不是断电/硬杀时绝对原子撤销的证明。

elapsed 的采样点应标明，不能将写前采样宣称为含最后 I/O 的精确总时长；写后预算检查结果可在返回/诊断中保留，但不得为了补记它再次进行无检查的成功文件写入。

## 5. T2：启动所有权与不确定性

在现有 msb_execution_safety.py 定义 `WorkerLaunchOwnership`，不增加执行模块。由父 run_job 在调用适配器之前创建，字段：

| 字段 | 初值及含义 |
|---|---|
| attempt_started: bool | False；紧邻 Popen 调用之前改 True，之后不得恢复 False |
| process: object 或 None | Popen 返回后立即记录，只允许同一对象，不由日志退出/恢复标志失败清空 |
| creation_uncertain: bool | False；已尝试而无可恢复对象、无法确认是否创建/退出时为 True |

接口变为 `NativeMSBAPI.start_worker(job: dict, child_handles: dict, *, ownership: WorkerLaunchOwnership) -> object`。假 API 使用相同签名。

日志上下文退出、继承标志恢复、普通异常和 KeyboardInterrupt/SystemExit 均通过覆盖整个启动阶段的 BaseException 路径，保留已知 process、原错误及次级恢复错误。现有 _WorkerStartError 扩展为携带这份 ownership；不能只覆盖 finally 中的恢复错误，也不能用次级异常抹去主因。

父端正常返回、启动异常和清理路径均从同一 ownership 取进程。已知进程必须执行原 Job 终止或 kill，再 wait 确认退出，并尝试关闭全部已拥有句柄；wait/释放不确定即失败。

若在调用期间被中断而没有可恢复进程对象，不能假定没有子进程。记录 attempt_started=True、creation_uncertain=True、cleanup_confirmed=False、alive=True（保守地表示未排除存活）、status=SUPERVISION_INTERRUPTED；不派下一 job，不声称清理成功。Popen 之前日志打开/句柄设置失败且 attempt_started=False 才可报告 attempts=0。

对解释器在调用返回与记录对象之间被硬中断的窗口，本规格只要求诚实标记不确定，不宣称 Python 字节码原子或必能取得 PID。禁止扫描用户进程、猜 PID、启动第二个 worker 或依赖子端断管自行退出作为清理证据。所有启动路径至多调用 Popen 一次。

## 6. T3/T4：显式终结握手

### 6.1 帧和阶段

控制帧仍为 4-byte 大端长度 + UTF-8 JSON、单条最多 8192 bytes、精确原字段集合；新增 kind 仅 CLOSE_REQUEST 和 CLOSE_GRANT。两者 job_assigned=True、power_released=True；保留所有 run/job/packet/protocol/nonce/PID 创建身份、start/deadline 检查。ACK/RENEW 的 power_released=False 不变。不能通过删字段或忽略序号“修复”。

子生命周期：BINDING -> WORKING -> PUBLISHING -> CLOSING -> CLOSED。
CLOSING 不读科学输入、不拟合、不做前向、不重新申请电源；开始前须已有真实 guard release 证据且结果文件写前/后检查通过。

父端在初始 GRANT 后使用纯控制状态 `MSBParentControl`，记录 issued=(sequence,lease)、last_acked、至多一个 outstanding renewal，以及 RUNNING / CLOSE_WAIT_ACK / CLOSE_GRANTED / DONE。它不处理科学数据和真实 OS。

### 6.2 确定的消息规则

1. 子主线程在同一 RLock 下检查 PUBLISHING、当前有效预算/租约，转 CLOSING，捕获当前已确认序号及 lease，发一次 CLOSE_REQUEST。所有 ACK、CLOSE_REQUEST、DONE 的构造/状态改变/整帧写入都在该锁下；不能仅锁低层 WriteFile。watchdog 仍是唯一读者，主线程不竞争读端。
2. 父接受 CLOSE_REQUEST 时冻结新 RENEW。请求 pair 必须精确等于 last_acked；只允许 issued==last_acked 或 issued.sequence==last_acked.sequence+1 且该次唯一续租尚未 ACK。后者不是放行任意旧序号，必须保留已确认 pair 与唯一在途 pair 的完整身份。
3. 没有在途续租时，父立即发当前 pair 的 CLOSE_GRANT。有在途续租时先进入 CLOSE_WAIT_ACK，继续接收该次精确 ACK，再发 CLOSE_GRANT；期间不再产生下一次续租。原预算、有效租约、接电与活性检查仍执行。
4. 子 watchdog 在 CLOSING 中仍处理已经在途的合法 RENEW，按原规则更新 pair 并串行 ACK；收到 CLOSE_GRANT 时必须与当前 pair 完全一致，设置终结许可事件。任何再次 GRANT、错 lease、错身份、跳序或重复终结许可都失败。
5. 主线程不持有 RLock 等待许可事件；由原 scope.check 与既有 API 等待适配维持检查，目标轮询 0.05 秒。等待不新增时间额度。收到许可后在锁下复查 phase/pair/预算，发唯一 DONE，转 CLOSED，再在锁外停止并确认 watchdog，最后释放句柄。
6. 父只在 CLOSE_GRANTED 下接受精确当前 pair 的 DONE 且 power_released=True。已接受 DONE 后不再读/续租该管道，继续监督原进程退出。done + exit_code==0 + wait 已确认 + guard release + 原预算有效才可能 COMPLETE。

在途 RENEW 先于 CLOSE_GRANT 写入同一父到子管道；子端 ACK 与 CLOSE_REQUEST/DONE 的写顺序由同一锁及 FIFO 保证。父需记住 last_acked，不能以一个 acked bool 丢掉上一已确认 pair。

### 6.3 两个必须覆盖的交错

| 事件次序 | 预期 |
|---|---|
| 子已发 CLOSE_REQUEST(q,Lq)，父尚未读取而发出 RENEW(q+1,Lq1) | 父读请求后冻结续租，等 ACK(q+1,Lq1)，再 CLOSE_GRANT(q+1,Lq1)，最后 DONE(q+1,Lq1) |
| 子先处理 RENEW 并写 ACK(q+1,Lq1)，后发 CLOSE_REQUEST(q+1,Lq1) | 父先确认 ACK，再接受请求并 CLOSE_GRANT，随后 DONE；不得无故失败 |

若先收到 CLOSE_GRANT 之前的 DONE、缺失在途 ACK、任意再老序号、延长 deadline、错误 packet/nonce、重放请求或迟到 RENEW，则失败。不以自动重试、忽略错误消息或宽松 seq<=current 判定替代状态机。

### 6.4 已退出进程的有界排空

新增内部 `MSBParentSupervisor._drain_control(channel: dict, control: MSBParentControl, binding, *, max_frames: int = 8) -> int`。每消费一帧前后检查原预算与父 guard；没有完整消息返回，不睡眠等待新消息；DONE 后停止读取。一次最多 8 帧，不读张量、不加科学任务。

主循环常规排空后观察进程；若 poll 显示退出，再做一次同样有界的排空，然后才判终态。这样覆盖“第一次读暂空 -> 子写 DONE 并退出 -> 父 poll 看到退出”的窗口。此时必须已有完整合法的 CLOSE_GRANTED/DONE 证据；非零退出、缺帧、坏帧/截断、控制序列超界或排空上限耗尽且未完成，均不成功。

不得将“看到退出”写进 read_message 的 fake 副作用。退出码变化由脚本化调度事件单独发生；测试必须在父读到 DONE 之前就可观察 exit_code=0。关闭后 EOF 仅在已验证 DONE 后可结束读；无 DONE 的断管不能视为正常完成。

## 7. 内部接口与执行身份

未来仅修改两项执行源码：scripts/msb_execution_safety.py 与 scripts/run_msb_role_mechanism.py。纯状态类仍在前者，不新建第 44 个执行文件；data、training、analysis、model、controls、audit、figure 及旧 34 源码保持字节身份。

新增/修改的具体接口由配套计划固定；基本职责是：
- WorkerLaunchOwnership：启动尝试与对象所有权。
- MSBParentControl：精确序号、续租与终结状态机，输出待发送帧；显式接收 granted_tick_ms 作为初始续租原点，不从截短的 lease 倒推。
- MSBWorkerScope._finish()：单次终结握手，watchdog 仍独占读端。
- _publish_terminal_state：成功发布的唯一尾部与失败撤销。

父 run_job 正常结果增加 end_handshake_verified 与 exit_confirmed，run_stage 必须检查两者为 True；旧 parent_binding_verified、job_assigned、attempts、alive、清理及电源证据不能被新字段替代。

未来执行身份升级为 `execution_contract="msb_parent_bound_v2"`，新增 `terminal_spec_sha256` 对本规格 LF 规范文本取 SHA-256；保留旧 repair_spec_sha256 与科学 spec_sha256 的原身份。不修改或复用已归档旧协议；旧 v1 协议不得自动升级。源码闭包数量仍为 43，但这两项字节变化必须重新登记全部闭包身份。

此次文档授权 ID、先前合成开发/复查授权都不能作为真实运行授权。协议版本与授权拒绝测试属于后续代码/合成范围；本轮不创建真实协议文件。

## 8. 后续合成验收合同

若用户另批开发，只能使用假 API/假时钟/假进程/假句柄及新建临时合成 fixture。新增 tests/run_msb_terminal_repair_acceptance.py 为固定白名单守卫入口，不修改旧验收器或旧日志。

输出根固定 outputs/msb_terminal_repair_synthetic_v1，阶段为 task1-red / task1-green / task2-red / task2-green / task3-red / task3-green / final，均独占且不覆盖；它们是同一获批开发阶段内的 TDD 证据，不是真实实验重跑。阶段不存在以外的网格/任意模块/任意输出路径选项。

禁止未模拟 NativeMSBAPI 构造/tick、WinDLL、Popen、os._exit；torch.load 只允许该计划合成 scratch 或 BytesIO；OS 库枚举仍模拟，不能用改网络计算取得绿灯。守卫应早于被测模块导入生效，必要的导入期接口以 mock 包装而非执行 OS 探查。

final 明确运行原九模块，不用 test*.py 广域发现；旧 121 项测试的合同语义保留，旧终结 fixture 可按新协议调整，但不得删除负例、将 seq 检查 mock 掉或让读 DONE 再驱动退出。计数按实际日志登记，不提前宣称新的通过总数。

反例、正常交错、预算等号、异常信息和清理不确定都要保留 red/green 与源码身份。失败证据不能用重建快照冒充当时原件；在每次运行前直接保存被运行验收器原文和 SHA。未模拟边界调用或白名单外输入被阻止也要报告为验收失败，不因异常被测试捕获而记成成功。

## 9. 文件保护、审阅与恢复研究主线

本轮不改变 CHANGES_REQUIRED / 原生 INSUFFICIENT_EVIDENCE。配套计划所有任务保持未开始；作者文档自审不是独立复查。

后续顺序：用户审核本规格与计划 -> 另批原生开发及合成验收 -> 独立只读复查另批 -> 原生无数据验收另批 -> 新真实协议登记与真实运行分别另批。不得自动重做六分钟空闲、计时探查或续跑旧失败目录。

工程修复仅移除验证障碍。研究主线仍是 B：验证既定 MSB-RRE 非线性角色视觉证据与间隔预算的条件性质，并对全部七个充分求解参照检验实际角色净纠错、误入目标及 Macro F1；C 阶段条件化决策保留后续方向。现有条件性公式、合成见证或工程安全不能代替原创性查新、真实强对照效果及泛化证据，也不能提前写入正式报告成果。
