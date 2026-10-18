# MSB原生验收驱动五项缺陷的最小修复规格

日期：2026-10-18。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档授权：`call_03a86c6e5b064586b7bb980fa04b2a53`，同时批准规格、计划及GitHub归档，不批准开发或运行。
基线：`92ff34eafdfdef7a7df15b4e4528f1b2357483bd`；受审源码：`c092f1cae59fc4187670bf4bff655375309d83b0`。
依据：[单次独立只读复查](../../experiment_records/2026-10-18_msb_native_supervision_independent_review.md)，2项P1及3项P2均接收、未修复。

## 1. 目标、取舍与不变边界

修复R1--R5，使新增无数据验收驱动不误接收无效N4证据、不核验另一checkout、不遗漏末端监督/电源异常、不因guard释放失败丢失终态，也不把零工作采样开销判成工作。
采用方案A：只修改原新增三文件的局部合同、接线与合成夹具。方案B为重写生产监督器/通用执行框架，扩大影响且无必要，不采用；方案C为放宽接收或只修改日志，会保留错误证据，不采用。
这是既有验收子系统的修复规格，不是新科学网络；本轮只写文档，不改任何代码，不导入项目、不跑测试/前向/拟合/计时/原生或空闲验收，不读真实数据/原图/缓存/权重，不改正式报告，不派代理。

未来开发修改范围只有：
- `scripts/check_msb_native_supervision.py`：五项修复及必要的局部接线。
- `tests/test_check_msb_native_supervision.py`：新反例及原45项测试的合法夹具适配，不删除或改名旧测试逃避覆盖。
- `tests/run_msb_native_supervision_synthetic_acceptance.py`：新固定证据根/scratch、原严格门及必要的守卫接线，不扩大模块。

旧43项执行源码、10项旧测试/验收器、全部旧日志/规格/复查原件保持原始字节。科学闭包不变，未来原生登记候选仍为43+3项；原始SHA-256、Git blob及LF内容分别核对。
共同RunBudget精确7200000ms、额外480秒累计窗口、N1--N4顺序、90/45/75/45秒场景上限、每例一次不重试、45秒工作/exit23/DONE后35秒迟退/父exit31均不变。
原30秒poll-gap、2秒双时钟差、30秒租约、15秒续租、5秒cleanup不改。N2/N4 ready-ACK上限5秒，N4退出观察上限5秒，等号拒绝；不做主动休眠、键鼠空闲或电源方案调整。
当前REQUEST_CHANGES不因写好本规格而改变；无原生协议/输出创建，无真实机制实验许可。

## 2. R2：登记root必须是实际执行来源

采用严格绑定而非只比较内容：`root.resolve()`必须等于当前驱动`__file__`解析后的仓库根，驱动真实路径必须为该root下的`scripts/check_msb_native_supervision.py`。两个内容相同但物理位置不同的checkout也不能混用。
新增纯接口`validate_execution_origins(root: Path, origins: dict[str, str]) -> dict`；`execution_origins() -> dict[str, str]`从真实`__file__`及`sys.modules`读取来源，不能由CLI指定。观测键固定为driver、module:<模块名>:file及存在时的module:<模块名>:spec；driver单独检查，不要求脚本__main__具有非空spec。
检查已加载的闭包项目模块：按SOURCE_FILES中的脚本名对应其实际`__file__`及存在的`__spec__.origin`，两者须指向登记root内对应源文件；缺来源、外部同名模块、来源互相矛盾均拒绝。不为检查而导入全部科学模块，不扫描图片/权重或读取.pyc。
纯返回包含`root`、`driver`和已验证的`module_origins`；它是路径来源证据，仍需既有SHA/远端blob核验，不能单独证明已登记内容，也不宣称抵御恶意篡改解释器或任意TOCTOU。

公开collector/controller先保留原许可校验，再验证来源，随后才可读固定协议、构造原生服务/线程/进程。项目导入后、原生构造前再检查新增加载的项目来源；`NativeServices(root: Path)`固定所绑定的root。
私有worker仍先校验实时能力字段；仅随后读取已发布的验收packet元数据、核对packet.root/输出路径与实际来源，再构造原生服务。不能因修复来源而把磁盘元数据当作实时能力。`run_controller`及`run_no_data_worker`实际入口也校验，不只保护CLI。
controller/worker命令使用已验证的驱动路径，不再自由使用来自另一root的`__file__`；每次阶段导入及启动前复查来源/原源哈希，不能只在collector开头检查一次。
合成测试可替换`execution_origins()`的观测输入，再执行真实纯验证器及接线；不能直接mock验证器返回verified=True冒充来源检查。

## 3. R5：零工作语义

`_wait_work(milliseconds, packet, services, scope=None)`保留接口；milliseconds为0时直接返回0，不调用pause、不把两次取时的差计作工作。周边累计预算/scope检查照常保留，不把零工作当作豁免监督。
45000/35000的正时长仍走现有等待与检查；负数、布尔及非整数拒绝。N3的work_elapsed_ms==0继续表示未执行工作等待体，不放宽为任意“小于某毫秒”的工作量。
反例使用每次读取都自然递增的fake tick，验证0仍返回0且pause次数0；另保留N1工作及N3真正DONE后迟退的原接线证据。

## 4. R1：N4有效证据与清理意图

collector保留PID+created匹配的查询/同步句柄后，先query_alive确认活着，再写身份ACK，记录`alive_at_ack=True`和`ack_tick_ms`。死进程、身份不明或查询异常均不ACK、不触发预定父故障。
父故障路径在验证collector ACK后，也用只查询句柄确认同一worker仍活着，记录`alive_before_fault=True`及`alive_tick_ms`；不复制Job/channel能力。新打开查询句柄正常失败路径必须释放，真实硬退出后的显式释放不能伪造。

controller内新增局部`N4FaultGate`，由故障线程和TraceAPI共享，不替换生产监督器：
- `note_cleanup(kind: str, pid: int | None) -> None`在真实terminate_job/kill_process调用之前记录清理意图并撤销N4故障接收资格，不能仅在调用成功后才记录。随后仍须转发原清理，日志失败不能吞掉生产清理调用。
- `commit_fault(ready: dict, ack: dict, trace_events: list[dict]) -> None`只有在无清理意图、无记录/同步异常、身份/活性/期限仍有效时，固化完整故障前trace和故障元数据，然后对当前controller执行exit31；正常返回属于失败。
- 清理意图登记与故障提交使用同一局部gate排序；锁获取最多1秒且受累计剩余约束。超时或日志异常标记INVALID并禁止新故障提交，不能以缺失记录推定“没有清理”。提交窗口不得等待observer停止或持有Job句柄。
- N4临界窗口的事件排序、并发清理/故障竞争和同步失败必须用受控fake调度覆盖；保留原1:1 native调用，不把模拟的SystemExit/finally当原生因果。任意OS抢占/底层阻塞仍非本次可证明范围，原生证据不齐只能拒绝。

故障记录新增上述活性、完整trace快照和`cleanup_before_fault=False`的可追溯证据。分类器验证它们与collector身份、binding、初始GRANT/ACK、WORKING/Job assigned相符；任何清理意图/terminate_job/kill_process、缺快照或INVALID均拒绝，不能只信一个布尔。
必需顺序为`ready_tick_ms <= ack_tick_ms <= alive_tick_ms <= fault.tick_ms < exit_tick_ms < fault.tick_ms+5000`。同一毫秒无法区分故障与退出时，不宣称排序已确认；等号5000拒绝。
父退出观测可以晚于子退出观测，不能要求相反顺序；删除max(0,负差)接收方式。父实际exit31、无DONE/完整结果、outer非超时、查询句柄关闭及监控停止等原条件仍必需。
故障前主动清理、ACK时已死、父故障前已观察退出、仅靠陈旧trace宣称无清理、时间相等/倒退和5秒等号均为反例；正常“故障后子退出、稍后才观察父退出”须可通过限定合同。
上述观测是四路径内的有限证据，不宣称数学上排除所有自发退出/任意并发原因。

## 5. R3：末端AC与监督连续性

根guard仍为原WindowsAwakeGuard+NativeEnvironmentAPI；每段末端源码/设置读取、记录加载等潜在阻塞操作前后调用真实`guard.check()`，退出根guard前再检查。只比较power_settings不等于检查AC。
guard释放后不能再调用已失活的guard.check，也不能重新进入一个guard重置累计起点来掩盖断层。新增`NativeServices.postflight_sample() -> tuple[float, float]`，直接调用原NativeEnvironmentAPI.sample，因而保留AC=1检查。
新增局部`FinalContinuity(binding: dict, last_sample: tuple[float, float], *, services)`及`check() -> dict`：从根guard最后有效sample接续，使用原30/2秒限值检查有限性、倒退、采样间隔及双时钟差，并检查原480秒剩余；不发起新的电源请求、不更改RunBudget。
终态分类/发布前、PASS写入前后均执行该检查，回执只记录已发生的观测；AC未知/断开、gap>30、双时钟差>2、累计480等号或读数非法均撤销内存PASS，停止后续动作。check返回verified、sample、gap_seconds、suspend_delta_seconds及elapsed_seconds；分类器要求outer_evidence.final_continuity包含有效的实际检查记录，缺失也拒绝。
发布后观测只能由实际返回/collector输出证明，不能把未来写后成功预填进已写回执；仅磁盘上有PASS而缺少最终成功返回/exit0及写后检查证据，不足以确认交付。不声称观测间隙绝无短暂断电或磁盘崩溃一致性。

## 6. R4：释放异常仍有终态

run_acceptance以外层try覆盖整个`with guard`及其退出；区分尚未创建RUNNING与已创建RUNNING。初始化/登记失败不为留回执而创建原生输出目录；已有RUNNING则无论body、guard释放或末端观察在哪失败，均尝试发布FAILED_NATIVE_SUPERVISION，delivery_verified=False，实际CLI非零。
body异常与release异常分别记录在errors中，不让__exit__覆盖原始原因；释放失败证据不得补造released=True。根guard保持同线程退出，正常路径只释放一次。
扩展`publish_acceptance(output, state, binding, *, services, final_check=None) -> dict`：PASS必须提供真实final_check并在写前/写后执行；非PASS的失败持久化不依赖接电恢复，不触发新case或重新获得计算预算。
若失败发布也失败，原AcceptancePublicationError扩展保存完整primary_errors及persistence_error并非零传播，保留primary_error为首项的兼容别名；不宣称磁盘RUNNING/PASS已撤销。不会无限追加写入、续跑失败目录或创建替补目录。

## 7. 未来合成验收与交接

另批后使用新根`outputs/msb_native_acceptance_repair_synthetic_v1`及独立scratch；本轮不创建二者。固定七phase：contract-red/green、worker-red/green、collector-red/green、final，每phase最多3次独占attempt含失败，不覆盖、不重置计数、不增phase。
前六phase只运行既有新模块test_check_msb_native_supervision；final仍固定原10模块，原202个测试名称全部保留，按实际新增数报告，不预填通过总数。所有原生进程/时钟/电源/线程启动/网络均fake或guard；数值bootstrap及原合成数学/梯度/小张量求解按既定边界，不扩大真实输入范围。
原始RED/错误/skip/守卫调用全部保留，不能把导入/夹具错误当成功功能RED；严格门仍要求0失败/错误/skip/expected/unexpected/未模拟调用，额度耗尽即停，不自动改bootstrap白名单或重派。
未来通过仍仅作者合成接收，独立复查、原生协议登记、原生运行及真实MSB实验均另批。旧202项与本次文档不自动放行。本轮文档许可及未来代码/复查许可均为非原生许可，未来默认拒绝测试应覆盖实际新许可，不能以任意非空ID代替人类原生授权。
科学主线仍是B、C后续；角色分支利用、间隔预算跨界、相对SHARED-BIL/CONCAT-MLP/充分求解凸参照的独立收益才是需要验证的网络贡献。11单元、340已暴露fit、三折三种子、100步主点/400步上限及七参照晋级门不变；不达门NO_ROLE_PROMOTION，不将工程公式/反例写成理论创新成果。
