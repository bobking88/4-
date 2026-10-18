# MSB原生无数据监督验收 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 当前只获文档授权，以下代码、合成运行和原生阶段均未开始；不自动派代理。

**Goal:** 用限定无数据驱动验证现有MSB原生监督器的四条执行路径，形成不越界的工程证据。

**Architecture:** collector调用既有LC外层超时回收，controller调用原MSBParentSupervisor，worker调用原open_worker_scope；不重写监督器。正常与故障路径均保留进程创建身份、原生消息摘要及退出观察，工程结果不等于网络效果。

**Tech Stack:** 既有Windows/CPython环境、stdlib、现有MSB/LC/NativeEnvironmentAPI；无新依赖。开发合成阶段所有原生进程/句柄/时钟/电源/线程与外部命令都fake。

**Spec:** [MSB原生无数据监督验收规格](../specs/2026-10-18-msb-native-supervision-acceptance-design.md)。

## Global Constraints

- 文档授权call_a42aed1b7bab474bb17246c3df69dc4e仅允许当前两份文件、记录和README归档；没有代码或运行许可。
- 现有43项执行源码、10项测试/验收器及全部旧证据不变；仅未来新增三个验收代码文件。
- 原生整体接收窗口固定480秒；RunBudget仍精确7200000ms，各controller/worker同一起点；等号过期，不倒移起点或按场景重置。
- N1/N2/N3/N4固定顺序，外层限额90/45/75/45秒；一次collector、最多4个controller和4个worker，每例一次，无重试、续跑或替补。
- N1工作45秒，N2退出23，N3正常DONE后等35秒，N4父故障退出31；N2/N4身份ready-ACK限5秒，N4退出观察限5秒。
- require_ac=True；原30秒poll-gap、2秒双时钟差、30秒租约、15秒续租及5秒cleanup不改，不做键鼠空闲/主动睡眠/改电源方案。
- 不读取真实prepared/fit/stop、原图/缓存/权重，不做真实训练/前向/PCA/T0，不改网络/损失/矩阵或正式报告。原生驱动不调用模型；未来合成开发仅按固定模块实际执行合成数学/梯度/小张量求解，不能宣称全部计算被mock。
- 原生协议/输出不在当前创建；实际登记、运行与真实MSB实验均另批。独立复查是否派发也另批。

## Review Focus

1. 正常45秒用例只有一条初始ACK却误称多次续租：必须真实RENEW/ACK至少两对且最终sequence>=3；Task2。
2. N4观察者复制Job句柄或先杀child导致假“父死即回收”：只能保留查询/同步进程句柄、精确创建身份，观察退出早于任何fallback；Tasks2/3。
3. MSB固定两小时被改成480、按例重置，或LC外层timeout被当MSB通过：共同原预算、另算480及分类接收；Tasks1/3。
4. 主流程只调用wasSuccessful或跳过缺例/异常发布后仍PASS：严格合成门、四例齐全且终结写后复查，双失败非零；Tasks1/3。
5. 不完整启动/AC未知/实例旧目录/未授权或远端身份差异：阻止后续原生动作，证据不足不重试、不读真实输入；Tasks1/3。

## Files And Interfaces

Future create:
- `scripts/check_msb_native_supervision.py`，生产验收设施，支持固定collector/controller/worker模式，默认拒绝原生执行。
- `tests/test_check_msb_native_supervision.py`，新增合同测试模块。
- `tests/run_msb_native_supervision_synthetic_acceptance.py`，新增固定范围合成入口。

Do not modify existing production/test files. 科学闭包43项不增加新驱动。
未来纯接口定义在新驱动：
- `validate_native_protocol(root: Path, protocol: dict, *, approval: dict) -> dict`：返回固定纯校验结果，拒绝旧许可/目录及真实输入字段。
- `remaining_acceptance_seconds(start_monotonic: float, start_tick_ms: int, *, now_monotonic: float, now_tick_ms: int) -> float`：480减两类已耗时的最大值，非有限、now小于起点、等号失败；跨调用倒退另由既有guard连续性检查，原生对象构造前的耗时由monotonic覆盖。
- `run_controller(root: Path, output: Path, case_id: str, binding: dict, *, services) -> dict`：仅生产MSB监督器入口，故障触发限定当前controller。
- `run_no_data_worker(packet_path: Path, handles: dict, *, services) -> dict`：仅生产scope及固定等待/故障，不进入科学worker。
- `classify_native_cases(case_records: list[dict], *, outer_evidence: dict) -> dict`：检查固定四例、真实因果/身份/清理证据，默认不通过。
- `run_acceptance(root: Path, protocol_path: Path, *, execute_native: bool, approval: dict, services) -> dict`：collector唯一入口，services注入仅供合成测试；真实模式固定原生services，不接受CLI自定义类。
services为新模块内部一处固定适配容器，承载现有监督器/LC wrapper/clock/文件/外部登记/观察接口；不得重写MSB控制状态机或允许任意真实命令。

### Task 1: 固定合同与严格合成验收入口

**Files:** Create 新驱动、新test模块、新synthetic验收入口。
**Consumes:** 本规格、既有157项测试及严格门定义。
**Produces:** 两个纯校验函数、固定case定义、fail-closed CLI和合成入口。

- [ ] Step 1: 先写纯合同反例：480秒等号/倒退/非有限拒绝；不同case使用同一原始MSB start/deadline；源代码/原生调用替身记录证明无许可时零构造/零派生/零真实文件读取。
- [ ] Step 2: 固定入口只接受七phase：contract-red/green、worker-red/green、collector-red/green、final。新根outputs/msb_native_supervision_synthetic_v1，每phase最多3个独占attempt含失败；前六phase仅新test模块，final为原九模块加新模块。守卫必须早于项目导入，数值bootstrap边界与既有验收一致。
- [ ] Step 3: 加纯严格门及实际main接线反例，拒绝空/skip/expectedFailure/unexpectedSuccess/守卫异常；外层用例自身不skip。阻止所有真实native服务、Popen/WinDLL/线程/clock/电源/网络，torch.load不得接触项目资产。
- [ ] Step 4: 获另批开发后仅运行固定contract-red，保留实际失败/快照。有效失败须对应合同，不以导入错误冒充功能RED；至多3次，额度耗尽即停。
- [ ] Step 5: 最小实现纯协议校验和fail-closed入口，固定值与规格一致；运行contract-green。快照直接捕获三新文件及现有53项身份，不能重建冒充运行前原件。
- [ ] Step 6: 记录真实计数和失败原文、提交；不预设测试总数，不改旧验收根/旧日志。计划复查不自动派发。

### Task 2: 真监督器复用与无数据worker的可验收接线

**Files:** Only new 驱动及test模块。
**Consumes:** Task1纯合同；原MSBParentSupervisor/WorkerScope和NativeMSBAPI。
**Produces:** run_controller/run_no_data_worker、只转发记录适配、ready/ACK与保留进程身份观察接口。

- [ ] Step 1: 用真实MSB状态机配fake native服务写四例反例，断言每例调用原run_job/open_worker_scope，不能用预制COMPLETE字典替代；不运行原80-job schedule，不引用科学worker_body。
- [ ] Step 2: N1断言WORKING持续45秒、两次RENEW和精确ACK以及正常release/close/exit；只有初始ACK或最终sequence<3即证据不足。
- [ ] Step 3: N2 fake退出23后无DONE，分别覆盖真实状态机缺关闭SUPERVISION_INTERRUPTED与原生pipe EOF异常映射FAILED，精确exit23/ready/无DONE/清理证据缺一拒绝；外层controller检查后exit0。N3真实DONE后fake推进原租约至等号、确认原MSB失败而非LC timeout；保留故障点顺序。
- [ ] Step 4: N4断言observer只保留PID+created匹配的查询/同步process handle，不复制Job/channel；先worker WORKING+Job assigned，再collector ACK，再父exit31，再5秒内观察child exit；fallback发生、缺ready、PID重用或未assigned一律失败。
- [ ] Step 5: 在worker-red记录上述反例，再实现最小接线。只转发代理不改tick/消息/返回值、不插入生产最终接收点I/O；故障线程仅限N4当前controller，所有线程在合成环境fake。
- [ ] Step 6: worker-green通过后，记录哪些fake数学/状态机实际执行、哪些native被模拟。停止未清理线程/句柄、重复launch或越权调用，提交，不运行原生。

### Task 3: collector有界编排、发布和证据验收

**Files:** Only new 驱动、test模块及新synthetic入口必要固定接线。
**Consumes:** Task1/2函数；既有LC run_supervised_job，原NativeEnvironmentAPI/WindowsAwakeGuard。
**Produces:** classify_native_cases/run_acceptance、终结证据与全部fake的工程验收。

- [ ] Step 1: 写反例锁定N1..N4顺序及每例一次；下一例启动前重新取min(case_limit,remaining)，全局耗尽、AC未知、远端identity差异、旧目录或缺快照时零新进程。
- [ ] Step 2: 外层LC返回TIMED_OUT/kill不能代替N2/N3/N4证据；N1/N2/N3的controller需正常exit0，N4需精确FAILED/exit31。同原始start/deadline穿过所有packet，拒绝人为倒移或新起点。
- [ ] Step 3: fake collector retained-handle观察覆盖N4 PID相同但创建时间变化、观察前误杀、Job句柄误留、5秒等号及无明确退出状态；每种拒绝PASS、停止后续，不扫描/终止无关进程。
- [ ] Step 4: 对所有正常guard记录acquired+released；N2/N4被硬退出的guard只记未显式观测，不能伪造release。观测线程不冒称拥有主线程guard；实际生产wrapper的guard parent链需有接线反例。
- [ ] Step 5: 最终证据四例齐全且根guard释放，重新核对时钟/设置/源码/零真实数据；写后480秒等号或写失败必须撤销内存PASS并非零，失败写再失败保留两异常，不声称磁盘撤销成功。空/缺例/skip均无PASS。
- [ ] Step 6: collector-red取得实际反例后最小实现，collector-green。新入口final只运行固定10模块，记录实际数和零非通过项；旧157个名称须保留，旧43源码/10测试及全部历史日志无diff，正式报告/六PNG身份不变。
- [ ] Step 7: 新代码、直接快照、合成日志和作者接收判断归档既有分支；此时仍为AUTHOR_SYNTHETIC_ACCEPTED_NATIVE_NOT_RUN。独立复查须另批一次，不能自动派代理或把合成通过改为原生通过。

Future commands after separate development approval, not run in this documentation stage:

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_native_supervision_synthetic_acceptance.py --phase contract-red
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_native_supervision_synthetic_acceptance.py --phase contract-green
```

其余固定phase按同一入口顺序使用；不能用unittest discover扩大范围，不能直接跑原生CLI验证代码。

### Task 4: 单独批准后的原生协议登记

**Files:** Future create docs/experiment_protocols/msb_native_supervision_v1.json、登记记录；不改生产。
**Consumes:** Tasks1--3真实完成且检查过的代码身份及另行登记授权。
**Produces:** 固定场景/身份/目录的待运行协议，非运行许可。

- [ ] Step 1: 获登记批准后，以实际新代码身份填入规范字段，不生成空许可或假批准；文档许可call_a42aed1b7bab474bb17246c3df69dc4e不可充当运行许可。
- [ ] Step 2: 快照固定43+3源项、零真实输入、480/四限额/1次、require_ac、旧目录拒绝和公开证据去能力化规则；保持MSB现有v2真实执行身份和原科学闭包语义不变。
- [ ] Step 3: 先推GitHub待运行稿并独立读回，再向用户申请一次明确原生运行许可；收到后只补实际授权信息、重登记并读回一致。无许可不创建acceptance_once目录。
- [ ] Step 4: 原生运行前，代码合成许可及复查许可ID同样是非原生许可；验证已登记的精确新批准，不拿任意非空ID当许可。记录登记失败即停、不自动修正协议绕过边界。

### Task 5: 另行批准的一次原生执行与研究交接

**Files:** Future outputs/msb_native_supervision_v1/acceptance_once及原生结果记录。
**Consumes:** Task4登记一致与原生运行明确许可。
**Produces:** 单次原生原件及限定PASS/FAILED/INSUFFICIENT结论。

- [ ] Step 1: 显式execute-native唯一collector入口，校验新登记/接电/新目录；持续480累计限，最多4+4进程，不扩大场景、不做空闲/待机实验。
- [ ] Step 2: 任例非预期或证据不足立即停，保留已消费目录，不重试，不因期望故障是负例而掩盖异常故障；应急cleanup动作只能确认失败，不能补成功。
- [ ] Step 3: 验收完成后核对原始时序、完整原件、身份、零真实读取、正常release及强杀路径的未观测限制，独立远端读回归档；不把原生结论回写旧157日志或正式报告。
- [ ] Step 4: 若限定原生通过，再另批真实MSB协议/一次机制实验；仍用既定11单元、三种子七参照门。无真实结果不能宣称角色网络有效，无达门则NO_ROLE_PROMOTION。

## 文档作者自审与交接

本文件是可审核计划，不是已实施清单；所有框均未勾选。任务1--3为未来新增设施与全fake合成阶段，任务4仅登记，任务5才原生运行；三种许可不可互换。
规格全部场景/时间/身份/保留事项分别落在Tasks1--5；Review Focus五项各有反例。没有预先声称未来测试数量或PASS。
文档经审核后建议原生方式逐任务实施，不自动派代理；具体开发/合成验收仍需新的明确批准。本轮只归档本规格、计划、文档记录及README，所有未来代码、协议、目录均不创建。
