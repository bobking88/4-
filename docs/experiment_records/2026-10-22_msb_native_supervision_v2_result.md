# MSB原生无数据监督验收v2：单次失败结果

授权工作包日期：2026-10-22。入口stdout最后写入时间（原文件元数据，UTC）：`2026-10-23T03:02:14.6116710Z`，不以本记录文件名前缀代替实际证据时刻。

结果：**FAILED_NATIVE_SUPERVISION，collector实际退出码1，delivery_verified=False。未通过，不放行真实科学实验。**
授权：`call_cdd31dd8986040fb8bd90d6e311f95fc`；运行前授权重登记提交：`732da9ea2355685649e97b74707c2d066aa9dd2f`。该提交7个实际文件内容已通过独立连接器回读，源码树不变，之后只调用一次collector。[执行授权登记](2026-10-22_msb_native_supervision_v2_execution_registration.md)保留，许可现已消费；不重试、不续跑、不改执行器挽救结果。

## 1. 实际经过

[根终态](../../outputs/msb_native_supervision_v2/acceptance_once/stage_state.json)记录登记与源码核验均为True，实际原生接口已执行。累计接收时间约33.547秒，低于480秒窗口；仅有N1目录，N2/N3/N4未启动。

根case_records有一份已载入N1记录，但它是在最终case校验前加入，**不等于N1已通过**。根错误为`ValueError: outer timeout or wrong controller exit`；[LC外层回执](../../outputs/msb_native_supervision_v2/acceptance_once/stage_state.json)实际为FAILED、controller exit1、alive=False、约12.969秒，不是90秒外层超时。

[controller.log](../../outputs/msb_native_supervision_v2/acceptance_once/cases/N1/controller.log)确认实际堆栈：`run_controller -> services.persist_trace -> _write_json -> temporary.open('x')`，触发`FileExistsError`，目标为`controller/trace.json.pending`。没有controller_result.json，根记录的msb与parent_power均为null，因此不能补造内层COMPLETE或正常父guard释放回执。

[trace.json](../../outputs/msb_native_supervision_v2/acceptance_once/cases/N1/controller/trace.json)和保留的[trace.json.pending](../../outputs/msb_native_supervision_v2/acceptance_once/cases/N1/controller/trace.json.pending)字节数及原始SHA-256相同，均记录spawn、assigned、HELLO、GRANT、ACK五类事件。此次已有握手/工作前置证据，与旧v1仅出现身份拒绝的现象不同，但未完成N1规定的45秒工作、两次续租及正常关闭，不能把它称为正常路径通过。

文件为何已存在尚未定位。并发写入、临时发布遗留或其他原因仍是待查假设；本轮不追加探针、复现、代码审计修复或重跑来确认根因。

## 2. 退出、清理与未证实范围

已记录controller PID23272、worker PID15516，worker保留身份句柄核对了创建身份、Job分配及退出，exit1、worker_alive=False、handle_closed=True、monitor_stopped=True。根collector和LC外层guard均记录power_request_released=True。

这些是本次实际记录所支持的限定观测。没有内层MSB回执或parent_power，不能宣称所有内层清理、worker显式电源释放或全部句柄零泄漏已经证明。本轮没有额外查询或终止进程，也没有把PID缺失当作Job因果证据。

[controller子stage_state](../../outputs/msb_native_supervision_v2/acceptance_once/cases/N1/controller/stage_state.json)仍为RUNNING，原样保留；不能据此续跑或认定进程仍活跃。实际工具session68067已经终结，collector返回1、stdout根终态与磁盘FAILED一致。

## 3. 原件与归档

- 原生目录共有12份原件，包括注册协议、根终态、ready/ACK、controller/worker日志、trace及其pending、packet和task_started。worker.log实际为空，未补写成功日志；pending不删除或覆盖。
- [入口stdout](../../outputs/msb_native_supervision_v2_native_result_v1/collector_stdout.log)含两条JSON记录（LC progress及最终失败），[stderr](../../outputs/msb_native_supervision_v2_native_result_v1/collector_stderr.log)为空。最终stdout对象与根磁盘状态一致；按行解析，没有因多行输出而重跑。
- 两份入口日志从私有捕获目录逐字节复制，私有原件保留。[raw_manifest](../../outputs/msb_native_supervision_v2_native_result_v1/raw_manifest.json)共14份证据，分别记录原始字节SHA-256、规范化LF SHA-256及Git blob，不把Git规范化身份冒充原始字节哈希。
- 公开前做了当前JSON能力键、日志/命令能力文本模式及现有日志的有限检查，未发现原始nonce或可复用句柄；nonce_sha256、协议/源码摘要与PID不是活动能力。此检查不是任意异常普遍脱敏的证明。
- 创建时间FILETIME在[结果摘要](../../outputs/msb_native_supervision_v2_native_result_v1/run_result.json)中用字符串保存，避免大整数精度损失，原始数值JSON不改写。
- [入口观察](../../outputs/msb_native_supervision_v2_native_result_v1/entry_observation.json)保留实际调用、初次工具session结果、终结观察来源、登记回读和元数据核对。[执行前保护](../../outputs/msb_native_supervision_v2_native_result_v1/protection_before.json)303条记录在运行后、归档前全部匹配；[归档后保护](../../outputs/msb_native_supervision_v2_native_result_v1/protection_after.json)仅允许README追加失败状态。正式报告及六张既有脏PNG不改、不暂存或还原。

## 4. 科学边界与下一步

packet明确为`native-supervision-no-data`，根记录real_data_opened=False、real_fits=0、model_forwards=0；没有读取真实prepared/fit/stop、原图、缓存、权重或张量，没有训练、模型前向、PCA/T0、额外计时/空闲验收、合成重验或代理派发。运行后不改协议/源码、网络、损失、公式、实验矩阵或正式报告。

这是原生执行工程失败，**不是MSB-RRE网络方法正负收益的科学证据**，不消费科学晋级门的有效实验，也不改变B主线与C后置的定位。先前253项合成通过和独立APPROVE保留其限定含义，不改写成原生通过。

本轮止于失败原件归档。建议下一步另批一次离线原因定位，仅使用此次原件及已登记源码，区分已证实的临时文件异常与待验证原因，形成最小修复建议。不自动修复、复现、测试、原生重试或扩场景；真实机制实验仍另批，已消费v2目录和授权不得复用。
