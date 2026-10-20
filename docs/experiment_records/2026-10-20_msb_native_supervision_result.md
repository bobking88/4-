# MSB原生无数据监督验收：单次失败结果

日期：2026-10-20。结果：`FAILED_NATIVE_SUPERVISION`，实际collector退出码1，`delivery_verified=False`。**未通过，不放行真实科学实验。**
真实授权：`call_3f51cdd2cc0040929106824e0c335950`；运行前登记提交：`6b32e84322b63a50d2e877590425858e64c4318e`，已独立读回7项登记文件和不变源码树，见[授权登记](2026-10-20_msb_native_supervision_execution_registration.md)。
仅调用一次既有collector，无重试、续跑或替补；`outputs/msb_native_supervision_v1/acceptance_once`已消费，保留不删除。本轮没有改执行器或协议来挽救失败。

## 实际经过与双层错误

[实际终态](../../outputs/msb_native_supervision_v1/acceptance_once/stage_state.json)显示登记和源码核验为真，进入N1后失败；累计记录35.7809999999954秒，低于480秒接收窗口。不是480秒超时负例通过。
只产生N1目录，collector已接收case_records为0，N2/N3/N4未启动，也不能声称N1完成45秒工作或通过续租/关闭。

1. **N1身份握手被拒绝。** [controller结果](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller/controller_result.json)的MSB状态为SUPERVISION_INTERRUPTED，错误为`worker HELLO identity differs`。原生[trace](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller/trace.json)记录spawn PID=11680、已分配Job，而HELLO中的worker_pid=23980、parent_pid=21844；parent_binding_verified=False，end_handshake_verified=False。trace只有spawn、assigned、HELLO、terminate_job，没有成功GRANT/ACK/RENEW/DONE链。
2. **collector观察器结束检查失败。** 根终态body错误为`SUPERVISION_INTERRUPTED: observer did not stop before final sample`，因而没有接收N1记录。[controller日志](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller.log)另保留实际CLI的`Unexpected inner case outcome`堆栈。不能只写外层观察器错误而丢掉内层身份问题。

以上是本次日志确认的现象，不是已证明的Windows、虚拟环境启动器或线程调度根因；本轮没有追加探针、改代码或复现。不能把拒绝错误身份的安全关闭改写成“正常路径成功”。

## 清理与证据不足

MSB回执针对**受跟踪的PID11680**记录exit_code=1、alive=False、exit_confirmed=True、cleanup_confirmed=True、creation_uncertain=False；trace记录terminate_job。N1父guard及collector根guard的power_request_released均为True。
由于实际HELLO身份绑定失败，不能据此宣称实际worker身份、Job因果、所有线程停止或所有句柄零泄漏均已证明。根观察线程的正常停止仍有明确错误，不能补写为已确认。
故障后只读检查本次记录的35876、21844、11680、23980四个PID，未发现仍存在的进程；没有额外终止进程。这是安全清理核对，不是新测试，也**不是保留句柄退出或Job清理因果证明**，不把PID缺失当通过证据。

[controller阶段文件](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller/stage_state.json)仍为旧RUNNING；保留原件，不伪造撤销或把它当仍在运行。实际collector工具session已终结、根终态失败、实际返回1。本授权即已消费，不因局部RUNNING或缺例而续跑。
[N1 worker日志](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller/runs/N1/worker.log)为空，这是实际原件，不能据此补造worker成功日志。

## 原件与归档检查

- [实际stdout](../../outputs/msb_native_supervision_native_result_v1/collector_stdout.log)为两条JSON记录：LC外层progress和collector最终失败；[stderr](../../outputs/msb_native_supervision_native_result_v1/collector_stderr.log)为空。根stdout终态与磁盘FAILED字段、run_id、登记提交及错误匹配，不只依赖磁盘判断。
- 最初将整个stdout按单JSON解析时因两条记录报额外文本；之后仅按行解析既有文件，未重跑原生任务、覆盖原文或修改代码。此归档解析问题不当作新的运行失败原因。
- [实际登记协议副本](../../outputs/msb_native_supervision_v1/acceptance_once/registered_protocol.json)与批准协议对象一致；[packet](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller/runs/N1/packet.json)仅为native-supervision-no-data元数据，不是矿物输入或训练结果。
- [原件清单](../../outputs/msb_native_supervision_native_result_v1/raw_manifest.json)含9份原生输出和2份入口日志；入口两日志为逐字节一致复制，所有内部原件不改写。只发布已检查的不含原始nonce/继承能力句柄的元数据和日志，trace保留nonce_sha256摘要；此检查不是一般安全证明。Git文本会规范化CRLF到LF，原始SHA-256、规范化LF SHA-256及Git blob分别记录，不混称远端原始字节哈希。
- [工具及观察记录](../../outputs/msb_native_supervision_native_result_v1/entry_observation.json)、[结构化结果](../../outputs/msb_native_supervision_native_result_v1/run_result.json)、[执行前保护](../../outputs/msb_native_supervision_native_result_v1/protection_before.json)、[归档后保护](../../outputs/msb_native_supervision_native_result_v1/protection_after.json)。175条执行前记录在运行后、归档前全部匹配；归档只允许README追加失败状态，其他174条匹配。两处正式报告均维持原哈希，不读正文、不编辑。

## 科学边界与后续

实际无数据入口记录real_data_opened=False、real_fits=0、model_forwards=0；没有读取真实prepared/fit/stop、原图、缓存、权重或.pt数据，没有做模型前向、训练、PCA/T0、额外计时/空闲探查、合成重验或代理派发。既有235项合成证据、独立复查、科学源码/损失/公式/矩阵及正式报告不改。
本次是原生工程失败，**不是MSB-RRE网络方法的正/负收益证据**，也不是角色模型科学晋级门的一次有效实验；不能把工程状态写成理论创新已验证、工业应用或网络无效。

本轮到失败原件归档为止。建议下一步仅获单独授权后做离线原因分析，区分进程身份不一致与观察器结束问题，先形成可审查的定位/最小修复建议；不先猜测修改PID校验、放宽身份/线程门、换解释器重跑或扩大实验。
任何修复、复现、合成或原生重验、真实机制实验仍另批；未来若运行必须使用新协议/新输出，不能删除或续跑此次已消费目录。
