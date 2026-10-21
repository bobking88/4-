# 实际解释器身份与观察器诊断修复：作者合成验收

日期：2026-10-21。基线`7d42505c300dd19258b72b505b917771833d62fb`。
人类授权`call_2790965a504c46dea799f87f8e9be324`：“认可文件，批准代码与有限合成验收（推荐）”。
状态：**AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING**。仅作者工程与合成接收；未做独立复查、原生验收或真实机制实验。

## 1. 完成范围

按[已审核规格](../superpowers/specs/2026-10-20-msb-python-launch-observer-repair-design.md)及[实施计划](../superpowers/plans/2026-10-20-msb-python-launch-observer-repair.md)，只改六个指定文件：
- [msb_execution_safety.py](../../scripts/msb_execution_safety.py)：实际解释器启动描述、venv上下文、OS映像/PID/创建时间核验及worker上下文前置门。
- [run_lc_role_adaptation.py](../../scripts/run_lc_role_adaptation.py)：仅native controller job opt-in，普通未标记job不改写。
- [check_msb_native_supervision.py](../../scripts/check_msb_native_supervision.py)：两层接线、v2命名、类型化停止、原始错误与受限序列化、一次停止缓存、期限耗尽的停止信号。
- [safety测试](../../tests/test_msb_execution_safety.py)、[driver测试](../../tests/test_check_msb_native_supervision.py)与[有限验收器](../../tests/run_msb_native_supervision_synthetic_acceptance.py)：固定12方法、六文件直接快照、精确保护豁免和runtime/thread守卫。

HELLO字段、PID/创建时间、nonce、Job、租约和消息强校验不放宽；不按HELLO自报PID重新认领，不新增后代进程搜索或补跑。原480秒/N1--N4/30秒gap/2秒漂移/7200秒科学预算不变。
新原生v2仅代码命名，不创建真实v2协议或目录，不复用已消费v1许可/输出。网络、损失、公式、实验矩阵和正式报告未变。

## 2. 实际验收与失败保留

所有调用均为既有venv运行固定入口`tests/run_msb_native_supervision_synthetic_acceptance.py --phase <固定阶段>`。
`contract/worker`阶段两模块、`collector`阶段一个模块、final固定十模块；没有unittest discover、LC整模块、独立计时/空闲/进程验收或代理。

| 阶段/attempt | 实际方法数 | 结果 | 非通过原因 |
|---|---:|---|---|
| contract-red/001 | 133 | FAILED，预期RED | 5处缺失helper/v2断言；0错误/跳过/未模拟调用 |
| contract-green/001 | 133 | PASSED | 全通过 |
| worker-red/001 | 135 | FAILED，预期RED | 5处缺标记/子上下文拒绝断言；0错误/跳过/未模拟调用 |
| worker-green/001 | 135 | FAILED | 15处NameError，校验语句误落到_worker_packet，runtime变量未定义；原件保留 |
| worker-green/002 | 135 | PASSED | 修正插入位置后全通过 |
| collector-red/001 | 86 | FAILED，预期RED | 4处缺失类型化停止结果断言 |
| collector-green/001 | 86 | FAILED | 新watch夹具未注入monotonic；6断言失败、7次调用被守卫拦截，未实际执行时钟接口 |
| collector-green/002 | 86 | PASSED | 加入局部fake clock后全通过 |
| final/001 | **247** | **PASSED** | 0失败/错误/跳过/预期失败/意外成功/未模拟受限调用 |

共七阶段、9个独占attempt，每阶段不超过3次。失败日志、summary和直接运行前source_snapshot均保留，没有覆盖或删除。原235个完整测试名全保留、新增12名，核对的是实际final日志，不是只看tests_run计数。
[最终summary](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/attempt-001/acceptance_summary.json)、[完整日志](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/attempt-001/synthetic_tests.log)、[直接快照](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/attempt-001/source_snapshot.json)。

日志中打印的“contract-green、单个skip”属于既有严格门反例在fake ROOT下的故意失败样本，不是本根额外阶段或实际final跳过；新T12也在各自fake ROOT里验证误触runtime/thread/time/network时会拒绝。这些子例不冒充真实native运行，也不抹去collector-green/001真实非零守卫计数。
最终记录2次scratch内合成张量读取，仍为获批的合成测试；不把它写成“从未读取任何.pt”，也不混同真实prepared/fit/stop。

## 3. 边界与证据

[本轮授权](../../outputs/msb_python_launch_observer_repair_synthetic_v1/authorization.json)、[来源身份](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/source_identity.json)、[范围和完整尝试表](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/preservation_and_scope.json)、[前保护](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/protection_before.json)、[后保护](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/protection_after.json)共同归档。

205条历史保护记录中，只允许六文件的精确路径及README状态追加；不把重复路径数当不同文件数。原41项其他执行源码、九项其他测试、历史合成/原生失败原件、旧v1协议、正式报告两处及六张原有脏PNG保持哈希。
后保护为196条不变、8条六文件变更记录（含历史重复路径）、1条README追加。另存[27份本轮原件的哈希清单](../../outputs/msb_python_launch_observer_repair_synthetic_v1/final/raw_evidence_manifest.json)，逐项区分本地原始SHA-256、规范化LF SHA-256和Git blob身份。
本轮46项源码闭包及额外safety测试摘要是作者来源快照，**不是新原生登记许可**；与final前直接六文件快照比对一致。Git文本规范化与原始文件哈希不混称。

第一次归档核对的JSON解析受到Git换行警告前缀影响；之后只重新执行只读文件/JSON核对，没有重新运行测试或任何原生入口，也未改失败原件。
所有原生进程/线程启动与join、runtime映像查询、时钟、电源/网络边界均在合成调用中模拟或由守卫阻断；框架数值bootstrap按原合同保留。没有读取真实图片、缓存、权重、prepared/fit/stop，没有真实训练、真实模型前向、PCA/T0或新环境探查。
这不是对本机venv配置、WinAPI路径兼容性、真实线程清理或任意异常字符串脱敏的完整证明；需在后续独立复查和另批的原生验收中确认剩余边界。

## 4. 作者裁定与交接

1. 直接批准的六文件/七阶段范围优先于通用技能中的额外命令、代理或scratch删除；本轮不派代理、不扩大验证范围。
2. 已审核规格和计划保留原文；私有ledger按实际顺序记录完成与失败，不回填历史文档为“早已完成”。
3. T12验证先行建立的守卫支撑，初次RED时它已通过，不能声称每个新增方法都在同一次RED中失败；它的负例实际误触守卫并断言拒绝。
4. runtime_venv_root采用明确的Windows绝对路径字段；本轮仅合成fixture值，不读取本机venv配置来补值。
5. controller/observer共用小型bounded-stop helper，在期限已尽时仍设置停止信号、零join预算并保留原始deadline错误，不引入新预算。
6. 上述归档解析问题只影响后处理，不归咎于模型或将其算成额外实验尝试。
7. 首次暂存后的通用空白检查发现原始unittest失败日志行尾空格，提交因此未执行；不修剪原件。仅对九份原始synthetic_tests.log免做源码空白门，其余32项仍检查，全部27份原始证据继续按哈希核对。此处不是测试接收门放宽。

下一步建议只单独批准一次只读独立工程复查，范围为六文件、12新方法、247项及全部失败证据、来源与期限/清理边界。未经新批准不派发。
独立复查、v2原生登记与验收、真实fit-only科学实验分别另批。科学目标仍是验证角色非线性证据与间隔预算相对强对照的收益，不能把247项工程通过写成网络理论创新有效或无效。
