# MSB原生验收驱动最小修复：作者实施与有限合成验收

日期：2026-10-19。状态：`AUTHOR_REPAIR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING`。
授权：`call_d538a55e3a7b401fa4709a0d3914bb24`，当前会话实施Tasks1--3及有限合成验收，不派代理。
基线：`2b8604a82ac052d383684e5cce59a8309f323b48`。
依据：[已认可规格](../superpowers/specs/2026-10-18-msb-native-acceptance-repair-design.md)、[已认可计划](../superpowers/plans/2026-10-18-msb-native-acceptance-repair.md)及[原独立复查](2026-10-18_msb_native_supervision_independent_review.md)。

## 实施范围与五项对应

仅修改三个既有验收文件：[驱动](../../scripts/check_msb_native_supervision.py)、[驱动测试](../../tests/test_check_msb_native_supervision.py)、[合成验收器](../../tests/run_msb_native_supervision_synthetic_acceptance.py)。Task1提交`4b851bd`，Task2提交`43fe043`；本记录与Task3一并归档。

| 原反馈 | 本次作者实现及合成反例 |
|---|---|
| R2/P1 | 将登记root绑定实际driver和已加载项目模块的file/spec来源；三入口、直接调用及导入前后检查。合法临时夹具只替换观察到的来源，不替换校验器结果；错误checkout即使内容相同也拒绝。 |
| R5/P2 | 零工作等待直接返回0；类型与负值校验、正时长等待和周边监督保留，不以宽松毫秒阈值掩盖采样开销。 |
| R1/P1 | N4故障提交与清理意图共享门；ACK前及故障前查询绑定worker活性，保存独立故障快照及trace。拒绝预先退出、缺失快照、清理先行和故障至退出越界；不要求父观察早于子退出观察。查询关闭报错时仍释放门，但不声称查询句柄已成功关闭。 |
| R3/P2 | 活跃段原根guard持续check；释放后从最后sample接续AC、30秒采样间隙、2秒时钟差与480秒累计期限检查，不重新进入guard或重置预算。PASS写前/写后均检查。 |
| R4/P2 | 外层覆盖guard退出异常；业务、释放和持久化错误分别保留。已发布RUNNING后的失败必须尝试非PASS终结，实际CLI非零；尚未登记/发布RUNNING的失败不为记录错误而建立输出目录。 |

PASS写后观察属于实际函数返回值/CLI stdout，不预写入磁盘收据当作未来事实。因此磁盘PASS文件本身不充分，也不宣称进程崩溃下可撤销已写磁盘成功状态。已失败收据只做一次失败持久化；候选PASS失败保留有界失败写入和双重错误。

## 固定阶段与全部尝试

根目录：[msb_native_acceptance_repair_synthetic_v1](../../outputs/msb_native_acceptance_repair_synthetic_v1)。固定七阶段，每阶段最多3次独占尝试（含失败）；实际11次，每阶段最多2次，没有重置额度或覆盖旧目录。

| 阶段/attempt | 测试数 | 失败 | 错误 | 实际结果 |
|---|---:|---:|---:|---|
| contract-red/001 | 52 | 11 | 1 | FAILED |
| contract-red/002 | 52 | 15 | 0 | FAILED |
| contract-green/001 | 52 | 0 | 0 | PASSED |
| worker-red/001 | 62 | 15 | 0 | FAILED |
| worker-green/001 | 62 | 2 | 2 | FAILED |
| worker-green/002 | 64 | 0 | 0 | PASSED |
| collector-red/001 | 72 | 16 | 0 | FAILED |
| collector-green/001 | 72 | 1 | 0 | FAILED；时钟守卫缺口见下文 |
| collector-green/002 | 72 | 0 | 0 | PASSED |
| final/001 | 230 | 1 | 0 | FAILED |
| final/002 | 230 | 0 | 0 | PASSED |

每次原始summary、log及运行前直接源码快照均保留。最终[summary](../../outputs/msb_native_acceptance_repair_synthetic_v1/final/attempt-002/acceptance_summary.json)、[完整日志](../../outputs/msb_native_acceptance_repair_synthetic_v1/final/attempt-002/synthetic_tests.log)、[直接源码快照](../../outputs/msb_native_acceptance_repair_synthetic_v1/final/attempt-002/source_snapshot.json)为本次接收依据。
固定十模块最终230/230通过：原202个完整测试名全部保留，新增28项；原九模块157项不变，新驱动模块73项。最终失败、错误、跳过、预期失败、意外成功及已守卫的未模拟边界调用均为0。仅有2次合成scratch张量回读，真实数据读取及真实拟合为0，未执行原生OS接口。

## 失败、仪器缺口与作者裁定

1. contract-red/001的1项错误为旧代码对None产生TypeError而非约定ValueError；源码未变，改为显式断言捕获异常类型，red/002取得15项有效断言失败，不把001改成有效绿灯。
2. worker-green/001暴露门异常类型不合约定，以及正例夹具使用固定假binding而非实际job binding；分别修正操作失败类型和夹具绑定，不放宽生产拒绝条件。随后补充清理抵达故障快照期间的同门反例，使用受控fake锁，不实际等待或启动线程。
3. **collector-green/001存在一次已知未受守卫的只读真实`time.monotonic()`调用**：CLI正例替换NativeServices工厂却漏替换入口时钟，真实起点与fake时钟混用，运行在RUNNING之前失败。旧验收器未拦截/计数该调用，因此原summary的0计数不能证明该次所有时钟都已模拟。未调用Windows进程/电源或读取真实研究输入。之后正例注入入口时钟并检查调用次数，验收器在数值库bootstrap后全局守卫monotonic；collector-green/002和两次final使用修正后的守卫。失败原件不改写。
4. final/001新增反例暴露close_query异常时门未释放，其余229项通过。用嵌套finally最小修复，final/002为230/230；没有新增phase或重置尝试额度。
5. 严格验收门测试会在隔离scratch中故意注入一项skipped并得到FAILED；这是外层测试验证拒绝行为，不是实际阶段跳过或额外尝试。
6. Task1完整diff-check曾报告5处原始RED日志尾随空白和测试文件空白EOF；原日志逐字保留，EOF在后续正常追加测试时消除。不能声称历史完整diff-check清洁；当前源码/文档单独检查，原始失败日志不为格式化而改写。

计数只覆盖实际仪器覆盖范围，不是对任何未知副作用的绝对证明。框架及sklearn必要数值运行库bootstrap依原约定装载，合成数学计算实际运行；只有被限制的原生接口、线程启动和网络边界被模拟/守卫。

## 身份、保护与限制

[来源身份清单](../../outputs/msb_native_acceptance_repair_synthetic_v1/final/source_identity.json)及[保护/范围核对](../../outputs/msb_native_acceptance_repair_synthetic_v1/final/preservation_and_scope.json)记录：旧43执行源码、10测试/验收器以及其余保护文件共75条既有哈希记录匹配；已认可的3份修复文档匹配；新三文件与final/002直接快照匹配。75为记录数而非去重文件数。
原202项日志、旧规格/计划、独立复查历史未改；15个历史计划复选框仍未勾选，实际实施状态由本记录和会话ledger说明。原6张脏PNG、两处正式报告哈希未变。
46项来源清单只是未来登记候选身份，不是原生协议。未创建`docs/experiment_protocols/msb_native_supervision_v1.json`或`outputs/msb_native_supervision_v1`。

没有读取真实prepared/fit/stop、原图、缓存或权重，没有真实训练、原生进程/电源/空闲验收或新独立复查；没有修改网络、损失、公式、实验矩阵或正式报告。fake SystemExit会展开finally，不能替代原生os._exit的清理证明；有限合成反例不能证明Windows构造、IPC、Job对象、任意并发或硬实时正确性。

## 当前结论与后续门

作者侧认为R1--R5已按批准规格实现并通过有限合成接收。原独立REQUEST_CHANGES保留历史含义，**新独立接收尚未取得**；原生证据仍不足，真实实验未获本轮放行。
下一步须另批一次只读独立复查；原生协议登记/实际验收和真实机制实验继续分别另批，失败证据不续跑或改写。
论文目标不变：MSB-RRE的角色证据利用、间隔预算跨界及相对强对照的独立收益尚未由真实实验验证。230项工程检查不是230项科学效果实验，不增加报告理论创新的完成度；科学晋级门与`NO_ROLE_PROMOTION`规则保持不变。
