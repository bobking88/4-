# MSB验收驱动修复：单次独立只读工程复查

日期：2026-10-20。工程结论：`REQUEST_CHANGES`；原生证据：`INSUFFICIENT_EVIDENCE`；科学效果：`NOT_TESTED`。
授权：`call_29684c57aed94222832ace218cd392be`，仅一次独立只读复查及GitHub归档，发现问题先记录，修复另批。
审查区间：`2b8604a82ac052d383684e5cce59a8309f323b48..90c017a931fc1c4f858194502b8cfb28ddf054b8`。

## 已确认问题

**P2：首次RUNNING发布后，初始化检查与guard释放的双异常可能丢失前一个原因。**
[驱动第933行](../../scripts/check_msb_native_supervision.py:933)先写RUNNING并调用check，下一行才进入捕获body异常的try。若首次check失败，再由[guard退出](../../scripts/lc_execution_safety.py:100)抛出释放异常，外层只把后者收集到结构化errors。Python异常上下文可能仍存在，但没有被分别放进回执errors或primary_errors。
这不是误报PASS：仍会尝试失败终态，实际CLI非零；问题是错误证据保留不完整，不满足[已认可规格第6节](../superpowers/specs/2026-10-18-msb-native-acceptance-repair-design.md:67)。
[既有联合测试](../../tests/test_check_msb_native_supervision.py:1124)从N1业务失败触发，未覆盖首次RUNNING检查这个窗口。

作者仅静态核对驱动896--1005行、guard80--119行、测试1100--1179行及规格50--76行，确认该触发和需求冲突；没有执行反例或修改源码。后续最小方向是把state建立后的初始化写入/检查纳入body错误捕获，并补双异常和叠加失败持久化反例，当前均未实施。

## 五项裁定

| 原项 | 本次独立裁定 | 作者接收 |
|---|---|---|
| R1/P1：N4活性与清理/故障排序 | ADDRESSED | 接收有限合同内结论，不推广为任意Windows并发证明。 |
| R2/P1：登记root与实际执行来源 | ADDRESSED | 接收实际来源绑定判断，不推广为恶意解释器安全保证。 |
| R3/P2：末端AC、时钟及累计期限连续性 | ADDRESSED | 接收有限采样与实际写后回调合同，磁盘PASS仍不充分。 |
| R4/P2：释放失败及完整错误保留 | PARTIAL | 新确认一项P2残留，UNFIXED；不因旧230项通过而豁免。 |
| R5/P2：零工作时长语义 | ADDRESSED | 接收零等待直接返回0及保留外围监督的判断。 |

独立审查未发现其他可确认的P0/P1问题，不等于证明不存在任何缺陷。[原230项作者合成验收记录](2026-10-19_msb_native_acceptance_repair_acceptance.md)和更早REQUEST_CHANGES均保持历史内容，未改写。

## 独立原件与证据范围

- [授权与单次派发](../../outputs/msb_native_acceptance_repair_independent_review_v1/authorization.json)、[完整审查请求](../../outputs/msb_native_acceptance_repair_independent_review_v1/review_request.txt)。
- [审查回复原文](../../outputs/msb_native_acceptance_repair_independent_review_v1/reviewer_response.txt)、[终态工具返回](../../outputs/msb_native_acceptance_repair_independent_review_v1/reviewer_terminal.json)。JSON保留原始completed字符串，文本文件仅补文件终止换行，不删减审查内容。
- [结构化裁定](../../outputs/msb_native_acceptance_repair_independent_review_v1/review_result.json)、[复查前保护记录](../../outputs/msb_native_acceptance_repair_independent_review_v1/preservation_before.json)、[复查后保护核对](../../outputs/msb_native_acceptance_repair_independent_review_v1/preservation_after.json)。

唯一审查代理Boole：`01a154ef-f310-75f2-aea7-854590001d7a`，新上下文派发一次；首次等待观察超时后继续等待同一handle，第二窗口收到完整终态，随后关闭，没有重派、追问或嵌套代理。
独立实际审读范围在原文逐项列出：三个修改文件及Git差异、认可规格/计划、历史复查、作者披露、相关原监督/guard源码段、11份summary判定字段与日志汇总/失败段、指定快照和身份字段；没有冒称完整审读全部旧43源码/10测试或全部保存记录。
复查确认旧202个测试名保留、新增28项及最终日志230/230；这是只读核对旧证据，没有重跑任何测试。collector-green/001已披露的一次未守卫只读monotonic调用仍按失败和仪器缺口保留，原0计数不能解读成完全模拟。

## 暂不裁定事项的逐项处理

1. 父退出观察晚于子退出观察：接收合法观测顺序，不新增缺陷。
2. run_outer导入疑点由审查代理按周边合同排除；保留其说明，不宣称父会话重新完整审查。恶意解释器、任意TOCTOU/字节码/伪造记录的安全证明不在本次范围，不增加安全保证。
3. N4查询/日志/同步失败的有限路径：接收限定判断；任意抢占、阻塞和Windows调度仍未知。
4. fake SystemExit与原生os._exit的差异，以及真实IPC/Job/线程/电源行为：保持原生NOT_TESTED，不用finally合成轨迹替代原生因果。
5. 磁盘PASS、崩溃撤销、采样间隙短暂断电：磁盘PASS不足以证明交付，仍需真实返回/CLI stdout；不要求规格未承诺的崩溃一致或连续观测保证。
6. 原始RED日志尾随空白保持原件；科学效果、真实数据和正式报告不在复查范围，不改写、不据此增加论文创新结论。

## 保护与后续边界

父会话负责保存审计：83条前置源码/文档记录中82条保持原始SHA-256，README仅追加本次状态说明；35份既有合成证据及2份历史参照逐项匹配。记录数并非去重文件数。三项受审代码、旧43执行源码/10测试、已认可规格与计划、原始日志、六张既有脏PNG及两处正式报告均未修改。复查后核对详情见上述preservation_after。
本轮仅归档及静态源位置/身份检查，不运行测试、前向、拟合、计时、原生/电源/空闲验收；不读真实prepared/fit/stop、图片、缓存、权重或正式报告内容，不改网络、公式、损失或实验矩阵。GitHub归档网络操作属于本次授权，不是运行网络接口测试。
当前仍需修复该一项P2；下一步最小修复规格/实施计划及后续开发均须另批，不在本轮实施。原生协议/验收和真实机制实验亦分别另批，未创建原生协议或原生输出目录。
研究目标仍是验证MSB-RRE网络中的角色非线性证据、间隔预算跨界及相对强对照的独立收益；当前工程复查不是该科学验证，正式报告和网络创新完成度不增加。
