# MSB原生无数据验收驱动：单次独立只读复查

日期：2026-10-18。授权：`call_7f515c71f122488da5242db3dc69dd60`。
审查范围：`a71d54821a482d50eb1fd182d4a5bcafe9219700..c092f1cae59fc4187670bf4bff655375309d83b0`。
结论：**REQUEST_CHANGES**，2项P1、3项P2均未修复，不能登记或运行原生验收，也不放行真实机制实验。

## 独立性与证据

按认可的[规格](../superpowers/specs/2026-10-18-msb-native-supervision-acceptance-design.md)和[计划](../superpowers/plans/2026-10-18-msb-native-supervision-acceptance.md)Tasks1--3，只派发一次独立审阅者Gibbs，ID为`01a14f8e-4be0-71c1-be51-dd7324bffd49`，未复制父会话历史、未重派或另派。
审阅者静态检查新增三文件、相关原监督器/worker/LC与环境合同，以及既有最终回执、日志、直接快照和指定失败日志；返回明确结论后关闭。
父会话只对五项反馈涉及的源码因果进行静态核对，另做哈希和归档核验，不冒称第二次独立复查。

- [授权与范围](../../outputs/msb_native_supervision_independent_review_v1/review_request.json)。
- [独立原文](../../outputs/msb_native_supervision_independent_review_v1/review_response.json)：`verbatim_response`保留返回字符串，包括原有Markdown换行空白，以JSON转义存储，未清洗审阅原文。
- [结构化结论与父会话裁定](../../outputs/msb_native_supervision_independent_review_v1/review_result.json)。
- [复查前保护记录](../../outputs/msb_native_supervision_independent_review_v1/preservation_before.json)及[复查后保护记录](../../outputs/msb_native_supervision_independent_review_v1/preservation_after.json)。

## 反馈与接收判断

以下行号均对应冻结的`c092f1c`，不是已实施的修复或已运行的新回归。

| 编号 | 严重度 | 已静态确认的问题 | 裁定 |
|---|---|---|---|
| R1 | P1 | `scripts/check_msb_native_supervision.py:538`把负退出时间差截为0；651--658行在检查存活前写身份ACK；N4未排除父故障前子进程已退出或显式清理。可能误把无效样本接收为Job清理证据。 | 接收，未修复 |
| R2 | P1 | 同文件359--365行核对传入root中的源文件，744行和258行却启动当前`__file__`；829行也只核对root。未绑定实际驱动及导入来源，可能核验B目录却执行A目录。 | 接收，未修复 |
| R3 | P2 | 同文件764--766行的末端检查不调用根`guard.check()`；`check_lc_native_environment.py:83`的power_settings不检查AC；`lc_execution_safety.py:100`退出仅释放请求，故末端断电或超过30秒采样间隔可能遗漏。 | 接收，未修复 |
| R4 | P2 | 同文件767行的except在`with guard`内部，根guard退出的POWER_RELEASE_FAILED在其外抛出，跳过769--779行终态发布，已写回执可能停在RUNNING。 | 接收，未修复 |
| R5 | P2 | 同文件277--283行即使0毫秒工作也多次读时钟，读数T、T+1、T+1会返回1；565行恰好要求0，可能把无等待体的N3采样开销误判为实际工作。 | 接收，未修复 |

后续修复应分别覆盖：死进程身份ACK、故障前退出/显式清理、不同checkout的执行来源、末端AC与采样断层、guard退出失败及双异常持久化、零工作但时钟自然递增。这里只记录必要反例，不编写修复规格或代码，也不运行反例。
R1尤其不能简单要求“子退出观察晚于父退出观察”：父进程的观察时刻可能本来晚于子退出，需区分故障事件时刻、进程活性和观察时刻，不把观察排序直接冒充物理因果。

## 既有通过与未验证边界

既有final/attempt-002仍是202/202合成通过，包含原157项及新增45项；本次不重跑、不修改或撤销其日志。五项反馈说明现有覆盖不充分，而不是新执行了五个失败测试。
原MSBParentSupervisor/open_worker_scope接线、共同7200000ms预算、额外480秒窗口、固定场景限额、查询句柄权限、公开入口默认拒绝及严格合成门等已有实现，不因局部缺陷否认这些事实，也不因这些事实忽略五项阻断。

审阅中保留以下未验证事项：真实Windows构造、OS调度、IPC、Job与电源/线程行为只能由以后获批的原生验收提供限定证据；假SystemExit展开finally不证明真实os._exit清理。这些不能本轮动态验证，作为后续边界保留，不擅自新增工程证明任务。
没有测试、项目导入、合成前向、拟合、计时/电源/进程探查、真实数据/原图/缓存/权重读取、原生验收或正式报告内容读取/修改。审阅者未使用网络或再次委派；父会话仅为获批结论归档使用GitHub。

复查后78项哈希记录匹配：旧43源码与10测试/验收器、新三代码、20项保护文件和2项本轮认可文档。六张用户既有脏PNG及两处正式报告身份保留；正式报告两处SHA仍为`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。
真实原生协议`docs/experiment_protocols/msb_native_supervision_v1.json`及`outputs/msb_native_supervision_v1`仍不存在。未修改源码、测试、网络、公式、损失、实验矩阵或历史证据。

## 当前状态

作者合成通过记录保持历史原貌；**当前独立工程闸门为REQUEST_CHANGES**，原生证据仍不足，科学效果未测试。旧执行器此前获APPROVE不适用于新增驱动。
下一步应另获人类授权后先界定五项的最小修复规格与实施计划；代码、合成验收、再次独立复查、原生协议登记/运行及真实机制实验仍分别另批，不能自动续跑。
本次是实验执行可靠性审查，不计为论文理论创新或模型效果提升。B主线/C后续方向及MSB-RRE科学规格不变，角色分支利用、间隔预算跨界和强对照独立收益仍待真实证据。
