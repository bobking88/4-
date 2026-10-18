# MSB完成接收与严格验收门：最小修复规格

起草：2026-10-17；静态校核与归档：2026-10-18。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档授权：`call_aa6d081524a94817b20cceb8fbd25053`，仅编写本规格与实施计划并归档GitHub。
当前基线：`655cf76a1c164052c87172b776bb0ed1aea21c96`；冻结被审实现：`a40567ab2f99fde695194e15fe726a6c3b9dd863`。

## 1. 目的、证据与范围

依据[一次独立只读复查](../../experiment_records/2026-10-17_msb_terminal_boundary_repair_independent_review.md)，处理R1/R2/R3及八帧排空覆盖缺口。它们是静态发现，不是已运行的Windows故障反例。原146项零跳过日志不改写，当前仍为CHANGES_REQUIRED、原生INSUFFICIENT_EVIDENCE。

| 编号 | 根因 | 后续验收目标 |
|---|---|---|
| R1/P2 | DONE后排空早返只查总期限，退出接收未再查冻结短租约 | 退出接收时原租约及累计期限均有效，等号失败；不能借DONE冻结后停止查租约 |
| R2/P2 | unittest.wasSuccessful不排除skipped，入口仍可PASSED | 未实际通过的合同不能冒充全绿；门函数及main接线都有反例 |
| R3/P3 | 失败记录也写不出时仅链接第二个异常 | 非零、不谎报持久化，同时保留首要异常/失败状态与记录写入异常 |
| G1/覆盖 | 目前仅拒绝max_frames=9参数，未计数验证8帧边界 | 实际被测排空循环最多消费8条，第9条不读；与真实状态机缺DONE拒绝分别证明 |

本轮仅生成待审核文档，不开发、导入或运行项目/测试代码，不读真实prepared/fit/stop、原图、缓存或权重，不做前向/拟合/PCA/T0、计时/空闲/电源/原生进程验收，不派代理，不改正式报告。
用户已明确同时批准规格与计划起草，因此一并交付两份待审核稿；这不等于书面文件已获开发许可。

## 2. 继承不变量与非目标

继承[终结握手规格](2026-10-17-msb-terminal-boundary-repair-design.md)、[执行合同规格](2026-10-16-msb-execution-contract-repair-design.md)及[科学规格](2026-10-14_margin_scaled_bilinear_role_evidence_design.md)，不回写其历史状态/哈希。

- 科学代码不变：h64/e23/log_anchor4，三个独立rank4分支，1083当前参数及1083固定参照；`R_ref=2*abs(m)+0.25`、原损失/优化器/初始化不变。
- 340个既有已暴露fit分组、[66,130,97,47]及三份原登记prepared；不读stop/原图；0次新PCA/T0。
- 11单元、3折、种子20261002/20261003/20261004；63+12=75拟合、80 job、400步上限、100步主点；615+12最终核验及七参照晋级门不变。
- 原累计预算7200秒、等号过期；短租约最多30000ms、续租间隔15000ms；清理wait最多5秒，不兑换计算额度；控制帧8192bytes及单次排空8帧不变。
- 43项执行源码闭包不增减；未来生产源码只动msb_execution_safety.py、run_msb_role_mechanism.py。其余41项、旧34项、四科学模块及data/training/analysis保护。
- 原83/121/146项日志、全部失败/合成目录、规格/接受的旧计划、六张用户PNG及两处正式报告保持身份；不续跑、不覆盖、不搬移旧目录。
- 不改IPC/Job框架、不加依赖、不改控制消息字段/种类、不扩大真实许可、不升级网络或实验矩阵；不宣称硬实时或崩溃一致性。

这是恢复验证条件的工程修复，不是论文网络创新。B仍为当前主线，C阶段条件化决策仍仅为后续方向。

## 3. 方案取舍

| 方案 | 判断 |
|---|---|
| 仅增加一条日志、延长租约或忽略跳过 | 不采用，不能修复最终接收/判定根因，且可能掩盖未完成合同 |
| 在既有接口加强完成校验、抽取严格门、结构化保留双异常 | 采用；改动集中，能逐项先红后绿，不改变科学模型 |
| 重写监督器或另建通用验收框架 | 不采用；超出三项问题需要，增加未经验证接口 |

## 4. R1：唯一退出接收点查冻结租约

将内部接口改为`MSBParentControl.require_complete(exit_code: int, *, now_tick_ms: int) -> None`。显式当前时间必填，不从DONE、上轮循环或last_tick倒推；先复用`_check(now_tick_ms)`检查原累计预算、非倒退及当前issued租约，再检查原DONE/CLOSE_GRANTED/ACK和exit_code==0结构条件。

父run_job观察退出、完成有界排空后，先完成既有父/guard检查，最后采样当前tick调用require_complete；紧接着才可设置COMPLETE。最后校验与成功接收之间不得插入可能耗时的I/O/guard/进程调用。无需改控制帧、续租或冻结后的phase语义，也不另设时钟原点。

设冻结租约L、总截止D，且L<D：退出接收tick=L-1可接受，L及L+1均SUPERVISION_INTERRUPTED；若先达到D则沿用TIMED_OUT。即使握手真实完成且exit_code=0，也不能用这些标志覆盖过期失败。失败仍走既有回收，不派下一job。
这里检查的是父确认接收的时点，不声称已获取OS物理退出精确时间，也不承诺Python字节码期间无异步硬中断。

R1反例必须独立驱动DONE与process.poll/退出：DONE于L-1被状态机接受，poll推进到L才返回0，且D尚远。不能在read(DONE)里设置退出，不能只测总期限等号。旧CloseControlTests调用迁移为显式now_tick_ms，原断言目的保留。

## 5. R2：严格结果门与独立证据目录

在现有`tests/run_msb_terminal_repair_acceptance.py`抽出纯函数`_acceptance_passed(result, *, forbidden_calls: list, harness_error: str | None) -> bool`，main必须实际使用其唯一结果决定status及返回码。
只有result存在、wasSuccessful为真、testsRun>0、skipped/expectedFailures/unexpectedSuccesses均为空、受限调用为空且harness_error为None时才通过。额外记录expected_failures与unexpected_successes实际数，不让“未跑任何测试”或“预期失败”被误称合同全通过；不预报未来通过总数。

TDD准备可将现有表达式原样移入门函数，暂不修正语义，并建立新固定phase/目录和回归。此步骤属于未来获批后的测试接缝准备，不是本轮可执行事项；有效RED必须看到skipped结果使旧判定返回通过，不能仅用“函数不存在”冒充原漏洞复现。
除纯门函数测试外，必须通过受控TestResult替身运行实际main判定/回执路径，验证非零skipped产生FAILED和非零返回；不能只测一个未接线helper。此为未来全模拟测试，外层测试本身不skip。

为不复用已耗尽task1-red，未来同一验收器改为固定新根`outputs/msb_completion_repair_synthetic_v1`；旧根完全不动，旧入口快照仍在历史提交/直接快照中可追溯。

| 固定phase | 唯一模块白名单 |
|---|---|
| gate-red / gate-green | test_run_msb_role_mechanism |
| lease-red / lease-green | test_msb_execution_safety |
| publish-red / publish-green | test_run_msb_role_mechanism |
| final | 原固定九模块 |

每phase最多3次独占attempt-001..003，含导入/守卫失败；任何旧目录不得覆盖或重命名，额度耗尽即停、另行报告。不增加任意模块/输出路径/真实stage参数；唯一CLI参数仍为--phase。新根本轮不创建。
仍只允许已安装torch/sklearn必要数值bootstrap，未用torchvision采用调用即失败替身；守卫早于项目导入，禁止真实NativeMSBAPI构造/tick、WinDLL、Popen、os._exit、线程启动及sleep；torch.load仅专用合成scratch/BytesIO。不放开原生库探查来获得绿灯。
每次运行前直接捕获五项源码/测试/验收器原文及原始字节哈希。固定九模块及普通合成数学运算保留，不以mock网络/求解器取得通过。

## 6. R3：双重失败的结构化诊断

在run_msb_role_mechanism.py新增本地`MSBTerminalPublicationError(RuntimeError)`，接口：`__init__(self, failed_state: dict, *, primary_error: BaseException | None, persistence_error: BaseException)`。
公开诊断属性为failed_state（外层dict快照，非整棵不可变对象）、primary_error及persistence_error；失败状态必须含delivery_verified=False、failure_record_persisted=False及原status/error。消息保留“not persisted”、失败状态及两次原因；异常链仍以persistence_error为cause。

在首个发布/写后预算检查异常尚可访问时保存原异常对象；失败记录再抛异常时附带两者，不用第二异常替换首要state.error。若输入原本就是失败状态、没有可访问的原异常对象，primary_error可为None，但原state.error文本必须保留，不伪造异常对象。
这条错误路径不再做额外写入、重试或“补成功”发布；CLI可继续传播异常而非零，不要求新增原生进程验收。没有证据时不能声称失败撤销已持久化。

## 7. G1：排空循环边界不冒充合法协议序列

单独以无数据ProbeControl检查真实`_drain_control`循环：队列放9个sentinel，probe只计数、done始终False、不发消息；断言实际读取/accept恰8次、第9项保留、返回8，且未宣布完成。此测试仅证明循环上界，不证明8条sentinel是可由生产协议合法生成的控制帧。
真实MSBParentControl的正常/拒绝路径仍使用真实accept/renew/require_complete，不以probe替代。另用真实状态机验证退出仍缺DONE不能完成，保留截断、再老请求、精确在途ACK及两种合法交错反例。
生产正常终结序列不要求凑够8帧；不得为强凑“8个合法帧”加入心跳消息、隐式续租、伪造ACK或删序列校验。G1计数测试可能在修复前已通过，应如实记为补覆盖，不制造红灯。

## 8. 身份、交付与停止条件

保持msb_parent_bound_v2及原科学/repair/terminal规格哈希语义。此次是实现缺陷修复，不无故增设v3或修改已接受规格；未来真实协议仍须新独立身份并重新登记全部43源码实际字节，不能复用旧登记。
文档/独立复查/将来代码合成许可均非真实许可；未来实现必须把届时已知非真实授权ID纳入既有拒绝回归。ID黑名单不是对任意字符串的语义认证，真实运行仍需人类明确批准并登记。

后续若获准原生实施，限定两生产文件、两测试文件和同一验收器；原生接口全部fake，新七phase有限尝试。最终记录actual counts、零跳过/预期失败/意外成功、零受限未模拟调用、real_data_opened=False及real_fits=0。
文档完成、开发/合成验收、独立复查、原生无数据验收、新真实协议与真实运行各自有授权边界，不自动派代理或恢复失败目录。

## 9. 恢复研究主线的真实标准

工程全绿仅移除执行障碍，不完成提高报告/论文网络创新的目标。后续科学检验仍须区分：间隔预算是否允许有效跨界、角色非线性证据是否真正被利用、角色分离的学习偏置相对SHARED-BIL/CONCAT-MLP/充分求解凸参照是否有独立收益。
既定真实实验若未达三种子七参照晋级门，保留NO_ROLE_PROMOTION，不用400步补救或更名宣称成功。340个已暴露fit分组也不能冒充未暴露泛化证据。
本轮不增加网络、损失或数学结论，不把租约/测试门公式写成论文理论创新，不向正式报告增加未经验证的成果。
