# MSB初始化异常保留修复：新一次独立只读复查

日期：2026-10-20。限定工程结论：`APPROVE`；R4：`ADDRESSED`；原生证据：`NOT_TESTED / INSUFFICIENT_EVIDENCE`；科学效果：`NOT_TESTED`。
授权：`call_abc05649de6c4a29a47489b19a41b00c`，新的一次独立只读复查及GitHub归档，工具不可用或无结论即停、不重派。
受审代码固定：`d158b178d42ba996c017be15e11b2504e6136d56..599643fbdc6782bc9f3c8827d4ac3587e788fc24`。归档前工作树HEAD为`0376efb009e5b21d7ed84e2c42052a1b13b45354`，比受审Head仅多前次失败归档，受审代码无漂移。

## 结论与作者接收

独立审查未发现新问题，R4在本次初始化异常保留合同内为ADDRESSED。接受这一限定结论，不作全系统无缺陷、原生就绪或网络创新有效的推论。
[驱动第933行](../../scripts/check_msb_native_supervision.py:933)的捕获边界覆盖两次初始化写入及首次check，body异常先存、guard退出异常后存；五个新方法覆盖实际合成RUNNING写后注入、原始异常对象与顺序、叠加持久化失败、写前失败、pre-state无输出及CLI非零/入口时钟注入。
父会话仅静态阅读认可规格与驱动896--1014行，核对上述结论的边界及第5项保留事项；没有重做整轮独立审查或运行反例。

原[历史REQUEST_CHANGES](2026-10-20_msb_native_acceptance_repair_independent_review.md)、[作者验收记录](2026-10-20_msb_initialization_error_retention_acceptance.md)及[前次额度终止记录](2026-10-20_msb_initialization_error_retention_independent_review.md)保持原件与历史语义；新结论仅作用于599643f的修复，不倒写前次代理已审读或已通过。

## 独立证据与原件

唯一代理Carver：`01a15869-1c05-7872-94a7-117c56fa3357`，新上下文派发一次，未指定模型覆盖。前三个观察窗口超时后均等待同一handle，没有重派；期间一次只读线程状态确认active/inProgress，未以部分评论代替终结结论。第四窗口取得完整completed结论后关闭同一代理。

- [授权与单次派发](../../outputs/msb_initialization_error_retention_independent_review_v2/dispatch.json)、[完整请求](../../outputs/msb_initialization_error_retention_independent_review_v2/review_prompt.md)、[归档前预检](../../outputs/msb_initialization_error_retention_independent_review_v2/preflight.json)。
- [审查原文](../../outputs/msb_initialization_error_retention_independent_review_v2/reviewer_response.md)、[原始终态返回](../../outputs/msb_initialization_error_retention_independent_review_v2/reviewer_terminal.json)、[关闭返回](../../outputs/msb_initialization_error_retention_independent_review_v2/reviewer_close.json)。原文文本仅补文件末尾换行，JSON保留completed字符串，不删减审查内容。
- [结构化接收](../../outputs/msb_initialization_error_retention_independent_review_v2/review_result.json)、[保护前记录](../../outputs/msb_initialization_error_retention_independent_review_v2/protection_before.json)、[保护后核对](../../outputs/msb_initialization_error_retention_independent_review_v2/protection_after.json)。

实际已读范围以独立原文为准：三个受审文件完整内容、认可规格/计划、作者验收和历史复查、README该次差异、guard与原生环境API有关段；指定11份既有summary/log/snapshot/metadata，旧230测试名日志及旧collector-green/001失败尾段。未冒称审读全部旧43源码/10测试，父会话的实物保护另列。
复查核对既有RED=78项9处功能断言失败0errors、GREEN=78/78、final=235/235；旧230完整名缺失0、新增指定五方法、固定十模块，三阶段均只有一个实际attempt。pre-state正例在RED已通过，不写成本次新修复带来的改善。
RED使用基线驱动，GREEN/final三文件正文匹配受审Head，快照中53项旧身份匹配。三阶段日志无skip/预期失败/意外成功/受守卫未模拟调用，不表示框架bootstrap及合成数值计算也全被模拟。
完整原始差异检查仍仅因RED原始日志35、41、65行尾空格非零，源码与作者文档限定检查清洁；不清洗原始日志、不改称完整检查全绿。旧失败尝试中的monotonic守卫缺口维持披露，不倒推完全模拟或原生成功。

## 未裁定事项的逐项处理

1. 真实Windows IPC、Job、线程、电源、硬退出：接收证据边界，继续原生NOT_TESTED；不能用fake的finally轨迹代替原生因果。
2. 任意调度和阻塞下的时限：接收有限合同结论，不增加普遍实时保证或新增修复任务。
3. 崩溃一致性及磁盘终态撤销：规格未承诺，接受限制；磁盘RUNNING或PASS均不足，实际返回/CLI stdout交付要求不变。
4. 采样间隙瞬时断电：接受未证明连续观测，不把采样证据写成持续物理接电证明。
5. state建立前再叠加释放失败的全原因聚合：本规格明确把捕获边界设在state成功建立之后，本次仅确认指定pre-state拒绝无输出；接收未裁定，不把state之后的全原因保证推广到此前复合故障。若未来要求额外聚合，需独立需求审核，本轮不新增承诺或暗中扩大修复。
6. 远端登记及原生运行资格：父会话会另行校验本次GitHub归档身份，但归档不等于原生协议登记或执行授权；当前没有新增原生协议/运行。
7. 六张脏PNG、其他未跟踪文件及正式报告：父会话对153条指定保护记录做哈希核对，README仅追加状态，其余152条匹配；两处正式报告只哈希、不读取正文。已有未跟踪产物未操作，不作其全量完整性证明。记录数不是去重文件数。

## 运行与研究边界

本轮只有只读审查、静态接收与归档；不重跑测试、前向、拟合、PCA/T0或计时，不做原生/进程/电源/空闲验收，不读真实prepared/fit/stop、矿物图片、缓存或权重，不改任何受审源码、网络、公式、损失、实验矩阵或正式报告。独立代理的无执行、无写入声明见原文，父会话保护核对另行留证。GitHub归档网络操作不属于实验接口验收。

下一步应回到验证网络创新的路径：先另批原生无数据监督验收协议登记及实际验收，再另批固定真实fit-only机制实验。当前不实施后两项，不自动复用此前失败运行或扩展网格。
MSB-RRE的角色非线性证据、间隔预算跨界以及相对强对照的独立收益仍须真实科学实验；本次APPROVE只是工程修复接收，不计作论文理论创新有效或研究总目标完成。
