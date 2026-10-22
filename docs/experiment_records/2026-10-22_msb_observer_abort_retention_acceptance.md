# MSB观察器提前异常回收：作者实施与有限合成验收

日期：2026-10-22。状态：`AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING`。

## 1. 授权和范围

授权为`call_1772c2605a3d40fabe5266d3b37ba35f`，答复“认可文件，批准代码与有限合成验收（推荐）”。执行依据为提交`c6ed89bf2189d948d68fede72cf87a0ddd6de19c`中的[规格](../superpowers/specs/2026-10-22-msb-observer-abort-retention-design.md)与[Task1实施计划](../superpowers/plans/2026-10-22-msb-observer-abort-retention.md)。两份冻结文档维持其起草时状态，由本记录描述后续实施，未回填为此前已完成。

代码仅修改driver、driver测试和原合成验收器三文件。未派代理、未读真实prepared/fit/stop或矿物原图、未真实训练、未做原生进程/空闲/计时验收；未修改网络、损失、公式、实验矩阵或正式报告。

## 2. 最小实现

新增纯汇总函数`_observer_abort_error`，消费一次close后的缓存快照与原始异常对象。primary按B（提前异常）、A（callback）、D（原stop_errors顺序）排列，仅在primary列表内按对象身份去重；cleanup独立保留原C，最终guard退出F和持久化P继续使用原发布链。不从message或JSON重建异常，不修改输入列表或快照。

无typed停止快照、无解释的停止失败、健康快照但close返回非True，分别追加明确的缺证据诊断；原B始终保留。异常回收只调用一次stop_observer，不调用finish、sample或新时钟/等待，不增加预算。原ObserverTerminationError快路径、正常finish/close/stop_fault和累计期限算法未改。

合成验收器仅开放collector-red、collector-green、final，使用新的独立输出根和scratch。以原247次最终验收快照合并的56路径为保护基线，仅豁免精确三文件；每次直接保存三文件运行前源码。新增三项文档/复查/本次开发许可拒绝项，不能将本轮许可当作原生运行授权。

## 3. 实际RED/GREEN结果

| 阶段 | 唯一尝试 | 方法数 | 结果 | 失败断言 | 错误/跳过/未模拟调用 |
|---|---|---:|---|---:|---|
| collector-red | attempt-001 | 92 | 预期功能RED | 15 | 0/0/0 |
| collector-green | attempt-001 | 92 | PASSED | 0 | 0/0/0 |
| final（原十模块） | attempt-001 | 253 | PASSED | 0 | 0/0/0 |

RED的15项为六个新方法的断言/子例失败，来源包括缺纯helper、B/A/D遗漏、缺诊断、False/缺快照未保留及三项非原生许可未拒绝。既有86项driver方法通过，RED无导入、夹具或守卫错误。因此是有效功能RED，不是执行器出错。三个阶段各只运行一次，没有额外重试。

六个新方法覆盖：check和run_outer提前异常、缓存callback/期限与close叠加、剩余0/0.25/2秒的单次有界停止与活句柄保护、缺快照/健康/False、原F/P及失败CLI，以及纯函数身份去重/输入不变。A1/A2/A5经过实际run_acceptance提前异常分支，不以fake finish成功记录替代。

最终完整方法名集合精确等于原247加固定六名，共253，无遗漏、意外新增或重名。所有阶段expectedFailures、unexpectedSuccesses及harness_error均为零/空。最终两次torch.load仅为scratch中的合成张量，不是读取真实矿物数据。进程、线程、时钟、电源、网络和runtime映像边界均由fake或守卫隔离。

终端中出现的嵌套collector-green/1项skipped/FAILED摘要来自既有“拒绝跳过输入”的fake-root测试，不是本轮真实collector-green失败尝试；权威结果为三个独立根下的acceptance_summary.json和synthetic_tests.log。

## 4. 证据和保护

本轮证据根为[outputs/msb_observer_abort_retention_synthetic_v1](../../outputs/msb_observer_abort_retention_synthetic_v1)，保留授权、保护前后检查、全部三个attempt的summary/log/源码快照及[名称与接收门核对](../../outputs/msb_observer_abort_retention_synthetic_v1/acceptance_check.json)。scratch原件保留本地，未删除。

最终运行后的三源码SHA-256分别为：

| 文件 | SHA-256 |
|---|---|
| scripts/check_msb_native_supervision.py | 557574eb6f0924096bcc880393c32c767c10da094df39ffe4d3979649f11b429 |
| tests/test_check_msb_native_supervision.py | d2de225bbd659314109826ede1e5917455f841f95db21f138422e8f957bf22ad |
| tests/run_msb_native_supervision_synthetic_acceptance.py | 1a91b6b96f3fe342ee873cf14ef128b6a169142f31c92fa30de7f51a0a3339fb |

原56项源码身份中仅三白名单路径有变；262条保护记录（含重复身份记录）在最终运行后均检查，除三代码文件及随后获批的README归档更新，其余原件不变。主目录与工作树正式报告及六张既有脏PNG保持原哈希；这些PNG不进入本轮提交。正式报告SHA-256仍为`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。

离线核对第一次输出过长，工具截断导致外层JSON解析失败；随后使用紧凑摘要完成同一只读核对。另一离线计数器最初仅统计43条scripts路径，改为同时包含三条tests路径后确认SOURCE_FILES仍为原46项且原字面量不变。这些核对未导入项目、未重跑测试或修改原件，不是额外合成尝试。

第一次提交前全范围空白检查被RED日志的四行unittest子例行尾空格拦截，提交尚未发生。原始日志按哈希保留，不做格式化；后续空白检查排除synthetic_tests.log原件，源码、文档及其他归档仍严格检查。这不改变RED/GREEN计数或接收门。

## 5. 结论与下一门

R1在作者有限合成范围内已处理；前次独立复查REQUEST_CHANGES保留为历史记录，本轮尚未获得新的独立复查结论。状态最多为作者合成通过、独立复查待批准，不能写成独立批准、原生已通过或网络创新有效。

下一步建议仅对本轮精确三文件改动及253项现有原件做一次独立只读复查，另行取得授权；复查不重跑测试或读取真实数据。原生无数据验收和MSB-RRE真实机制强对照仍分别另批，不复用已消费许可或失败目录。

科学目标仍是角色非线性视觉证据与判别间隔预算是否在强对照下产生实际收益。本轮仅提高后续实验的失败证据可靠性，没有新的准确率、网络优势或理论有效性证据，不升级正式报告结论。
