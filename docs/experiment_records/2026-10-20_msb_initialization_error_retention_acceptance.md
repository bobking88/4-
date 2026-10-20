# MSB初始化异常保留：作者实施与有限合成验收

日期：2026-10-20。状态：`AUTHOR_REPAIR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING`。
授权：`call_a1c023c657114dfcb443cdec85dfdebe`，当前会话原生实施Task1及有限合成验收，不派代理。
基线：`d158b178d42ba996c017be15e11b2504e6136d56`。
依据：[认可规格](../superpowers/specs/2026-10-20-msb-initialization-error-retention-design.md)、[认可计划](../superpowers/plans/2026-10-20-msb-initialization-error-retention.md)、[独立复查原发现](2026-10-20_msb_native_acceptance_repair_independent_review.md)。

## 修复与实际行为

仅修改原三验收文件。驱动将既有body try上移至state成功建立后，覆盖registered_protocol写入、首次RUNNING写入及check；原except保留body对象，外层独立保留释放对象，发布/异常重包合同不改。另按规格将两个既知只读/文档许可ID加入NON_NATIVE_IDS。
测试新增五个固定方法，以实际合成JSON写入和原guard/check/remaining/exit为对象；受限底层接口仍fake。验收器只修改新输出与scratch路径，原十模块、PHASES可选项、bootstrap、严格接收门及守卫不变，隔离验收门夹具同步预期路径。

| 反例或边界 | 受本次合成证据支持的结论 |
|---|---|
| RUNNING真实写入后AC、31秒gap或480秒等号失败，再释放失败 | errors按body、collector_scope_or_release顺序保留实际类型/消息，无case/observer启动，失败交付且不伪造released。 |
| 上述AC双异常再叠加失败收据持久化失败 | primary_errors中两个原始对象按is保存，primary_error首项兼容，persistence_error为实际注入对象；失败写入一次，磁盘原RUNNING可仍存在，不谎称已撤销。 |
| registered_protocol或首次RUNNING写入前失败，再释放失败 | 原始写入原因不丢失，不伪造已完成RUNNING写入，不启动case。 |
| main真实控制路径 | 入口monotonic显式注入且调用一次，code=1，stdout含两因，不重复历史时钟缺口。 |
| state建立前登记、来源、guard进入拒绝 | 保持原异常传播，输出不存在、write/launch/observe为0。该正向保护在RED前已成立，不冒称新增功能。 |

## 固定三阶段原始结果

新根：[msb_initialization_error_retention_synthetic_v1](../../outputs/msb_initialization_error_retention_synthetic_v1)。三个实际阶段各一次独占attempt，未使用剩余额度、未运行其余四阶段，全部原件保留。

| 实际阶段 | attempt | 测试数 | 失败 | 错误/跳过/受限未模拟调用 | 结果 |
|---|---:|---:|---:|---:|---|
| collector-red | 001 | 78 | 9 | 0/0/0 | FAILED，有效功能红灯 |
| collector-green | 001 | 78 | 0 | 0/0/0 | PASSED |
| final | 001 | 235 | 0 | 0/0/0 | PASSED |

RED的9处断言失败：AC/gap/deadline三处原因丢失、CLI一处、三异常对象长度一处、两个初始化写入原因两处、既有许可方法中新增两ID拒绝子例两处。不是导入或夹具错误；原有方法中的新增子例失败不能解释成原73项旧子例回归。生产捕获修改只在有效RED之后进行。
最终[summary](../../outputs/msb_initialization_error_retention_synthetic_v1/final/attempt-001/acceptance_summary.json)、[完整日志](../../outputs/msb_initialization_error_retention_synthetic_v1/final/attempt-001/synthetic_tests.log)、[直接源码快照](../../outputs/msb_initialization_error_retention_synthetic_v1/final/attempt-001/source_snapshot.json)记录235/235，固定十模块，原230个完整测试名全保留、新增五方法。失败、错误、跳过、预期失败、意外成功及受限未模拟调用均为0，仅2次合成scratch张量回读。
日志中隔离验收门故意注入skip并输出FAILED的fake contract-green属于外层测试验证，不是实际新根下额外阶段、真实跳过或重试。既有数值库bootstrap及合成数学计算实际执行，不能笼统称所有计算模拟。

## 来源、保护与作者接收

[来源身份](../../outputs/msb_initialization_error_retention_synthetic_v1/final/source_identity.json)及[保护/范围核对](../../outputs/msb_initialization_error_retention_synthetic_v1/final/preservation_and_scope.json)保存实际来源与范围：三修改文件匹配final运行前直接快照；原43执行源码/10测试保持身份。46项来源只是未来原生登记候选，不是已登记协议。
基线131条记录中3条属于获准修改代码，其余128条受保护；归档前127条哈希不变，README仅追加本次状态。记录数不是去重文件数；已认可规格/计划及文档记录、原230项日志、独立复查、六张既有脏PNG和两处正式报告均保留。正式报告只核对保护哈希，不打开正文。
全量新证据保留原始UTF-8日志、summary和直接快照；完整暂存diff-check指出本次RED日志35、41、65行有三处unittest子例标题尾随空格，按原样保留，不清洗失败原件。此前按Raw文本统计漏计CRLF前空格，现已用逐行检查纠正为3；源码与作者文档单独检查清洁，不宣称完整diff-check通过。上轮collector-green/001的一次漏守卫monotonic失败仍按原披露保留，不能用本次绿灯或旧0计数将它改写为完全模拟。
作者自审认为R4初始化异常保留缺口已按规格得到有限合成验证；旧独立REQUEST_CHANGES保留历史意义，**新独立复查尚未进行**。有限fake调度不证明Windows原生构造、IPC/Job/线程、电源及任意并发或崩溃一致性；磁盘PASS不单独证明写后检查或交付。

## 未做事项与后续门

未读取真实prepared/fit/stop、图片、缓存或权重，没有真实训练、原生/电源/空闲/进程验收、额外计时探查或新独立复查；没有修改网络、损失、公式、实验矩阵及正式报告。原480秒/30秒/2秒及7200秒科学边界不变；原生协议和输出目录仍未创建。
只完成作者实施与合成接收，不把工程235项检查当235项科学实验。MSB-RRE角色证据利用、间隔预算跨界及强对照独立收益尚未验证，不能提升报告/论文理论创新结论或放行真实实验。
下一步若获另批，才开展一次只读独立复查；原生登记/验收、真实机制实验继续分别另批，不自动重派、续跑或扩大实验矩阵。
