# MSB-RRE原生无数据监督验收v2：待运行协议登记

日期：2026-10-22（Asia/Shanghai）。状态：`REGISTERED_AWAITING_NATIVE_APPROVAL`，`NOT_RUN`。
本轮授权：`call_be3d89be53b944ec8acf9cd1e30a0058`，答复“批准仅编写与登记协议（推荐）”。该[授权](../../outputs/msb_native_supervision_v2_registration_v1/authorization.json)只允许文档及GitHub登记，实际原生验收和真实机制实验仍未获批。

基线为`487af614ad4ee0a70870cb39e8b00b34e9beb3b7`；已复查源码提交为`35fed93746ae380578d54dd89183235d67b886e4`。依据[原生验收规格](../superpowers/specs/2026-10-18-msb-native-supervision-acceptance-design.md)、原计划Task4、现有v2执行器和[最新独立APPROVE](2026-10-22_msb_observer_abort_retention_review.md)。本轮不修改这些冻结资料。

## 1. 新协议、目录与非执行状态

新[协议JSON](../experiment_protocols/msb_native_supervision_v2.json)位于`docs/experiment_protocols/msb_native_supervision_v2.json`，拟用独占输出为`outputs/msb_native_supervision_v2/acceptance_once`。登记前两个路径未被使用；本轮只创建协议，不创建该运行输出、RUNNING状态、case或worker目录。登记证据另存于`outputs/msb_native_supervision_v2_registration_v1`，它不是运行输出。

严格沿用当前执行器16字段，包含`DIRECT_CPYTHON311_VENV_V1`启动合同和已登记的绝对venv根。没有增加实验输入、求解器或运行参数。`status`为`REGISTERED_AWAITING_NATIVE_APPROVAL`，`approval_call_id`及`approval_answer`均为`NOT_GRANTED`，不把本次登记ID、独立复查ID或旧原生许可填作执行许可。

这是刻意不可执行的待批准稿：源码校验要求`APPROVED_NATIVE_ONLY`、新的真实call_许可、精确授权原文及远端登记提交；当前状态与授权标志均不满足。这里只作源码/JSON静态核对，未导入或调用执行器/校验器，不将登记写成原生接收通过。

未来须先获得新的一次真实人类原生授权，再只补`status`、`approval_call_id`、`approval_answer`三个字段，重新登记并核对实际远端内容。源码、目录、场景和限额保持不变；登记提交号由外部授权传入，不把自身提交号写进协议形成循环身份。任何来源或身份漂移先停，不静默更新哈希绕过门限。

旧v1协议、已消费授权及N1失败原件保持不变，不重新运行、不续跑或覆盖。当前v2独立命名空间不意味着旧失败已经被原生实测修复。

## 2. 固定N1--N4与证据要求

| 场景 | 既定工作或故障 | controller外层上限 | 核心接收要求 |
|---|---|---:|---|
| N1 normal-renew-close | begin_work后45秒有界等待，正常关闭退出 | 90秒 | 至少两对真实RENEW/ACK，最终sequence至少3；MSB COMPLETE、精确关闭、exit0、退出/清理确认及正常电源请求释放 |
| N2 worker-exit-no-done | ready及身份ACK后worker硬退出23，无DONE | 45秒 | 精确exit23和故障前WORKING；只接受对应缺关闭/原生pipe EOF路径；controller检查后exit0、父释放及完整清理；任意FAILED或外层timeout不算通过 |
| N3 done-linger | 正常DONE后故意迟退35秒 | 75秒 | 原冻结租约不续延，租约失效导致原监督器拒绝并回收；不是用LC外层75秒超时替代 |
| N4 parent-hard-exit | 身份/Job/WORKING及collector ACK后controller硬退出31 | 45秒 | 精确父exit31、故障前worker确在Job且存活；保留身份句柄在5秒内确认退出；不预先杀worker、不持有Job句柄延长寿命、不补造DONE或结果 |

一个collector，顺序固定N1、N2、N3、N4；每例controller和worker各一次，最多4+4个工作进程。N2/N4的ready-ACK最多5秒；前置身份或活性证据不足时不触发预定故障，转为失败清理。

证据齐全的预定负例属于该例通过；任何非预期失败、证据不足、缺例、outer kill/timeout、未知所有权、清理或正常释放失败，立即停止，不进入后例。不重试、不续跑、不替补、不扩大场景，不另做计时或空闲探查。N2的worker及N4硬退出进程缺少显式释放观测，不能补填released=True。

## 3. 累计480秒与接电条件

480秒累计接收窗口包括入口校验、依赖准备、源码及远端核对、对象/进程创建、四例、诊断、清理及最终发布；不清零起点，等号过期。每例外层上限为min(90/45/75/45,累计剩余)，后续例前及最终发布仍须检查剩余。

各层继续继承首次MSB起点和原7200000ms生产预算；480秒是额外接收上限，不修改生产预算、不声称实测两小时超时。保留30秒最大采样间隙、2秒双时钟差、30秒租约、15秒续租和最多5秒生产清理；清理不获得新计算额度。OS抢占、底层阻塞或崩溃下不承诺普遍硬实时完成，超期或证据不足不得通过。

要求全程AC=1接电，不主动休眠、断电，不修改电源方案/待机设置，前后只读设置须一致。**键鼠可以正常使用**，不重做旧360秒空闲验收或0.25秒超时测试。只报告实际观测到的正常路径与collector显式电源释放；退出不等于所有句柄零泄漏。磁盘PASS本身不足，还须最终检查及实际返回/CLI交付一致；失败收据无法保存时保留原始异常并非零，不宣称旧磁盘候选已撤销。

## 4. 已复查源码身份与保护

[源码身份](../../outputs/msb_native_supervision_v2_registration_v1/source_identity.json)固定43项原科学执行闭包及三项验收基础设施，共46项，与当前SOURCE_FILES字面量及旧协议的路径集合一致。原始字节SHA-256供运行时严格比较；规范化LF SHA-256及Git blob SHA-1分列，不将Git对象哈希冒充原始文件哈希。

全部46项当前规范化Git blob匹配本地基线提交，并通过独立GitHub连接器核对基线树下scripts/tests的46个blob。当前driver/测试/harness分别与既有253项合成最终快照一致；只沿用已有合成与独立APPROVE证据，不重新运行测试。venv根只作已知路径登记，没有查询本机解释器映像或执行环境探查；实际身份仍由未来获批运行的原严格门检查。

[前保护](../../outputs/msb_native_supervision_v2_registration_v1/protection_before.json)固定前轮280条状态和最新复查提交8文件，共288条记录（含重复路径）；[后保护](../../outputs/msb_native_supervision_v2_registration_v1/protection_after.json)只允许README追加本登记状态。正式报告、网络/损失/公式/实验矩阵、所有旧原件和六张既有脏PNG不改、不暂存或还原。其他未跟踪产物不操作，不将本保护清单宣称为工作区所有文件的全量审计。

[静态登记检查](../../outputs/msb_native_supervision_v2_registration_v1/registration_check.json)只记录JSON字段/类型、路径、源码/远端身份与保护核对，不是新的测试或原生接收证据。报告正文不读取，只保留两份正式DOCX哈希。

## 5. 本轮禁止与交接

本轮仅新建协议及登记记录、授权、源码身份、保护/静态检查证据，并追加README。无代码开发、项目导入、测试、前向、拟合、PCA/T0、计时/进程/线程/电源/原生/空闲验收；不读prepared/fit/stop、矿物原图、真实缓存、权重或张量，不派代理，不修改正式报告。Git与归档网络访问不是实验执行。

后续仍分两道门：先另批一次按固定协议的原生无数据验收，补授权并重新登记、远端核验后才运行；只有该限定验收通过后，再另批真实MSB-RRE固定fit-only机制协议与实验。任一失败或证据不足保留原件即停，不自动修复或重试。

本登记不是网络理论创新、原生通过、论文阳性结果或工业能力证明。B主线、C后置及既定MSB-RRE强对照设计不变；角色非线性视觉证据与判别间隔预算是否有实际独立收益，仍须科学实验检验，尚未验证的结果不写入正式成果。
