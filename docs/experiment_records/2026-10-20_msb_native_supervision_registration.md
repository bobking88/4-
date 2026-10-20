# MSB-RRE原生无数据监督验收协议登记

日期：2026-10-20。状态：`REGISTERED_AWAITING_NATIVE_APPROVAL`，仅登记待运行稿，`NOT_RUN`。
本轮授权：`call_409c008b909c4a259fbd54f3c0ca6d87`，答复原文“批准仅编写与登记协议（推荐）”。[授权原件](../../outputs/msb_native_supervision_registration_v1/authorization.json)仅允许文档及GitHub登记，不是原生执行许可。
基线：`b6a4ebd287f66391d69357efea238372839d7d70`；受审代码：`599643fbdc6782bc9f3c8827d4ac3587e788fc24`。依据[认可规格](../superpowers/specs/2026-10-18-msb-native-supervision-acceptance-design.md)、[认可计划Task4](../superpowers/plans/2026-10-18-msb-native-supervision-acceptance.md)、[最新限定独立APPROVE](2026-10-20_msb_initialization_error_retention_independent_review_v2.md)。

## 1. 协议与未授权状态

[待运行协议JSON](../experiment_protocols/msb_native_supervision_v1.json)只使用既有执行器的14个字段，不新增输入或运行参数。路径固定为`docs/experiment_protocols/msb_native_supervision_v1.json`，拟用输出固定为`outputs/msb_native_supervision_v1/acceptance_once`；已核对该输出不存在，本轮不创建。
`status`明确为REGISTERED_AWAITING_NATIVE_APPROVAL，`approval_call_id`和`approval_answer`均为明确非空的`NOT_GRANTED`标志，不是假批准、不拿本次登记ID或旧许可充当运行许可。

这是一份刻意不可执行的待批准稿：既有校验器只接受APPROVED_NATIVE_ONLY及真实call_许可，当前三个标志不满足合同。上述为静态字段/源码核对，未调用或导入校验器，不把“待运行稿已登记”写成“原生协议已获执行接收”。
未来必须先获得本验收的新一次真实人类批准，再仅补实际授权原文与对应状态、重新提交及远端读回；46项源码身份、场景/限额和目录保持不变。登记提交号由外部执行授权传入，不写入本JSON自身形成循环身份。任何源码漂移先停，不自动修改身份表或重写协议绕过门限。

## 2. 固定四场景

| 场景 | 既定工作或故障 | controller上限 | 核心接收证据 |
|---|---|---:|---|
| N1 normal-renew-close | begin_work后45秒有界等待，正常关闭退出 | 90秒 | 至少两对真实RENEW/ACK，最终sequence至少3；MSB COMPLETE、精确关闭、exit0、退出及清理确认、正常电源请求释放。 |
| N2 worker-exit-no-done | ready及身份ACK后worker硬退出23，无DONE | 45秒 | 精确exit23和故障前WORKING；仅接受相应缺关闭/pipe EOF路径，controller负例检查后exit0、父释放及完整清理；任意FAILED或外层timeout不算通过。 |
| N3 done-linger | 正常DONE后故意迟退35秒 | 75秒 | 冻结租约不续延，原监督器因租约失效拒绝并回收，非以LC外层75秒超时替代。 |
| N4 parent-hard-exit | 身份/Job/WORKING及collector ACK确认后controller硬退出31 | 45秒 | 精确父exit31，故障前worker确实在Job内且存活，5秒内由保留进程句柄确认退出；不得先杀worker、持有Job句柄延长寿命或补造DONE/成功结果。 |

顺序固定N1、N2、N3、N4，每例controller/worker各一次，最多4+4个工作进程；一个collector，无重试、续跑、替补、扩场景或额外计时探查。N2/N4的ready-ACK最多5秒；故障前所需身份或活性证据不足，不触发预定故障而按失败清理。
预定故障证据完整属于该负例通过；非预期故障、缺证据、outer kill/timeout、未知所有权、缺例、清理或正常释放失败，整体验收停止，不进入后续场景。N2硬退出worker及N4硬退出两进程的显式电源释放不可观测，不补造released=True。

## 3. 480秒、电源与交付

累计验收窗口480秒，包含入口校验、依赖准备、源码及远端登记核对、进程/观察器创建、四例、诊断、清理和最终发布；不重新起点，等号过期。
各例外层限额为min(90/45/75/45,累计剩余)。所有层仍继承同一生产MSB起点和7200000ms预算；额外480秒验收窗不改变生产预算，也不宣称实测两小时超时。
保留30秒最大采样间隙、2秒双时钟差、30秒租约、15秒续租及最多5秒生产清理；清理不获得新计算预算。OS抢占、底层阻塞及崩溃下不承诺普遍硬实时完成，缺证据不通过。

持续接电AC=1，不主动休眠/断电，不改电源方案或待机设置；前后只读电源设置须一致。**无需保持键鼠空闲**，不重做旧360秒空闲验收或旧0.25秒超时测试。
只正常路径及collector可报告已观测的显式释放；进程退出不能证明所有句柄零泄漏。磁盘PASS单独不足，仍要求最终检查、实际返回/CLI stdout和交付状态一致；失败持久化再失败保留全部原因并非零，不能声称旧磁盘状态已撤销。

## 4. 身份与既有证据

[源码身份清单](../../outputs/msb_native_supervision_registration_v1/source_identity.json)含固定43项原科学执行闭包加3项验收基础设施，共46项；不是把验收驱动加入科学闭包。
逐项原始字节SHA-256匹配既有235项final的身份清单；另单列规范化LF SHA-256、Git blob SHA-1。GitHub基线树的scripts/tests子树通过独立连接器核对全部46个blob，未把Git blob当原始字节哈希。
科学模块仅读取源码字节以核对身份，未导入、调用或检查真实输入。两个正式报告只做原文件哈希保护，未读取正文。

[保护前记录](../../outputs/msb_native_supervision_registration_v1/protection_before.json)、[保护后核对](../../outputs/msb_native_supervision_registration_v1/protection_after.json)覆盖163条既有记录；README仅追加本次登记状态，其余162条匹配。计数为记录数，不是去重文件数；既有六张脏PNG不暂存、不还原，其他未跟踪产物不操作、不作全量哈希审计。
[静态登记检查](../../outputs/msb_native_supervision_registration_v1/registration_check.json)只记录JSON、字段、路径、源码/远端身份和保护核对，不是测试日志或原生接收结果。既有RED/235项日志、三处尾空格、历史时钟守卫缺口和历次独立原件均保持原样。

## 5. 本轮禁止与后续交接

本轮不开发代码、不运行任何验收或测试，不导入项目、不新增前向/拟合/PCA/T0或计时，不读取真实prepared/fit/stop、矿物图片、缓存、权重或.pt文件，不启动原生进程/线程/电源/空闲探查，不改网络、损失、公式、实验矩阵或正式报告，不派代理。Git提交和归档网络访问不是实验运行。
只创建待运行协议、登记/保护证据及README状态；不创建acceptance_once、RUNNING状态、case目录或科学输出。

后续分两道独立门：新一次原生运行批准后补许可并重新登记，再开展一次四场景验收；仅其限定通过后，另批真实MSB-RRE固定fit-only机制协议及运行。任何失败或证据不足留存并停止，不自动修复、重试、扩网格或续跑旧目录。
这项登记不是网络理论创新、原生通过、论文实验结果或工业能力证明。B路线及11单元、340已暴露fit、三折三种子、100步主点/400步上限、三种子七参照晋级门均不改；真实收益未验证仍不写入正式成果，未达门仍NO_ROLE_PROMOTION，C路线继续后置。
