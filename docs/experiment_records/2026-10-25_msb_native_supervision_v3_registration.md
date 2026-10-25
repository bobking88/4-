# MSB-RRE 原生无数据监督验收 v3：待运行协议登记

日期：2026-10-25（Asia/Shanghai）。状态：`REGISTERED_AWAITING_NATIVE_APPROVAL`，`NOT_RUN`。

本轮授权 `call_9c7c2737ea8440299d7799d573181300`，答复“批准仅编写与登记待运行协议（推荐）”，仅允许文档与 GitHub 登记。基线为 `6982bcacb650d19a26251679fd365f16a14dbd9d`；已复查源码固定 `4312036b33c05a9b8c29f4849860562eb7c68b87`。依据原 N1–N4 规格、固定 v3 适配规格及[最新独立工程 APPROVE](2026-10-25_msb_native_namespace_v3_review_v2.md)，不修改已归档资料；实际原生验收及真实机制实验仍未获批。

## 1. 新协议与未授权状态

新[协议 JSON](../experiment_protocols/msb_native_supervision_v3.json)固定为 `docs/experiment_protocols/msb_native_supervision_v3.json`，拟用独占运行输出为 `outputs/msb_native_supervision_v3/acceptance_once`。登记前均不存在；本轮只创建协议，不创建运行输出、RUNNING claim、controller/case/worker 目录。登记证据另存于 `outputs/msb_native_supervision_v3_registration_v1`，不作为运行输出。

沿用执行器精确 16 字段、原 46 项 SOURCE_FILES、`msb_parent_bound_v2` 与 `DIRECT_CPYTHON311_VENV_V1`。venv 根仍登记为既有 `D:\成信工科研\人工智能选矿\.venv-training`，不查询或启动解释器。

协议中的 `status=REGISTERED_AWAITING_NATIVE_APPROVAL`、`approval_call_id=NOT_GRANTED`、`approval_answer=NOT_GRANTED`；`real_training_authorized=false`。本次文档许可、复查许可和旧 v2 原生许可均不作为执行许可。

这是刻意不能通过执行门的待批准稿：源码要求 `APPROVED_NATIVE_ONLY`、新的精确人类原生许可及匹配远端登记提交。本轮只静态比较字段与源码，不导入或调用校验器，不尝试执行，不消费新的原生许可。

未来另获真实原生授权后，仅补状态及两个授权字段，重新登记并实际远端回读后才进入唯一 collector；登记提交号由外部传入，不将自身提交号嵌入协议形成循环身份。源码、目录、场景及限额保持固定，身份漂移先停，不静默更新哈希放行。

旧 v1/v2 协议、已消费许可、失败目录与原件保持不变，不删除、不续跑。新命名空间不代表旧 Windows 首发 trace 原因已完全定位，也不代表修复已经原生实测。

## 2. 固定 N1–N4 与接收边界

| 场景 | 既定行为 | controller 外层上限 | 核心接收要求 |
| --- | --- | ---: | --- |
| N1 normal-renew-close | begin_work 后等待 45 秒并正常退出 | 90 秒 | 至少两对真实 RENEW/ACK、最终 sequence 至少 3；MSB COMPLETE、关闭握手、exit0、退出/清理及正常电源释放证据 |
| N2 worker-exit-no-done | ready/身份 ACK 后 worker 硬退出 23，无 DONE | 45 秒 | 精确 exit23 和故障前 WORKING；仅对应缺关闭/原生 pipe EOF 路径，controller 检查后 exit0、父释放与清理完整；不以任意 FAILED 或外层 timeout 替代 |
| N3 done-linger | 正常 DONE 后故意迟退 35 秒 | 75 秒 | 原冻结租约不续延，由原监督器租约失效拒收并回收，不改为 LC 外层超时 |
| N4 parent-hard-exit | 身份、Job、WORKING 及 collector ACK 后 controller 硬退出 31 | 45 秒 | 精确父 exit31；故障前 worker 在 Job 且存活；保留身份句柄并在 5 秒内确认退出；不预杀、不持有 Job 延长寿命、不补造 DONE/结果 |

一个 collector，固定顺序 N1、N2、N3、N4；每例 controller 与 worker 各一次，最多四加四个工作进程。N2/N4 ready-ACK 最多 5 秒；前置身份或活性不足，不触发预定故障而进入失败清理。

只有证据齐全的预定负例可记为该例通过。任何意外失败、证据不足、缺例、外层 kill/timeout、所有权不明或正常释放/清理失败，立即停止，不启动后例，不重试、不续跑、不替补、不扩场景。N2/N4 硬退出缺少显式释放观测时，不补填 released=True。

不重做 360 秒空闲验收、0.25 秒超时测试或其他计时探查。沿用既有 trace 首因、内层结果/最终发布错误分离、失败回执及拒收规则，不为登记修改执行代码。

## 3. 累计期限与接电条件

未来实际入口的 480 秒累计接收窗口包括入口校验、依赖准备、源码及远端核验、创建、四例、诊断、清理和最终发布，不清零起点，等号过期。每例上限为其固定上限与累计剩余的较小值；后例前及最终发布仍受同一累计期限约束。本次登记不是该运行入口，没有开始计时验收。

各层继承首次 MSB 起点与原 7200000ms 生产预算；480 秒是额外接收上限，不修改生产预算、不宣称实测两小时超时。原 30 秒最大采样间隙、2 秒双时钟差、30 秒租约、15 秒续租和最多 5 秒生产清理不变；清理不获得新计算额度。OS 抢占、底层阻塞或崩溃下不承诺普遍硬实时，超期或证据不足不得通过。

要求未来运行全程 AC=1 接电，不主动休眠/断电，不改电源方案或待机设置，前后只读设置应一致。**键鼠可正常使用**，不要求空闲。只报告实际观测释放，不由退出推导所有句柄零泄漏。

磁盘 PASS 候选不足以单独通过：末端检查、实际返回与 CLI 交付须一致；失败回执不能保存时保留原异常并返回非零，不宣称已有磁盘候选自动撤销。

## 4. 源码身份与静态核验

[源码身份](../../outputs/msb_native_supervision_v3_registration_v1/source_identity.json)保留原 43 项科学执行闭包与三项验收基础设施，共 46 项，路径集合与当前 SOURCE_FILES 纯字面量及旧 v2 协议一致。旧 v2 的 43 项原始源码哈希不变，三项验收文件更新为已审 4312036 的身份，没有修改任何源码。

46 项原始字节 SHA-256、规范化 LF SHA-256、Git blob SHA-1 分开列出，不混用。全部 Git blob 同时匹配已审 4312036 与归档基线 6982bca；通过实际 GitHub commit/root/scripts/tests tree 核对 6982bca 下 46 项 blob。未来仍须严格核对本地原始字节，远端对象规范化不能替代本地原始哈希。

本轮未查询解释器映像、进程、电源或计时；venv 只登记已知路径，未来实际身份仍由原严格链检查。既有 GREEN 124/124、final 285/285 仅引用已存日志和独立静态复查，不重新测试、不据此宣布原生就绪。

## 5. 本轮保护与文档范围

[前保护](../../outputs/msb_native_supervision_v3_registration_v1/protection_before.json)由前 480 条按本轮 README 身份归一化，再加复查归档 6982bca 十文件组成，共 490 条，保留重复路径。[后保护](../../outputs/msb_native_supervision_v3_registration_v1/protection_after.json)仅允许 README 索引更新。两处正式报告、网络/损失/公式/科学矩阵、列明旧日志与原生失败原件、六张预存脏 PNG 保持原哈希，不暂存或还原其他改动。

仅创建协议、登记记录、授权、源码身份、保护和[静态登记检查](../../outputs/msb_native_supervision_v3_registration_v1/registration_check.json)，追加 README。没有代码开发、项目导入、测试、前向、拟合、PCA/T0、计时/进程/线程/电源/原生验收，不读 prepared/fit/stop、原图、真实缓存/权重/张量，不派代理、不改正式报告。Git 归档网络访问不是实验执行。

保护清单是列明文件的固定前后身份核对，不是工作区全量审计，也不能证明不存在所有其他历史调用或瞬时修改。

## 6. 后续关口与科学边界

登记并实际远端回读之后，仍须另获一次按固定协议的原生无数据验收许可，补授权、重登记与远端核验后才运行。失败或证据不足保留原件即停，不自动修复或重试。

只有限定原生门通过，才另行登记、批准真实 MSB-RRE fit-only 机制协议与实验。B 主线、C 后置和既定强对照不变；间隔缩放预算及独立角色非线性证据能否产生独立收益，必须通过科学实验验证。工程登记不构成理论新颖性、网络优越性、论文阳性结果或工业能力证明。

本文创建时，本次提交与实际远端逐文件回读尚待完成；不能用本文代替后续核验。
