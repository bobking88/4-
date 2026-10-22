# MSB原生无数据监督验收v2：单次真实授权重登记

日期：2026-10-22（Asia/Shanghai）。登记时状态：`AUTHORIZED_NOT_STARTED_PENDING_REREGISTRATION_READBACK`。
真实授权：`call_cdd31dd8986040fb8bd90d6e311f95fc`，答复“批准一次固定协议原生验收（推荐）”；[授权原件](../../outputs/msb_native_supervision_v2_execution_registration_v1/authorization.json)。

待运行稿基线：`4918610f470644e1c2f2d567fd305fbf53b16bb6`；受审源码仍为`35fed93746ae380578d54dd89183235d67b886e4`，限定独立APPROVE归档为`487af614ad4ee0a70870cb39e8b00b34e9beb3b7`。既有[登记记录](2026-10-22_msb_native_supervision_v2_registration.md)及[工程复查](2026-10-22_msb_observer_abort_retention_review.md)保留，不改写为此前已有运行许可。

## 1. 仅补实际许可并先重新登记

只对[固定协议](../experiment_protocols/msb_native_supervision_v2.json)补三个字段：status=`APPROVED_NATIVE_ONLY`、approval_call_id为本次真实ID、approval_answer为上述原文。其他13字段、46项原始源码SHA-256、runtime_venv_root和DIRECT_CPYTHON311_VENV_V1合同保持不变。

本次只批准原生无数据验收，不是网络或科学实验许可。先提交并独立回读核验实际协议、登记内容及源码树，再把该登记提交号从CLI外部传入，启动唯一collector。登记时尚未调用入口、未创建运行目录，也不执行前置计时/进程/电源探查；实际AC和原生身份由已登记驱动在唯一运行中检查。

## 2. 单次执行边界

拟用独占输出仍为`outputs/msb_native_supervision_v2/acceptance_once`，登记前确认不存在；不提前创建，仅由原驱动创建。一旦调用collector即消费本次许可，包括入口拒绝或未生成输出的情形，不自动重试。旧v1失败目录、协议及许可不复用、不删除或续跑。

固定N1、N2、N3、N4，每例controller和worker各一次，最多4+4个工作进程，一个collector。controller上限90/45/75/45秒，整体480秒累计接收窗包含入口准备、依赖/源码/远端核验、四例、诊断、清理和最终发布，不重置起点，等号过期。原MSB生产7200000ms预算及共同起点不变，不将验收写成实测两小时超时或普遍OS硬实时保证。

N1为45秒等待及至少两次续租/精确关闭；N2为工作身份确认后worker退出23且无DONE；N3为DONE后故意迟退35秒，须由冻结租约失效拒绝并回收；N4为完整身份/Job/WORKING/collector ACK后父退出31，并由保留身份句柄确认worker退出。预定负例须证据完整才通过；非预期失败、证据不足、outer timeout、未知所有权或清理/正常释放失败即停，不进后例、不替补、不扩大场景。

持续AC=1接电，键鼠可正常使用。不主动休眠/断电，不修改电源方案、待机或阈值，不重做旧360秒空闲及0.25秒超时测试。只报告实际观测的显式释放，硬退出进程缺证据不能补写released=True。

## 3. 原始日志与保护

通过现有`D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe`调用已登记collector公共入口，固定root为当前工作树，传入协议、真实许可原文和回读验证后的登记提交号。只临时设置PYTHONDONTWRITEBYTECODE=1、PYTHONUTF8=1以避免字节码并稳定日志编码，结束后恢复父shell值，不改模型、求解器或电源配置。

入口stdout/stderr先保存在新私有元数据目录`.superpowers/sdd/2026-10-22-msb-native-supervision-v2-execution`。父会话仅观察同一工具session，记录实际退出码/终态，不重启collector；多行JSON输出按行只读解析，不因解析问题重跑。正常依赖导入不冒称所有原生接口仍被mock。

内部原生元数据和原始日志保持原件；公开GitHub前检查活动能力、原始nonce及可复用句柄，必要时只导出去能力化摘要并保留内部哈希，不覆盖原件、不提交秘密能力。只只读观察本次无数据输出，不追加独立探针或用外层timeout代替原生证据。

[前保护](../../outputs/msb_native_supervision_v2_execution_registration_v1/protection_before.json)固定296条既有记录；[登记后保护](../../outputs/msb_native_supervision_v2_execution_registration_v1/protection_after.json)只允许README追加及协议三个授权字段变化。[静态登记检查](../../outputs/msb_native_supervision_v2_execution_registration_v1/registration_check.json)不是新增测试或原生接收结果。正式报告、旧原件、网络/公式/矩阵及六张既有脏PNG保持原样；不作其他未跟踪产物全量审计。

## 4. 禁止与科学边界

不修改源码或执行器，不重跑合成测试，不派代理，不读真实prepared/fit/stop、原图、权重、缓存或张量，不训练/模型前向/PCA/T0，不改网络、损失、公式、实验矩阵或正式报告。无数据packet不是矿物输入或科学结果。

运行后依据实际stdout/工具退出码、根终态与四例原生证据归档。失败或证据不足保留原件即停，不边运行边修复或重新登记挽救结果。即使原生限定通过，也只解除该工程门；后续真实MSB-RRE机制强对照须单独登记、另行批准，角色证据和间隔预算的科学收益仍未验证。
