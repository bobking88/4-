# MSB-RRE 原生无数据监督验收 v3：单次真实授权重登记

日期：2026-10-25（Asia/Shanghai）。登记时状态：`AUTHORIZED_NOT_STARTED_PENDING_REREGISTRATION_READBACK`。

真实授权 `call_d7ead69a9778437890a84a9c099e586a`，答复“批准一次固定协议原生验收（推荐）”。[授权原件](../../outputs/msb_native_supervision_v3_execution_registration_v1/authorization.json)只允许按固定协议执行一次原生无数据监督验收，真实机制实验仍未授权。

待运行稿基线 `f5fe149962c004ad1b1414eeee84fd55746bfc36`；已复查源码仍为 `4312036b33c05a9b8c29f4849860562eb7c68b87`，独立工程 APPROVE 归档为 6982bca。原[待运行登记](2026-10-25_msb_native_supervision_v3_registration.md)和[独立复查](2026-10-25_msb_native_namespace_v3_review_v2.md)保留，不改写为此前已有运行许可。

## 1. 仅补真实许可并先重新登记

只补[固定协议](../experiment_protocols/msb_native_supervision_v3.json)三个字段：status=`APPROVED_NATIVE_ONLY`、approval_call_id=`call_d7ead69a9778437890a84a9c099e586a`、approval_answer 为上述精确原文。其余 13 字段、46 项原始源码 SHA-256、venv 根、启动与监督合同均不变。

先提交并独立回读核验实际协议、登记内容与不变源码身份，再从 CLI 外部传入该登记提交号，启动唯一 collector。本文创建时未调用入口、未创建运行输出，不进行前置计时/进程/电源探查；实际 AC 与解释器/父子身份由已登记驱动在单次运行中核查。

## 2. 单次执行边界

固定独占输出 `outputs/msb_native_supervision_v3/acceptance_once`，登记前确认不存在，不提前创建，仅由原驱动创建。任何 collector 调用即消费本许可，包括入口拒绝或没有输出；不重试、不续跑、不自动选新目录。旧 v1/v2 许可、目录及原件不复用、不删除或改写。

固定 N1–N4，一个 collector、每例 controller/worker 各一次，最多四加四个工作进程。外层上限 90/45/75/45 秒；480 秒累计接收窗口包含入口准备、依赖/源码/远端核验、四例、诊断、清理和最终发布，不重置起点，等号过期。继承原 7200000ms MSB 预算与共同起点，不将验收写成实测两小时超时或普遍 OS 硬实时保证。

N1 为 45 秒正常等待、至少两次续租与精确关闭；N2 为工作身份确认后 worker 退出 23、无 DONE；N3 为 DONE 后故意迟退 35 秒，须由冻结租约失效拒收回收；N4 为身份/Job/WORKING/collector ACK 后父退出 31，保留身份句柄确认 worker 退出。预定负例须证据齐全才能通过；意外失败、证据不足、外层超时、所有权不明、清理或正常释放失败即停，不进入后例、不替补、不扩场景。

要求全程 AC=1 接电；键鼠可正常使用。不主动休眠或断电，不改电源方案、待机或阈值，不重做 360 秒空闲或 0.25 秒超时验收，不追加探针。只报告实际释放观测，硬退出缺乏显式释放证据时不能补写 released=True。

## 3. 原始入口与证据保存

使用原已登记 venv 的公共 collector 入口，root 固定为当前 worktree；传入固定协议、真实许可原文及实际回读后的登记提交。仅临时设置 PYTHONDONTWRITEBYTECODE=1 和 PYTHONUTF8=1，结束恢复父 shell 值，不改模型、求解器或电源设置。

入口 stdout/stderr 保存到独立私有元数据目录 `.superpowers/sdd/2026-10-25-msb-native-supervision-v3-execution`。父会话只观察同一工具 session，保留实际退出码/终态，不重启。多行 JSON 按行只读解析；解析问题不成为重跑理由。原生执行中正常依赖导入不宣称仍为全部 mock。

原生元数据与日志保留原件；公开 GitHub 前检查原始 nonce、可复用句柄与活动能力。若含能力，仅公开去能力化摘要并保存内部原始哈希，不覆盖原件、不上传秘密。保留原始错误、pending 文件及失败回执，不删除失败痕迹或补造成功。

[前保护](../../outputs/msb_native_supervision_v3_execution_registration_v1/protection_before.json)为前 490 条加待运行登记八文件，共 498 条，保留重复项。[登记后保护](../../outputs/msb_native_supervision_v3_execution_registration_v1/protection_after.json)仅允许 README 与协议三个授权字段变化。[静态登记检查](../../outputs/msb_native_supervision_v3_execution_registration_v1/registration_check.json)不是新的测试或原生通过证据。报告两处、源码、旧证据与六张预存脏 PNG 保持原样，不暂存或还原其他改动。

## 4. 禁止与科学边界

不改源码或执行器，不重跑合成验收、不派代理；不读真实 prepared/fit/stop、原图、权重、缓存或张量，不训练/模型前向/PCA/T0，不改网络、损失、公式、实验矩阵或正式报告。无数据 packet 不是矿物输入或科学结果。

运行后依据实际 stdout、工具退出码、根终态与四例证据归档。失败或证据不足保留原件即停，不边运行边修复、重登记或重跑挽救结果。即使限定原生验收通过，也仅解除该工程门；后续真实 MSB-RRE 机制强对照仍须另行登记批准，网络理论收益尚未验证。

本文创建时实际远端回读与唯一运行尚未发生，必须以之后实际工具/原始证据为准。
