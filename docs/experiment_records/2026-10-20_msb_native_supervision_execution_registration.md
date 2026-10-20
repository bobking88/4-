# MSB原生无数据监督验收：真实单次执行授权登记

日期：2026-10-20。状态：`AUTHORIZED_NOT_STARTED_PENDING_REREGISTRATION_READBACK`。
真实授权：`call_3f51cdd2cc0040929106824e0c335950`，答复原文“批准一次固定协议原生验收（推荐）”；[完整授权](../../outputs/msb_native_supervision_execution_registration_v1/authorization.json)。
待运行稿基线：`2b8f5a28ba7c00d7e3dfd1790018c252800ae219`，受审代码仍为`599643fbdc6782bc9f3c8827d4ac3587e788fc24`，限定独立复查为[APPROVE / R4 ADDRESSED](2026-10-20_msb_initialization_error_retention_independent_review_v2.md)。

## 唯一获批范围

只对[既定协议](../experiment_protocols/msb_native_supervision_v1.json)补三个字段：status=APPROVED_NATIVE_ONLY、approval_call_id为本次真实ID、approval_answer为上述原文；其余字段和46项源码哈希保持原样。
此前仅登记的ID、NOT_GRANTED标志及旧许可均不得使用。本提交只完成授权重登记，完成独立远端读回后才启动唯一collector；登记提交号由实际CLI外部传入，避免协议自引用。当前没有提前创建运行目录或启动验收。

按[认可规格](../superpowers/specs/2026-10-18-msb-native-supervision-acceptance-design.md)和[登记记录](2026-10-20_msb_native_supervision_registration.md)，固定N1、N2、N3、N4，每例一次，最多4个controller及4个worker。
整体480秒验收窗口含入口准备、源码/远端核验、四例、诊断、清理和最终发布，等号过期；每例外层上限90/45/75/45秒。原MSB生产7200000ms预算及共同起点不变，不把本验收说成实测两小时超时。
N1工作45秒并核对至少两对续租；N2 worker退出23、无DONE；N3 DONE后迟退35秒触发冻结租约；N4 parent退出31并由既有Job/保留进程句柄核对worker退出。只能依据实际因果与正常清理/释放证据判定，不以LC外层timeout代替负例。

要求持续AC=1，键鼠可正常使用；不主动睡眠/断电，不改电源方案、待机设置或采样阈值，不重做旧360秒空闲和0.25秒测试。接电未知、设施不可用、任例非预期失败或缺证据即停；一次调用即消费本授权，即使入口拒绝或尚未建输出也不自动重试。
预期故障符合负例合同才算该例通过；任何应急清理不补写为负例成功，硬退出进程的显式释放未观测则保留未知。
拟用新输出`outputs/msb_native_supervision_v1/acceptance_once`，已经核对不存在；仅由获批驱动创建，不预建、不续跑、不覆盖旧目录。

## 保护与执行方法

[保护前记录](../../outputs/msb_native_supervision_execution_registration_v1/protection_before.json)包含170条既有记录；[登记后核对](../../outputs/msb_native_supervision_execution_registration_v1/protection_after.json)只允许README状态追加和该协议三个授权字段变化，其他168条已匹配。记录数不是去重文件数；正式报告只哈希不读正文，既有六PNG与未跟踪产物不操作。
[静态登记核对](../../outputs/msb_native_supervision_execution_registration_v1/registration_check.json)不是新的测试日志；本轮不重跑235项合成测试、不作新计时/空闲探查、不派代理。

只调用现有`scripts/check_msb_native_supervision.py`的collector公共入口，固定项目root、协议路径、真实授权原文及回读核验的登记提交号；使用现有`D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe`。
临时设置PYTHONDONTWRITEBYTECODE=1及PYTHONUTF8=1，仅避免派生字节码、固定日志编码；不改变任何模型/求解器或电源设置。依赖正常载入不冒称全部原生接口模拟。
原生进程、IPC、Job、时钟及临时防待机请求由已有驱动/监督器管理；父会话保存实际入口stdout/stderr、工具退出码及终态，只只读观察本次输出，不追加探针或替代监督器。

不读取真实prepared/fit/stop、原图、缓存、权重或.pt数据，不训练、模型前向、PCA/T0，不改代码、网络、损失、公式、实验矩阵或正式报告。验收packet只是新目录中的无数据元数据。
运行后原件留在唯一目录；公开归档只保留不含活动能力、原始nonce或可复用句柄的证据，必要去能力化另存摘要，不覆盖内部原件。
本授权不包含真实MSB-RRE科学实验。原生通过也只解除限定工程证据障碍；角色非线性证据、判别间隔预算和相对强对照的收益仍未验证，不能写成网络理论创新已取得实证。
