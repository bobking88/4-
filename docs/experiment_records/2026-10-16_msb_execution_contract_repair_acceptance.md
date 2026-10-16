# MSB 执行合同修复：原生实现与合成验收

日期：2026-10-16。授权：`call_3d401d8b809843e0b7103995a772caad`。
用户批准 GitHub `8de3804` 中的规格与计划，限定原生代码及合成验收 Tasks1--5。
状态：`AUTHOR_REPAIR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING`。

## 已完成范围

按[修复规格](../superpowers/specs/2026-10-16-msb-execution-contract-repair-design.md)及[实施计划](../superpowers/plans/2026-10-16-msb-execution-contract-repair.md)原生实施，未派实现或审阅代理。
这是作者实现与合成验证，不是新的独立工程复查，也不是本机原生进程验收。

| 任务 | 提交 | 交付 |
|---|---|---|
| 1 | f73596f | train/validation 精确 projector_fit；无张量身份摘要及共享三折校验器 |
| 2 | 139a69d | 不可变入口 RunBudget、严格控制消息、父句柄身份与 worker scope |
| 3 | e3cf21e | MSB 专用管道/Job Object 适配及有序 GRANT/ACK/RENEW/DONE 监督 |
| 4 | 22af013 | 三次 load 后、任何 fit 前的父门及逐 worker 子门；分析/导出继承同一期限 |
| 5 | 本记录及同次归档提交 | 九模块守卫验收、源码/保护身份、失败历史及复现入口 |

活跃通道与原父进程创建身份取代磁盘 RUNNING 作为启动依据；子 guard 不重置预算。先关联 KILL_ON_JOB_CLOSE，再授予工作许可；清理或释放证据不全不能成功，过期结果不交付。具体 Windows 句柄继承、嵌套 Job 和父硬退出行为尚未进行原生验证。

已有三次 load 输出精确 JSON 身份摘要；联合三折通过才生成 joint_preflight 证书并派发 fit。后续 worker 重验摘要哈希、登记、联合语义与证书；没有第 81 个任务，没有新增模型前向。

## 合成验收与证伪证据

[最终摘要](../../outputs/msb_execution_repair_synthetic_v3/acceptance_summary.json)与[完整日志](../../outputs/msb_execution_repair_synthetic_v3/synthetic_tests.log)：121 项通过，0 失败、0 错误、0 跳过，受限接口未模拟调用计数为 0。

| 模块 | 实际通过数 |
|---|---:|
| 执行安全 | 25 |
| 数据合同 | 12 |
| 网络 | 9 |
| 强对照 | 10 |
| 条件理论审计 | 13 |
| 固定训练流程 | 9 |
| 执行器 | 22 |
| 结果分析 | 17 |
| 公式结构图 | 4 |
| 合计 | 121 |

最终[固定验收入口](../../tests/run_msb_execution_repair_acceptance.py)显式列出九个模块，拦截 NativeMSBAPI 构造及 tick、WinDLL、真实 Popen 和 os._exit；张量反序列化只允许临时合成目录或 BytesIO。进程、句柄、时钟、电源与 OS 枚举均模拟；threadpool_limits 的 OS 枚举用 nullcontext 替代，合成张量计算、损失和求解仍真实执行。两次临时张量读回仅用于合成 round-trip，不是项目 prepared/缓存。

关键反例与观察：

- 原 runner 在三折各自合法、联合 OOF 重复时会进入首个 trajectory；修复后最多完成三次 load，神经与凸拟合调用均为零。[有效修复前失败日志](../../outputs/msb_execution_repair_synthetic_v2/task4-preflight-red-valid.log)保留原 15 项通过、新两项失败的结果。
- 缺失、错误或带空格的 scope 在 train 与 validation 中都拒绝，不用另一折的 fit 标记掩盖训练记录。
- 陈旧 RUNNING、缺少继承通道、PID 创建身份不符、延长 deadline、租约重放、Job 关联失败和无法确认 wait 均不成功交付。
- 已验收摘要被替换，在下一 worker 打开本折输入前失败；准备耗尽预算不派任务；analysis 与 final 导出超时均清除 delivery_verified。
- 有效 DONE 后管道可能先于进程退出关闭；先复现错误拒绝，再修复为继续监督退出而不再读/续租该管道。[该反例红灯](../../outputs/msb_execution_repair_synthetic_v2/task3-pipe-red.log)及[23 项绿灯](../../outputs/msb_execution_repair_synthetic_v2/task3-final-green.log)均保留。

末轮作者审查另发现“Popen 已返回进程，但恢复句柄继承标志失败”时进程对象可能丢失于异常栈；已用两项 fake 故障回归先红后绿，启动异常显式携带 spawned_process，父端接收后仍 kill/wait。此前 v2 的 119/119 作为旧源码快照保留，当前源码以加入两项回归的 v3 为准。

原 83 项合成测试语义保留；121 不是 121 次真实训练。75 fit / 80 job 全流程通过记录是 mock 编排，不能写成完成真实矩阵。部分单元测试在新建小合成张量上做短轨迹、100 步检查或凸求解，这是被批准的合成验收，不是现有 340 分组真实实验。

## 偏差、失败与处理

1. 沙箱 OS 临时目录的原子替换曾被拒绝。改用本计划隔离 scratch 下的 TemporaryDirectory，未更改生产原子写入，也未扩大文件访问权限。
2. 早期两次 runner 集成回归中，simulate 之外的非法入口未继承假时钟，进入 run_stage 时采样了只读 GetTickCount64 后才拒绝非法元数据/路径。没有启动监督进程、申请电源、读真实输入或拟合；但不满足“时钟接口全部模拟”，不能抹去或将早期日志标为完全模拟。未逐次插桩，不虚报采样次数。现已在 setUp 统一注入假接口，并在最终入口禁止真实 API 构造/tick。[早期集成日志](../../outputs/msb_execution_repair_synthetic_v2/task4-runner-green.log)与[后续守卫集成](../../outputs/msb_execution_repair_synthetic_v2/task4-guarded-integration.log)分开保存。
3. [首轮严格验收 v1](../../outputs/msb_execution_repair_synthetic_v1/acceptance_summary.json)为 FAILED（9 失败、1 错误）。原因是验收器将 Path.name 误作完整路径、未区分 BytesIO，且 threadpoolctl 的 OS 枚举触发 WinDLL 禁用守卫；被拦截的 DLL 调用没有执行。修正的是验收器的路径类型处理和 OS 枚举模拟，未更改模型或放开真实数据。v1 原日志、源码身份及[可哈希核对的验收器原文](../../outputs/msb_execution_repair_synthetic_v1/acceptance_harness_source.json)均保留，后续 v2/v3 使用独立目录，不覆盖 v1。

因此“最终 121 项通过、受限接口无未模拟调用”仅指 v3 最终验收，不反向覆盖以上历史偏差。没有重复六分钟空闲/计时验收，没有创建或续跑真实实验目录。

源码/说明文档空白检查通过；原始 task1-red.log 第 2 行包含 unittest 输出自带的尾随空格，原样保留，不为消除警告改写失败证据。

## 身份保护与科学不变性

[43 项源码及九模块测试身份](../../outputs/msb_execution_repair_synthetic_v3/source_identity.json)、[旧 34 源码保护](../../outputs/msb_execution_repair_synthetic_v3/legacy_source_preservation.json)、[10 项保护文件](../../outputs/msb_execution_repair_synthetic_v3/protected_artifact_preservation.json)及[规格/旧日志保护](../../outputs/msb_execution_repair_synthetic_v3/spec_and_history_preservation.json)分别登记。
原网络、损失/强对照、理论审计和结构图四个源码逐项不变；原科学规格、修复规格、83 项历史日志、六张用户 PNG 和两处正式报告保持原身份。

保留：三个 rank4 分支、1083 当前参数及固定参照、R_ref=2|m|+0.25、原损失、11 单元、三种子/三折、63+12 拟合、400 步上限、100 步主点、7200 秒累计预算、615+12 最终核验及原七参照晋级门。唯一新增执行源码是安全模块，未来协议必须登记 43 项闭包，不能沿用旧 42 项身份；本次代码授权已明确拒绝作为真实运行许可。

## 复现与下一闸门

从同一 worktree 用既有固定 Python 执行下列入口可复现同一限定验收；默认输出独占，已有回执时拒绝覆盖，重新验收需先确定新的独立输出身份：

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' tests/run_msb_execution_repair_acceptance.py
```

本轮只提升实验可信度，不增加网络原创性或真实识别效果证据。未重新文献查新，未证明真实角色分支优于充分求解的七参照，也未验证泛化或工业分选。正式报告不写入新效果结论。

下一步建议另批一次独立只读工程复查，重点检查继承句柄/管道/Job 生命周期、累计 deadline、两级 preflight 与此次隔离偏差记录；不读真实输入、不重跑测试。原生无数据验收、新真实协议登记及真实运行仍各自另需批准，不能自动进入原计划 Task9。
