# LC-RFA-B v1 有限开发实验超时归档

## 结论与边界

本批次 **TIMED_OUT，开发实验未完成**。不能根据这些部分训练结果宣称角色路由、分支专门化或有限预算网络有效，也不能宣称它们无效。未生成选择锁，未开展最终拟合、stop 评价、默认推理复验或训练重放。正式报告、规格、网格和冻结源码均未修改。

用户真实授权为“批准该有限真实开发实验”，仅覆盖既定340个fit分组及340个已暴露stop分组、最多422拟合加4次温度拟合、21600秒累计额度和锁定后一次stop评价。失败即停，不允许自动重试。本次已执行停止规则；不以此前“继续”或持续目标授权重跑。

## 登记与终止证据

- 执行前远端登记提交：`0d98c1b0f18ec4ca199d0421a4736cb882f7040d`，分支 `codex/theory-aware-report`。
- 协议blob：`7d454d513f5944182e4e6e255d1c2d84e11a8cd9`；规范化协议SHA-256：`ef78934dff3b9728afed1968fb64a80b70e63f77f88d2b13ea04290bb345b0c0`。
- 执行命令：`python scripts/run_lc_role_adaptation.py --protocol docs/experiment_protocols/lc_role_adaptation_v1.json --output-dir outputs/training/lc_role_adaptation_v1/development_fold_0 --stage fit`。
- 主执行会话73077退出码1；最后worker为 `cv-U1-0345`，PID33860，退出码1，`alive=false`，`attempts=1`，状态 `TIMED_OUT`。
- 全部350个已启动worker：349个完成、1个超时。其中4个预处理/温度任务完成；345条CV轨迹完成、1条CV尝试超时；没有最终fit。尚未运行P/E凸强对照。
- H0/F0/S0/S1/R0/R1各完成54条CV轨迹，U1仅完成21条。CV原排程369条；不能把345条的部分网格拼成完整选择结果。
- 最后worker观测耗时9955.407秒；累计计时12226.891秒。累计值低于21600秒，但**单worker观测耗时超过登记的600秒，不能称本批次完全满足硬时限协议**。
- `selection.lock.json`、`fit_complete.json`、`oof_predictions.pt` 和 `evaluation/` 均不存在。因完整fit门未满足，evaluate/verify不执行。
- 终止后再次核验全部44项登记来源SHA，均一致；只读进程检查未发现本实验残留python进程。

原始账本和状态位于 `outputs/training/lc_role_adaptation_v1/development_fold_0/`。完整控制台traceback保存在 `outputs/lc_role_implementation_qa/development_fit_console.log`；中间状态、packet及结果权重仅本地保存并在归档清单中列hash。小型日志/CSV/JSON不按阳性结果筛选上传。

归档目录为同输出根下的 `abort_archive/`：

- `abort_audit.json`：由终止账本计算的次数、状态、预算和访问范围，不是模型评价。
- `source_snapshot.json`：终止后重验的44项登记来源hash。
- `artifact_inventory.csv`：1746项原始产物的路径、字节数、SHA-256及归档位置，其中700个`.pt`仅本地。
- `partial_worker_evidence.zip`：全部1040份worker日志/历史/梯度CSV，5118023字节；不含权重、packet或冻结特征。ZIP逐成员内容hash与原文件一致，目录安全、无重名，CRC检查通过。
- `worker_evidence_manifest.json`：ZIP全部成员hash及ZIP本身hash。ZIP容器采用固定1980时间戳，仅为字节稳定，不代表实验发生时间。
- `publish_manifest.json`：全部待同步小型文件的本地SHA-256及Git过滤后blob身份；不对清单自身作循环hash。

worker日志压缩归档避免上千个零散文件的网络写入，但保留了全部内容，不是挑选若干成功轨迹。可用 `python -m zipfile -e partial_worker_evidence.zip 任意新的解压目录` 解压核查；不得把它解压回原失败目录后重跑。默认推理复验未运行，归档完整性检查不能冒称 `VERIFIED_INFERENCE_ONLY`。

## 超时原因核查

Windows System日志记录：2026-10-10 00:28:42进入Modern Standby；失败任务packet在00:28:44.965创建；03:14:40.229记录退出待机；阶段状态和账本在03:14:40.378写入。间隔约9955秒，与worker观测耗时吻合。10:52又有唤醒事件，用户界面工具直到此后才取得终止结果。

上述时间证据支持“宿主待机期间监督器无法持续调度，恢复后判定超时”的解释，而不是证明神经网络计算耗时2小时46分钟。失败worker没有结果文件、没有心跳和非空日志；因此不能从现有证据精确恢复它的活动计算时长或每个暂停区间。事件摘录见同目录 `host_power_events.json`。

Python进程内监督只能在宿主被调度时检查截止时间。现有合成timeout测试不覆盖Windows Modern Standby，不能证明跨待机的绝对wall-clock终止保证。此次不修改登记代码或系统电源设置，也不将超时改标为成功。

## 对理论创新的含义

本批次没有真实完整模型评价和A/B/C逐样本理论审计，故没有新增效果证据。此前10000行合成界检查、464项实现测试及实际3024参数结构图仍可引用为“解析/实现层证据”，不能变成真实泛化收益证明。三个研究假设保持待验证：角色路由相对同参数S1的作用、角色辅助监督与分支机制、限幅相对U1的净效益。

没有独立确认，也没有同行评审。已有stop反复用于开发，本轮即使完成也只具备开发观察资格。不能将部分CV历史写为独立测试提升，不能把界成立写为召回保证。

## 后续建议与未执行事项

下一次开发应先明确不被睡眠中断的执行环境，以及待机/暂停时的计时与中止语义，再通过合成验收和新协议登记获得**新的完整有限运行授权**。保留本失败目录，不在其中续跑；是否重跑同一有限网格、如何处置计算预算，应由用户明确批准。不能在本次失败后自动补做余下轨迹或读取stop。

尚未执行：重跑、扩大网格、其他折、新图片、独立确认、训练重放和正式报告整合。正式报告SHA-256保持 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。

本阶段只做终止验收、失败归档和GitHub同步。此前实施裁定为实际preflight/合成benchmark延后至Task7，以包含完整分析/绘图源码快照；代价是计时证据到实现齐备后才取得。代码验收是作者自审，弱于独立评审；无隐藏的小项延期或新的模型修复。
