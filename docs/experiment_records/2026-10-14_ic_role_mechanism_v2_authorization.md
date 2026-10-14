# IC-RFA-B v1 修复后 v2 协议：单次真实机制实验授权

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent run。
- Origin Date: 2026-10-14，Asia/Shanghai；Version Label: `ic_role_mechanism_v2_once`。
- Verification Status: `AUTHORIZED_PENDING_EXECUTION`，不是效果或完整验收结论。
- 协议登记提交：`bd97d317d04574a9004f207a96cdeee37ab38891`；限定独立工程复查提交：`e64f80f38b5c3ea902e87573616de129ef62f519`。
- 模型、执行和分析源码仍为修复快照 `da176fc5d247affde7b2ceb766aa5a5727aac658`，完整34文件身份不改。

## 1. 直接人类批准

问题项 `["request_user_input_async","call_RHuePD3CQpoUvp4t88mzQnSM",0]` 的直接人类回答为 **“批准一次固定协议的真实机制实验”**。

获批问题原文：是否批准按 GitHub 提交 e64f80f 保留的 v2 协议执行一次真实 fit-only 机制实验？仅补真实运行授权并重新登记；限既有340个fit分组、108条每条最多400步轨迹、最多3次温度拟合，100步为固定主点；准备、训练、诊断、导出及受监督最终推理验收合计7200秒硬上限。要求持续接电；失败或证据不足即停，不重试、不续跑、不扩网格，不重做计时探查或空闲验收，不读stop/原图，不改网络、损失、矩阵或正式报告。

授权记入 UTC 为 `2026-10-14T02:15:35Z`，取自本次时钟工具，是录入时间，不是平台原始人类消息时间戳。实际问题项身份存入机器协议，不另造消息ID。

补录后机器协议字节 SHA-256：`332dd67fec71b123a0b88eea1d0c255e34af11d251f9f48f672065d78679852c`。静态核对相对e64f80f仅status/authorization两项顶层值改变，真实授权门通过；未调用run_stage/worker或真实反序列化。本记录及JSON承载提交远端回读一致后方可启动。

[v2 JSON](../experiment_protocols/ic_role_mechanism_v2.json)仅更新 status/authorization 为 APPROVED_REAL_ONCE、human_approved=true、实际问题项身份及录入UTC。[v2 中文协议](../experiment_protocols/ic_role_mechanism_v2.md)、[登记记录](2026-10-13_ic_role_mechanism_v2_preregistration.md)和[限定复查记录](2026-10-13_ic_role_state_publication_independent_review_v2.md)保留登记/审阅时的未授权历史文字；当前单次许可由本记录及重新登记的JSON决定，不将历史文字改成效果验证。

## 2. 前置身份核对

现有训练解释器中只读核对全部34源码字节SHA、Python3.11.5/torch2.12.1+cu130及CPU float64合同，符合登记。其他包保持numpy2.4.6、scipy1.17.1、scikit-learn1.9.0、threadpoolctl3.6.0、matplotlib3.11.1，不安装或升级。

获准后只读输入字节哈希已核对：

| 输入 | SHA-256 |
| --- | --- |
| 既有 fit_raw.pt | `d1edbe1f87977ce0a91639bc8cb455708d4c2e17d53340c48d11d2e8b2a568b1` |
| 既有 inner_assignment.json | `e603ba0b603bc903f82649dd19edaebb69f7e171be0dec2092d69048c5469927` |

尚未反序列化输入、拟合温度/PCA或开展神经优化。ACLineStatus快照为1；这不是全程接电证据，既有执行监督继续核对。独占新目录 `outputs/training/ic_role_mechanism_v2/development_fold_0` 仍不存在，本次不继承v1失败目录/许可。

## 3. 一次入口与最终验收监督

工作目录为既有 theory-aware-report 工作树，解释器为主项目 `.venv-training`。先将真实授权JSON及本记录 exact bytes 登记GitHub并回读核对，然后仅调用冻结 `scripts/run_ic_role_mechanism.py` 的 `run_stage(root, protocol_path, output, 'fit-mechanism')`。入口再次核对远端协议、34源码、环境和输入，按固定三个内折准备、108轨迹和临时分析执行，不使用其他真实训练入口。

从本次执行器入口起累计7200秒，包含远端检查、输入核对、准备、训练、梯度/Jacobian/核诊断、临时分析/导出及最终公开验收。旧合成估算仍属于旧源码，不重计时、不增预算或略去诊断。状态发布修复和限定静态复查不保证此次真实实验成功。

只有阶段MECHANISM_COMPLETE、108/3计数、全部身份及父/子释放证据完整，才做一次 `verify_mechanism_delivery(root, output)`：复用现有 `run_supervised_job` 将它放入独立子进程，时限为7200秒减去从入口起全部已用时间。父级用 `WindowsAwakeGuard(api=NativeEnvironmentAPI())` 持续接电/连续性监督并退出释放；最终子进程不另建电源请求。只加载同一fit包及保存预处理/权重并前向核验，不优化、不重新拟合PCA/T0或追加轨迹。

将阶段、最终推理、父级电源、全流程耗时及结果写入同一独占目录内的 `final_acceptance/` 与 `whole_experiment.json`。临时 analysis/evidence.json 的 delivery_verified=false 不等于最终接受。剩余时间不足、阶段或最终验收失败、监督/释放证据不足或全流程超限，整体失败；证据原样保留，不重试、不续跑、不重复运行验收。

## 4. 科学与交付边界

范围固定为已暴露340fit分组、12x3x3=108轨迹、最多3次训练折T0、每轨迹400预算及100固定主点；完整交付需全部既定轨迹完成。主比较C-S-R/C-S-U，其余既定对照全部保留。不得选择较好seed/检查点、扩大网格、增加温度、全fit重训或CNN重训。无一致角色增量如实保留负结果并停止晋级，不为收益追加实验。

不读stop、原图、组合缓存、新分组或独立测试，不改网络/损失/矩阵/预算/种子/公式，不改正式报告，不把探索结果称为工业分选或品位预测。不永久改电源方案，不重做计时探查或空闲验收。

主项目/工作树唯一正式报告只读SHA仍为 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`，六张用户既有PNG改动不暂存、不覆盖。按项目原有可复现归档要求，仅同步授权和轻量状态/日志/指标/如实结果，不上传原始张量、逐状态权重或无关数据。Task8及完整报告/论文目标不能在授权阶段标记完成。
