# IC-RFA-B v1 单次真实机制实验授权

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent run。
- Origin Date: 2026-10-13；Version Label: ic_role_mechanism_v1_once。
- Verification Status: `AUTHORIZED_PENDING_EXECUTION`，不是效果验证结果。
- 预登记提交：`1b91d0e83eac2481a7675fc67b394a5787d1b5a7`。
- 网络/执行/分析源码身份仍为 `b412dfca50a3b26b4371537b44b388587d786fc6`，全部 34 文件哈希未改。

## 1. 真实人类批准

问题项 `["request_user_input_async","call_d12ae77d482e46f28498bfaed5810de7",0]` 的直接人类回答为 **“批准一次真实机制实验”**。

问题原文：是否批准按 GitHub 提交 1b91d0e 登记的 IC-RFA-B v1 协议执行一次真实 fit-only 机制实验？仅补运行授权信息并重新登记；限现有340个fit分组、108条每条最多400步轨迹、最多3次温度拟合，准备、诊断、导出及最终推理验收合计7200秒硬上限。要求持续接电；失败或证据不足即停，不重试、不续跑、不扩网格，不读stop/原图、不改网络或正式报告。

授权记入时间 UTC：`2026-10-13T05:27:22Z`；这是本次录入时间，不冒充平台提供的原始消息时间戳。机器协议 user_message_id 保存实际回复问题项身份，不捏造其他消息 ID。

[机器协议](../experiment_protocols/ic_role_mechanism_v1.json)只将 status/authorization 从未授权草稿更新为 APPROVED_REAL_ONCE 与上述真实人类批准；输入、源码、环境、矩阵、预算和输出身份不变。[中文设计](../experiment_protocols/ic_role_mechanism_v1.md)和[预登记记录](2026-10-13_ic_role_mechanism_preregistration.md)保留登记时的历史未授权表述，不能将它们误认为当前授权状态或新实验结果。

## 2. 前置核验

使用现有训练解释器，只读检查全部34源码字节 SHA、锁定环境、fit_raw.pt 与 inner_assignment.json 字节 SHA，均符合预登记。尚未反序列化输入、拟合或评价。目标目录仍不存在，当前 ACLineStatus=1；执行器继续实时核对接电与监督连续性，不把此快照当全程证据。

已有合成验收/审核证据保留；不重复合成计时、空闲验收或训练回放。历史估算未测得修复后的额外校验开销，本轮依然硬截止，不因结果或时间改变矩阵。

## 3. 固定调用与累计监督

工作目录是既有 `theory-aware-report` 工作树；现有 `.venv-training` 解释器。第一阶段仅调用冻结 `scripts/run_ic_role_mechanism.py` 的 run_stage(root, protocol_path, output, 'fit-mechanism')。它核对远端登记字节，按固定顺序准备三个内折、执行108轨迹及临时分析，不调用其他真实训练入口。

从该入口开始累计计时。只有阶段 MECHANISM_COMPLETE、108/3 计数及父/子终态完整时，才对最终公开 verify_mechanism_delivery(root, output) 做**一次**保存状态的推理验收。该函数放在现有 run_supervised_job 的独立子进程中，时限为 7200 秒减去已经消耗的全部时间；父级用现有 NativeEnvironmentAPI/WindowsAwakeGuard 持续接电/连续性监督，结束释放。最终验收只载入同一 fit 包和已保存预处理/权重，不优化、不重拟合、不追加真实轨迹。

最终验收的临时系统防待机由监督父进程承担；工作子进程不额外建立电源请求。训练阶段原有父/子请求全部须释放。剩余时间不足、子进程/监督失败、电源释放证据不足、最终核验失败或总耗时超限，整体实验失败；临时分析不能冒充最终交付，也不重试。

独占目录仍为 `outputs/training/ic_role_mechanism_v1/development_fold_0`。最终推理只增加该目录内的 `final_acceptance` 与 `whole_experiment.json` 等生成证据，不建立新的训练目录或修改任何源文件。逐任务实际消费与执行失败均保留，失败不重新使用当前批准启动完整实验。

## 4. 未授权事项

不读 stop、原始图片、组合缓存或新分组，不改网络/损失/网格/种子/主点，不全fit重训，不自动扩展、续跑或重试。不永久修改电源方案，不写正式技术报告，不把探索指标当工业回收率或品位。只同步协议、代码身份、轻量监督/指标和如实结果说明，不上传原始张量/模型权重。

本次前置检查主项目及工作树的唯一正式报告保持 SHA-256 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`；六张用户既有图改动不暂存、不覆盖。若角色净收益不成立，保留负结果并停止晋级，不追加实验寻求正结果。
