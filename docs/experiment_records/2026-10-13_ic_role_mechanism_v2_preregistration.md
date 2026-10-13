# IC-RFA-B v1 修复后 v2 执行协议登记记录

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent plan / preregistration。
- Origin Date: 2026-10-13；Version Label: `ic_role_mechanism_v2_preregistration`。
- Verification Status: `SCOPED_STATIC_PROTOCOL_CHECK_PASSED_REAL_UNAUTHORIZED`。
- GitHub 完成条件：承载本记录的提交与远端 `codex/theory-aware-report` exact commit 一致，新协议/设计字节回读一致；提交哈希不在文件内自引用。
- 独立工程复查状态：`PENDING_NO_CONCLUSION`；本记录不是独立审核或真实效果报告。

## 1. 直接人类批准

问题项 `["request_user_input_async","call_b7aaa1e4e1764728b80858311c0bf2a6",0]` 的人类回答是 **“批准新协议登记，暂不运行”**。

问题范围：仅编写并在 GitHub 登记修复后的 IC-RFA-B v1 新执行协议；新源码身份及独立输出目录；网络/损失/矩阵不变，限340fit分组、108条最多400步轨迹、最多3次温度拟合和7200秒；不读真实输入、不训练、不重复计时/空闲验收、不改正式报告；注明独立复查尚未完成，真实运行仍需另行批准。

[v2 中文协议](../experiment_protocols/ic_role_mechanism_v2.md)和[v2 JSON](../experiment_protocols/ic_role_mechanism_v2.json)只落实这一登记授权。执行模型仍为 IC-RFA-B v1，JSON 的 `protocol=ic_rfa_b_v1` 不改；`status=DRAFT_NOT_AUTHORIZED`，`human_approved=false`，`user_message_id=null`、`approval_utc=null`。本次批准身份只在文档记载，不填进真实运行 envelope，不继承旧消费许可。

## 2. 身份与冻结设计

| 身份 | SHA-256 / 提交 |
| --- | --- |
| 修复源码提交 | `da176fc5d247affde7b2ceb766aa5a5727aac658` |
| 修复源码树 | `b2dbba7714ef359c44dd085ac5160a0ca35e43e0` |
| v2 机器协议字节 | `6ba9631420694d96af5df13db2bdc830663aa715776c8a6b265d7a7ea636801a` |
| v2 中文协议字节 | `7ac621aa2883cb7450189e88cda3b2204eda1bb15a589eed5cfe15ae53a47a60` |
| 34 源码 SHA 表规范 JSON | `89aa86065729315e0714491ba9394cf1f5b62814cdc789127d2c647a03859a3a` |
| 108 轨迹规范 JSON | `8ae883853d9b82ac2ee936632a0900a43318a7117c76fe6a1d6c61655a9c1e89` |
| 主项目/工作树唯一正式报告 | `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c` |

相对旧 v1 机器协议仅五个顶层值变化：status、authorization、source_sha256、protocol_path、output_relative。34 文件源码中仅执行器从旧 SHA 更新为 `423fad79bee6d5a351589319cb03b98a8097b958615a9b9e1fd274ef921f302a`；另33文件逐项不变。全部工作树源码字节符合新 SHA 表，Git clean blob 身份符合 da176fc。没有改任何源码或测试。

输入两项 SHA 继承旧规格/协议，本轮未读取其内容或重新哈希。新目录 `outputs/training/ic_role_mechanism_v2/development_fold_0` 静态检查不存在，本次未建立；旧输出和四份失败文件 SHA 仍逐项符合失败归档。旧 v1 JSON SHA 仍为 `f2311d538a8410dca3c525e91235edbaf7f45b214243c6e386d4eccb960fd3aa`。

固定环境、12x3x3矩阵、三个内折、三种子、检查点0/1/2/5/20/50/100/200/400及100主点核对一致。C-S-R/C-S-U首要配对、P1/P2/P3条件、目标新增/损失与Ti/G/M误入的原分母、无自动晋级和负结果停止规则保留。未新增创新/效果结论。

## 3. 本轮静态核验结果

现有 `.venv-training` 中以 `python -X utf8 -B -c` 做只读断言：JSON exact schema、未授权 envelope、冻结常数、34源码SHA/Git blob、纯字典矩阵和规范哈希、环境、新目录边界、旧协议/失败证据及正式报告哈希。不是一次合成训练、原生计时、空闲验收或真实输入验证。

调用 `validate_protocol(root, protocol, 'fit-mechanism')`，预期并实际得到：

```text
Fresh explicit real approval required; drafts and consumed approvals cannot run.
```

该检查将 torch.load、fit加载器、prepare、fit_trajectory、run_stage、worker 替换为调用即失败哨兵；调用计数均为0。以下是本次工具输出的结构化摘要，不冒充操作系统级文件访问审计或完整控制台转录：

```json
{
  "status": "SCOPED_STATIC_PROTOCOL_CHECK_PASSED_REAL_UNAUTHORIZED",
  "source_count": 34,
  "trajectory_count": 108,
  "environment_matches": true,
  "sentinel_calls": {
    "torch_load": 0,
    "fit_loader": 0,
    "prepare": 0,
    "fit_trajectory": 0,
    "run_stage": 0,
    "worker": 0
  },
  "real_input_contents_or_hashes_read": false,
  "new_output_exists": false,
  "old_protocol_and_four_failure_files_unchanged": true,
  "independent_review": "PENDING_NO_CONCLUSION",
  "new_timing_or_idle_acceptance": false
}
```

仅提交两份新协议、本记录、README导航、计划状态和为新协议保留字节的三条 .gitattributes 路径。六张用户既有PNG及其他非本任务文件不暂存。实际源码闭包、网络/损失/矩阵、旧证据和正式DOCX不改。

## 4. 尚未完成的工作

先前一次独立工程复查因额度限制未返回结论，作者自审不替代它。本轮仅明确记录缺口，不重派代理。尚无完整真实准备/108轨迹/最终验收，也没有角色净纠错效果证据。

历史合成预算估算只属于旧源码；修复后未重新计时。未来即使另获准，也要把远端检查、准备、拟合、诊断、临时分析/导出及一次受剩余预算监督的最终推理验收合并计入7200秒。预算/监督或证据不足即失败，不重试、不续跑、不扩网格，不无界调用公开最终验证器。

GitHub预登记完成并不授予运行。后续须处理独立复查事项，真实单次执行仍需单独明确人类批准及重新登记真实授权envelope；如果源码或范围变动则重新书面登记。stop、新图像、未暴露确认、数学新颖性评估和正式报告修改均未获准。Task8仅推进登记，整体保持未完成。
