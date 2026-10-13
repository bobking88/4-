# IC-RFA-B v1 机制协议登记记录

日期：2026-10-13。状态：`PREREGISTRATION_DESIGN_ONLY_REAL_UNAUTHORIZED`。

## 1. 人类授权与交付

用户在问题项 `["request_user_input_async","call_621575a3608b411981c864956e245d93",0]` 回答“批准协议编写与登记（推荐）”。问题明确限制为只写协议并在 GitHub 登记，不开始真实训练。没有继承 LC-RFA 旧授权，也没有把这次回答填成真实运行批准。

- [中文预登记](../experiment_protocols/ic_role_mechanism_v1.md)：研究问题、理论证据、强控制矩阵、指标/分析、时间及停止规则。
- [机器协议](../experiment_protocols/ic_role_mechanism_v1.json)：DRAFT_NOT_AUTHORIZED，human_approved=false，真实授权身份/UTC 为空。
- 执行源码：`b412dfca50a3b26b4371537b44b388587d786fc6`，34 文件完整闭包，网络/损失/网格未改。
- 目标登记分支：`codex/theory-aware-report`。承载本记录及两份协议的 Git 提交即为登记提交；不在文件内自引用尚未生成的提交哈希。远端须与本地 exact commit、协议字节身份相符后才算 GitHub 登记完成。

## 2. 固定身份

| 身份 | SHA-256 |
| --- | --- |
| 机器协议字节 | `1c99a79a0e15c9c7aafec098c2f5ed22a03285ec9bfb0be92b6782e2b0aeddb6` |
| 中文设计字节 | `fe2c14c663c547ff10c2d179aa2d0ead08177192ffcda1d574bd8a1ebdea2526` |
| 源码 SHA 字典规范 JSON | `07fd6a29962aaa439d6ad2ceaf1640e9da02737c83dd4ef84cb7bd220a2de7a6` |
| 108 轨迹规范 JSON | `8ae883853d9b82ac2ee936632a0900a43318a7117c76fe6a1d6c61655a9c1e89` |
| 唯一正式报告 | `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c` |

规范 JSON 为 Python `json.dumps(value, sort_keys=True, separators=(',', ':')).encode()`。源码表是机器协议的 source_sha256，轨迹表由冻结 build_fit_matrix() 产生。未登记任何新效果数据。

## 3. 本次只读核验

在现有训练解释器中解析 JSON、核验固定字段/环境/34 文件哈希，对每源码用 Git clean blob 身份比对 b412dfc；生成纯字典运行矩阵并核对 12 单元 x 3 内折 x 3 种子、400 更新及 100 主点；均通过。

对 validate_protocol(root, protocol, 'fit-mechanism') 的授权门检查返回预期拒绝：

`Fresh explicit real approval required; drafts and consumed approvals cannot run.`

检查时将 torch.load、load_fit_bundle、fit_trajectory、prepare_inner_fold 替换为一旦调用即失败的哨兵，调用数全部为 0。只调用协议校验，不调用 run_stage、worker、真实预处理或优化。哨兵属于该校验的调用边界证据，不冒充操作系统级文件访问审计。

未打开 fit_raw.pt 或 inner_assignment.json 内容；其输入 SHA 继承已认可规格，未来获准运行才先检查字节哈希、再核对内容。未读取 stop/原图/组合缓存、未加载模型权重、未重复计时/空闲验收。真实输出目录 `outputs/training/ic_role_mechanism_v1/development_fold_0` 仍不存在。

主项目和工作树正式 DOCX 只读哈希均保持上述身份。六张用户既有 PNG 改动不覆盖、不暂存。只提交本协议/记录、README 导航和实施计划授权状态，不提交无关图片/临时输出或历史结果。

## 4. 研究与执行限制

首要 C-S-R/C-S-U 比较、100 主点、目标新增/损失、Ti/M 误入及三 seed 方向一致性已事前固定。没有自动选模或自动晋级阈值；无一致角色增量即保留负结果，不扩网格/种子/数据寻找收益。

历史 3767.654 秒合成预算估算来自 f933153，b412dfc 校验加固后未重测；真实新增开销和临时/最终两次核验成本没有被伪装成实测。7200 秒硬限制仍绑定全流程。

现有公开最终验证器不能单独为自身设置剩余时限。协议要求以后复用已有监督器、按从单次入口开始的剩余累计预算完成最终核验；无法提供这一证据就停止、不接受完整交付。本次不开发执行器、不真实运行。

协议登记完成之后，仍须就这一固定协议单独取得一次真实运行批准，登记真实授权 envelope 后方可执行。正式报告修改、独立确认、stop/新图像和完整效果协议不在此范围。Tasks1--7保持合成验收完成；Task8仅推进预登记，整体未完成。
