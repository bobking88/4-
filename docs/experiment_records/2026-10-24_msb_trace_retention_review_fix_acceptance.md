# MSB Trace Retention Review Fix：作者实施与有限合成验收

日期：2026-10-24。状态：**AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING**。

## 1. 授权与范围

人类在 `call_d010267bd19d42f0b34c89e9822bd6db` 批准按提交 `0c3fe0172e41eb1ce66e07f1101af903e1dc8596` 的[规格](../superpowers/specs/2026-10-24-msb-trace-retention-review-fix-design.md)与[计划](../superpowers/plans/2026-10-24-msb-trace-retention-review-fix.md)原生会话内实施 Task1。仅 driver、driver 测试与原验收器三文件改变；未派代理、未启动独立复查或原生/真实科学实验。

独立复查的 R1/P1、R2/P2、R3/P2 原记录保留，不把历史 REQUEST_CHANGES 改写为 APPROVE。这里的结论是作者代码及合成验收，不是独立审查结果。

## 2. 实际变化

- R1：固定能力标识或十种分行字符触发整消息替换；普通单行继续数值脱敏/截断。保留原异常对象，只导出安全记录。不宣称可识别任意未标识秘密。
- R2：_bounded_fault_stop 接可选调用方 errors 列表，空列表身份保留；停止自身抛错不能再丢掉已捕获预算原因。ControllerTracePublicationError 保有停止、预算、返回结果和已缓存 callback/snapshot 原引用。缓存独立标为 diagnostic-only，停止异常时绝不能据缓存判成功。无新 join/query/guard/时钟，非 trace 路径原异常优先级不变。
- R3：实际 _read_controller_failure 先读；坏/部分回执为非空失败哨兵。有失败时专用 diagnostic_only case record 能先进入 collector 汇总，再由原第一条规则拒收；正常 N4 快照/前缀检查不放宽。
- 联合覆盖：row 已返回，scope 退出、N4 stop/callback、final trace 和部分 receipt 后续错误仍分角色保留。实际 _write_json 函数体配内存文件流，确实写出前缀后再抛错；不是打开前即失败的替身。

通用 _write_json、NativeServices.stop_fault、N4FaultGate、底层监督、网络/损失/公式/矩阵和原生协议均未修改。验收器只改 output/scratch 两处路径；两个旧 fixture 的期望输出路径随之调整，不删除/改名旧测试。

## 3. 实际三阶段结果

| 阶段 | 尝试 | 方法数 | 结果 |
| --- | --- | --- | --- |
| collector-red | 001，唯一一次 | 116 | 30 个缺行为断言失败，0 errors/skips/未模拟调用；9 个新增方法失败，旧106和新增兼容方法通过 |
| collector-green | 001，唯一一次 | 116 | 116/116，零失败/错误/跳过/预期失败/意外成功/未模拟调用 |
| final | 001，唯一一次 | 277 | 277/277，同上；十个模块不变 |

RED 的 driver 原始哈希仍为批准基线 `16577cdf...`；RED 后测试与验收器冻结，GREEN/final/当前三源码哈希相同。逐名比较保存日志，旧267完整名全部保留，新增恰好规格 Q01--Q10 的十名，final 277 个唯一全名；不是只比较总数。合成日志、summary、source snapshots 和六份入口 stdout/stderr 均保留，空 stderr 也保留。

Final 保留既有两次 scratch 合成张量读取及十模块契约；这些不是读取真实 prepared/fit/stop、图片、权重或科学模型实验。新增网络训练、科学前向、计时/空闲/原生验收均为零。

## 4. 作者静态核对与裁定

五类 Review Focus 已按 Q01--Q10 核对；R1--R3 的路径、错误对象身份、原一次停止/累计期限、collector 失败关闭及实际 helper 接线有相应断言。作者另读 driver/harness diff；代码符合本次规格，但这不消除独立复查的价值。

两项执行裁定：
1. 人类明确限定三阶段且禁止代理/自动清理，因此不运行 skill 额外 baseline/task-done 测试、不派最终审查者、不移除工作目录。代价是当前没有独立复查，不能自动放行实际运行。
2. 保留原两个 harness fixture 全名，仅更新期望输出路径，和原验收器新根一致。代价是这两处断言文本有必要变化；保留旧267方法名不等于全部旧测试逐字不变。

没有新依赖/bootstrap、额外模块或第四源码需求；三阶段各一次，无重试。执行 ledger 是作者记录，不能独立证明所有操作历史。实际原生日志首因、Windows 文件替换可靠性和网络科学收益均未判断。

## 5. 原件与保护

原始合成目录为 `outputs/msb_trace_retention_review_fix_synthetic_v1/`。16 份原件/作者 ledger 快照的 raw SHA256、LF canonical SHA256 和 Git blob SHA1 分开记录；Git 规范化文本换行不等于原件重写。入口原始日志还保留于本次私有工作目录，公开副本字节哈希一致。

383 条保护记录在实施前全部匹配，归档后只允许三源码和 README 索引变化。历史56源码闭包中53不变，旧原生登记46源码中43不变；其余3项正是本轮批准变化。旧14份原生失败原件、新旧合成日志、两处正式报告及六张预存脏 PNG 均不修改。旧原生 v2 源码身份不匹配本次变化且许可已消费，不能复用。

## 6. 科学边界与下一关

本轮没有提高论文中的已证实理论创新水平，只为安全执行后续科学实验补工程前置条件。MSB-RRE 的机制收益、相对强对照的改进及对报告结论的支持仍未验证；不改正式报告，不声称工业分选或品位预测完成。

下一步仅可另批一次独立只读工程复查。新原生登记/执行及真实机制实验仍分别另批；只有科学实验完成后才能决定是否将网络方法与公式的实证结论写入报告/论文。

本记录创建时 GitHub 同步尚待外部内容回读；同步结论以实际 commit/tree/ref 及逐文件内容核验为准，不用本文件的意图代替远端证据。
