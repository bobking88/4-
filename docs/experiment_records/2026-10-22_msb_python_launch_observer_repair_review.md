# MSB解释器身份与观察器修复：独立只读工程复查

日期：2026-10-22（Asia/Shanghai）。
人类授权：`call_9831ed712c9348a19b4a71960443c76d`，答复“批准一次只读复查并归档（推荐）”。
受审提交：`16fb16d6c9293e9c37c700d5e5f28e081993ac67`，树`37549ab790022385df073285f70fbe8815c85569`，父提交`7d42505c300dd19258b72b505b917771833d62fb`。
结论：**REQUEST_CHANGES**；状态：**INDEPENDENT_REVIEW_REQUEST_CHANGES_NO_FIX**。

## 1. 首要发现与作者核对

**R1 / P2：异常回收分支遗漏观察器停止诊断及原始原因。**

[stop_observer](../../scripts/check_msb_native_supervision.py#L557)直接返回[NativeObserver.close](../../scripts/check_msb_native_supervision.py#L937)的布尔值；[run_acceptance异常回收](../../scripts/check_msb_native_supervision.py#L1059)忽略该返回值。若此前已有回调异常A，check或run_outer再抛出B，close返回False后只重抛B，未保存observer_diagnostics；若关闭句柄另抛C，组合也只包含B/C，没有并入缓存停止快照中的A或期限错误。

这是静态反例，**没有执行或通过实验复现**。该路径仍失败关闭，问题不是把失败变成PASS，而是丢失调试/清理证据。[现有T10](../../tests/test_check_msb_native_supervision.py#L1391)覆盖finish_observer正常进入后的组合失败，没有覆盖此前的异常回收入口。

作者只读核对557、937--978、1050--1077行，认可该P2；状态为`ACCEPTED_OPEN_NOT_FIXED`，不降级、不豁免、不修改源码。复查者建议异常回收也消费缓存停止快照并保留主错误、callback、stop-budget和close原因，且不能再次join；这只是发现中的修复建议，**不是已批准的修复规格或实现**。

## 2. 单次复查与范围

只派发一次代理：`01a161d1-dbab-7290-af17-f1054dc67418`（Leibniz），从独立上下文读取固定提交。未创建其他复查者或下级代理，完成后关闭该代理。
第一次等待接口发生观察超时、随后host generation失效；未据此判定代理结束或重派。仅继续等待同一代理，最终收到完整REQUEST_CHANGES结论。

[授权](../../outputs/msb_python_launch_observer_repair_independent_review_v1/authorization.json)、[完整派发要求](../../outputs/msb_python_launch_observer_repair_independent_review_v1/review_request.md)、[原样返回全文及作者处理](../../outputs/msb_python_launch_observer_repair_independent_review_v1/reviewer_result.json)一并保存。未通过改写审查意见制造APPROVE。

复查者报告实际检查：
- 六文件完整差异、启动/ownership/HELLO/停止/发布调用链、T1--T12正文及harness；
- 冻结规格/计划、作者实施记录及私有ledger；其他注册闭包源码只核对身份，不评模型设计；
- 九份既有summary/log、九份snapshot绑定、46项源码及额外safety测试摘要、27份原件和205条保护记录。

五项Review Focus中，直接启动环境与错误runtime拒绝、启动后ownership、外层显式opt-in及子入口前置门静态满足所审路径；正常停止链具备类型化、单次缓存和有界join。但T10/异常回收的原始原因保留不完整，因此不能用其他项满足抵消R1。

## 3. 既有合成证据与本轮保护

复查者独立比较既有日志，确认final/001的247项为原235个完整名称加指定12名，无失败、错误、跳过、预期失败或意外成功；七阶段实际9次attempt，每阶段不超过3次。
两次非预期失败原件保留：worker-green/001的15个NameError，collector-green/001的6个断言失败及7次被守卫拦截的monotonic调用。没有重跑，也没有把嵌套fake-root/skip反例算成额外根attempt或final跳过。

复查确认源码/snapshot/原始SHA-256、规范化LF身份与Git blob匹配，205条原实施保护为196不变、8条批准代码变更记录（含重复路径）、1条README追加。那是**上一轮实施相对其基线**的数量，不是本轮又改了8项代码。
本轮[前保护](../../outputs/msb_python_launch_observer_repair_independent_review_v1/protection_before.json)固定当前205条历史状态和受审提交41文件；[后保护](../../outputs/msb_python_launch_observer_repair_independent_review_v1/protection_after.json)只允许README新增复查状态，其他受保护内容保持原样。[归档核对](../../outputs/msb_python_launch_observer_repair_independent_review_v1/review_check.json)记录实际静态检查，不能当新测试日志。

本轮没有源码、网络、损失、公式、实验矩阵或正式报告变更，没有项目导入、测试/前向/拟合/计时/原生/进程/空闲验收，没有读取真实图片、权重、缓存、prepared/fit/stop或张量。两份报告和六张既有脏PNG只核对哈希。原native v1失败原件与已消费许可保留，无实际v2协议或输出目录。

## 4. 复查未作结论的边界及处理

| 未作结论的范围 | 原因与处理 |
|---|---|
| Windows实际venv/base互操作、WinAPI和进程归属 | 本轮禁止原生运行；仍未验证，不宣称根因或兼容性已证明 |
| 真实线程调度、join、锁和句柄清理 | 现有静态/fake证据不能证明原生收敛，后续须另批 |
| 任意异常字符串的完整脱敏 | 当前有限规则/测试无普遍保证，不增加“完全脱敏”宣称 |
| 模型效果、数学创新、真实科学可复现性 | 授权范围之外，创新仍未证明，不写入正式报告为阳性结果 |
| GitHub远端状态 | 复查者只核对本地Git；本轮归档由主会话另行上传并独立回读，不归功于代理 |

归档后处理的首次JSON解析受到Git换行警告前缀影响；之后只重新读取哈希/差异，将stdout与stderr分离并在后保护中保留警告，没有重跑测试或复查。此事不计为新的合成attempt，也不涉及源码修复。

## 5. 交接

本轮仅归档上述发现和结论；**R1未修复，未放行原生或科学实验**。下一步建议先另批仅编写该单项的最小修复规格与实施计划，随后开发/有限合成、独立复查和实际实验仍按人类批准的边界分别进行。
不把247项既有工程通过或本轮静态审查计为MSB-RRE理论、网络或论文效果的验证。论文目标仍为既定角色非线性视觉证据与判别间隔预算相对强对照的科学检验。
