# MSB观察器异常回收原因保留：规格与计划登记

日期：2026-10-22（Asia/Shanghai）。
文档授权：`call_a6f6d55427584c8da99bbba75fef5a29`，“批准仅编写修复规格与计划（推荐）”。
基线：`c555dba71e03e0307f59b8de890a882a71b18bac`。
状态：**DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN**。

## 1. 本轮交付与未完成状态

本轮按直接授权同时起草[最小规格](../superpowers/specs/2026-10-22-msb-observer-abort-retention-design.md)及[一个工作包、六步骤计划](../superpowers/plans/2026-10-22-msb-observer-abort-retention.md)，二者都是待人类审核文件，不是已经批准的实现。

依据[独立复查R1/P2](2026-10-22_msb_python_launch_observer_repair_review.md)，当前异常回收仍可能丢失原始callback/stop-budget原因和observer_diagnostics；R1保持`ACCEPTED_OPEN_NOT_FIXED`、已有复查保持REQUEST_CHANGES。
源码没有变化，本轮未用动态实验复现反例，也没有新的通过测试结果。

## 2. 拟定的最小方案

采用原一次stop_observer/close之后的纯汇总函数，保留捕获主错误B、缓存callback A、缓存stop_errors D及独立close C，再沿原失败发布保留guard退出F和持久化P。
只在primary_errors内部按对象身份去重，不按文本去重，不抹掉cleanup角色；缺快照或停止未确认增加明确缺证据标记，而不伪造已结束。
不复用可能继续采样的finish，不改底层stop_fault/close缓存、身份/Job/握手或累计期限，也不通过再次join获取更好的停止证据。

未来只改driver、driver测试和native synthetic harness三个文件。新方法六名，driver预计92、final预计253；保留原247完整名称。只拟定collector-red、collector-green、final三个全fake阶段，每阶段最多3次独占attempt含失败，原件保留。
历史247运行快照中的53条保护与六份直接源码去重后确为56个唯一路径，未来仅三文件可变、其余53不变。这个56是静态去重核对，不是本轮运行数。
新合成输出/scratch及实际native v2协议/目录都未创建。

## 3. 只读文档自审

- 问题链、正常finish与异常回收差异已按现有源码和复查意见定位，没有将静态反例写成运行复现。
- 原因优先级、原始对象、None/False与健康close语义、活句柄和零预算行为在规格中明确；纯helper返回组合对象，由调用方保存并抛出，接口在计划中一致。
- A1--A6覆盖提前check/outer、缓存A/D与C叠加、期限/活线程、缺快照/健康停止、F/P/CLI及身份去重/无副作用；Review Focus逐条映射。
- 计划六个步骤全未开始，三个未来phase、六方法、253预期数及三文件白名单一致；没有TBD/TODO/FIXME。接口代码块中的省略号仅表示尚未实现的签名，不是漏掉的决定。
- 保持原480秒、N1--N4、30秒gap/2秒漂移、7200秒科学预算，不创建新原生授权、不复用已消费许可。
- 本轮直接授权覆盖两份文档一起起草，故不重复请求是否允许写计划；实现仍须在两份书面文件审核后新批。未派代理进行自审。

[文档授权](../../outputs/msb_observer_abort_retention_planning_v1/authorization.json)、[自审与范围核对](../../outputs/msb_observer_abort_retention_planning_v1/planning_check.json)、[前保护](../../outputs/msb_observer_abort_retention_planning_v1/protection_before.json)及[后保护](../../outputs/msb_observer_abort_retention_planning_v1/protection_after.json)同时归档。
保护对象为此前205条历史记录、41个原实施归档文件及8个复查归档文件，共254条记录，包含重复路径；不能声称254个不同文件。只允许README新增本轮文档状态，其余源码、证据、正式报告两处和六张既有脏PNG维持哈希。

## 4. 明确不包含的工作

无代码修改、项目导入、测试/前向/拟合/训练、计时/原生/进程/键鼠空闲验收、真实图像/权重/缓存/prepared/fit/stop或张量读取，无代理，无网络/损失/公式/实验矩阵/正式报告变更。
这不是网络或理论创新的证明，也不撤销R1。下一步建议人类审核两文档后，另批当前会话原生实施及限定合成验收；独立复查、原生登记/验收与真实机制实验仍分别另批。最终科研目标不变：验证既定MSB-RRE角色视觉证据与判别间隔预算相对强对照的收益。
