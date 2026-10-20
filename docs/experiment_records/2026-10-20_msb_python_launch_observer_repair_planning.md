# MSB解释器身份与观察器修复：文档起草记录

日期：2026-10-20；状态`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
授权`call_15418f42581f439da850f0e8cfa0c004`：“批准仅编写修复规格与实施计划（推荐）”。
基线`73bf05bb9668aa0ead6e9244be6d9334a3d5df34`，依据[离线诊断](2026-10-20_msb_native_failure_offline_diagnosis.md)。这是作者文档自审，不是独立工程复查。

## 已交付

- [待审规格](../superpowers/specs/2026-10-20-msb-python-launch-observer-repair-design.md)：选择实际解释器直接启动并保留venv上下文，保留身份/Job/失败门；停止结果拆分线程状态、回调异常及预算，保留原始错误。
- [待审计划](../superpowers/plans/2026-10-20-msb-python-launch-observer-repair.md)：三任务15步，未来六文件白名单、固定12新增方法、七阶段有限全fake验收。所有实施步骤未开始。
- [授权](../../outputs/msb_python_launch_observer_repair_planning_v1/authorization.json)、[文档自审索引](../../outputs/msb_python_launch_observer_repair_planning_v1/planning_check.json)、[前保护](../../outputs/msb_python_launch_observer_repair_planning_v1/protection_before.json)、[后保护](../../outputs/msb_python_launch_observer_repair_planning_v1/protection_after.json)。

## 关键边界

Windows venv解释仍是假设；窄配置为拟定支持范围，不冒充本机鉴定。所引用的CPython实现不能当直接可用补丁。未来runtime采集需在另批的单次原生入口内部计入480秒，不在本轮调用。
不采用放宽HELLO、自报PID认领或增加线程超时。普通LC任务不变；新原生v2仅文档预留名字，旧v1目录/授权已消费。
计划明确排除LC整模块中真实spawn用例，原235项名字保留，247仅为未来预计；没有运行或确认新的方法数量/通过率。
错误和清理不确定性仍保留，状态仍未修复、原生未通过、科学效果未验证。当前没有新的运行许可，也不自动替换正式报告。

保护前198条（含历史重复路径记录，不是198个不同文件）；只允许README追加，其他197条保持哈希。46项既有源码仍与旧协议一致，两处正式报告仅核对哈希、不读正文。
本轮无代码编辑、项目导入、测试/前向/拟合、时钟/进程/原生探查、真实数据读取或代理；六张历史脏PNG和全部失败证据不改。

## 交接

书面规格与计划交付后等待审核及另行开发/合成授权；不得直接执行计划中的命令。独立复查、v2原生协议登记和验收、真实科学实验仍另批。
本轮只补执行可信度设计，不把工程文档当理论创新或网络效果证据。
