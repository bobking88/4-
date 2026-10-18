# MSB原生无数据监督验收：规格与计划归档记录

日期：2026-10-18。授权：`call_a42aed1b7bab474bb17246c3df69dc4e`。
状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档基线：`779a4c669e647d38798b35e9df535b4bc06129ce`，当前合成工程APPROVE、原生INSUFFICIENT_EVIDENCE。

## 交付

- [规格](../superpowers/specs/2026-10-18-msb-native-supervision-acceptance-design.md)。
- [实施计划](../superpowers/plans/2026-10-18-msb-native-supervision-acceptance.md)。
- README状态索引。仅此四个文本路径进入本次提交。

用户已同时批准两份文档，故合并交付待审；没有将“规格可写”解释为开发许可。本次按设计/计划技能原生起草、自审，不派代理。

## 读源得到的关键调整

1. RunBudget固定7200000ms；MSB run_job不消费per_fit_seconds，不能假造0.25秒原生MSB超时。
2. 现有synthetic入口有完整80-job schedule，不适合作为本次限定原生验收；选择专用无数据驱动调用现有监督器，未来只新增三个验收源文件，原43项生产闭包不改。
3. N1正常45秒、N2 worker exit23、N3 DONE后迟退35秒、N4 controller exit31四例顺序固定；整体480秒，外层限额90/45/75/45秒，一例一次。
4. 真实WinAPI/Job/能力握手不替换为fake成功；collector只保留精确身份查询句柄，不复制Job或通信能力。父硬退出负例不得先杀worker再报Job清理成功。
5. 没有键鼠空闲门，不主动休眠或更改电源方案；持续接电，正常guard显式释放与强杀路径未显式观测严格区分。
6. 原生通过仅限定四条路径，不承诺精确毫秒边界、两小时实测、任意并发/崩溃或全面零泄漏；真实科学实验仍另行批准。
7. 自审进一步核对_terminal映射：N2的原生管道EOF可能映射FAILED而非缺关闭分支的SUPERVISION_INTERRUPTED，因此只在exit23/ready/无DONE/明确失败路径/清理证据齐全时接收相应负例，不修改生产分类，也不把任意FAILED都当通过。

## 作者静态自审

已逐项核对现有MSB run_job/open_worker_scope/NativeMSBAPI、LC run_supervised_job与WindowsAwakeGuard/NativeEnvironmentAPI接口；没有导入或执行这些模块。
比较完整80-job smoke、专用无数据驱动及重写执行框架三种路径，采用第二种。规格与计划共享准确场景、期限、输出位置与停止条件，不以实现方便缩小将来证据要求。
计划保持任务1--3未来代码/全fake合成、任务4另批协议登记、任务5另批原生运行；全部复选框未勾选，没有预造批准或源码哈希。
本轮没有创建拟议验收脚本/测试、原生协议JSON、合成根或原生输出根，没有测试/前向/拟合/计时/电源/原生进程验收，不读真实prepared/fit/stop、图片、缓存或权重，正式报告不变。
规格、计划的本地链接/固定接口/数值、保护哈希与提交白名单在归档前以文本/JSON/Git静态检查核对；这些检查不是程序或实验验收。
首次静态JSON提取被Git换行警告混入输出而解析失败；随后临时禁用autocrlf的diff-check又将既有CRLF显示为尾随空白。均为只读命令口径问题，没有改仓库配置或规范化文件；最终以仓库默认换行规则下的暂存diff-check和四路径白名单为准，没有运行或重跑项目测试。

## 研究目标与下一步

此前独立复查提供了新的限定接收证据；等待下一批准的自动继续没有新科学进展，不视为批准。当前文档为恢复真实网络机制检验提供明确路径，但仍不是网络创新效果证据，广义目标未完成。
下一步由用户审核两份文档，决定是否另批原生新增验收代码与限定全fake合成阶段Tasks1--3。登记、独立复查、原生运行、真实MSB机制实验不自动启动，也不重复旧idle验收。
