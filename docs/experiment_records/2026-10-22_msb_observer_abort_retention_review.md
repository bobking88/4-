# MSB观察器异常回收：独立只读工程复查

日期：2026-10-22（Asia/Shanghai）。
人类授权：`call_b7c247344b84483a92f3cdaa4490086b`，答复“批准一次只读复查并归档（推荐）”。
受审提交：`35fed93746ae380578d54dd89183235d67b886e4`；树`cf7acbc07811872f8f724755aabaa51bd0df31f3`；父提交`c6ed89bf2189d948d68fede72cf87a0ddd6de19c`。

结论：**APPROVE，仅限工程修复；此前R1/P2为ADDRESSED。**
状态：`INDEPENDENT_REVIEW_APPROVED_ENGINEERING_ONLY`。

## 1. 发现与结论

独立复查未发现本次修复引入的待整改P0--P3问题，没有形成新的可操作静态反例。此前“提前异常回收遗漏缓存callback/期限原因与停止诊断”的R1在所审内部调用路径中已处理。

作者核对修复差异与只读保护状态，接受该限定结论。没有新增整改项，因此本轮不修源码、不补测试、不扩大工程范围，也不将前次REQUEST_CHANGES历史改写为当时已通过。

| Review Focus | 独立结论及依据 |
|---|---|
| A1：check/run_outer提前异常 | 同一分支只停止一次，读取缓存、保存组合诊断再抛出 |
| A2/A3：A/D缓存与C叠加 | 原B/A/D对象按角色顺序保留；C独立为cleanup；已有快照不再次停止 |
| A3/A4：活线程/期限/缺证据/健康close | 状态区分明确；不编造缺失线程字段；健康停止不抵消原B |
| A5：F/P及发布/CLI | 原列表沿发布链保留；F追加、P独立；失败不变成PASS |
| A6：身份去重/纯函数 | primary按is去重，同文不同对象保留，cleanup不跨角色合并；无采样、时钟、close或join副作用 |

这是静态工程复查，不是本轮重新执行反例，更不是网络、理论或原生有效性结论。

## 2. 单次复查与原件

只派发一次独立代理：`01a163e8-1e78-7703-9e84-334b0aab0ae1`（Mendel），使用独立上下文。两次等待达到观察超时后继续等待同一代理，没有重派；最终完整返回APPROVE，随后关闭已完成代理。未派发下级代理或第二位复查者。

完整[派发要求](../../outputs/msb_observer_abort_retention_independent_review_v1/review_request.md)、[授权](../../outputs/msb_observer_abort_retention_independent_review_v1/authorization.json)和[原样返回全文](../../outputs/msb_observer_abort_retention_independent_review_v1/reviewer_result.json)均保留。reviewer_result同时记录作者处理与边界，不删改原意见或将限制转换成整改/放行授权。

复查者报告检查了：三文件完整差异、六个新增测试及夹具、完整harness；observer/stop/发布/CLI调用链及必要WindowsAwakeGuard、run_supervised_job文本；冻结规格/计划、上次发现、作者实施记录、私有ledger和README差异；三阶段全部summary/log/snapshot、旧247 final三份原件及本轮280条保护基线。

## 3. 既有合成证据与保护

复查者独立解析既有日志：RED为92个方法、15个预期断言/子例失败；GREEN为92/92；final为253/253，完整名精确等于旧247加规格固定六名，无重复、遗漏或额外方法。三阶段均只有attempt-001，GREEN/final各接收门为零，summary未模拟守卫调用为零。

这些数字均来自已归档原件，不是本轮新测试结果。嵌套fake-root拒绝测试的skipped/FAILED摘要不算额外真实attempt；RED日志行尾空格保持原字节。

历史源码保护为53项加六份直接快照、重叠一致后56路径，仅精确三文件变化；SOURCE_FILES原46项字面量未变。GREEN/final直接快照匹配受审文件。作者实施的262条保护记录为252不变、10条获准变化记录（含重复路径），只涉及原三代码文件与README，不能解释成这次复查又改了10项代码。

本轮[前保护](../../outputs/msb_observer_abort_retention_independent_review_v1/protection_before.json)固定当前262条历史状态和受审提交18文件，共280条记录；复查者核对280/280匹配。[后保护](../../outputs/msb_observer_abort_retention_independent_review_v1/protection_after.json)只允许README追加本次复查状态，其他保护原件不变。[归档检查](../../outputs/msb_observer_abort_retention_independent_review_v1/review_check.json)记录作者的只读身份/哈希检查，不作为新测试日志。

主目录及工作树两份正式DOCX和六张既有脏PNG仅核对哈希，保持不变；原生v1失败原件及旧授权不复用。没有实际v2原生协议或输出目录创建。本轮没有真实数据、权重、张量、缓存、prepared/fit/stop或报告正文读取，没有项目导入、前向、拟合、测试、计时/进程/电源/空闲或原生验收。

## 4. 覆盖边界及未作结论的范围

复查者注明一个非阻断覆盖边界：A5中持久化失败子例直接检查run_acceptance的原始异常对象；此时CLI失败由静态异常传播确认。实际main返回1的测试覆盖的是失败收据成功保存的子例，不把它写成两个持久化路径都重新运行了CLI。本轮保留此限制，不扩展测试。

| Declined to judge | 作者处理 |
|---|---|
| Windows原生解释器互操作、线程调度和清理收敛 | 仍未验证，后续原生验收另批；静态APPROVE不代表本机可运行 |
| 任意异常的普遍脱敏 | 有限规则无普遍保证，不增加“完全脱敏”宣称 |
| 底层从未交付的期限原因D | 不声称已恢复；本修复只汇总实际可取得缓存 |
| 网络、数学理论及科学有效性 | 尚未验证，不增加论文或报告阳性结论 |
| 归档后的GitHub状态 | 由主会话单独上传并回读，不归功于独立代理 |

## 5. 后续门与科学目标

本轮只完成独立只读复查与归档，不放行原生或真实机制实验。建议下一步另批**仅编写与登记新的原生无数据监督验收协议**，绑定本次已复查源码及独立输出目录；固定N1--N4、原480秒累计接收上限、接电和失败即停要求不变。实际原生验收、随后MSB-RRE真实机制强对照仍分别另批，不续跑旧失败目录，不把文档或复查许可当运行许可。

工程修复及复查只为恢复科学验证前提。论文主目标仍是检验角色非线性视觉证据与判别间隔预算是否在强对照下产生实际收益；本轮没有新的网络改进效果或理论成立证据，也没有修改正式技术报告。
