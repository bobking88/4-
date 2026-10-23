# MSB-RRE 原生无数据验收 v2：离线原因定位

日期：2026-10-23。性质：作者离线诊断，不是独立复查、修复验收或科学实验。

## 1. 授权与输入边界

本轮人类授权为 `call_a491933e40b140d3becdb5aecd12e7e5`，答复“批准离线定位及结论归档（推荐）”。只读此次 v2 失败原件、已登记源码及必要官方文档；仅新增诊断归档和 README 索引。不导入项目、不修改代码、不运行测试或复现、不做进程/计时/电源/文件共享探查，不读取真实数据，不派代理，不修改正式报告。

- 原件归档基线：`beb3e218b3721b5f26a12c979137618bbebc84e9`。
- 原单次运行登记：`732da9ea2355685649e97b74707c2d066aa9dd2f`。
- 受登记代码实现：`35fed93746ae380578d54dd89183235d67b886e4`；本轮逐项核对 46 个登记源码哈希。
- 运行授权工作包日期为 2026-10-22，原入口日志实际最后写入时间为 2026-10-23 UTC；沿用历史文件名，不倒改原件日期。

## 2. 结论先行

**已经定位直接失败机制和错误记录缺口，尚不能恢复第一次操作系统错误。**

直接失败不是网络训练、HELLO 身份错误，也不是外层 90 秒超时：控制器在 `finally` 中调用最终 trace 发布，固定临时文件 `trace.json.pending` 已存在，`open('x')` 抛出 `FileExistsError`，控制器退出 1。当前必须继续保持 `FAILED_NATIVE_SUPERVISION`，不能放行真实科学实验。

更早为何留下 `.pending`，原日志没有保存创建、写入、关闭、替换各阶段的首发异常和 WinError。Windows 文件共享冲突是合理候选，但不能写成已证实根因；也不能直接断言有两个内部写线程。

## 3. 原件证据

| 已保存证据 | 能证明的事实 | 不能推断的内容 |
| --- | --- | --- |
| `cases/N1/controller.log` | 栈指向 driver:483 → :648 → :267，最终独占创建 `.pending` 失败 | 不是第一次 `replace` 失败的栈；没有首发 errno/winerror |
| `controller/trace.json` 与 `controller/trace.json.pending` | 均为 883 字节，完整 JSON，原始 SHA-256 相同 | 字节相同不等于两个不同写线程，也不证明替换曾成功 |
| 五个 trace 事件 | spawn、assigned、HELLO、GRANT、ACK 已保存 | 不证明完成 45 秒工作、两次续租、正常关闭 |
| 根 `stage_state.json` | 累计约 33.547 秒，N1 外层约 12.969 秒，退出 1；N2--N4 未启动 | `outer timeout or wrong controller exit` 是合并校验消息，不能据此断言超时 |
| 查询观察记录 | worker 退出已观察，查询句柄已关闭，观察线程停止已记录 | 缺失内层 MSB 和 parent power 回执，不能宣称全部内部句柄/guard 均已证明释放 |
| 子目录 `stage_state.json` | 最后保存为 RUNNING | 不能把残留状态文字当成仍有活进程；本轮未另做进程查询 |

两个 trace 文件的原始 SHA-256 均为 `0a9db7a39b56d7c5f16d26fd6f36ee78449ce101461d93bdbae024309c1979da`。原始 14 文件、空日志和 `.pending` 均保留；本轮不删除、覆盖、补写或重新格式化。

## 4. 静态调用链及错误可见性

1. driver 的 `_write_json`（259--270 行）对可更新文件使用固定 `<name>.pending`：独占创建、序列化、关闭、`replace`；函数没有局部错误阶段记录或失败清理。
2. `TraceAPI.pause`（438--440 行）在每次父监督暂停前经 `NativeServices.persist_trace` 写同一个 trace。
3. `MSBParentSupervisor.run_job`（safety:620--656）捕获 `BaseException`，保存为内存中的 `row.status/row.error`，执行清理并返回记录。**这不是说本次第一错误内容已知，而是源码中确有这条转换路径。**
4. driver 的 `run_controller`（479--496 行）在保存 `controller_result.json` 前，先在 `finally` 无条件再次发布 trace。若前次失败留下临时文件，最终发布会再次独占创建而失败。
5. 该最终发布失败后，后面的内层结果和 parent evidence 不会落盘。因此首发错误即使已被转换为 `row.error`，也可能随本地变量一起丢失。当前只有第二次 `FileExistsError`，无法从现有原件恢复第一次错误的 WinError。

这里不能笼统说“Python finally 永远删除了原始异常”。Python 对仍在传播的异常保留上下文；本项目的特殊缺口是更早的异常可能已经被内层捕获并转成返回值，然后这个返回值又未能发布。[Python finally 语义](https://docs.python.org/3.11/reference/compound_stmts.html#finally-clause)。

## 5. 假设分级

| 候选解释 | 判断 | 理由及缺口 |
| --- | --- | --- |
| 前一次创建/写入后，关闭或替换失败留下 `.pending`，最终发布再碰撞 | 较强静态推断，有前提 | 在登记单写者及无外部改动前提下符合调用链；缺失首发异常，不能把关闭和替换二者完全区分 |
| 观察器或其他读取者使 Windows 替换发生共享冲突 | 合理但未证实 | 观察器读取同一目的文件；官方说明共享模式可能限制重命名；未保存冲突句柄、首发 PermissionError 或 WinError |
| N1 两个内部 trace 写线程并发争用 | 现有调用图不支持 | N1 不启动 N4 故障线程；父循环和 finally 写在同一控制流，collector 观察器只读，worker 不写该 trace；不排除未记录的外部活动 |
| 复用旧目录、JSON 非法、HELLO 再次失败或外层超时 | 原件不支持 | 新目录独占创建、两份 trace 可解析、已保存握手与身份核验，实际用时小于相应上限 |

Python 的 `x` 模式本来就应在文件存在时失败；改成 `w` 强行覆盖，不是保留证据的修复。[Python open 文档](https://docs.python.org/3.11/library/functions.html#open)。`Path.replace` 提供替换语义，不保证每一次 OS 调用都成功。[pathlib 文档](https://docs.python.org/3.11/library/pathlib.html#pathlib.Path.replace)。

官方 CPython v3.11.0 源码中，Windows replace 经 `MoveFileExW(..., MOVEFILE_REPLACE_EXISTING)`，错误会返回 Python；这是版本化机制参考，不是对本机已安装二进制的反编译验证。[CPython 源码](https://raw.githubusercontent.com/python/cpython/v3.11.0/Modules/posixmodule.c)、[Microsoft MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)。Windows 已打开句柄的共享选项会持续生效，rename/delete 相关访问需要相容共享设置；这只支持上述候选机制，不证明此次实际命中了它。[Microsoft CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)。

## 6. 现有合成覆盖的明确局限

静态阅读 `tests/test_check_msb_native_supervision.py:186--206` 可见，`FakeDriverServices.persist_trace` 直接用 `Path.write_text` 写 trace，没有走原生 `_write_json` 的固定 `.pending` 和替换链。因此，原有 253 项通过记录不能证明该实际链的创建/写入/关闭/替换异常与最终发布交互均已覆盖。本轮不重跑、不改变旧日志，也不撤销旧测试对其已模拟边界的意义。

## 7. 最小修复建议，尚未授权实施

1. **先保全首发原因。** 区分内层 row/body/guard 结果与最终 trace 发布失败，单独保留错误阶段、异常类型及 errno/winerror。保留已有监督清理路径，日志失败不能跳过清理，也不能变成通过。
2. **定义 trace 发布所有权与失败策略。** 第一发布失败后停止继续发布该 trace，保留其 `.pending` 证据，不在 finally 对同一路径再次制造碰撞；明确读取端和发布端的 Windows 共享契约。只改成唯一临时名不能解决目的文件共享冲突；盲删 `.pending`、吞掉错误或无限重试都不可取。
3. **补最小反例覆盖。** 若另批开发，使用全模拟的 create/write/close/replace 失败和最终发布双错误情形，经实际 helper 检查首发错误保留、原件不覆盖、失败关闭及原累计期限；不要只测试直写替身。本轮未编写或运行这些测试。

这三点是供下一次审批的修复建议，不是已批准规格、实施计划或新运行许可。不扩大到网络结构、损失函数、实验矩阵、数据集或科学结论。

## 8. 本轮检查与研究状态

归档保护采用可追溯组合基线：既有 303 条保护记录（将 README 更新为失败归档后的真实哈希）加该失败归档的 21 条文件记录，共 324 条，重复记录保留。写入诊断正文前 324 条匹配，46 项登记源码匹配，14 原始文件及其公开日志副本匹配；最终归档后检查见同目录 `protection_after.json`，仅允许 README 索引变化。

此次仅做文本阅读、原件 JSON 解析与哈希核对，没有测试、复现、项目导入、前向、拟合、进程探查或原生运行。一次过长的只读哈希核对命令因沙箱启动长度限制未启动，改用已有清单引用完成核对；它不是原生验收或实验重试。

**MSB-RRE 的理论/网络效果仍未验证。** 工程故障不构成网络无效证据，修复建议也不构成创新有效证据。旧 v2 运行许可和输出目录已经消费。合理的下一步仅是另行审批最小修复规格与实施计划；任何代码、合成验收、新原生验收或真实机制实验仍分别另批。

机器可读摘要、授权与文献证据见 `outputs/msb_native_supervision_v2_offline_diagnosis_v1/`。
