# MSB-RRE v3 trace 替换失败的离线原因定位

日期：2026-10-27。授权：`call_eae1cfb62b064c1b95398ac87368af2d`。基线提交：`06bc2fdc52069a6928d953143db8f1c169a1d538`；受审执行源码仍为 4312036。

**结论：已确认失败操作、传播链和一个具体的文件读写并发风险；尚不能确认触发本次 WinError 5 的唯一操作系统原因。** 当前是只读诊断结论，不是代码修复、原生复测通过或网络科学效果验证。

## 1. 实际失败链

已归档 N1 的 controller_failure.json 记录 REPLACE / PermissionError / errno=13 / winerror=5，根终态 FAILED_NATIVE_SUPERVISION，工具退出 1。collector 仅运行一次，N2–N4 未启动。记录的 attempts=2 是两次 trace 发布，不是两次验收。

源码链如下，行号以固定 4312036 源码为准：

1. `scripts/msb_execution_safety.py:619` 在等待 HELLO/后续消息的轮询中调用 pause(.05)。
2. `scripts/check_msb_native_supervision.py:560` 的 TraceAPI.pause 先持久化整份 events，再等待；即使 events 没变化也再次写入。
3. `scripts/check_msb_native_supervision.py:302` 序列化、独占创建 .pending、写入、关闭，最后在 317 行替换目标；本次记录的是替换失败，不是创建失败。
4. `TracePublicationState` 在 327–349 行锁存首发错误。外层 MSB 在 620–656 行转换错误并清理，controller 在 611 行起保留诊断回执；最终不再正常重发 trace。
5. collector 在 1189 行优先读取失败回执，1281 行虽加入 case_records，但后续验收拒收。因此“有一条 case record”不代表 N1 通过。

trace.json 与 .pending 的原始字节 SHA-256 均为 `e83afb10a7abd2c5961a95bb32b27af2dab416c80072ccc0f1965cd936263743`，内容仅 spawn 和 assigned。与“一次成功后，相同事件快照再次替换失败”一致。没有 HELLO/ACK/关闭证据，不应推断完整监督已经建立。`lease_sequence=1` 在 control 尚未建立时也可能是默认返回值（msb_execution_safety.py:654），不能作为已经授租的证据。

写入方源码在替换前调用 stream.close，且无 CLOSE 异常记录；没有证据支持“忘记关闭自身 .pending 写句柄”。失败后清理可能追加内存事件但不再发布，故 trace 中缺少清理事件也不能单独证明未执行清理。

## 2. 已确认的读写风险，不等于已确认根因

observer 在 controller 启动前建立（driver:1262），循环中每约 0.05 秒 sample（1127）。sample 在 1100–1102 行依次读取 worker_ready、controller_ready、trace；到 1105 行才判断 ready 是否缺失。**因此 worker_ready 尚不存在时也会打开已存在的 trace 文件。**

读取通过 `_read_json_if_present`（864）调用普通 Path.read_text；写入方反复替换同一个目标路径。observer 的 threading.RLock 只保护自身状态，不跨进程协调 controller 文件替换。代码没有为 trace 读取明确建立允许删除/重命名共享的 Windows 句柄契约。

这使“观察器持有目标文件的读取句柄时，controller 替换失败”成为**优先排查假设 H1**。但原件不包含失败瞬间的读句柄、共享掩码、重叠时间和外部打开者信息，所以不能将 H1 写成已经复现实证的根因。

## 3. 官方依据与适用限度

Windows 将删除共享单独定义为 FILE_SHARE_DELETE，删除访问同时涉及重命名；普通读写共享不能直接替代这个约定。[CreateFileW 官方文档](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)

目标替换还受 ACL 等条件约束，因此第一次创建/发布成功不能证明以后替换必然成功。[MoveFileExW 官方文档](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)

WinError 5 是访问被拒绝，不能据此指定唯一锁持有者，更不能直接写成 WinError 32 的共享冲突。[Windows 错误码](https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-)

失败记录报告 CPython 3.11.5。该版本 Path.replace 调用 os.replace，Windows 分支通过 MoveFileExW 和 MOVEFILE_REPLACE_EXISTING 实现；Path.read_text 则是普通打开、读取、关闭。[pathlib 源码](https://raw.githubusercontent.com/python/cpython/v3.11.5/Lib/pathlib.py)，[posixmodule 源码](https://raw.githubusercontent.com/python/cpython/v3.11.5/Modules/posixmodule.c)

默认 Windows FileIO 走 _wopen；此处没有显式 FILE_SHARE_DELETE 设置。这些源码与接口说明支持检查共享语义，但**不构成本机失败瞬间实际共享掩码的观测**。[FileIO 源码](https://github.com/python/cpython/blob/v3.11.5/Modules/_io/fileio.c)，[CRT 共享选项](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/sopen-wsopen?view=msvc-170)

“成功时替换具有原子性”和“替换在并发读取下必定成功”是两件事，不能用前者证明后者。[os.replace 文档](https://docs.python.org/3.11/library/os.html#os.replace)

## 4. 假设与未获证据

| 项目 | 当前结论 | 尚缺什么 |
|---|---|---|
| H1：observer 读取与替换冲突 | 有明确代码路径、时间窗口与平台规则支持；优先假设 | 实际共享掩码及重叠证据，或未来单变量验证 |
| H2：ACL/文件属性/外部过滤器或第三方打开者 | 未排除，不可归因于某个软件 | 失败时权限、属性、句柄或过滤器证据 |
| 自身 pending 未关闭 | 本次无该证据，源码已关闭后再替换 | 不能据现有记录主张它是原因 |
| 命名空间混用/复用旧目录 | 协议与当前独占输出绑定正确，原件来自新 v3 | 不能把该错误重新解释为旧 v2 目录复用 |
| 第二次完整运行 | 没有发生 | 两次 trace 发布不能混同两次 collector |

本轮没有读取实时 ACL、文件属性、进程句柄表或杀毒配置，也没有导入项目、重放测试或进程探查。不要求关闭防护、不更改全盘权限、不改电源设置、不延长原期限。

## 5. 合成验收为什么没有提供这项保证

`tests/test_check_msb_native_supervision.py:1729` 的 TraceMemoryIO 是内存字典模型；1754 行的 replace 通过字典赋值/移除完成，或由预设异常直接失败。1808、1827、1842 附近的反例能够验证首因保留、调用次序、失败回执和不再发布，却没有建模“读者持有一个缺少删除共享的句柄，再让另一进程替换目标”。

因此，既有 285 项合成结果只证明当时各自覆盖的断言，**不能据此声称真实 Windows 文件共享语义已被验证**。本轮只读测试源码，没有重跑旧测试，也不把覆盖缺口自动升级为真实根因。

## 6. 最小后续建议，尚未实施

建议下一步先写一个局限于 trace 读取的共享语义适配规格，而不是继续增加异常包装或放宽成功条件。

- 首选候选：明确只读、非继承的 trace 读取适配，允许与目标替换兼容的共享方式，例如 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE；及时关闭句柄，保留原始错误与累计期限。共享方式不是授予删除权限，也不能绕过 ACL，不能作为当前已证实修复。
- 减少无效读取：ready/controller 前置条件未满足时，不打开 trace；这只能减少冲突机会，不能单独证明后续读写安全。
- 备选而非当前直接采用：不可变版本化快照可避免替换正在读取的目标，但会改变发布和读取协议，范围更大，需单独设计。
- 不采用：无限重试、删除失败 .pending、忽略 PermissionError、关闭安全软件、把 controller 自报清理当作独立验收通过，或为通过测试降低 N1–N4 要求。

未来规格至少应要求确定性模拟“读者持有/释放”“删除共享有/无”“目标替换”“读取/关闭双异常”“既定期限耗尽”，并将恒定事件快照、ready 缺失及 N4 读写关系纳入覆盖。这里只列需要验证的性质，不冻结新测试矩阵，不编写或执行测试；开发、真实平台确认和重新运行仍另批。

## 7. 研究边界与归档

此次工作仅定位执行障碍，不能提升或证明 MSB-RRE 的理论创新。科学问题仍须由获批的真实机制强对照回答，不能把工程日志写成精度提升、理论增益或结题成果。网络、公式、损失、实验矩阵和正式报告保持原样。

[结构化诊断](../../outputs/msb_native_supervision_v3_offline_diagnosis_v1/findings.json)、[官方来源](../../outputs/msb_native_supervision_v3_offline_diagnosis_v1/official_sources.json)、[授权](../../outputs/msb_native_supervision_v3_offline_diagnosis_v1/authorization.json)与保护记录随本次归档。530 条保护记录保留重复项，只允许 README 索引变化；旧失败原件和此前原生结果不改写。本轮新增测试/原生验收/科学前向/拟合/代理均为 0。

**下一步权限边界：只读诊断已经完成；最小修复规格、开发及任何新实验未自动获批。**
