# MSB trace 共享读取：实现与有限合成验收

日期：2026-10-27。状态：**AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING**。本记录是作者实施与证据审计，不是独立工程复查、原生验收或科学效果结论。

## 1. 授权与历史

原 Task1 开发批准为 `call_4490c98e36aa4667a22d7065e677fb99`。首轮无效 RED 的完整原件、未完成草稿及说明已归档 `43ddb021d9cd5e253ac1b9e5b9a117f83e922506`，不覆盖、不删除，也不把该轮改判通过。

用户随后通过 `call_3dd1ecba80c644358636e38b4f36ac60` 明确批准修正草稿并在原范围、剩余额度内继续。本轮执行前另存540条保护记录，包含上次537条（保留重复项）和首轮3份原始输出；不再冒用回溯比对作为本轮前快照。见[继续授权](../../outputs/msb_trace_shared_read_synthetic_v1/continuation_authorization.json)与[前快照](../../outputs/msb_trace_shared_read_synthetic_v1/continuation_protection_before.json)。

依据仍为[原规格](../superpowers/specs/2026-10-27-msb-trace-shared-read-design.md)和[单工作包计划](../superpowers/plans/2026-10-27-msb-trace-shared-read.md)。源码仅限原 driver、driver 测试、原合成验收器三文件。相对43ddb02，本轮只再修改 driver 和测试；验收器沿用首轮草稿中的新路径与读取工厂守卫，没有追加接口、bootstrap例外或阶段。

## 2. 最小实现

- `_bind_trace_reader_kernel` 共用实际 ABI 设置，真实/注入内核均声明 HANDLE 指针宽、DWORD计数和BOOL返回；`_trace_reader_kernel` 延迟创建 DLL，不在模块导入时打开句柄。共用绑定函数是私有实现细节，不增加文件、依赖或原生入口。
- `_TraceReadFileIO` 使用 GENERIC_READ=`0x80000000`、share=`7`、OPEN_EXISTING=`3`、FILE_ATTRIBUTE_NORMAL=`0x80`、NULL security/template。只读、非继承，不申请写或DELETE权限；失败后立即取得原始系统错误。读取上限4096字节/块，正常EOF后只关闭一次；失败关闭不重试或伪装成功。
- `_read_trace_json_if_present` 在打开前、块前后、关闭后和返回前调用同一个继承 `check`。读取/期限失败仍进入关闭；首因与关闭/末端期限异常按发生顺序保留。仅 OPEN WinError2/3作为暂缺，errno=2但没有WinError不被吞掉。
- 先读字节并关闭句柄，再严格UTF-8解码和JSON解析；解析失败也检查末端预算。有效期限内的JSONDecodeError保留原轮询语义；若同时到期，保留原JSON异常对象与CHECK原因，不能被观察器的JSON等待分支吞掉。
- `TraceReadError` 保留原异常对象与 OPEN/READ/CHECK/CLOSE 顺序。`_fault_error_record` 只为该类型增加至多8项的脱敏记录，子项不递归扩展，errno/winerror只接受范围内的真实整数。普通错误记录仍保留原三字段形状。
- `NativeObserver.sample` 缺 worker/controller ready 即返回；先验证已有case/working/job/binding及必要身份/时间字段，再经 `NativeServices.read_trace` 读取当前case的trace。assigned、ACK序号与父PID、spawn身份、活性、5000ms窗口及只查询句柄均保留。

未修改通用 `_read_json_if_present`、trace写入器/首因锁存、N4杀父/故障快照/失败回执优先级、原监督器、协议常量或46/56来源集合。旧v3目录和旧许可仍已消费；没有生成新的可执行协议。

## 3. 固定测试与TDD证据

原124个driver方法名保留，只修正了两处旧验收夹具的输出路径。固定12个新增方法未增删：R01实际适配ABI与参数；R02条件共享模型；R03缺失/访问错误/JSON轮询及联合期限；R04分块EOF/读错/解码/关闭先于解析；R05首因对象及清理顺序；R06七处期限检查及等于期限；R07缺ready零trace读取；R08错误字段/binding前置拒收；R09实际sample→service→helper→adapter接线与ACK拒收；R10实际callback与停止/期限传播及诊断边界；R11实际写入/锁存与N4失败回执先读；R12实际harness内默认工厂被阻断而注入fake可执行。

| 顶层阶段 | 尝试 | 结果 | 解释 |
|---|---|---|---|
| collector-red | 001 | 无效，保留原件 | 先前9个旧夹具error entries、1个R11字段error及R12覆盖不足；不算有效RED |
| collector-red | 002 | 有效RED，136项 | 19个明确assertion failure entries（含subtests），零errors/skips/未模拟调用；旧124项通过，R11通过，必需R01/R07/R09/R10/R12均报缺行为 |
| collector-green | 001 | 136/136通过 | 零failures/errors/skips/expected/unexpected/未模拟调用 |
| final | 001 | 297/297通过 | 原十模块；完整测试名集合严格等于历史285项并集规格固定12项 |

RED002后冻结测试与验收器，之后仅改driver。测试 SHA-256=`7d6e11deb08fedd8bad41ba4ba12bcbbaaaa4ef5864ed6bf45f4eeafdf3f61da`；验收器=`a9217b5b96c7e5cffda0d6180b80adb081f01be4a72137365b53e53072a3e036`。GREEN/final当前56项来源身份一致，driver=`2ecb7085e510c282fe7c31031fdb8a1c2a1406c30df7975b25a5d8009948f000`。见[有效红灯门](../../outputs/msb_trace_shared_read_synthetic_v1/valid_red_gate.json)。

日志中嵌套的“collector-green、1项skip、FAILED”是原拒收门自检在临时ROOT下故意注入的结果，不是顶层失败或额外顶层尝试。依据各顶层独占目录summary判断。本轮未追加RED003、GREEN002或final002；没有额外pytest、单测、进程/计时/共享探查或原生运行。

本轮RED002/GREEN001/final001均保存入口stdout/stderr，并逐字节复制到对应attempt目录；首轮缺失的分离入口日志不事后伪造。每轮完整合成日志、summary和当时三文件source snapshot全部保留。索引见[产物清单](../../outputs/msb_trace_shared_read_synthetic_v1/continuation_artifact_manifest.json)。

## 4. 作者静态审计与保护

对计划五类Review Focus逐项检查：ABI与last-error/关闭语义由R01/R04/R05覆盖；旧快照与新打开共享由R02/R03/R09覆盖；读/关/期限/停止联合异常由R05/R06/R10覆盖；ready/身份/工厂门由R07/R08/R12覆盖；相同事件发布、首因锁存、失败回执先读由R03/R11及历史285项覆盖。未发现需改动冻结测试或扩大规格的事项。此为作者自查，不能替代独立复查。

540条前后记录中528条匹配，12条差异只对应driver和测试两条路径的重复记录；未批准变化为0。主目录与工作树两份正式报告、六张预存脏图、旧失败原件（包含RED001）均相同。此前已有的其他未跟踪训练产物不读取、不暂存或删除。见[后保护及完整方法集合核对](../../outputs/msb_trace_shared_read_synthetic_v1/continuation_protection_after.json)。

原始日志保留行尾空格，未为格式检查改写失败原件。源码/手写记录可单独作格式核对；所有原始合成输出在暂存时命令级关闭autocrlf并核对blob/原始字节，不修改Git配置、gitattributes或历史数据。工作账本为前轮Git Bash helper失败后已建立的同名ignored路径，不宣称该helper本轮运行成功。

## 5. 没有被验证的事项

- R02仅为Windows规则的条件模型，不是原WinError5故障在本机被复现；share=7不绕过ACL、外部持有者或过滤器。
- R01使用fake API验证实际绑定代码和适配器，不证明真实Windows ABI调用已验收；默认工厂被合成守卫阻断，没有真实文件API调用。
- 期限检查不保证能硬实时中断任意阻塞的同步系统调用。原480秒累计门、失败关闭和单次执行契约不变。
- 无原生进程/空闲验收、真实prepared/fit/stop或矿物原图读取、真实拟合和RGB骨干构建；final只有原有两个scratch合成张量读取，未扩大豁免。没有派代理。
- 未修改网络、损失、数学公式、科学矩阵或正式报告；**没有任何新的神经网络理论收益、泛化收益或指标提升证据**。

共享与错误语义依据：[CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)、[ReadFile](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile)、[CloseHandle](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-closehandle)。只将其用作工程接口依据，不据此证明本机故障根因。

## 6. 后续门槛

Task1的代码与有限合成验收现为作者验收通过、独立复查待批。下一步应只读独立复查已归档三文件、固定12项/297项既有日志、原件及期限/清理边界，不重复运行。独立复查、新原生协议登记和单次运行、真实fit-only机制强对照分别仍需批准，不能用本轮合成通过替代。

对研究目标仍应检验MSB-RRE的间隔缩放预算是否允许纠错、是否损害原正确样本、三个角色非线性证据分支能否超出强对照的等价表达。只有经批准的真实固定矩阵与强对照提供证据后，才决定其能否写入报告和第一篇论文的理论/网络创新；本次不预写阳性结论。
