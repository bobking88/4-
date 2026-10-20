# MSB原生N1失败：离线根因定位与最小修复建议

日期：2026-10-20。依据失败归档提交`950d3c9237fe68f10361355c094f93f7f42e249b`及其运行登记`6b32e84322b63a50d2e877590425858e64c4318e`。本轮授权为`call_0ed0a8dae29d4d6aaa949f2752ff89ce`，人类答复“批准离线定位及结论归档（推荐）”。
结论状态：**离线定位完成，未修复，未复现，未取得新的原生或科学实验结论。** 这是作者诊断，不是独立复查。

## 1. 核心判断

1. **已确认的内层直接原因：进程身份契约不匹配。** 父监督器按启动返回的进程PID构造预期HELLO，worker按自身PID构造HELLO；本次记录分别为11680和23980。因此严格相等校验必然拒绝。这不是应当放宽的错误身份检查。
2. **有力但未最终证实的环境解释：Windows venv重定向层。** 官方历史实现能够产生“启动器进程不等于实际解释器进程”的情形，两层PID差异与此一致；现有原件没有本机解释器/启动器的完整身份证据，不能写成唯一根因已经证实。
3. **已确认的观察器诊断缺口：一个布尔值合并两种失败。** `stop_fault`同时要求线程结束及回调无异常；任一不满足，`finish`都报“observer did not stop”。原件未保存`watch.error`，无法判断本次是回调异常、线程仍活着，还是两者兼有。不能直接称为“线程卡死”，也不能断言它由PID错误引起。

## 2. 原件约束与事实

[失败结果记录](2026-10-20_msb_native_supervision_result.md)、[根终态](../../outputs/msb_native_supervision_v1/acceptance_once/stage_state.json)及[实际stdout](../../outputs/msb_native_supervision_native_result_v1/collector_stdout.log)一致：`FAILED_NATIVE_SUPERVISION`、实际退出1、仅进入N1，已接收case记录为0；N2--N4未开始。记录累计35.7809999999954秒，不能解释成480秒预算超时。

| 层级 | 已有原件 | 记录 | 证据边界 |
|---|---|---|---|
| collector启动controller | stdout首条progress | Popen PID=35876 | 该记录不是实际controller自身身份 |
| controller自身 | controller_ready.json | PID=21844，创建时间134369819590203162 | 与外层Popen PID不同 |
| controller启动worker | trace的spawn/assigned | PID=11680，创建时间134369819644636078 | 身份来自启动返回的进程句柄 |
| HELLO内容 | trace的HELLO | worker_pid=23980，parent_pid=21844 | 尚未通过身份认证，不能据此认领实际发送进程或其创建时间 |

[内层trace](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller/trace.json)仅有spawn、assigned、HELLO、terminate_job；[内层结果](../../outputs/msb_native_supervision_v1/acceptance_once/cases/N1/controller/controller_result.json)保留`worker HELLO identity differs`，没有成功GRANT/ACK/RENEW/DONE链。PID差异已经足以解释拒绝，但日志没有保存完整HELLO，不能说“其余所有字段都匹配”。

受跟踪PID清理、guard释放、故障后PID未见残留等旧证据维持原有有限含义；不能升级为真实worker认证、Job归属因果、线程或句柄完全清理证明。本轮没有再次查询进程。局部旧RUNNING文件及两个实际空日志保持原样，不改写为通过或运行中。

## 3. 身份链的源码追踪

以下行号以`950d3c9`中的未改源码为准。

| 步骤 | 源码位置 | 语义 |
|---|---|---|
| 两层命令构造 | [验收驱动](../../scripts/check_msb_native_supervision.py)第379、941行 | controller与worker均通过`sys.executable`启动 |
| 外层进程记录 | [LC执行器](../../scripts/run_lc_role_adaptation.py)第207、219、235行 | 保存并监督`Popen`返回进程，未解析实际解释器后代身份 |
| 内层启动/归属 | [MSB执行器](../../scripts/msb_execution_safety.py)第665、695行 | 返回Popen并把该进程句柄分配给Job |
| 启动身份来源 | [trace适配](../../scripts/check_msb_native_supervision.py)第326行 | 对返回进程句柄调用process_identity，不是随意填写PID |
| 父侧预期身份 | [MSB父侧](../../scripts/msb_execution_safety.py)第500行 | `process.pid`进入binding |
| worker侧身份 | [MSB子侧](../../scripts/msb_execution_safety.py)第377行 | `api.current_pid()`进入binding，随后发送HELLO |
| 拒绝点 | [HELLO比较](../../scripts/msb_execution_safety.py)第512行 | 全消息严格比较，差异即SUPERVISION_INTERRUPTED |
| 尚未到达的外层门 | [case验收](../../scripts/check_msb_native_supervision.py)第651行 | 要求外层Popen PID等于观察的controller PID；本次35876/21844也不兼容 |

因此修复不能只盯着内层worker，忽略collector到controller的一层。科学worker入口[run_msb_role_mechanism.py](../../scripts/run_msb_role_mechanism.py)第241行也采用`sys.executable`，同一启动假设需要纳入未来规格边界；本轮不扩展为科学代码改造或真实执行。

### 官方依据与不确定性

Python 3.11的[venv文档](https://docs.python.org/3.11/library/venv.html)说明Windows虚拟环境使用重定向可执行文件，并说明符号链接配置存在差别。固定版本[CPython v3.11.9 launcher.c](https://github.com/python/cpython/blob/v3.11.9/PC/launcher.c)的VENV_REDIRECT与run_child路径展示了查找原解释器、创建独立进程并等待其退出的机制。它们支持H1，不证明本机二进制采用该路径。

[同版本multiprocessing实现](https://github.com/python/cpython/blob/v3.11.9/Lib/multiprocessing/popen_spawn_win32.py)专门处理Windows venv重定向：基础解释器与虚拟环境上下文成对设置。这说明不能只把命令换成“基础python.exe”而丢掉依赖环境。它是参考，不是对现有受监督Popen/继承句柄/Job契约的直接替换方案。

本轮未读取本机`pyvenv.cfg`、解释器二进制或运行时配置，未调用解释器获取版本、路径或进程树。源码中的“CPython 3.11”注释不等于本机版本鉴定。历史3.11的`PC/venvlauncher.c`链接未取得可用证据，实际采用可读取的`PC/launcher.c`；不把失败网页或main分支实现冒充本机证据。

## 4. 观察线程结束路径

[NativeServices](../../scripts/check_msb_native_supervision.py)第552--564行在线程体捕获异常，把`repr(error)`放进内存`watch.error`。停止时设置Event、join最多1秒，再返回：

```text
stop_ok = (not thread.is_alive()) and (watch.error is None)
```

[NativeObserver.finish](../../scripts/check_msb_native_supervision.py)第867--872行对false结果统一发布停止失败；回调异常没有写入当前终态。由此得到下面的源码逻辑表，**不是本轮运行的线程实验**：

| 线程已结束 | 回调无异常 | stop_fault | 当前报错能否区分 |
|---|---|---|---|
| 是 | 是 | true | 正常停止 |
| 是 | 否 | false | 不能与线程未结束区分 |
| 否 | 是 | false | 不能与回调失败区分 |
| 否 | 否 | false | 仍然是同一泛化错误 |

官方[Thread.join/is_alive文档](https://docs.python.org/3.11/library/threading.html#threading.Thread.join)区分join结束与超时，线程也可能因异常结束。本代码确实调用了is_alive；**不是误用join返回None**。NativeServices第540行及NativeMSBAPI第702行也已有pause；**不是缺少pause适配器**。

本次worker_ready未产生，观察器sample第826行会在缺就绪信息时返回；它不直接读取MSB的HELLO拒绝原因。不能把内层身份失败自动当成观察线程报错的完整因果解释。原始异常不在原件中，离线诊断应在这里承认信息不足，而不是猜测具体异常类型或增加线程超时。

## 5. 为什么既有合成通过仍未覆盖本次情形

只读查看[启动夹具](../../tests/test_msb_execution_safety.py)第256、270--278行：假进程pid=8，合成HELLO也按8构造，预设启动进程就是worker。[假服务](../../tests/test_check_msb_native_supervision.py)第188--197行直接返回stop成功；第815行的负例注入false，验证失败关闭，但不区分“线程已退且回调失败”与“线程仍活着”。

这些局部检查有原先的有限价值，不应撤销既有235项日志；同样不能把它们当原生解释器层级和真实线程诊断的证明。本轮只读源码，不重跑、不重算、不改写这些历史验收。

## 6. 最小修复建议，尚未实施

**R1：先明确受监督进程到底是谁。** 优先设计“实际解释器就是被拥有/被分配Job/被认证的进程”的启动方式，同时保留准确的虚拟环境上下文、继承通道、父身份与创建时间。若必须保留启动器，则需要单独设计可认证的启动器—后代链，而不是直接接受HELLO自报PID。这不是删掉一条比较即可安全完成的修复。两层启动必须同时考虑。

**R2：将停止证据拆开。** 未来记录thread_terminated、join_timed_out、callback_error的类型和原因，并分别保留首发异常及清理异常。线程未停止和回调异常仍均为失败，不通过清空异常、忽略错误、延长超时或改PASS来“修好”。若期限不足以完成清理证据，就保留证据不足结论。

**R3：让未来反例覆盖真实差异。** 至少覆盖两层wrapper/interpreter PID差异、伪造或错误HELLO仍拒绝、线程已退但回调异常、join超时仍存活、缺worker_ready和清理双异常。这里只列验收要求；未编写测试、未运行，也不是新的执行许可。

不建议现在换解释器试跑、按PID追加kill、删除失败目录或恢复旧RUNNING。此次授权与输出已消费；未来任何运行使用新的协议、源码身份和独立目录，继续失败即停，不在失败目录上补跑剩余N2--N4。

## 7. 保护与研究含义

[授权](../../outputs/msb_native_failure_offline_diagnosis_v1/authorization.json)、[结构化判断](../../outputs/msb_native_failure_offline_diagnosis_v1/diagnosis.json)、[官方来源索引](../../outputs/msb_native_failure_offline_diagnosis_v1/official_sources.json)、[前保护](../../outputs/msb_native_failure_offline_diagnosis_v1/protection_before.json)、[后保护](../../outputs/msb_native_failure_offline_diagnosis_v1/protection_after.json)共同归档。

前保护192条记录，包含175条旧保护及950d3c9新增的17项失败归档；不按唯一路径数冒称192个不同文件。只允许README追加本轮状态，其余191条不变；46项登记源码匹配，两处正式报告仅核对哈希、不读正文。

本轮未修改源码/网络/损失/公式/实验矩阵/正式报告，未导入项目、未启动代理，未读真实prepared/fit/stop、原图、权重或缓存，未做测试、前向、拟合、计时/进程探查或原生重验。离线归档和Git核对不算科学验证。

**科学判断仍是：MSB-RRE实际增益未验证。** 当前结论既不能写成理论创新有效，也不能写成网络无效。下一步建议仅单独申请“最小修复规格与实施计划”的授权；开发、合成验证、原生重验和真实科学实验仍分别另批。
