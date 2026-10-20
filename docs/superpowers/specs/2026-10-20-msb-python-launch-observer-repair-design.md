# MSB实际解释器身份与观察器诊断修复规格

日期：2026-10-20。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档授权：`call_15418f42581f439da850f0e8cfa0c004`，答复“批准仅编写修复规格与实施计划（推荐）”。
基线：`73bf05bb9668aa0ead6e9244be6d9334a3d5df34`；依据[离线诊断](../../experiment_records/2026-10-20_msb_native_failure_offline_diagnosis.md)及其不确定性，不把Windows venv解释写成已证实的本机根因。
本轮直接授权同时起草规格和计划，两份均为待审核文件，不代表认可尚未存在的实现。无代码、项目导入、测试、探查、代理或新协议运行。

## 1. 目标、范围与方案比较

目标：消除“启动返回的进程必然等于执行Python代码的进程”这一未经约束的假设，保留严格HELLO身份与Job归属；同时区分观察回调异常和线程真正未结束，保留原始错误。只恢复可信的工程验证条件，不增加科学结论。

| 方案 | 判断 | 理由 |
|---|---|---|
| A：直接启动已核对的实际CPython解释器，同时传递准确venv上下文 | 推荐，待本规格审核 | 保持被拥有/被Job跟踪/发送HELLO的是同一进程，改动集中；运行时不符合支持配置就拒绝 |
| B：保留redirector，认证launcher到实际解释器的后代链 | 不采用 | 涉及继承、后代创建时序、Job成员及新身份委托合同，不是本次最小修复 |
| C：接受HELLO自报PID、去掉比较、增大超时 | 禁止 | 会放宽安全门，且不能解决观察器诊断缺失 |

A仅参照[CPython v3.11.9的venv启动处理](https://github.com/python/cpython/blob/v3.11.9/Lib/multiprocessing/popen_spawn_win32.py)；不能只替换为基础解释器而丢失venv。当前不宣称本机版本或该方案原生可用。

未来实现白名单严格为六文件，不新增通用启动框架：
- `scripts/msb_execution_safety.py`：小型启动描述/验证helper及Windows适配；原HELLO、Job、期限合同保留。
- `scripts/run_lc_role_adaptation.py`：仅对明确opt-in的native controller job接入共同helper；普通LC job路径不变。
- `scripts/check_msb_native_supervision.py`：两层接线、运行时上下文证据、停止结果/错误保留及新native身份命名空间。
- `tests/test_msb_execution_safety.py`、`tests/test_check_msb_native_supervision.py`：限定反例与已有fake更新。
- `tests/run_msb_native_supervision_synthetic_acceptance.py`：六文件快照、精确保护豁免、新输出根及必要全fake注入。

旧43执行源码中仅前两项执行适配源码可改，其余41项不变；网络、损失、公式、矩阵和正式报告全不变。原十测试/验收器仅`test_msb_execution_safety.py`有本轮功能变更，其他九项不改。三个原native文件均可在上述边界内修改。
不改科学runner的模型/训练逻辑，也不重构LC、IC、MSB全体系。若六文件不足，先停下记录，不自动扩界。

## 2. 实际解释器启动合同

### 2.1 支持配置与运行时身份

仅支持标准Windows CPython 3.11 venv配置：`os.name == "nt"`、`sys.implementation.name == "cpython"`、`sys.version_info[:2] == (3,11)`，`sys.prefix`匹配下一版原生协议明确登记的现有`.venv-training`根，且不同于`sys.base_prefix`。
这是拟定的受支持配置，不是本轮本机检测结果。不同版本、嵌入式/冻结程序、无法确认的别名/符号链接关系均失败关闭，不探测替代解释器，不搜索PATH/注册表、不安装、不切换环境。

未来用当前进程的Python元数据和OS模块路径形成`PythonRuntimeIdentity`：
`implementation, version, venv_executable, base_executable, module_executable, venv_prefix, base_prefix`。
版本保存完整数值三元组；`base_executable`来自`sys._base_executable`但必须存在且与OS查询的当前实际模块路径一致。不得仅凭Python私有属性或“CPython 3.11”注释认定真实映像。
路径要求绝对路径、确定性Windows大小写/分隔符规范化；不通过文件名相同或后缀相同接受不同路径。不能确定实际路径等价就拒绝，不添加通用别名兼容层。

`NativeMSBAPI.current_python_runtime() -> PythonRuntimeIdentity`负责原生采集；其中`GetModuleFileNameW`、`QueryFullProcessImageNameW`等Win32签名/长度/失败返回按官方合同明确配置。合成阶段完全fake，不实际调用。
限定一次32768个WCHAR缓冲，不动态重探：GetModuleFileNameW以NULL模块句柄查询当前进程，DWORD返回0或大于等于容量即拒绝；QueryFullProcessImageNameW用进程句柄、flags=0、DWORD长度指针，BOOL为0或返回长度越界即拒绝。句柄需有查询权限；不能把DWORD长度返回误作BOOL。依据[GetModuleFileNameW](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-getmodulefilenamew)与[QueryFullProcessImageNameW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-queryfullprocessimagenamew)官方合同，访问日期2026-10-20；本轮只是查阅文档，未调用API。
`build_python_launch(command: list[str], runtime: PythonRuntimeIdentity, parent_env: dict[str,str]) -> PythonLaunch`为纯结构化转换，不读配置/二进制/训练数据、不创建进程。`PythonLaunch`含`argv, executable, env, runtime_digest`，digest不包含PID或秘密。
command为参数数组，首项严格匹配当前venv executable，脚本来源仍过原注册root/source门；拒绝shell字符串、空命令或其他解释器，不把“任意命令”规范化为Python。

### 2.2 参数、环境与身份核验

未来Popen同时使用绝对实际解释器作为`executable`及argv[0]，其余脚本/参数逐项原样保留，不经shell。
只在子进程env副本中设置`__PYVENV_LAUNCHER__=runtime.venv_executable`及固定、非秘密的预期上下文元数据；移除子env中的`PYTHONHOME/PYTHONPATH`以免外来路径覆盖。保留既有线程限制及UTF8/无缓冲设置，不修改父进程环境或磁盘venv。
上下文元数据是功能校验材料，不是授权能力；不得替代nonce、继承管道、父句柄、创建时间及注册协议。

原生collector在未来一次获批入口内部采集并记录实际runtime；该准备时间计入480秒，不另开“环境探查”任务。下一原生协议登记支持配置和venv根，实际runtime摘要由入口记录；不把运行时记录冒充预先登记的解释器二进制哈希或包环境完整性证明。
controller及worker在任何WORKING、ready或模型/数据路径之前核对自身runtime与父侧预期：实际映像、venv/base prefix、sys.executable及版本均须匹配；任一不匹配拒绝，不修补环境后重试。
worker核对接入`open_worker_scope`前置门，不能修改HELLO字段集合或通过env独立运行。假的runtime描述不能授予工作权限，原通道身份仍必须通过。

`verify_python_process(process, launch, *, api) -> dict`在启动后核对Popen PID、进程句柄PID、创建时间及实际进程映像。创建时间使用整数或十进制字符串无损保存，不经IEEE-754数值中转。
启动成功至核验失败仍由原ownership/finally拥有该进程并清理；不能在校验中抛错而丢失句柄或新增失管进程。无重试、后代搜索、额外kill策略或放宽Job约束。
启动核对不证明所有包版本或硬件状态可复现；后续科学协议仍须单独登记环境与源码。

### 2.3 两层接线

- collector到controller：仅该job标记`python_launch_contract="DIRECT_CPYTHON311_VENV_V1"`，LC的`run_supervised_job`用共同helper构建Popen并记录真实PID/创建时间/映像；普通未标记LC job保留原路径，不默默重写既有任务。
- controller到worker：`NativeMSBAPI.start_worker`对原固定Python worker命令使用同一helper，保留STARTUPINFO的精确handle_list、close_fds、CREATE_NO_WINDOW、原日志独占及WorkerLaunchOwnership。
- 外层PID等于controller自身PID、内层Popen PID等于HELLO worker_pid仍是硬条件；不能把不同PID当alias接纳。父创建时间、消息/packet/protocol摘要、nonce、租约及结束握手都不放宽。
- 该MSB底层适配也供未来科学worker使用；现有科学协议因源码身份改变不可直接复用。生产真实运行许可不由本修复提供，也不顺手修改科学runner。

## 3. 观察线程停止及错误合同

### 3.1 类型与状态分离

将内存watch保存的`repr(error)`改为实际`BaseException`对象，并保留无局部变量的堆栈位置资料。序列化仅输出type、受限message及文件/函数/行号，不输出nonce、继承句柄、完整env或局部变量。
停止接口改为`stop_fault(watch, *, remaining_seconds: float) -> FaultStopResult`，字段固定：
`thread_terminated, join_attempted, join_timed_out, deadline_exhausted, callback_error`；
其中callback_error为内部原始对象或None，对外诊断另标`callback_error_unknown`（线程未结束时不能把尚未观察到的异常当最终无异常）。

`stop_ok`只在thread_terminated为真、callback_error为None、deadline_exhausted为假时为真。
join等待不超过`min(1.0, remaining_seconds)`；无预算则不join、不称为join timeout，记录deadline_exhausted并失败。检查前后仍受原累计期限和连续性门约束，不能用结束诊断扩出第二份预算。

| 结果 | 必须记录 | 接受 |
|---|---|---|
| 线程结束、回调正常、期限内 | terminated=true、timeout=false、callback_error=null | 可继续原有最终采样/验收，不能直接PASS |
| 线程结束、回调异常 | terminated=true、timeout=false、原始异常 | 失败；不能叫“线程未停止” |
| join后线程仍活着 | terminated=false、timeout=true | 失败；回调最终状态未知 |
| 预算不足未join | attempted=false、deadline_exhausted=true | 失败/证据不足，不伪装已清理 |

同一个watch只做一次正式停止采样并缓存该不可变结果，后续close/异常回收复用它，不重复join累加等待、不把稍后退出改写为首轮停止成功。缓存只记录本轮实际采样，不为未开始的watch补成功记录。
`close`只有确认线程结束后才能关闭它可能仍使用的query handle；否则保留未确认清理字段，继续失败关闭，不能unsafe close或宣称零泄漏。线程已结束但回调有异常时仍安全关闭自有句柄，同时保留异常，不能清空后变成功。
controller的故障观察watch同样消费类型化结果；旧`fault_monitor_stopped`兼容字段由stop_ok计算，不能直接bool(result)。

### 3.2 原始错误、失败诊断及发布

引入仅用于本驱动的`ObserverTerminationError`，保留`stop_result`、有序原始`primary_errors`及独立cleanup_errors；不能从字符串重建异常对象。
在collector首次捕获观察器失败时，把同一实际snapshot序列化为根state的`observer_diagnostics`，包含停止字段、callback/cleanup错误及阶段，不要求额外旁路文件或重试写入。原inner/controller结果原件照常保留，不把观察器错误当内层唯一原因。
首发观察器原因、回收错误、guard退出错误和最终持久化错误分别保存；现有body/release/postflight errors语义及AcceptancePublicationError的原始对象保留合同不退化。
最终发布仍复用现有有限发布器。序列化、清理或guard释放失败均不能输出PASS；如果失败收据也写不出，保留未交付状态和原始异常，不反复写、重入guard或启动下一case。
未取得worker_ready时不能编造WORKING/ACK；即使线程正常停止，该case仍证据不足，不能因诊断修好而接纳失败的身份链。

## 4. 身份与权限迁移

保留旧v1协议、目录、失败状态、原件和Git提交。本修复未来保留`execution_contract="msb_parent_bound_v2"`的消息合同，原生驱动命名空间迁至`msb_native_supervision_v2`及新协议路径`docs/experiment_protocols/msb_native_supervision_v2.json`；拟用新原生目录`outputs/msb_native_supervision_v2/acceptance_once`。
未来FIXED/路径校验/合成夹具同步新名字及DIRECT_CPYTHON311_VENV_V1字段；本轮不创建任何实际协议或该输出目录。旧v1授权和输出不得复用。
SOURCE_FILES仍为现有46项，因未来三执行源码及两个native测试/验收文件变动，全部重新计算身份；额外记录未在46项中的`tests/test_msb_execution_safety.py`摘要。不能照抄旧235项快照或旧46哈希称为新代码验收。
将已知非运行授权ID纳入拒绝集：本轮`call_15418f42581f439da850f0e8cfa0c004`、诊断`call_0ed0a8dae29d4d6aaa949f2752ff89ce`和此前已消费native许可`call_3f51cdd2cc0040929106824e0c335950`不得触发v2运行；未来实际开发/复查许可也不是原生许可。

## 5. 有限未来合成验收

仅在新开发与合成授权后，使用新根`outputs/msb_python_launch_observer_repair_synthetic_v1`及scratch `.superpowers/sdd/2026-10-20-msb-python-launch-observer-repair/test-tmp`。本轮不创建二者。
固定七阶段名称不变：contract-red/green、worker-red/green、collector-red/green、final；各最多3次独占attempt，失败和夹具错误计数，目录递增、原件保留、耗尽即停。
为使每个RED覆盖所属边界，contract及worker阶段仅运行`test_msb_execution_safety`与`test_check_msb_native_supervision`两模块；collector仍仅driver测试；final仍原十模块。不加入LC整模块、不unittest discover。
保留原235个完整测试名；新增固定12方法如下，子例不限为“新测试项数”冒充。最终预计247方法，以真实日志核对，当前不宣称通过。

| ID | 所在文件 | 固定方法名及断言 |
|---|---|---|
| T1 | test_msb_execution_safety.py | test_direct_python_launch_preserves_venv_context：fake venv/base路径不同；argv及executable为base、__PYVENV_LAUNCHER__为venv、父env不变、清除子PYTHONHOME/PYTHONPATH |
| T2 | 同上 | test_direct_python_launch_rejects_unsupported_runtime：错误实现/版本/prefix/module映像/命令首项/空描述，均在Popen前拒绝 |
| T3 | 同上 | test_direct_worker_process_identity_and_wrong_hello：原spawn模拟wrapper 11680/实际23980拒绝；新direct模拟二者相等成功；错误nonce/PID/父创建时间仍拒绝，无工作许可 |
| T4 | 同上 | test_direct_launch_post_spawn_failure_keeps_owned_cleanup：启动后映像查询或句柄PID核验失败，保留同一process owner，原清理一次，无第二spawn |
| T5 | test_check_msb_native_supervision.py | test_direct_controller_launch_uses_shared_helper_and_actual_identity：调用真实LC函数但Popen/时钟/guard全fake；opt-in用同helper且outer PID=self PID；未标记job保留原路径 |
| T6 | 同上 | test_child_runtime_context_mismatch_never_reaches_ready_or_work：两层子入口缺失/错误预期上下文，不产生ready/WORKING，不读数据，无后备解释器 |
| T7 | 同上 | test_stopped_fault_with_callback_error_is_not_join_timeout：实际NativeServices.start/stop逻辑，Thread/Event/join全fake；终止线程+原始ValueError保留、stop_ok=false、timeout=false |
| T8 | 同上 | test_live_fault_after_join_is_timeout_and_closes_nothing：fake线程join后存活，stop_ok=false，不关可能被使用句柄，callback最终状态未知 |
| T9 | 同上 | test_fault_stop_is_once_and_bounded_by_remaining：fake剩余0/0.25/2秒分别不join/0.25/1秒，多次close不再join，不重置累计起点 |
| T10 | 同上 | test_observer_callback_cleanup_release_and_publication_causes_survive：实际失败收集/发布路径，注入不同对象；callback、cleanup、release及持久化原因保留，实际main非零、不能PASS |
| T11 | 同上 | test_missing_ready_and_old_permission_never_pass_native_v2：缺ready不编造通过；旧v1/已消费许可和两个文档ID均拒绝，输出不存在 |
| T12 | 同上 | test_synthetic_scope_covers_runtime_and_thread_adapters：误调用新路径采集/进程映像/Thread.start/join/时钟/网络均被守卫并判失败；全fake正例零未模拟调用 |

真实OS进程、实际Thread.start/join、路径映像查询、时间/电源/网络全fake；既有数值bootstrap保持。不要执行`tests/test_run_lc_role_adaptation.py`的整模块，其中有真实子进程用例，不能用“synthetic”名字视为安全。
快照在每次attempt运行前直接捕获六文件。对旧保护身份只允许六文件白名单中的确切变动，其余仍逐字节核对；保存旧哈希与新哈希，不改旧基线原件、不按通配符移除整个scripts/tests保护。
严格接收为全部旧名称+12方法齐全、0失败/错误/跳过/预期失败/意外成功/未模拟受限调用，来源匹配且无额外attempt。只保留235作历史，不把它预填为247新通过。

## 6. 后续门与研究边界

当前只交付待审规格/计划、作者文档记录及保护证据并归档GitHub，不修改上述六文件、不创建测试根、不导入项目或实际查询运行时。
后续先另批六文件原生实施及有限全fake验收；独立工程复查、v2原生协议登记与单次原生验收、真实fit-only机制实验分别另批，不自动串行执行。
原480秒、N1--N4参数、30秒gap/2秒连续性和7200秒科学预算不变；不重做键鼠空闲验收、不扩大训练/网格、不读stop或新图。
工程修复不是论文创新。本次只解决执行可信度，MSB-RRE视觉证据、间隔预算和强对照收益仍未验证，不写入正式报告作为已实现的科学改进。
