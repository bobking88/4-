# MSB解释器身份与观察器诊断修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 当前仅起草；不实施、不运行、不派代理。

**Goal:** 在不放宽身份/期限/失败关闭合同的条件下，修复两层Python启动身份假设与观察器错误信息丢失。
**Architecture:** 共同的纯启动描述helper配合严格Windows runtime适配；LC仅opt-in，MSB worker使用同一转换。观察watch保留原始异常，停止结果类型化并缓存单次正式采样，复用现有失败发布器。
**Tech Stack:** 既有Python、dataclass、ctypes Windows API、unittest/mock及现有全fake验收器；不安装依赖。
**Spec:** [待审修复规格](../specs/2026-10-20-msb-python-launch-observer-repair-design.md)。
**Baseline:** `73bf05bb9668aa0ead6e9244be6d9334a3d5df34`。仅文档授权`call_15418f42581f439da850f0e8cfa0c004`，状态`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。

## Global Constraints

- 当前只写文档、JSON保护和GitHub归档；以下所有步骤未开始，命令只供未来审核后执行。
- 实施白名单六文件：三个脚本`msb_execution_safety.py/run_lc_role_adaptation.py/check_msb_native_supervision.py`及两个测试`test_msb_execution_safety.py/test_check_msb_native_supervision.py`、原native synthetic harness。均位于原scripts/tests目录。
- 不修改网络、损失、公式、科学实验矩阵、正式报告、其他41项原执行源码、九项原测试或全部历史原件/六张脏PNG。六文件不足即停，不自动扩界。
- CPython3.11/Windows标准venv窄配置；不支持就拒绝，不换解释器探查，不弱化PID/创建时间、nonce、继承句柄、Job或消息校验。
- 输出新根`outputs/msb_python_launch_observer_repair_synthetic_v1`；scratch `.superpowers/sdd/2026-10-20-msb-python-launch-observer-repair/test-tmp`；本轮不创建。
- 七阶段各最多3次独占attempt含失败，原件递增保留；contract/worker两模块，collector一个，final原十模块；不加LC整模块或discover。
- 全部进程、线程启动/join、时钟、电源、路径映像和网络接口fake；原数值bootstrap及严格零跳过/零未模拟门不放宽。
- 保留旧235全名，新增固定12方法，预计247；实际计数以新日志为准。快照记录六文件的真实运行前字节，豁免仅精确白名单，不改旧snapshot。
- 原生v2仅预留名字，不创建/登记/运行实际协议；旧v1授权和目录不复用。480秒/N1--N4/30秒gap/2秒漂移/7200秒生产预算不变。
- 建议未来由本会话原生实施。即使以后批准实施，也不自动派代理、复查或进入原生/科学实验。

## Review Focus

1. venv与base解释器不同且存在PYTHONHOME/PYTHONPATH污染时，子环境准确、父env不改，错误版本/映像在spawn前拒绝：T1/T2。
2. spawn已返回而进程映像核验抛错，不能失去ownership；wrapper差异与错误HELLO均仍拒绝：T3/T4。
3. 外层LC必须opt-in，不能只修内层或误跑LC真实进程用例；两个子入口上下文不匹配不工作：T5/T6。
4. 线程已经退出但回调抛错，与join后仍活着必须不同；诊断不增加累计等待、不unsafe close：T7/T8/T9。
5. callback/cleanup/guard释放/最终写入四种原因及旧授权边界必须同时保留，守卫本身漏接线不能报全fake：T10/T11/T12。

---

### Task 1: 共同直接解释器启动及两层接线

**Files:** 修改三个脚本及`tests/test_msb_execution_safety.py`、`tests/test_check_msb_native_supervision.py`；harness仅由Task3作必要支持。
**Consumes:** 原WorkerLaunchOwnership、NativeMSBAPI、run_supervised_job、NativeServices及原消息/packet绑定。
**Produces:** `PythonRuntimeIdentity`、`PythonLaunch`、`build_python_launch(command: list[str], runtime: PythonRuntimeIdentity, parent_env: dict[str,str]) -> PythonLaunch`、`verify_python_process(process, launch, *, api) -> dict`，都定义在msb_execution_safety.py；`NativeMSBAPI.current_python_runtime() -> PythonRuntimeIdentity`负责适配。

- [ ] **Step 1: 固定快照与T1--T4、T11/T12反例。** 按规格表准确方法名分别写入safety及driver测试；T11/T12也须在首次contract-red前存在，不能在最终阶段无RED地追加。复用FakeAPI，增加可注入runtime/模块路径/进程映像方法。T3的旧wrapper和错误HELLO反例要证明仍拒绝，新direct正例证明同一进程；T4在process已被owner记录后抛出映像/句柄核验异常，断言一次清理且spawn=1。先准备Task3的六文件快照/全fake守卫，不改启动行为。
- [ ] **Step 2: 运行contract-red。** 仅未来新开发授权后用文末固定入口；预期新增helper/启动契约断言失败，旧安全断言保留。导入、夹具或守卫错误占attempt但不是有效功能RED；每阶段最多3次，耗尽即停。
- [ ] **Step 3: 实施纯转换及Windows适配。** helper拒绝不支持配置/命令，构造实际executable+原参数+子env；在NativeMSBAPI.start_worker内部及明确opt-in的LC分支复用。配置GetModuleFileNameW/QueryFullProcessImageNameW签名、固定32768-WCHAR缓冲及越界拒绝，原生调用留给future授权，全fake验收不运行它们。同步driver FIXED/协议路径及合成夹具的v2命名与launch_contract，追加规格的非运行ID拒绝，使T11/T12在contract-green前闭合。保留原ownership、句柄继承和Job指定对象，核验失败走原清理。
- [ ] **Step 4: 运行contract-green。** T1--T4、T11/T12及既有同阶段方法零非通过项；未标记LC行为由后续T5确认。不得为通过测试接纳不匹配HELLO或去掉runtime检查。
- [ ] **Step 5: 两层接线T5/T6及worker-red/green。** driver测试写T5/T6后先worker-red，随后controller job opt-in、子入口runtime核对及root运行时证据接线，再worker-green。T5只在全fake内调用真实LC函数，显式注入Popen/monotonic/sleep/guard，不加载其整个测试模块；T6在ready/WORKING之前拒绝上下文错误，不注入成功结果。旧FakeAPI补合法合成runtime，不以simulated标志绕过实际验证逻辑。

Task1完成标志仅为共享启动契约合成验证，不是确认本机venv机制或原生启动成功；未来真实环境不支持则仍应失败关闭。

### Task 2: 类型化停止结果与失败原因保留

**Files:** `scripts/check_msb_native_supervision.py`、`tests/test_check_msb_native_supervision.py`。
**Consumes:** 原NativeServices watch、NativeObserver、run_acceptance、AcceptancePublicationError及剩余累计预算。
**Produces:** `FaultStopResult`、`ObserverTerminationError`；`stop_fault(watch, *, remaining_seconds: float) -> FaultStopResult`；原state增加observer_diagnostics，不增加旁路重试发布器。

- [ ] **Step 1: 写T7--T10反例。** 使用实际start_fault/stop_fault/finish代码，Thread/Event/join及clock全fake。保留不同原始异常对象，T7确认已终止+callback错误不是timeout；T8确认活线程不关可能被使用handle；T9以0/0.25/2秒检查不join/0.25/1秒及重复close不二次join；T10叠加callback、cleanup、guard release和失败持久化，依次断言is对象与main非零。
- [ ] **Step 2: 运行collector-red。** 新反例应在停止语义/错误对象断言失败；不是用真实线程重现。保留旧235日志与全部本轮失败原件，不手改为红灯通过。
- [ ] **Step 3: 实施最小类型化与错误接收。** watch保存原始异常，单次stop正式结果缓存；finish/close及controller fault watch明确检查stop_ok，不bool(object)。只有线程确认终止才关query handle；callback错误仍保留。run_acceptance在现有失败路径把snapshot写入state，复用原发布器与body/release/postflight对象序列，不挖掘字符串重建异常，不反复join或发布。
- [ ] **Step 4: 运行collector-green。** 正常线程结果只允许继续原验收，缺ready/握手失败不能PASS；回调异常与线程未结束可区分、均失败，T10中持久化失败仍保留原始cause及未交付状态。期限和连续性异常优先保持失败，不让诊断延长预算。
- [ ] **Step 5: 静态核对接口使用点。** 源码所有start/stop/finish/close使用点均消费同一类型，FakeDriverServices及既有负例同步类型而不改原测试名；新增快照字段不暴露nonce/句柄/env/局部变量。此步骤不执行原生API。

### Task 3: 身份迁移、严格合成总验收与作者归档

**Files:** driver、其测试、synthetic harness；不新建实际原生协议。
**Consumes:** Task1/2接口、原46项SOURCE_FILES、旧235完整名、七阶段/attempt合同。
**Produces:** 新合成证据、实际源码身份、作者接收记录；原生v2待未来独立登记。

- [ ] **Step 1: 给Task1提前提供有限harness改动。** 首次RED前完成新输出/scratch两根、六文件运行前快照、原保护身份的精确六路径豁免及全部新受限接口守卫。contract/worker仅固定两模块，collector一个，final原十模块。保留原严格接收门、bootstrap、零跳过要求；不将整个scripts/tests从保护移除。六文件之外仍按旧身份核对；白名单的旧哈希亦保留供审计。
- [ ] **Step 2: 静态确认v2接线及T11/T12保留。** Task1已完成相关RED/GREEN；此处核对driver FIXED/协议路径/夹具的v2名字与launch_contract一致，旧v1/消费许可/文档ID仍拒绝。SOURCE_FILES集合仍46项，额外归档safety测试摘要；T11/T12不能在后续改动中被移除。当前项目根下不得产生v2协议、v2原生目录或任何native授权。
- [ ] **Step 3: 运行final一次起始attempt，最多三次上限。** 严格检查原235全名保留、12方法齐全，预计247项、0失败/错误/跳过/预期失败/意外成功/未模拟调用；不只看wasSuccessful。不重新运行未授权阶段补证据，不改raw日志。
- [ ] **Step 4: 核对证据与保护身份。** 六文件直接pre-run快照、新46项源码身份+额外测试摘要、7阶段尝试计数、红绿有效差异、正式报告/六张脏PNG/旧失败原件均匹配。runtime路径/版本样本全部标合成，不能声称检测了本机；少证据即作者不接收。
- [ ] **Step 5: 显式白名单提交并独立读回。** 仅代码六文件、真实合成日志/summary/快照、作者记录和README归档现有分支；不暂存无关变化、不force推送、不改旧原件。作者合成接收不等于独立复查/原生通过，后续分别申请，不自动派代理。

## 未来唯一合成入口

以下仅供未来获批后执行，本轮不运行。始终使用已存在的venv入口，不安装或切换解释器；合成阶段不会调用新原生采集函数。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' '.\tests\run_msb_native_supervision_synthetic_acceptance.py' --phase contract-red
```

后续同一入口只按上述步骤把phase依次替换为contract-green、worker-red、worker-green、collector-red、collector-green、final。每个phase最多3次含失败；不使用其他入口、LC整模块、discover或额外原生验证。实现若需要超出此边界，先记录并等待新批准。

## 自审与交接

本计划为三项交付、15个未开始步骤；Task3 Step1是首次RED前的有限harness支持，不是额外运行阶段。推荐实施顺序：Task3 Step1，Task1，Task2，Task3 Step2--5。
T1--T12逐一覆盖规格，Review Focus映射见各Task。类型、字段、预算、六文件边界与目录一致；当前全部`- [ ]`保持未开始。
本轮已明确授权同时写规格/计划，因此一并交付待审；不是实现批准。请先审核书面内容，后续若批准由当前会话原生实施，仅限六文件及全fake有限验收；独立复查、实际原生验收和科学实验仍各自另批。
本计划不增加MSB-RRE数学创新，也不提供模型有效性结论。工程问题闭合后，研究重点仍应回到既定理论命题与强对照实验，不能以更多执行器步骤替代科学验证。
