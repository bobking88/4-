# MSB Trace Shared Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans only after separate human approval. Proposed execution is inline in the current session; no agents are authorized. All unchecked steps below are future work.

**Goal:** 用最小三文件修复明确活跃 trace 的 Windows 只读共享、句柄生命周期和 ready 前置契约，保留原监督门。

**Architecture:** driver 内新增延迟绑定的私有读取适配器和受期限约束的 JSON helper，通过 NativeServices 接入观察器。固定 12 项合成反例覆盖实际 helper/sample；原发布器、监督器和科学网络不变。

**Tech Stack:** 现有 Python 3.11、ctypes Win32 API、unittest/mock 与原十模块合成验收器；无新依赖。

**Spec:** [最小修复规格](../specs/2026-10-27-msb-trace-shared-read-design.md)。当前 `call_2e34a6fa4f8f4207ad5cf4476536803a` 仅批准编写与归档；并未批准本计划执行。

## Global Constraints

- DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN；文档基线 cf2e62e，源码基线 4312036，本轮不开发、不导入项目、不测试或探查、不读真实数据、不派代理、不改正式报告。
- 未来仅 driver、driver 测试、原合成验收器三文件；原 46/56 来源路径集合不变，其余 43/53 源码不改。
- 原 msb_parent_bound_v2、DIRECT_CPYTHON311_VENV_V1、N1–N4、480 秒累计、[90,45,75,45]、接电、单次及失败关闭不变。
- 读权限 GENERIC_READ=0x80000000、share=7、OPEN_EXISTING=3、FILE_ATTRIBUTE_NORMAL=0x80、NULL security/template、不继承；4096 字节分块；原同一累计期限，无新计时或重试。
- 原 trace 写入/首因锁存、普通 JSON helper、N4 失败先读/快照约束不改，无删除 pending、忽略访问错误或 ACL 操作。
- 固定合成根 outputs/msb_trace_shared_read_synthetic_v1，scratch .superpowers/sdd/2026-10-27-msb-trace-shared-read/test-tmp；三阶段各最多 3 次独占尝试，含失败，无自动重试。
- 原 124/285 完整测试名保留；固定新增 R01–R12，未来 136/297 仅目标；所有受限原生接口全模拟，原 bootstrap/十模块/严格通过谓词不放宽。
- 不复用已消费 v3 原生目录或授权，不生成新可执行协议。网络、损失、公式、矩阵、正式报告不改；独立复查、原生和科学实验另批。

## Review Focus

1. HANDLE 截断、ReadFile/CloseHandle BOOL 失败或 last-error 被后调用覆盖：R01/R04/R05。
2. 读取过程中目标替换，旧快照不能变成虚假 ACK，允许删除共享也不能绕过权限：R02/R03/R09。
3. 读取失败叠加关闭/期限/停止失败，首因和脱敏证据必须完整，句柄只关闭一次：R05/R06/R10。
4. ready 缺失与坏 binding 不能打开 trace 或查询进程，新增接口不能引入未守卫原生 I/O：R07/R08/R12。
5. 相同事件再发布、N4 失败回执与 JSON 部分内容的交互，不能恢复正常发布或弱化拒收：R03/R11及原285项。

---

## Task 1: 共享读取最小接线及有限合成验收

**Files:**
- Modify: `scripts/check_msb_native_supervision.py`，当前 _fault_error_record、NativeServices、_read_json_if_present 邻近新增私有 helper 及 NativeObserver.sample；不改通用 helper。
- Test: `tests/test_check_msb_native_supervision.py`，新增 TraceSharedReadTests 的固定 12 方法，旧 fake services 注入及两处 harness 路径预期局部更新。
- Modify: `tests/run_msb_native_supervision_synthetic_acceptance.py`，仅 output/scratch 两路径、新读取工厂守卫及对应诊断名称。

**Interfaces:**
- `_trace_reader_kernel()`：延迟绑定三项 Win32 API；合成 harness 必须 guard 此工厂，默认真实分支不得触达。
- `_TraceReadFileIO(kernel=None)`：open_shared_read(path)、read_chunk(handle, size) -> bytes、close(handle)，注入 fake kernel 才能在合成环境使用。
- `_read_trace_json_if_present(path, *, check, file_io=None)`：只有 OPEN Win2/3 返回 None；字节读取/关闭成功后解码 JSON；其余按规格保留错误。
- `TraceReadError(errors)`：有序 (phase, original_exception)；_fault_error_record 仅对该类型添加有界 trace_read_errors。
- `NativeServices.read_trace(path, binding)`：绑定原 _remaining；sample 在原 ready/binding 检查通过后调用。
- 测试 helper `SharedReadKernelFake` 模拟句柄、共享掩码、旧/新快照和故障；不写代码到仓库真实原生路径，不访问真实 Windows API。

### 步骤，当前均未执行

- [ ] **Step 1: 取得独立开发授权并核对起点。** 审核用户届时批准的文档提交、三文件白名单和空 index；核对原 46/56 来源、旧 285 全名、v1/v2/v3 失败原件、六张脏图和两处正式报告。两个新合成目录必须仍不存在，否则停，不自选后缀。保存开发前保护记录。

- [ ] **Step 2: 写固定 R01–R12 及边界模型。** 名称和全部断言取规格第6节；R02 必须先展示同一模型在缺 DELETE 共享时拒绝替换，再通过实际读取适配的 share=7 允许替换。fake kernel 记录实际 API 参数、关闭次数和操作次序；R05/R10保存并核对原异常对象，不能仅比对拼接消息。不得直接 mock 被测 helper/sample 为成功。

- [ ] **Step 3: 局部适配旧夹具和合成守卫。** 在 RED 前完成旧 fake service 的显式 read_trace 注入及两个新 harness 路径预期；不更名/删除旧方法，不削弱旧断言。harness guard _trace_reader_kernel，保留 ctypes.WinDLL 等原守卫，不能为了 fake 测试移除全局守卫。先登记旧夹具逐处差异，固定新增12方法/helper与harness，不引入新模块或 bootstrap 例外。

- [ ] **Step 4: 运行 collector-red 并判定有效性。** 仅未来获批后运行下列命令。应收集原124加12共136；R01/R07/R09/R10/R12的缺行为必须产生明确断言失败，旧124应通过；其他新增项可部分已通过。缺接口用 getattr 后 assert/callable 断言，不以 AttributeError、导入错误、未模拟调用或错误收集数冒充有效 RED。保存 attempt、summary、source_snapshot、全部日志与入口 stdout/stderr；无效则记录原因，不自动重试。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-red
```

- [ ] **Step 5: 实现规格第4节读取适配。** 在 driver 内延迟绑定指针宽 HANDLE 和 DWORD/BOOL ABI，精确设置只读/share7/非继承参数，立即捕获系统错误；helper 以继承 check 驱动分块读取与一次关闭。OPEN Win2/3 以外失败不转 None；CloseHandle 失败不伪装成功。读取结束后先关闭，再检查期限、解码、解析、返回；失败路径仍收集关闭和末端预算错误，不启动额外时钟或等待。

- [ ] **Step 6: 接线错误保留与 ready 门。** TraceReadError 只保存 OPEN/READ/CHECK/CLOSE 的原对象序列，首因不被清理覆盖；_fault_error_record 对它作有界且沿用原规则的脱敏记录，保留整数 errno/winerror。NativeServices.read_trace 使用原 _remaining；sample 先返回缺失 ready/controller，再检查既有语义/binding，最后读 trace。保留 assigned/ACK/spawn、活性、5000ms 窗口和原查询句柄逻辑，不改写入器、N4或终结后读取。

- [ ] **Step 7: 运行 collector-green。** 仅下列原入口。136项全通过、零errors/skips/expected failures/unexpected successes/未模拟调用；核对R01–R12确实调用实际适配/helper/sample及错误发布链。RED后新增测试/helper、harness和旧夹具冻结，若必须改变则停下记录并另行审核，不能调整断言追绿。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-green
```

- [ ] **Step 8: 运行 final 并核对覆盖。** 仅下列十模块入口，297项全通过；完整方法集合应为旧285并集固定12项，GREEN/final 来源哈希一致。严格通过谓词保持原样；R02结果只标为条件模型，不当作本机故障复现。不另跑 pytest、单测、文件共享/计时/进程探查或原生验收。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase final
```

- [ ] **Step 9: 作者验收并限定归档。** 对照规格1–7节和五类Review Focus逐项裁定；核对三文件、旧方法/拒绝门、46/56来源、期限/原件/脏图/报告保护。保留所有失败尝试，不改旧日志。只有全部符合才标 AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING；限定代码/证据提交并远端实际内容回读，不夹带报告/图片。独立复查须另批，不在本工作包派代理。

## 文档自检与交接

规格的共享语义/生命周期/前置门/错误保留/边界保护均在 Task1 明确接口和 R01–R12 中对应。136/297是未来验收目标，未在本轮收集或运行。权限、路径、字段、三文件和不变量已作作者静态一致性检查，不是独立工程复查。

下一步建议仍由当前会话本人实施这个单工作包；先审核并独立批准代码与有限合成验收。本轮不生成运行协议、不读原始矿物图、不拟合、不运行模型、不改正式报告。工程修复完成后，仍须另批真实机制强对照，才能判断 MSB-RRE 是否具有可写入报告/论文的网络创新收益。
