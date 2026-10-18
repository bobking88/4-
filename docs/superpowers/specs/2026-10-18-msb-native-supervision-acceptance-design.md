# MSB原生无数据监督验收规格

日期：2026-10-18。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档授权：`call_a42aed1b7bab474bb17246c3df69dc4e`，只编写规格与计划并登记GitHub。
当前代码/证据基线：`779a4c669e647d38798b35e9df535b4bc06129ce`；获限定独立接收的实现：`ee6078a859a90006a78ba473c5d150c7d35412c0`。

## 1. 目的与授权边界

依据[独立只读工程复查](../../experiment_records/2026-10-18_msb_completion_acceptance_repair_independent_review.md)，补充fake无法提供的Windows进程、IPC、Job和退出观测。原结论为合成工程APPROVE、原生INSUFFICIENT_EVIDENCE；157项既有合成日志不等于原生验收。

本轮只写文档及README索引，不创建验收代码、运行协议JSON或原生输出目录，不导入项目/运行测试，不做原生时钟、进程、电源、空闲或计时探查，不读取真实prepared/fit/stop、原图、缓存或权重，不训练、不新增前向、不改科学网络或正式报告，不派代理。
用户已同时批准规格和计划起草，因此两份均作为待审文件交付；这不是开发或运行许可。

本验收的未来通过只解除限定执行证据障碍，不证明网络创新有效、新颖、泛化可靠或适用于工业选矿。

## 2. 当前实现约束与方案选择

| 源码事实 | 设计影响 |
|---|---|
| msb_execution_safety.RunBudget要求deadline-start精确为7200000ms | 不把它改为480秒，不倒移start伪造短超时 |
| run_msb_role_mechanism.run_stage的synthetic也排完80个job | 不直接运行完整synthetic入口冒充小规模烟测 |
| MSBParentSupervisor.run_job不使用per_fit_seconds强制短超时 | 不照搬旧LC的0.25秒用例或声称已测MSB累计7200秒超时 |
| NativeMSBAPI为Job配置KILL_ON_JOB_CLOSE，worker需继承实时能力 | 可通过既有监督器及专用无数据worker，观察原生握手与父退出 |
| 既有WindowsAwakeGuard为线程归属；NativeEnvironmentAPI.sample要求接电 | 主线程嵌套既有guard，观察释放，保持原限值，不修改电源方案 |

候选A为完整80-job原生synthetic，成本高且不便定向注入；候选B为专用无数据驱动调用现有监督器，采用；候选C为重写通用Windows执行框架，超出需要，不采用。
这是既有执行器上的限定验收子系统，采用规格/计划路径。不得因实现困难悄悄切换为新监督器或只看fake日志。

## 3. 未来文件与依赖边界

仅在另批开发后新增：
- `scripts/check_msb_native_supervision.py`：collector/controller/worker三种固定模式、纯协议校验、证据收集和接收判断。
- `tests/test_check_msb_native_supervision.py`：全部原生接口fake的合同反例。
- `tests/run_msb_native_supervision_synthetic_acceptance.py`：固定合成入口、守卫、直接源码快照及严格结果门。

现有43项执行源码及10项测试/验收器全部保持原始字节，不改MSB/LC生产监督器、科学模块或旧验收入口。新驱动是验收设施，不加入真实MSB科学执行闭包。
未来验收登记必须绑定现有43项源码、新驱动及两项新测试/入口的原始SHA-256；Git远端以blob和规范化LF内容单独核对，不能把Git blob SHA冒充原始字节SHA。
原生验收驱动允许已安装依赖的正常导入/数值运行库初始化，不承诺没有原生库加载；驱动本身不调用神经网络前向、求解器、torch.load、PCA/T0或训练方法。未来开发合成阶段的既有九模块数学/梯度/小张量求解照常实际执行，限专用合成scratch的round-trip，不是假装所有计算都被mock；当前文档阶段一概不运行。

## 4. 进程结构和真实能力

collector位于被测MSB Job之外，使用既有`run_lc_role_adaptation.run_supervised_job(job, protocol, remaining_seconds)`限定controller存活窗口；它仅为外层看门与回收，不作为MSB通过证据。
controller使用原始`MSBParentSupervisor.run_job`，worker通过原始`open_worker_scope`完成实时继承通道、父创建身份、HELLO/GRANT/ACK、续租与关闭。

worker只执行有界等待、scope.check、验收元数据及固定故障，不调用`run_stage`、`_schedule`或科学worker_body；不创建result.pt。验收packet与RUNNING状态按现有_published合同在新目录构造，其stage标记为`native-supervision-no-data`，不是向生产run_stage添加新stage。
每个worker仍先有真实Job分配，再接受GRANT；不得用磁盘快照构造_VerifiedBinding，或通过monkeypatch让生产监督器宣布COMPLETE。

驱动中的记录代理只能一对一转发真实NativeMSBAPI调用、在实际成功后记录返回值/事件；不替换tick、消息、wait结果或Job动作。故障仅由固定worker分支或独立的controller故障线程触发，不通过fake API注入。
事件缓存在内存并在明确检查点写入，不能在生产最终校验与COMPLETE之间插入I/O。公开日志仅保留nonce摘要和进程身份，不公开活动能力、原始nonce或可复用句柄。

collector观察线程只读取本次独占目录及已登记worker身份；用SYNCHRONIZE/QUERY_LIMITED_INFORMATION句柄核对PID和创建时间后保留至退出检查。
它不得复制Job句柄、通信句柄或可继承能力，否则会延长Job寿命并使父硬退出用例失真。进程活性以保留句柄的Wait结果为准，不以“PID搜索不到”或单张tasklist证明。
退出检查前不得先杀worker再声称Job自动清理成功。需要应急回收则本例失败，记录实际动作，不可转为通过。

## 5. 固定场景与接收条件

顺序固定N1、N2、N3、N4，每例controller一次、MSB worker一次，无重试、续跑或替补。最多4个controller及4个worker；观察线程不是额外工作进程。任何例意外失败/证据不足立即停止，不进入后续例。
计划内故障符合下表预期且证据齐全，属于该负例通过，不是违反“失败即停”。

| 场景 | 固定故障/工作量 | controller外层上限 | 必需证据 |
|---|---|---:|---|
| N1 normal-renew-close | worker在begin_work后持续45秒有界等待，然后正常publish/close/exit0 | 90秒 | 两次以上真实RENEW及匹配ACK，最终sequence至少3；MSB COMPLETE、exact close、exit0、exit_confirmed/cleanup_confirmed为真、alive为假；父及worker正常路径电源请求实际释放 |
| N2 worker-exit-no-done | worker已begin_work、发布ready并获collector身份确认后立即os._exit(23) | 45秒 | worker保留句柄确认退出，exit23；MSB SUPERVISION_INTERRUPTED（缺关闭）或FAILED（原生pipe EOF）、无完整关闭；controller完成负例检查后exit0；父guard释放。worker释放未显式观测，不能伪造released=True |
| N3 done-linger | worker正常完成关闭并发送DONE，离开scope后等待35秒而非立即退出 | 75秒 | 真实DONE已被接收、原冻结租约不续延；worker退出接收前租约过期，MSB SUPERVISION_INTERRUPTED且记录lease原因、Job回收与exit_confirmed；不能用外层超时冒充该反例 |
| N4 parent-hard-exit | worker已begin_work，collector保留身份句柄并回ACK；controller固定故障线程os._exit(31)，不走finally | 45秒 | 外层精确exit31而非超时；原生Job已分配；collector在5秒内观察worker退出，无DONE/完整结果、无成功交付；collector未持有Job句柄、未预先杀worker。controller/worker显式电源释放未观测，不能补造 |

N2/N4的ready-ack等待上限5秒，超时不触发预定故障，转为失败并走现有清理。N4故障线程只等本例ready和collector ACK，不能向其他PID发信号；collector必须先完成保留句柄身份校验再ACK。
N2不能把任意FAILED视为通过：必须有原生已开始工作、exit23、无DONE、失败发生在该异常退出的管道/缺关闭路径、正常父guard释放及完整清理证据；TIMED_OUT、POWER_RELEASE_FAILED或其他原因失败均拒绝。现有_terminal对不含分类关键字的WinAPI异常返回FAILED，本规格不为统一标签修改生产代码，也不只用fake的缺关闭分支冒充原生EOF。
N4的controller-ready/worker-ready、Job分配、ACK和故障启动均须有可排序事件；没有证明child确在Job内时，不得只凭随后退出归因于Job。
N4故意没有正常MSB返回回执及完整worker结果；这是预定负例结构，不可补造。它只以故障前完整身份/分配/WORKING证据、精确父exit31及保留句柄后续退出观察接收，其他例缺少必需回执则失败。
N3的35秒是关闭后故意迟退时间，预期由监督器更早终止；35秒工作完成或外层75秒到期不算负例通过。精确L-1/L/L+1与最终guard耗时位置仍由既有合成反例支持，本原生例不声称覆盖毫秒边界。

## 6. 累计期限与电源

未来原生整体接收窗口固定480秒，包含入口纯校验、依赖准备、源码核对、远端登记核对、原生对象构造、四例、身份观察、诊断、清理和最终发布。首次入口取monotonic，首次实际MSB tick另记；已发生的准备时间不能清零。
所有controller/worker继承同一首次MSB tick及其加7200000ms的生产RunBudget，不按场景重新起点。480秒是额外的验收上限，不能拿它改生产预算或宣称实测了两小时边界。

每例实际外层限额为min(表内上限,全局480秒剩余)，调用既有LC wrapper；各层最终接收还须重新检查全局剩余。等号过期；不得先保存旧remaining再等待耗尽后启动下一进程。
生产cleanup wait最多5秒保留；清理不获新计算额度，超480秒只能记失败。OS停调度、底层阻塞或崩溃时不承诺硬实时完成，证据缺失即不通过，不将“设置了480”写成已保证的外部硬截止。

collector主线程持有NativeEnvironmentAPI版WindowsAwakeGuard；同线程LC外层guard的parent链检查collector，MSB父及worker沿用原guard。持续AC=1，否则失败；轮询间隔上限30秒、双时钟差限2秒不改。
不主动休眠、拔电、改电源方案/待机时间，不要求300秒待机设置，不要求用户停止键鼠操作。前后只读power_settings必须一致，键鼠空闲不作为本验收指标；不重用旧LC idle验收的已消费许可。
N4进程终止可释放OS所持资源，但不是显式SetThreadExecutionState返回值证据；只正常路径和collector guard可报告实际observed released。不能把进程已死推导为已观测到所有句柄零泄漏。

## 7. 新目录、协议与状态

未来原生运行拟用`docs/experiment_protocols/msb_native_supervision_v1.json`及`outputs/msb_native_supervision_v1/acceptance_once`；本轮二者均不创建。任何既有目录都不覆盖、删除、重命名或续跑，首次尝试含失败即消费。
实际协议仅在代码合成验收及必要复查后另批登记：固定场景/时间/顺序、代码身份、目录、require_ac=True、max_attempts=1、real_training_authorized=False及许可原文。先登记待运行稿，再另获原生运行批准、补许可并重新登记，通过只读远端核对后才能创建运行目录。

当前文档许可、代码合成许可、复查许可及旧LC/IC许可均不能被填为原生运行许可；新ID仍须真实人类明确批准，黑名单不代替语义授权。
运行状态为RUNNING后唯一终结：PASS_NATIVE_SUPERVISION_SCOPE / FAILED_NATIVE_SUPERVISION / INSUFFICIENT_EVIDENCE；预期负例只在case记录中判通过，不能冒充科学COMPLETE。
纯校验或登记失败、原生设施不可用、outer kill/超时、原生消息异常、未知创建所有权、清理/释放证据缺失、任何skip/缺例均不得发布PASS。失败记录若也写不出，要保留两异常及非零退出，不声称已撤销磁盘候选。
不重新读取真实MSB协议或prepared列表作为本验收输入，不借核对身份打开.pt数据。

## 8. 交付证据与禁止扩大结论

未来单次目录至少保留注册协议原件及SHA、受测源身份、整体起止时钟、四例原始controller/worker日志、Job/进程身份和消息摘要、LC外层回执、MSB内层回执、ready/ACK事件、保留句柄退出观察、正常路径电源证据及终结判断。失败缺例原样留存，不补造后续例。
公开归档前不携带活动nonce/句柄；原始内部元数据仅在本独占目录验证，不为公开图表删改原始证据，单独导出去能力化的摘要。

接收只覆盖表内四种已观测路径。未测：主动挂起/恢复、任意父子/线程竞争、管道缓冲耗尽与任意分片、任意创建失败窗口、设备崩溃、精确租约毫秒边界、两小时累计超时、持久撤销、所有句柄泄漏及所有环境兼容性。
原生通过仍须保留上述限制，未来真实实验另行登记并批准，不能宣称“所有原生问题已解决”。如果已观测行为不符，先记录，不自动扩场景、修生产或重跑。

## 9. 与科学主线的对应

B路线不变，C仅后续研究。工程结束后真正要验证的仍是：间隔预算是否允许跨界、角色非线性证据是否被利用、相对SHARED-BIL/CONCAT-MLP/充分求解凸参照是否有独立收益。
沿用11单元、340已暴露fit、三折三种子、100步主点/400步上限、三种子七参照晋级门；不达门即NO_ROLE_PROMOTION，不把本验收的四个负例或时间公式计作网络理论创新。
