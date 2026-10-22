# MSB观察器异常回收原因保留：最小修复规格

日期：2026-10-22。状态：`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。
文档授权：`call_a6f6d55427584c8da99bbba75fef5a29`，“批准仅编写修复规格与计划（推荐）”。
基线：`c555dba71e03e0307f59b8de890a882a71b18bac`；依据[独立复查R1/P2](../../experiment_records/2026-10-22_msb_python_launch_observer_repair_review.md)。受审代码仍为`16fb16d`，本规格不是修复完成或开发许可。
直接授权同时起草规格与计划，因此一起交付待审；不把这次文档许可视为认可尚未存在的实现。

## 1. 问题、目标与不变范围

静态链：`NativeServices.stop_observer -> NativeObserver.close -> _stop`返回布尔停止结果；collector异常回收忽略返回值，只保留外层异常B，或在close再抛C时只组合B/C。已缓存callback A、期限/停止错误D可能遗漏，根state可能没有observer_diagnostics。现有247项通过不能覆盖此提前异常入口。

目标只是在**现有一次close后**汇总可取得的缓存原因并走原失败发布器，不修改底层停止算法、正常finish采样、身份校验或预算。
未来实现白名单仅：
1. `scripts/check_msb_native_supervision.py`：小型纯原因汇总函数、异常回收接线及已知非运行许可拒绝项；
2. `tests/test_check_msb_native_supervision.py`：六个固定新方法、原方法的必要fake路径更新；
3. `tests/run_msb_native_supervision_synthetic_acceptance.py`：新合成根、三阶段、精确三文件保护及原守卫。

其余源码与测试，尤其msb_execution_safety.py、LC runner和safety测试，保持原字节。网络/损失/公式/实验矩阵/正式报告、旧原生及合成原件均不改。三个文件不足则停止记录，不自行扩界。

| 方案 | 决定 | 理由 |
|---|---|---|
| A：在原异常分支一次close后，用纯函数汇总缓存及原始对象 | 采用，待审核 | 改动最少，无新采样、join、线程或服务接口 |
| B：异常时改调finish_observer | 不采用 | finish可能继续sample/pause，不适合作为已失败路径的回收 |
| C：stop返回False即再join、延时或重启observer | 禁止 | 增加等待、改变首次停止证据，不能解决原因保留 |

本规格基于只读源码反例；当前未动态复现。未来RED必须在这条实际异常路径失败，不能把静态推理写成已运行证据。

## 2. 唯一新增内部接口与原因顺序

未来在driver定义：
```python
def _observer_abort_error(
    primary: BaseException,
    *,
    stop_result: FaultStopResult | None,
    stop_errors: list[BaseException],
    close_returned: bool | None,
    close_error: BaseException | None,
) -> ObserverTerminationError:
    ...
```

这是纯汇总函数：只消费传入快照/对象，不调用_remaining、close、finish、sample、时钟或任何原生适配器，不修改传入列表/快照，不从message/JSON重建原始异常。
调用方从observer的现有stop_result和stop_errors取得缓存；不存在时用None/空列表，不能伪造已终止或stop_ok。
close_returned为一次close正常返回的布尔值；抛错时为None、close_error为原始C。

确定性排列如下，顺序表示诊断角色优先级，不声称时间先后：
- primary_errors先B，再快照中的callback A，再缓存stop_errors中的D（保留原列表顺序）；
- primary_errors内部仅按对象身份去重；不同对象即便type/message相同也不能合并；
- cleanup_errors独立保留C；即使C与某个primary对象相同也保留清理角色，不跨两种角色去重；
- root guard退出F继续由原collector_scope_or_release接收；最终持久化P仍是原AcceptancePublicationError.persistence_error。新增汇总不吞并或重建它们。

不完整停止证据必须显式失败：
- 有typed result但stop_ok=False、没有A/D解释时，追加一个“observer termination not confirmed”的新诊断RuntimeError；
- 无typed快照时追加“observer stop snapshot missing”的新诊断RuntimeError，stop_result仍None，不能编造线程字段；
- typed result正常但close正常返回非True时，追加“observer close not confirmed”；
- C已说明close抛错时，不另造同义close错误；B始终保留，健康停止也不能抵消B或让流程PASS。

这些新诊断对象只是对缺证据的标记，不能冒充原A/B/C/D。
底层stop_fault在快照形成前自身抛错时，只保留实际可取得的B、C及缺快照标记；不虚构未被底层交付的D，不在本修复重构底层故障保存。
只支持当前内部类型合同，不增加通用异常/任意脱敏框架。

## 3. Collector接线与失败发布

仅替换原run_acceptance中非ObserverTerminationError的异常回收分支：
1. 保留捕获的B；调用services.stop_observer(observer)恰好一次，捕获返回值或原始C；
2. 读取已有缓存，调用上述纯函数；
3. 将同一组合异常的record保存到state.observer_diagnostics，再抛出该组合异常；
4. 外层沿原observer_primary/observer_cleanup归并、guard退出和publish_acceptance完成失败收据；持久化失败时原对象沿primary_errors保留，CLI仍非零。

已经捕获ObserverTerminationError的现有分支保持原样，不再调用close/join；正常finish和stop_fault/close缓存合同不改。
异常回收不等待新ready/ACK、不再sample、不调用finish，也不启动下个case。活线程不关闭其可能使用的query handle；已确认终止才沿现有close处理。原始绑定与累计起点不变，不增加任何剩余预算。

原480秒、N1--N4、30秒gap/2秒漂移及7200秒科学预算不变。已消费native v1与旧失败目录不能复用；native v2仍只保留现有代码命名，本轮及未来合成阶段均不创建实际协议或原生输出。
未来实现把本次文档许可、上次复查许可和届时的开发许可列为非native许可；不放宽现有scope/source/registration gate，不把新合成身份当原生授权。

## 4. 六个固定合成反例与验收边界

未来仅在新开发批准后，在`ObserverAbortRetentionTests(unittest.TestCase)`增加六方法，不继承其他TestCase类以免复制原方法数：

| ID | 方法名 | 必须断言 |
|---|---|---|
| A1 | test_abort_before_finish_retains_callback_and_body_errors | check/run_outer各提前抛B；真实NativeObserver.close+fake watch得到A；finish不调用，close一次，原B/A进入最终primary_errors且observer_diagnostics存在 |
| A2 | test_abort_retains_callback_budget_and_cleanup_objects | 缓存A/D，close抛原C；顺序B/A/D、独立cleanup C，stop_result为原快照；原始is对象均保留 |
| A3 | test_abort_live_or_expired_observer_never_rejoins_or_closes_live_handle | fake剩余0/0.25/2秒，join为0/0.25/1秒；缓存后不再join，活线程不close handle，timeout/expired不误标成功 |
| A4 | test_abort_missing_snapshot_and_clean_close_still_fail_closed | 缺快照与健康停止分别验证；缺证据不造线程字段，健康停止仍保留B且整体失败；False不能被忽略 |
| A5 | test_abort_errors_survive_release_publication_and_cli | A/B/D/C叠加原guard退出F、持久化P；最终原对象及顺序保留，persisted失败和写不出两类均非PASS，实际main非零 |
| A6 | test_abort_aggregation_deduplicates_identity_without_sampling | 纯函数同对象仅primary去重、不同同文错误均保留、cleanup角色不丢；传入列表不变、无_remaining/采样/停止/时钟调用 |

A1/A2/A5必须经过实际run_acceptance的提前异常回收，不能用完全fake的finish成功记录代替该分支；原始对象断言可用现有失败收据注入让AcceptancePublicationError暴露primary_errors。持久化成功子例检查真实main返回值及失败JSON。
复用现有fake runtime、thread/event、guard和CollectorTests.protocol_file等；不运行LC整模块，不新增真实线程或进程。

固定未来根：`outputs/msb_observer_abort_retention_synthetic_v1`；scratch：`.superpowers/sdd/2026-10-22-msb-observer-abort-retention/test-tmp`。当前不创建两者。
唯一未来入口仍是既有native synthetic harness，但仅允许collector-red、collector-green、final三阶段；前两者仅driver测试，final原十模块。每阶段最多3次独占尝试（含失败），递增目录，不覆盖旧证据，耗尽即停。
原全部247完整方法名保留；新增恰六名，driver预计92、final预计253，以实际名称和日志核对，不预填为已通过。

harness以原247 final/attempt-001/source_snapshot.json为只读保护基线：合并其protected_identity中53项当前sha256及sources中六项sha256，重叠值必须一致，得到56个唯一受保护路径。只允许上述三个文件有新字节，其余53项保持基线；每次运行前直接快照三文件源码，旧原件不改。46项SOURCE_FILES集合不改，未来新摘要另存，不复用旧摘要称新验收。
原test_check_msb_native_supervision中的两处fake-root contract-green夹具只更新为collector-green和新输出根；原方法名、负例及严格零跳过/未模拟门不变。未授权阶段在新harness CLI中拒绝，不打开旧根继续写。

原生API、process、thread start/join/is_alive、clock/power/network/runtime映像查询全部fake或守卫阻断，既有torch/sklearn数值bootstrap不扩展；仅scratch合成张量可读，不能称读取真实矿物数据。
接收必须零失败/错误/跳过/预期失败/意外成功/未模拟调用；新RED应为预期功能断言失败。夹具、导入或守卫失败占attempt，但不是有效功能RED。不能为绿色日志删除A1或放宽身份/期限/清理合同。

## 5. 当前交付及后续门

本轮只写规格、[实施计划](../plans/2026-10-22-msb-observer-abort-retention.md)、作者文档记录/保护证据和README并归档，不导入项目、不写上述三代码文件、不运行任何测试或探查、不派代理。
R1仍`ACCEPTED_OPEN_NOT_FIXED`，既有复查仍REQUEST_CHANGES；本规格没有把247历史通过提升为253。
书面文件审核后才可另批当前会话原生开发和有限全fake验收；独立复查、原生登记/验收和真实科学实验仍另批。
这是执行诊断修复，不是网络或数学创新。后续科学目标仍是既定MSB-RRE角色视觉证据与判别间隔预算的强对照验证，不能以更多工程步骤替代该目标。
