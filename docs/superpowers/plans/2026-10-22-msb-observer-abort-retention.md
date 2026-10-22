# MSB观察器异常回收原因保留实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 当前仅文档，不实施、不运行、不派代理。

**Goal:** 修复R1/P2，让collector提前异常回收保留一次停止的缓存诊断与原始原因，而不增加采样或等待。
**Architecture:** 原stop_observer调用和NativeObserver.close缓存不变；新增一个纯异常汇总函数，在原非ObserverTerminationError分支统一保存诊断并沿原失败发布器退出。
**Tech Stack:** 既有Python、dataclass、unittest/mock和全fake合成入口；不安装或切换依赖。
**Spec:** [最小修复规格](../specs/2026-10-22-msb-observer-abort-retention-design.md)。
**Baseline:** `c555dba71e03e0307f59b8de890a882a71b18bac`。仅文档授权`call_a6f6d55427584c8da99bbba75fef5a29`。
**Status:** `DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`。本次直接授权同时起草规格/计划，不代表书面方案已审核或开发获批。

## Global Constraints

- 只为R1计划未来三个文件：driver、driver测试、原native synthetic harness；不改其他代码、网络/损失/公式/实验矩阵/正式报告和旧原件。
- 原一次停止缓存、累计起点/期限、活线程不关query handle、失败关闭全部保留；不复用finish作回收，不采样、不重试join、不续跑原生失败目录。
- 原480秒、N1--N4、30秒gap/2秒漂移及7200秒科学预算不变。v2仅代码命名；无新实际原生协议/目录。
- 未来合成根`outputs/msb_observer_abort_retention_synthetic_v1`，scratch`.superpowers/sdd/2026-10-22-msb-observer-abort-retention/test-tmp`；当前均不创建。
- 仅collector-red、collector-green、final；各最多3次独占尝试含失败，原件递增保留；前两者driver模块，final原十模块，不discover或跑LC整模块。
- 原247完整名保留，新增规格固定六名，预计driver92/final253；不是当前通过结果。零非通过项/零未模拟守卫规则不变。
- 所有受限原生/进程/线程/时钟/电源/网络/runtime查询fake；保留原数值bootstrap，仅scratch合成张量可读，不碰真实输入。
- 规格与计划维持本轮未开始记录；未来执行另建ledger记录实际RED/GREEN，不回填旧规格为早已完成。
- 推荐将来由当前会话原生实施本唯一工作包；独立复查及实际实验分别另批，不自动串行派发。

## Review Focus

1. check在observer创建后、outer前抛B，与run_outer抛B，必须走相同诊断合同：A1。
2. 缓存已含A/D，close又抛C，不能只保留B/C或重复停止：A2/A3。
3. 活线程、零剩余预算、缺快照、健康close四类不可混同；B都使流程失败：A3/A4。
4. B/A/D/C与guard退出F/持久化P叠加仍保留原对象和CLI失败，不能靠字符串复原：A5。
5. 按错误对象身份去重、不按文本去重，cleanup角色与原列表不得被抹去；纯函数无副作用：A6。

---

### Task 1: 单项异常回收修复及有限合成验收

**Files:**
- Modify: `scripts/check_msb_native_supervision.py`，原1055--1064异常分支、小型纯helper及非native许可拒绝集；
- Test/Modify: `tests/test_check_msb_native_supervision.py`，新ObserverAbortRetentionTests六名及两处原fake-root路径；
- Modify: `tests/run_msb_native_supervision_synthetic_acceptance.py`，新根/三阶段、三路径保护和直接快照；不改严格接收规则。

**Interfaces:**
- Consumes: `FaultStopResult`、`ObserverTerminationError(stop_result, primary_errors, cleanup_errors=())`、observer.stop_result/stop_errors、`NativeServices.stop_observer(observer)`及原run_acceptance发布链；
- Produces: `_observer_abort_error(primary: BaseException, *, stop_result: FaultStopResult | None, stop_errors: list[BaseException], close_returned: bool | None, close_error: BaseException | None) -> ObserverTerminationError`；
- 纯helper只返回组合对象，不调用native/时钟/observer方法；调用方保存record后抛出。旧stop_observer/finish/close签名及正常行为不变。

- [ ] **Step 1: 先准备三文件快照边界及六个失败反例。**

按规格A1--A6的精确方法名写入新的unittest.TestCase类，复用现有CollectorTests协议夹具、FakeCollectorServices、DirectLaunchBoundaryTests.fault_watch和ObserverTests.make，不继承已有测试类复制方法。
A1分check/outer两种提前抛B；A2用缓存原A/D与close原C；A3用fake剩余0/0.25/2秒；A4分缺快照、正常typed close及False；A5叠加原F/P，实际main子例非零；A6直接测纯汇总。原对象断言采用如下语义，不以message相等代替：
```python
self.assertIs(combined.primary_errors[0], body_error)
self.assertIs(combined.primary_errors[1], callback_error)
self.assertIs(combined.stop_result, original_stop_result)
self.assertIs(combined.cleanup_errors[0], close_error)
self.assertEqual(finish_calls, 0)
self.assertEqual(close_calls, 1)
```
这些变量由各测试用既有fake在该次调用中创建，不是fixture返回的预制成功state。A1/A2/A5必须运行实际run_acceptance分支，用原失败收据注入观察最终异常对象，另有成功持久化子例核对失败JSON/CLI。
harness在RED前切换新根和三阶段；保护基线为原247运行快照中53+6去重成56路径，只豁免三指定文件，直接捕获三文件运行前文本。原两处fake-root contract-green调用/路径改为collector-green及新根，原方法名及负例不变。非native许可拒绝集仅追加已知文档/复查/届时开发ID，不创建运行协议。

- [ ] **Step 2: 运行collector-red并保留原件。**

仅在新开发批准后执行文末唯一入口，phase为collector-red。预计92方法，新异常回收断言/缺helper断言失败、既有方法保留；0错误/跳过/未模拟受限调用才是有效功能RED。每阶段最多3次含失败，未通过的attempt保留，不能用额外命令补跑或覆盖日志。

- [ ] **Step 3: 实施纯helper及原异常回收接线。**

按规格顺序B、A、原stop_errors排列primary_errors，按is在该列表去重；cleanup C独立保留，传入列表/快照不变。缺快照、未解释的停止失败或反常close返回按规格追加明确缺证据标记，原对象不重建。
原非ObserverTerminationError分支只调用stop_observer一次，保存close_returned/close_error，再读缓存构造组合异常，保存observer_diagnostics并抛出。原typed异常分支不重复close；不改finish、stop_fault、原_remaining与publish_acceptance算法，不增join或新预算。
若需第四文件或底层改造，停止记录并另请批准，不顺手扩展。

- [ ] **Step 4: 运行collector-green。**

同一未来入口phase为collector-green，预计92方法；A1--A6原对象、缺证据、单次停止、活句柄及CLI断言全部通过，0失败/错误/跳过/预期失败/意外成功/未模拟调用。失败计入该phase的3次上限，不能删反例、改原异常对象为字符串或接受False来凑绿色结果。

- [ ] **Step 5: 运行final并核对完整名称。**

同一入口phase为final，原十模块。预计253方法；将实际log完整名与原247日志及固定六名作集合/计数比较，零漏名/重名；所有严格门为零，保护56项除精确三路径外不变。最终源码必须匹配本次运行前直接快照；不运行其他phase、LC整模块、单独原生或科学探查。
原始summary/log/snapshot、所有失败attempt及原始哈希均保留；实际新增源码身份另记，原46集合不变、extra safety测试仍独立核对。红灯结果不算green成功，253只是预期。

- [ ] **Step 6: 作者归档及交接。**

只提交三文件、实际合成证据/授权/保护、作者记录及README到既有分支；独立读取GitHub实际内容并核对Git blob，不强推、不暂存六张既有脏PNG或其他变化。
状态最多为作者有限合成通过、独立复查待批准；不声明R1已独立复查/原生通过、网络创新有效或理论成立，不自动派代理。后续门分别另批。

## 未来唯一命令模板

以下只是待审核的计划文本，**本轮不运行**。未来同一入口仅替换phase为上文三个固定值：
```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' '.\tests\run_msb_native_supervision_synthetic_acceptance.py' --phase collector-red
```

## 自审与执行交接

本计划为一个工作包、六个未开始步骤；A1--A6覆盖R1、原因顺序、缺证据、预算/句柄、原发布对象及纯函数副作用，Review Focus逐项映射一致。接口为返回组合对象、调用方保存并抛出，没有“helper自身抛出”的歧义。
当前只有规格/计划/作者文档及保护归档，R1仍未修复，253项未运行，原生/科学效果均未验证。请审核两份书面文件；建议之后由当前会话原生实施此限定工作包，不派代理。开发及有限合成仍须新批准，独立复查/实际实验另批。
