**Verdict: REQUEST_CHANGES（仅工程审查）**

以下为静态可达的条件缺陷，不代表已经在 Windows 原生运行中观察到。

**Findings**
1. **P1｜设计缺口，由新增诊断路径继承：多行能力信息可能泄露。**  
   [driver:284](<D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/check_msb_native_supervision.py:284>) 复用 [driver:90](<D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/check_msb_native_supervision.py:90>) 的脱敏器，但其 `.*` 不跨换行。例如异常文本 `environment=\nAPI_KEY=SECRET` 只会删掉第一行，第二行仍进入失败回执和 CLI JSON。数字替换及 256 字符截断不会消除该短字符串。计划要求复用既有脱敏器，却未处理这一实际泄露边界。

2. **P2｜实现缺陷：停止自身异常时，已捕获的停止原因可能丢失。**  
   [driver:587](<D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/check_msb_native_supervision.py:587>) 只有在 `_bounded_fault_stop` 完整返回后才取得结果与错误列表。如果预算前检先失败、随后 `stop_fault` 再抛异常，helper 内已经保存的预算异常不会返回；控制器仅保存后一个 `stop_exception`。同样，[driver:602](<D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/check_msb_native_supervision.py:602>) 不读取已经缓存的 `watch.error` 或停止快照。与 trace 失败并发时，汇总因此可能遗漏既有 callback／预算原因，而非完整保留角色分离的证据。

3. **P2｜实现缺陷：N4 失败回执可能无法进入 collector 汇总。**  
   新增读取位于 [driver:1166](<D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/check_msb_native_supervision.py:1166>)，但之前的 N4 分支先要求有效 `parent_fault` 和完整快照。trace 发布失败使 N4 未提交故障快照时，即使 `controller_failure.json` 已成功保存，[driver:1158](<D:/成信工科研/人工智能选矿/.worktrees/theory-aware-report/scripts/check_msb_native_supervision.py:1158>) 仍先抛错。collector 得不到含失败回执的 case record，主要诊断变成“缺少 N4 快照”。磁盘原件仍保留、接收仍失败关闭，但新增的诊断保留承诺在该路径不成立。

**Missing Coverage**
- T08/T09 没有联合覆盖“已获得 row、scope 退出异常、最终 trace 异常、回执异常”；T09 的写入替身在创建文件前就抛错，也不覆盖部分回执。
- T11 使用 N1，明确没有 fault watch，不能证明 N4 停止异常时的原因保留或停止次数。
- T10 只加载 N1；T12 只有单行敏感文本，未覆盖上述多行边界。T14 的新 helper 路径仅覆盖 N1–N3。

**Strengths And Evidence**
独占 CREATE、不删除既有 pending、单次 CLOSE、WRITE/CLOSE 原对象有序保留及首错锁存均有清晰实现。T05 和控制器 fixture 确实调用原 `NativeServices.persist_trace`，只替换文件适配层。

保存日志显示 RED 为 92 通过、14 个接口缺失断言失败，GREEN 为 106 通过，final 为 267 个唯一全名；我独立比较确认旧 253 名全部保留、新增恰为计划十四名。快照确认 RED driver 对应基线，测试／验收器冻结，GREEN/final 源码相同。16 份原件哈希匹配；展开的 339 条保护记录与作者汇总一致，包括六张 PNG 和两处报告的哈希保护。

**Scope And Limits**
实际检查了规格、计划、验收记录、三个修改源码的 diff 与相关正文、三份必要 callee 的调用路径、被引用的测试 fixture，以及三阶段日志／summary／source snapshots、acceptance_check、identity、manifest、authorization、ledger 和保护元数据。

不额外测试、不自动清理、磁盘 `ATTEMPTED` 与内存／CLI `PERSISTED` 分离，以及保留既有合成张量契约的裁定，与可见代码一致。但“历史上只执行三次入口”“ledger 使用 apply_patch”“复制前发生解析错误”仅能确认有作者记录，不能从这些归档独立证明完整操作历史。

未判断 Windows 共享冲突首因、原生时序／释放／送达可靠性或科学收益：这些需要本次禁止的运行或输入。未读取报告正文、真实数据、权重、张量或缓存。**未运行任何测试、Python 或项目导入；未编辑文件、移动分支或派出其他审查者。**
