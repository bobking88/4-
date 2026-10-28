# MSB Fixed Native Namespace v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans only after separate human approval. Inline execution is proposed; no agents are authorized by the current documentation approval. All unchecked steps are future work.

**Goal:** 仅将原生无数据验收登记位置一致适配为固定v4，保留既有共享读取、监督与科学数据边界。

**Architecture:** 复用driver现有两个路径常量及FIXED协议名，不改函数体。原harness只更换两个固定证据路径；独立4项v4反例先RED，再按封闭清单迁移旧夹具。

**Tech Stack:** 既有Python3.11、unittest/mock、原三阶段合成验收器，无新依赖或CLI。

**Spec:** [固定v4规格](../specs/2026-10-27-msb-native-namespace-v4-design.md)。本轮`call_6912f128ec5a4fb0a8e847242f3c76cf`只批准文档，尚未批准以下步骤。

## Global Constraints

- DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN；文档基线d2fa040，源码仍为已审8175134。本轮不开发、不导入项目、不执行项目函数/测试/前向/拟合/计时/原生验收，不读真实数据、不派代理、不改正式报告、不生成执行协议。
- 未来只改原driver、driver测试和原harness三文件，保留46/56来源集合，其他43/53源码原始哈希不变。
- driver仅3个固定值：协议名msb_native_supervision_v4、协议路径docs/experiment_protocols/msb_native_supervision_v4.json、输出outputs/msb_native_supervision_v4/acceptance_once；派生函数体、权限策略、共享读取、写入/N4/监督器均不改。
- msb_parent_bound_v2、DIRECT_CPYTHON311_VENV_V1、N1–N4、480秒累计、[90,45,75,45]、接电、单次及失败关闭保持。
- 合成根outputs/msb_native_namespace_v4_synthetic_v1；scratch为.superpowers/sdd/2026-10-27-msb-native-namespace-v4/test-tmp；原三个阶段各最多3次独占尝试，含失败，全部保留，不自动重试、扩模块或bootstrap。
- 所有受限原生接口全模拟，原十模块、通过谓词、守卫及2个scratch合成张量例外不变；不新增真实数据/骨干路径。
- 原136/297完整方法名保留，固定新增G01–G04，未来140/301仅目标；31项旧字面清单=4项RED前预期+26项有效RED后迁移+1项保留负例。
- 新增4项/helper、harness与4项旧harness预期在有效RED后冻结；允许的26项旧字面迁移必须逐项核对，不冻结整个测试文件跨RED/GREEN的哈希。GREEN/final完整来源相同。
- 不复用旧v3输出或许可，不创建真实v4协议/原生目录；独立复查、协议登记及实际运行仍另批。

## Review Focus

1. v4协议名/输出/本地路径混搭旧v3，不能被放宽等值比较接受：G01。
2. packet/case跨执行root、旧v3前缀、错误case/job或坏继承能力，不能读取元数据或启动服务：G02。
3. v4登记时远端旧tree/contents、head或raw/blob/内容身份不一致，不能回退：G03。
4. 旧v3哨兵与新v4独占目录共存时，既不能误拦新验证，也不能覆盖/续跑旧或已存在新输出：G04。
5. 路径迁移不能削弱共享读取、callback/期限/清理及N4失败优先；旧297断言结构及原R12实际工厂守卫必须保留：G04及原297。

---

## Task 1: 原三文件固定v4适配与有限合成验收

**Files:**
- Modify: `scripts/check_msb_native_supervision.py`第80–82行三个固定命名值，其他生产代码保持字节不变。
- Test: `tests/test_check_msb_native_supervision.py`末尾追加独立NativeNamespaceV4Tests及fixture_v4_protocol；旧字面值只按[静态清单](../../../outputs/msb_native_namespace_v4_plan_v1/fixture_migration_inventory.json)分阶段修改，不改旧完整方法名。
- Modify: `tests/run_msb_native_supervision_synthetic_acceptance.py`第23、24行EVIDENCE_ROOT/SCRATCH_ROOT值，其他代码不变。

**Interfaces:**
- 消费既有`_protocol_path(root,path)`、`_case_output(root,case_dir,case_id)`、`_worker_packet(packet_path,handles)`、`NativeServices.verify_registration(root,path,protocol,approval,check)`；签名/返回/失败契约不变，无新生产函数。
- 新测试`fixture_v4_protocol() -> dict`：deepcopy旧fixture_protocol，仅独立覆盖`protocol="msb_native_supervision_v4"`及`output_relative="outputs/msb_native_supervision_v4/acceptance_once"`。
- 新类直接继承unittest.TestCase；`protocol_file(tmp)`按现有方式注入observed_origins(root)，仅写临时root下v4协议并返回(root,path,protocol,fixture_approval(protocol))。
- `packet(root,case_id)`组合WorkerTests.packet，显式覆盖case_root为临时v4根，注册doCleanups；纯flags构造可复用既有方法。`success(operation)`把应成功却因旧namespace失败的异常转为明确assertion failure，不将失败分支变成成功。
- 全部新方法名及强制行为取规格G01–G04，不继承旧TestCase复制收集，不mock被测helper/主链为恒定成功。

- [ ] **Step 1: 新开发许可与前保护。** 先核对将来获批文档提交、空index、原三文件/46/56来源、旧297完整方法、旧协议/所有失败原件、六张脏图和两处报告。新合成根及test-tmp须不存在，否则停。记录新before，不把本轮规划保护当作未来开发前快照。

- [ ] **Step 2: 只追加独立4项v4反例与helper。** 旧正向v3夹具暂不动，新类及helper追加在文件末尾。按规格构造独立字面预期和原真实helper；缺行为须assertion failure，不以AttributeError/导入错误/未模拟调用为RED。至少落实以下独立断言（完整拒收子情形取规格）：

```python
# G01: hand-written v4 paths, not computed from native.FIXED.
self.assertEqual(native.NATIVE_PROTOCOL_RELATIVE,
                 "docs/experiment_protocols/msb_native_supervision_v4.json")
self.assertEqual(native.NATIVE_OUTPUT_RELATIVE,
                 "outputs/msb_native_supervision_v4/acceptance_once")
self.assertEqual(self.success(lambda: native.validate_native_protocol(
    root, protocol, approval=approval))["protocol"], "msb_native_supervision_v4")

# G02: actual helper; exception on missing v4 becomes assertion failure.
self.assertEqual(self.success(lambda: native._case_output(root, case, "N1")), case)

# G03: actual registration, only endpoint/filesystem boundary faked.
self.assertTrue(self.success(lambda: service.verify_registration(
    root, path, protocol, approval, lambda: None))["verified"])
self.assertFalse(any("msb_native_supervision_v3" in q for q in requests))

# G04: actual all-fake collector, preserving inherited limits/order.
self.assertEqual(self.success(lambda: native.run_acceptance(
    root, path, execute_native=True, approval=approval, services=services))["status"],
    "PASS_NATIVE_SUPERVISION_SCOPE")
self.assertEqual([c[1] for c in launches], ["N1", "N2", "N3", "N4"])
self.assertEqual([c[2] for c in launches], [90, 45, 75, 45])
```

G01另逐项拒绝旧v1/v2/v3、混搭、越界和额外字段。G02对N1–N4逐一验证精确packet结构，坏能力/前缀时读取和发布校验调用为0。G03仅在scratch造46份伪源码，fake head/tree/contents严格绑定v4并逐项突变raw、source/protocol blob、实际内容和head。G04临时v3/v4哨兵哈希前后相同，验证新目录存在即拒绝，实际controller/worker及claim/来源拒收复用既有fake，不启动线程/进程。

- [ ] **Step 3: 只迁移两处harness值及4处旧预期。** EVIDENCE_ROOT/SCRATCH_ROOT按Global Constraints；仅同步清单原行133、1195、1196、2099的harness路径预期。其余26项v3字面值与2709负例仍不动。保存旧文件完整diff与AST/完整方法名清单，确认新类仅4方法，原136保留。

- [ ] **Step 4: collector-red。** 将来仅获批后运行下列原入口；首次尝试必须独占，入口stdout/stderr分别保存。预期140项，旧136通过，G01–G04均有明确缺v4行为的assertion failure；零errors/skips/expected/unexpected/未模拟调用。无效即保留原件停止，不自动重试。有效后保存新增4方法/helper源码或AST指纹、4个旧预期和harness完整哈希并冻结。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-red
```

- [ ] **Step 5: driver3值及26项旧字面最小迁移。** driver只更改规格固定值；不改函数体、NON_NATIVE_IDS或其他字段。测试按封闭清单迁移26项，保留方法/helper/类名、断言结构、计数和历史负例，原行2709不动。记录每项旧/新值及语义；wrong-root、case/job结构的负例随当期v4前缀迁移，保留原错误条件。清单外变化、新增4项/helper或harness改动即停、另批。

- [ ] **Step 6: collector-green。** 仅下列原入口；预期140/140及全部严格门为零。核对新增逻辑/4项旧预期/harness冻结指纹，26项迁移逐条吻合，无旧断言弱化；三份源码存完整快照。若需纠正遗漏，只能按批准清单并在剩余额度内处理；不得改变冻结测试追绿。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-green
```

- [ ] **Step 7: final。** 仅下列十模块入口；预期301/301，完整方法名集合=旧297并集固定G01–G04，不能只看数量。GREEN/final三文件及56项身份一致；旧R01–R12实际读取/callback/守卫仍通过。保存所有attempt、summary、source_snapshot、完整log与入口stdout/stderr；不额外运行pytest、单测、进程/计时/共享探查或原生验收。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase final
```

- [ ] **Step 8: 作者验收与保护。** 按五类Review Focus核对固定身份、实际helper、冻结内容与26项旧夹具差异、46/56闭包、全部旧失败原件、报告和脏图保护。未来只有规格和证据同时满足才标AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING；无协议生成、原生或科学运行。

- [ ] **Step 9: 限定提交与远端实际回读。** 只归档原三文件及合成/作者证据，非强制更新既有分支，核对commit/tree/ref和实际原始blob，不夹带报告/图片。新独立复查、执行协议登记与任何运行分别另批；新源码必须重新登记，旧v3许可不迁移。

## 规格覆盖、自检与交接

规格固定值/三文件范围落入Step1/3/5；新路径/拒收与五类Review Focus落入G01–G04和原297；分阶段冻结落入Step3–7及31项静态清单；保护/归档与后续门落入Step8/9。常量/接口签名和140/301目标一致，无未定义生产接口、无第四源码文件。

当前只完成文档作者自检，不派builder/reviewer，不运行上述任何命令，不创建未来合成/test-tmp/原生目录或v4 JSON。建议由当前会话原生实施这一单工作包，但须先审核文件并另行批准代码与有限合成。后续真实fit-only机制强对照才能裁定网络理论收益；本计划不预写科学成功。
