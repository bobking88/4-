# MSB Fixed Native Namespace v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans only after separate human approval. Inline execution is proposed; no agents are authorized. All unchecked steps are future work, not permission to execute now.

**Goal:** 将原生无数据验收统一到固定 v3 命名空间，为另行登记提供一致路径；不改变监督行为或科学网络。

**Architecture:** driver集中两个固定路径常量；collector/controller/worker及远端登记复用同一身份。只局部更新旧测试夹具/预期，新增8项合成验收；原验收器只更换两处证据路径。

**Tech Stack:** 既有Python 3.11、unittest/mock、三阶段合成验收器，无新依赖。

**Spec:** [固定v3规格](../specs/2026-10-25-msb-native-namespace-v3-design.md)。当前授权仅为 `call_7e2c43cf35794bd5b76ae09dca272a86` 的规格/计划编写与GitHub归档，尚未批准以下步骤。

## Global Constraints

- 状态 DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN；文档基线06829bc，源码固定dee2be4；本轮不导入项目、不改代码、不测试、不读真实数据、不派代理、不改正式报告、不生成执行协议。
- 未来仅原 driver、driver测试、原合成验收器三文件；原46项与56项路径集合不变，其余43/53源码原始哈希不变。
- 固定协议名msb_native_supervision_v3；协议路径docs/experiment_protocols/msb_native_supervision_v3.json；输出outputs/msb_native_supervision_v3/acceptance_once。
- 监督契约msb_parent_bound_v2及DIRECT_CPYTHON311_VENV_V1不变；N1-N4、480秒累计、[90,45,75,45]、接电、单次、失败关闭不变。
- 所有受限原生接口模拟；原十模块、bootstrap、通过谓词和守卫不改。无科学输入或真实拟合/前向，原scratch合成例外不扩大。
- 合成根outputs/msb_native_namespace_v3_synthetic_v1；scratch为.superpowers/sdd/2026-10-25-msb-native-namespace-v3/test-tmp；三阶段各最多3次独占尝试，含失败，全部保留，不自动重试或扩范围。
- 旧116/277完整方法名不删、不改名；新增固定8项，未来124/285。含v2的旧方法名保留，正向预期按新规格局部更新，负向旧版本检查不删。
- 旧协议/原生目录不删除、改名或复用；无权限迁移/路径回退；独立复查、协议登记、原生与科学运行另批。

## Review Focus

1. v3协议名与旧输出混搭、未知字段或任意路径，必须拒绝：N01/N02。
2. controller/worker指向旧根、错误case/job或其他执行root，不得读元数据或启动原生服务：N03/N04/N08。
3. 本地v3路径与远端旧tree键/contents路径不一致，不能回退接收：N05/N06。
4. 本地原始哈希、远端blob或协议实际内容不符，以及已存在新输出，不能被命名升级掩盖：N06/N07。
5. 新路径下完整模拟主链仍保留原累计期限、四例顺序、来源/claim、清理与失败关闭：N08及原277项。

---

## Task 1: 三文件适配与有限合成验收

**Files:**
- Modify: `scripts/check_msb_native_supervision.py`，当前80-86、202、391-395、765-794、856-859行附近；只调整固定命名空间与派生路径。
- Test: `tests/test_check_msb_native_supervision.py`，正向路径夹具/预期与两个harness输出预期，新增类NativeNamespaceV3Tests及一个独立v3夹具helper。
- Modify: `tests/run_msb_native_supervision_synthetic_acceptance.py`，当前77、90行的output/scratch两个字面路径。

**Interfaces:**
- 新常量 `NATIVE_PROTOCOL_RELATIVE: str`、`NATIVE_OUTPUT_RELATIVE: str`，值精确取自规格第3节。
- `FIXED["protocol"]`为v3，`FIXED["output_relative"]`引用输出常量；其余字段不变。
- `_case_output(root, case_dir, case_id)`、`_protocol_path(root, path)`、`_worker_packet(packet_path, handles)`及`NativeServices.verify_registration(root, path, protocol, approval, check)`保持现有签名与失败契约，按规格派生唯一位置。
- 测试helper `fixture_v3_protocol() -> dict`：deepcopy现有fixture_protocol，独立覆盖v3协议名/输出字面值；不能从待测FIXED复制预期值。无新TestCase子类继承旧测试导致重复收集。

### 固定测试清单

完整名前缀：`test_check_msb_native_supervision.NativeNamespaceV3Tests.`。每行一个方法，子情形用subTest。

| ID | 方法名 | 固定验收断言 |
|---|---|---|
| N01 | test_fixed_v3_constants_and_protocol_preserve_native_contract | 两常量精确等于规格值；v3纯校验成功且不读assets；FIXED其余字段/类型及46项路径集合不变 |
| N02 | test_old_mixed_and_arbitrary_namespaces_are_rejected | v1/v2/未知协议、v3配旧输出、旧协议配新输出、任意输出或额外配置字段均拒绝；不修改原字典 |
| N03 | test_case_output_uses_only_fixed_v3_root | N1-N4正确v3路径逐一返回；v2、其他根、错case和未知case拒绝，无目录写入或服务构造 |
| N04 | test_worker_packet_binds_v3_prefix_and_inherited_capability | 实际_worker_packet对四例合法合成packet成功；缺继承能力、旧根、错误五段结构或路径case/job段不一致先拒绝，再无read_text/_published；复用既有来源及发布校验不改 |
| N05 | test_local_protocol_path_rejects_old_and_unregistered_paths | _protocol_path只接受固定v3解析后位置；旧文件/其他目录/其他root位置拒绝；collector及controller错误路径在读协议/NativeServices前拒绝 |
| N06 | test_remote_registration_requires_v3_path_and_exact_identities | 实际verify_registration与合成46文件、严格假API接线成功，只请求v3contents；旧tree键、raw哈希、source blob、protocol blob及实际内容分别突变即拒绝，无网络调用或回退 |
| N07 | test_consumed_v3_output_is_rejected_without_touching_history | scratch中旧v2哨兵可与尚不存在的新输出共存，v3纯校验通过；新输出存在即拒绝；新旧哨兵字节/哈希不变，无自动后缀/删除/续跑 |
| N08 | test_v3_simulated_entries_preserve_claim_budget_and_case_order | 实际collector完整模拟N1-N4一次且顺序/原累计绑定不变；有效controller/worker接线复用已有fake；错误来源或RUNNING claim阻止服务/工作，四例证据不足仍拒收，guard与句柄原断言保留 |

N04/N06不得把待验证helper本身替换为固定成功值；只模拟原生/远端边界和合成元数据。N06的source写入仅为scratch伪源码，不读取真实数据。所有测试禁止创建仓库中的真实v3协议文件或原生输出目录。

### 步骤，当前均未执行

- [ ] **Step 1: 核验新开发授权与起点。** 审核通过且获得独立开发/合成授权后，核对届时文档提交、空index、三文件白名单；比对原46/56源码、既有277日志、旧原生原件、六张预存脏图及两处正式报告。新合成根/scratch必须尚不存在；否则停，不另选未批准后缀。

- [ ] **Step 2: 只写8个反例及v3测试helper。** 旧fixture_protocol和正向v2路径在RED前保持；N01/N03/N05/N06的缺行为异常须转为明确assertion failure，缺常量用getattr后精确断言，不能以AttributeError、导入失败或未模拟调用作为RED。其余方法可在RED已有部分通过，不强求全部失败。

- [ ] **Step 3: 更换原验收器两处路径。** output/scratch取Global Constraints；两个旧harness路径预期同步更新。除此不改harness。记录旧116方法源码变化清单；RED之后冻结新增8方法/helper、harness及两处harness预期。保留原全部断言。

- [ ] **Step 4: collector-red。** 未来仅按下列入口运行。应收集旧116加8项共124；N01/N03/N05/N06必须有缺行为断言，旧116应通过；零errors/skips/预期失败/意外成功/未模拟调用。不是有效RED则保留原件，不能宣称验证了缺陷。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-red
```

- [ ] **Step 5: 最小接线与旧夹具迁移。** 按规格集中两个常量，FIXED改v3，case/本地协议/远端协议路径引用常量；worker沿用FIXED前缀。仅局部更新旧正向v2协议/输出预期，不能全局替换负向v2检查。生成逐处差异清单，确认类型、拒绝条件、计数和方法名未弱化；不改权限策略或源码闭包。

- [ ] **Step 6: collector-green。** 仅下列原入口，124项均通过且所有严格守卫为零；核对N01-N08实际helper、端点、读取/启动计数和历史哨兵。GREEN与final源码身份相同。若需改变已冻结新增断言，不在本工作包临时放宽规格，应记录并重新审核。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase collector-green
```

- [ ] **Step 7: final。** 仅下列十模块入口，285项均通过；完整方法名集合为旧277并集固定8项。保存全部attempt、summary、source_snapshot、synthetic_tests.log、入口stdout/stderr；不另跑pytest、单测、计时探查或原生验收。

```powershell
& 'D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe' .\tests\run_msb_native_supervision_synthetic_acceptance.py --phase final
```

- [ ] **Step 8: 作者逐项静态验收并归档。** 核对五类Review Focus、固定身份、三文件/两harness路径、旧测试局部差异、46/56闭包和全部原件保护；记录任何失败，不由绿色数量推导原生或科学可用。只有规格和证据同时满足，才能标AUTHOR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING。

- [ ] **Step 9: 限定提交并核验远端。** 仅归档获批的代码、合成与作者证据，不带脏图或正式报告；非强制更新现有分支，远端commit/tree/ref和实际文件内容回读一致。此后独立复查、新执行协议登记及任何运行分别另批。

## 文档自检与交接

接口、常量、路径及124/285计数与规格一致；五类Review Focus分别落入N01-N08和原277项。旧行为测试完整名、错误保留和N4严格门保留；RED/GREEN阶段允许的旧夹具差异已显式界定，不虚称全部测试源码跨阶段冻结。

本轮仅文档自检，未执行任何Step、未创建拟议synthetic/scratch/原生根或v3协议，不派代理。建议未来仍由当前会话原生实施这一单工作包；请先审核规格和计划，开发与有限合成验收尚待独立批准。

本计划不是论文创新交付。完成工程门后，还须另行批准真实fit-only机制实验，用间隔预算的纠错/损伤条件、角色证据贡献和强对照结果检验网络创新；负结果也须保留，不预写模型优越结论。
