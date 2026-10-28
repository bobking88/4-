# MSB 原生无数据验收固定v4命名空间最小适配规格

日期：2026-10-27。状态：**DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN**。

授权 `call_6912f128ec5a4fb0a8e847242f3c76cf`，答复“采用固定v4，批准仅编写规格与计划（推荐）”。用户已选择固定v4；本规格和计划仍待审核，未授予开发、合成运行、原生验收或真实实验许可。

## 1. 目的、依据与选择

依据[兼容性核对](../../experiment_records/2026-10-27_msb_native_registration_compatibility_v2.md)。文档基线 `d2fa0408e36d9b1118d9a33b9678bb3eaea3a412`，执行源码仍与已审 `817513471f665256e747f02e07db8b0aad69bc8c` 一致。v3目录和许可已消费，当前不能只新建JSON后运行。

选择A：一次固定v4命名空间适配，复用已有路径常量，维持严格等值和来源校验。不选B通用可配置执行器，因其扩大路径/权限组合且非本次所需。禁止重用、清空、改名、覆盖或续跑旧目录，也不以换worktree或根别名绕开独占门。

v4是登记命名空间版本，不是MSB-RRE算法版本。本工作不增加网络、公式、损失、科学矩阵、依赖或CLI功能，不重构监督器，不修复其他问题，不修改正式报告。既有共享读取修复不动，原WinError5唯一根因仍未实证。

## 2. 未来文件边界与固定值

未来获开发许可后，仅原三文件：

| 文件 | 允许变化 |
|---|---|
| `scripts/check_msb_native_supervision.py` | 仅第80–82行三个固定命名字面值，其他函数体/权限策略不变 |
| `tests/test_check_msb_native_supervision.py` | 固定4项独立v4反例及其helper；清单内旧夹具/预期的分阶段字面迁移 |
| `tests/run_msb_native_supervision_synthetic_acceptance.py` | 仅EVIDENCE_ROOT与SCRATCH_ROOT两个固定路径值 |

driver未来值：`NATIVE_PROTOCOL_RELATIVE="docs/experiment_protocols/msb_native_supervision_v4.json"`，`NATIVE_OUTPUT_RELATIVE="outputs/msb_native_supervision_v4/acceptance_once"`，`FIXED["protocol"]="msb_native_supervision_v4"`。`FIXED["output_relative"]`继续引用输出常量，不复制字面路径。

保持原46项SOURCE_FILES和历史56项来源集合不变，其他43/53项源码原始哈希不变。保持`msb_parent_bound_v2`、`DIRECT_CPYTHON311_VENV_V1`、`APPROVED_NATIVE_ONLY`、`NATIVE_SUPERVISION_NO_DATA`、480秒累计接收上限、N1–N4顺序、case limits `[90,45,75,45]`、require_ac=True、max_attempts=1、real_training_authorized=False及其类型。

既有`_worker_packet`、`_case_output`、`_protocol_path`、`NativeServices.verify_registration`已经共享上述常量/FIXED；只随新固定值改变有效位置，不改签名、分支、继承能力/身份/nonce/摘要检查。原始SHA-256、Git规范化blob、远端实际内容和两次源码校验全部保留。不改NON_NATIVE_IDS或权限逻辑，旧许可和本轮文档许可都不得充作真实运行许可。

## 3. 新证据位置与未授权事项

未来合成根固定为`outputs/msb_native_namespace_v4_synthetic_v1`，scratch为`.superpowers/sdd/2026-10-27-msb-native-namespace-v4/test-tmp`。当前不创建。未来开发前须重新核对两者仍不存在；存在即停，不换后缀、不续跑。

未来阶段仍只有collector-red、collector-green、final；每阶段最多3次独占尝试，含失败，保留全部原件，不自动重试、扩模块或放宽门。所有受限原生进程、线程启动/join、时钟、电源、网络、runtime及Windows文件API全部模拟；沿用原十模块、bootstrap、严格通过谓词和全部守卫，原两个scratch合成张量例外不扩大。

不生成真实`msb_native_supervision_v4.json`，不创建真实原生输出根，不运行原生验收/计时探查，不读真实prepared/fit/stop或矿物原图。所有未来合成协议、伪源码、哨兵仅在隔离scratch的临时ROOT内，不构成真实许可或登记。

## 4. 固定4项增量回归与保留规则

完整名前缀`test_check_msb_native_supervision.NativeNamespaceV4Tests.`，只直接继承unittest.TestCase，不继承含旧测试的TestCase，避免重复收集。原297个完整方法名（driver136）全部保留；未来目标driver140、final301，不是本轮结果。

| ID | 方法名 | 必须检验的行为 |
|---|---|---|
| G01 | test_v4_fixed_protocol_paths_and_legacy_rejection | 两固定路径/协议名为v4，纯验证与本地固定协议路径实际成功；旧v1/v2/v3、混搭输出、任意路径、额外字段拒绝，原字典不改；不读资产或构造服务 |
| G02 | test_v4_case_worker_roots_keep_inherited_identity | N1–N4的实际case/packet helper只接受同一执行root的v4精确结构；旧v3、错case/job段、其他根、坏继承能力在读取元数据/发布校验/服务启动前拒绝 |
| G03 | test_v4_registration_uses_exact_remote_identity | 实际verify_registration配46项临时伪源码与严格fake API，成功时只读v4 tree/contents；旧v3 tree键、raw哈希、source/protocol blob、实际协议内容、head分别突变均拒绝，无回退或真实网络 |
| G04 | test_v4_single_use_collector_preserves_supervision_and_history | scratch旧v3哨兵不阻碍新v4验证，新v4输出存在即拒绝且原件不变；实际collector/controller/worker全模拟主链保持顺序、限额、根绑定、RUNNING claim/来源拒收、guard释放/句柄清理及证据不足拒收 |

`fixture_v4_protocol() -> dict`从旧fixture_protocol作deepcopy，独立写明v4协议名/输出字面值，不从待测FIXED导出预期。新类`protocol_file(tmp)`只建临时v4协议并注入observed_origins(root)；`packet(root,case_id)`可组合WorkerTests.packet但必须显式覆盖v4 case_root并注册清理，不继承旧类收集其测试。`success(operation)`将缺命名空间的受控异常转为明确assertion failure，不吞掉原本应拒绝的分支。参数flags可复用现有纯构造helper。

G02/G03/G04必须调用实际项目helper/主链；只模拟原生/远端边界与scratch元数据，不能把被测helper替换为恒定成功。G03独立fake应明确拒绝未登记端点。旧R01–R12的共享读取/实际callback/工厂守卫断言全部保留，G01–G04不是替换它们。

## 5. 分阶段夹具迁移，避免无效RED

详见[31项静态迁移清单](../../../outputs/msb_native_namespace_v4_plan_v1/fixture_migration_inventory.json)，基线测试SHA-256为`7d6e11deb08fedd8bad41ba4ba12bcbbaaaa4ef5864ed6bf45f4eeafdf3f61da`。这份清单是未来变更要求，当前未应用。

1. RED前保持旧正向v3夹具及其他旧断言不变，只追加独立G01–G04和v4 helper。验收器先切换两个证据路径；同步4处旧harness路径预期（原行133、1195、1196、2099）。不能再漏掉R12两条路径断言。
2. collector-red应收集旧136+4=140项；旧136通过，G01–G04都有明确缺行为assertion failure；零errors/skips/expected/unexpected/未模拟调用。导入错误、AttributeError或夹具缺文件不能冒充有效RED，无效即保留原件并停止。
3. 有效RED后冻结新增4方法及其helper、整个harness和上述4处预期。driver改3个固定值；只允许清单26项旧namespace字面迁移，方法/类/helper名称、断言结构、计数、拒绝条件不变。
4. 原行2709的`dict(namespace="msb_native_supervision_v3")`为额外未知字段的负例，保持原样；旧v1/v2负例不全局替换。wrong-root、wrong-case/job等负例中代表“当期版本”的26项清单字面值随v4迁移，以继续检查原错误而非仅靠旧版本不符拒收。
5. 历史NativeNamespaceV3Tests及fixture_v3_protocol等名称保留，合法夹具值在GREEN阶段按清单迁移；不因命名陈旧删改方法ID。这里只冻结新增逻辑与指定预期，**不声称整个测试文件RED/GREEN哈希相同**。GREEN/final三文件完整哈希必须相同。

清单外字面值、旧断言结构、生产函数体或新测试冻结逻辑若需变更，先停止并记录，另行批准，不靠改断言追绿。每个旧夹具变动须有语义和diff对应；保留全部失败及入口stdout/stderr，不覆盖目录。

## 6. 接受条件与后续门槛

作者验收需要：三文件边界与固定值符合；有效RED；GREEN140/140及final301/301，所有严格门为零；完整方法集合精确等于旧297并集固定4；新增逻辑/4个预期/harness按规定冻结，GREEN/final来源一致；旧原件、报告及六张预存脏图不变；代码和证据实际远端原字节回读一致。

当前只交付规格、计划、迁移清单与作者静态自检，没有合成/原生/科学运行或代理。开发、独立复查、真实可执行协议登记、单次原生验收及真实fit-only机制强对照仍分别另批。已审8175134不自动批准未来适配源码；未来必须重新登记实际身份。

本适配不是论文创新成果。MSB-RRE间隔预算的纠错/损伤条件、角色非线性证据的非等价贡献及强对照效果仍未科学验证；工程绿灯不能代替网络理论收益，也不能预写阳性结论。
