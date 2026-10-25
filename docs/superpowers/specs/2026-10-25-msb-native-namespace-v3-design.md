# MSB 原生无数据验收固定 v3 命名空间适配规格

日期：2026-10-25。状态：**DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN**。

授权：`call_7e2c43cf35794bd5b76ae09dca272a86`，答复“批准仅编写规格与计划（推荐）”。用户已选择固定 v3 方案；本规格和实施计划仍待审核，尚无开发、合成运行、原生运行或真实实验许可。

## 1. 目的、依据与非目标

解除[兼容性核对](../../experiment_records/2026-10-24_msb_native_registration_compatibility.md)发现的固定 v2 路径冲突，为以后另行批准的新登记提供一致的固定位置。当前文档基线为 `06829bcf6cdd2afd5b156d83019c7105a473a20f`，三个执行/验收文件仍与已审 `dee2be4045d325b417995da6d56943ccf06d1ee1` 相同。

不改变 MSB-RRE 网络、损失、公式、监督契约、实验矩阵、N1-N4 行为或科学数据边界。不引入可配置命名空间、不增加 CLI 参数、不升级依赖、不修复其他问题。原始 Windows trace 故障原因仍未完全解释。

本轮不生成 `docs/experiment_protocols/msb_native_supervision_v3.json`，不创建原生输出根，不执行计划中的任何命令。文档登记不是可执行协议登记。

## 2. 选择与范围

用户选择：**一个新固定版本命名空间**。将相关固定路径集中为两个模块常量，消除三处重复字面路径；不增加泛化路径抽象或兼容旧版本的回退。

不选通用配置接口，因为当前只需恢复一次独占验收，而通用接口会扩大允许的输入与路径面。禁止删除、改名、覆盖或续跑旧 v2 输出，也禁止修改旧协议、改根目录或跳过源码身份检查。

未来若获开发批准，仅可修改：

| 文件 | 唯一职责 |
|---|---|
| `scripts/check_msb_native_supervision.py` | 固定协议/输出命名空间及派生路径的一致接线 |
| `tests/test_check_msb_native_supervision.py` | 必要路径夹具/预期更新，以及固定8个新增验收方法 |
| `tests/run_msb_native_supervision_synthetic_acceptance.py` | 仅更换合成 output/scratch 两处固定路径 |

原46项 SOURCE_FILES 和历史56项保护闭包的路径集合不变；前者43项、后者53项非白名单源码保持原始哈希不变。文档/证据可按另行授权归档，但不增加第四个执行源码文件。

## 3. 固定身份与接口

未来 driver 的固定值：

| 名称/字段 | 固定值 |
|---|---|
| `NATIVE_PROTOCOL_RELATIVE: str` | `docs/experiment_protocols/msb_native_supervision_v3.json` |
| `NATIVE_OUTPUT_RELATIVE: str` | `outputs/msb_native_supervision_v3/acceptance_once` |
| `FIXED["protocol"]` | `msb_native_supervision_v3` |
| `FIXED["output_relative"]` | 引用 `NATIVE_OUTPUT_RELATIVE` |
| `execution_contract` | 仍为 `msb_parent_bound_v2` |
| `python_launch_contract` | 仍为 `DIRECT_CPYTHON311_VENV_V1` |

其余 FIXED 字段及类型不变：`status=APPROVED_NATIVE_ONLY`、`scope=NATIVE_SUPERVISION_NO_DATA`、`max_seconds=480`、`case_order=["N1","N2","N3","N4"]`、`case_limits=[90,45,75,45]`、`require_ac=True`、`max_attempts=1`、`real_training_authorized=False`。

函数签名及返回契约不变：

- `_protocol_path(root, path)`：从 root 与 NATIVE_PROTOCOL_RELATIVE 派生唯一协议路径，其他解析后位置拒绝。
- `_case_output(root, case_dir, case_id)`：从 root、NATIVE_OUTPUT_RELATIVE、cases、case_id 派生，继续拒绝未知 case 或不匹配位置。
- `_worker_packet(packet_path, handles)`：仍先验继承能力和执行来源，再从 FIXED 输出派生 packet 前缀；精确五段结构及“case与job一致”不变，不新增路径输入。
- `NativeServices.verify_registration(root, path, protocol, approval, check)`：远端协议路径改为 NATIVE_PROTOCOL_RELATIVE；远端 head/commit/tree/blob/实际内容及两次原始源码校验全部保留。
- `_validate_protocol_fields`、`validate_native_protocol`、`_check_approval`：除引用的新 FIXED 身份外，不改字段集合、权限逻辑、原始哈希检查或已存在输出拒绝策略。

不扩大 NON_NATIVE_IDS 或更改权限策略。旧运行许可不得迁移使用；本轮及未来开发许可不能被填为真实原生许可，后续登记仍须从真实人类答复单独核验。合成夹具的假授权仅存在于新 scratch 内。

## 4. 父子与登记链一致性

collector、controller、worker 必须落在同一固定 v3 输出根。同一 root 下保留旧 v2 原件不应妨碍新 v3 的纯校验；但 v2 路径不得被当作 v3 输入。每个入口继续保留当前来源、继承句柄/nonce、RUNNING claim、protocol_digest 与登记提交绑定。

本地固定协议路径、远端 tree 键、远端 contents 请求必须一致为 v3；禁止在远端 v3 缺失时回退 v2。同一原始源码哈希错误、规范化 Git blob 错误或协议内容错误仍必须拒收，不因为命名升级而重新定义“匹配”。

新原生输出存在即拒绝，不自动选新后缀、不续跑、不清理腾空。现有累计期限、父监督、ACK/DONE、N4故障证据、guard/句柄清理、错误首因与脱敏逻辑均不改。

## 5. 未来有限合成验收设计

仅提出以下新固定位置，当前不创建：

- 合成根：`outputs/msb_native_namespace_v3_synthetic_v1`。
- scratch：`.superpowers/sdd/2026-10-25-msb-native-namespace-v3/test-tmp`。
- 阶段仍为 collector-red、collector-green、final；每阶段最多3次独占尝试，含失败，保留全部原件；不自动重试、不扩模块。具体实施仍须新授权。

原生进程、线程启动/join、时钟、电源、网络、runtime 查询全部模拟。沿用原十模块、通过谓词、bootstrap 与所有守卫；原两项 scratch 合成张量例外不扩大，不读真实输入。原验收器除 output/scratch 两处字面路径外不得变化。

新增类 `NativeNamespaceV3Tests`，固定8个方法 N01-N08，详见配套计划。保留既有116个 driver 完整方法名和277个 final 完整方法名，未来应为124和285；旧名称含 v2 的历史方法也不改名，只更新其正向固定身份预期。

RED 时先保留旧 v2 夹具，仅新增独立的 v3 测试夹具；合成输出路径的两条旧预期可提前更新。N01/N03/N05/N06须体现真实缺行为断言，不能以导入错误或名称缺失 error 冒充。GREEN 才将旧正向路径夹具/预期局部迁移为 v3，旧负向 v1/v2 用例仍保留；新增8个方法及其 helper 在RED后冻结。旧测试不是任意可改，须列出每个命名空间字面值改动并审查其断言未弱化。

## 6. 接受与失败标准

全部同时满足才可记录“作者合成验收通过、独立复查待做”：

1. 三文件范围、固定值和父子/远端路径接线符合规格；无其他监督行为变化。
2. RED提供指定缺行为证据；GREEN与final零失败、错误、跳过、预期失败、意外成功和未模拟边界调用。
3. 完整名集合精确等于原集合加固定8项，不以数量替代断言；来源快照、summary、日志、入口 stdout/stderr 与所有失败尝试保留。
4. 历史保护闭包、旧协议/输出、既有267及277合成原件、六张预存脏图和两处正式报告不变。
5. 不生成执行协议、不开原生进程、不读科学数据；代码和合成证据实际远端回读一致。

发现第四源码文件、路径回退、原件覆盖、权限重用、守卫放宽、超出尝试上限或未模拟调用即停止并记录。不能为使测试通过改网络/公式，也不能把证据不足当作通过。

## 7. 当前交付与后续门槛

本轮只交付规格、计划、静态文档自检和保护证据，不派 builder/reviewer。436条保护记录保留重复项，仅允许README追加说明；没有执行合成、原生或科学实验。

审核规格与计划后，开发/有限合成验收、独立复查、可执行协议登记、单次原生验收及真实机制实验分别另批。新命名空间的代码尚未实现，原先APPROVE只适用于旧受审源码，不能提前覆盖未来变化。

工程适配不构成网络或理论创新。它仅解除实验准备障碍；报告和论文的创新强度最终取决于MSB-RRE的间隔预算、角色证据非等价性及强对照收益能否得到科学验证，当前不得宣称已验证。
