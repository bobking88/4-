# IC-RFA-B v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 实现恒等中心化有界角色适配，并用可证伪的导数验收和同修复强对照区分结构修复、通用保护与角色净纠错贡献。

**Architecture:** 保留冻结 CNN 特征与概率锚点，在三个瓶颈分支和共享读出上使用固定初始函数参照。原始 h 上的参照与适配后输入分离，角色/均匀路由、参数化和正则化中心独立控制；先合成验收，再单独申请 fit-only 机制探查。

**Tech Stack:** 现有 Windows / PowerShell、`.venv-training`、Python、PyTorch、NumPy、scikit-learn、threadpoolctl、unittest、JSON/CSV、Python matplotlib；不升级环境或新增框架。

**Spec:** [IC-RFA-B v1 书面规格](../specs/2026-10-12-identity-centered-role-adaptation-design.md)，提交 `dfee7b592e274bb78fd4d4a3d8a1c3c283e1419e`。该提交中的待审核文字保留为历史快照；认可事件以本计划 Material Passport 为准。

## Global Constraints

- 类别顺序 T/Ti/G/M，即 target_mineral / ti_bearing_negative / gangue_negative / metallic_hard_negative；B 主线，C 阶段条件化、真实品位/回收率与送检经济成本不进入本轮。
- 冻结 CNN 的 H1280、E18、矛盾量及后验；每内层训练折拟合 PCA64、标准化 E23/h64 和 T0。标准差下限 `1e-8`、full SVD、规范载荷符号、不 whitening。
- 首个真实阶段只能加载规格列明的 fit_raw 和既有 inner_assignment；340 唯一 fit 分组，T=66/Ti=130/G=97/M=47。不加载双子集 cache、stop、原图或其他分组。
- 分支 `64 -> 4 -> 64`，上投影无 bias；C 分支为 `(A-A0)/16`，`epsilon_f=1`。读出 `87 -> 16 -> 4`，参照输入原始 h/e，不是适配后输入。
- C 核心3024可训练+3024函数参照标量；HC为1476+1476，FC为3132+3132；Z 的 SP 快照只用于正则，不伪造前向参照。
- 种子 `20261002/20261003/20261004`；head 使用 seed+100、分支 seed+200+j。一次 Linear 构造、c2 置零、深复制，不另行重初始化。
- C 的 U/W2 非零；Z 把 U/W2/c2 按旧规则置零。Z/C 是参数化整块比较，不是纯初始化因果对比。
- 角色路由 `g_j=4*pbar_T*pbar_j=b*w_j`，普通路由 `g_j=b/3`；预算 `alpha_max=beta_max=.5`，不得增加门值下限、可学习预算或另外的二分类阈值。
- NLL不加类权重；pair三对各1/3、系数 `.10`（noaux为0）。lambda=`.001`，O惩罚θ，S惩罚θ−θ0；包括全部当前权重及偏置，不包括固定参照。
- CPU float64、全批次 Adam lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=0；固定单线程。无调度、BatchNorm、Dropout、增强、后处理温度或 STE。
- 所有数值在 log-space；输入非有限、实体概率下溢、错身份、缺类和非法范围即失败，不静默裁剪或重试。
- 拟议真实阶段12单元×3既有内折×3种子=108轨迹，另最多3次温度拟合；每轨迹最多400更新，检查点0/1/2/5/20/50/100/200/400，100为主要描述性点，不选模、不全fit重训。
- 拟议累计上限7200秒，包括准备、训练、梯度/Jacobian/核诊断和导出。不是已获准预算；旧392拟合和6小时授权不能复用。
- 初始等价绝对容忍度1e-12；界容忍度 `1e-10+1e-10*abs(bound)`；有限差分步长1e-6、容忍度 `1e-7+1e-5*abs(derivative)`；构造样例Jacobian范数>1e-8，不对任意真实数据承诺该下界。
- 正式报告只读，SHA256 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`；保留旧负结果和已登记源文件，不创建第二正式版本。

## Review Focus

1. 当前/参照值相同但共享可写存储：更新当前不能改变参照；保存重载仍保持无别名（Task2）。
2. 参照错误接收适配后输入，或 detach 产生伪梯度：有限差分和错误构造负例必须揭示，不能只看 autograd 非零（Task3）。
3. 验证数值、同形状重排行或旧全fit预处理混入：训练状态只能依赖该内折训练行，所有行身份严格对齐（Task1）。
4. 检查点混淆、正则梯度当数据学习、静默挑选最佳轮次：保存完整梯度向量与完成更新数，100主点固定（Tasks4、6）。
5. 超时、目录重用、监督/电源失败后继续或重放训练：终止留痕、释放防待机请求、无续跑和重试（Task5）。

## Material Passport

- 日期：2026-10-12；模式：writing-plans / academic-research-suite experiment-agent plan。
- 人类确认原文：“确认上方的规格审核选项”；所确认的问题限定为“认可规格，仅编写实施计划”。因此规格认可，**本轮只写计划**。
- 状态：`CODE_SYNTHETIC_ACCEPTANCE_COMPLETE_REAL_UNAUTHORIZED`。用户于2026-10-12明确批准“认可计划，批准原生代码与合成验收”；只执行Tasks1--7，Task8和正式报告修改未获准。
- 2026-10-13 授权更新：用户回答“批准协议编写与登记（推荐）”，仅允许Task8的预登记部分。形成[新机器协议](../../experiment_protocols/ic_role_mechanism_v1.json)和[中文协议](../../experiment_protocols/ic_role_mechanism_v1.md)；运行、真实数据读取和正式报告修改仍未获准，Task8整体不标记完成。
- 随后单次运行授权：用户在call_d12ae77d482e46f28498bfaed5810de7明确回答“批准一次真实机制实验”。当前状态为`REAL_MECHANISM_ONCE_APPROVED_PENDING_EXECUTION`；仅按已登记340fit/108轨迹/3温度/7200秒范围执行Task8。正式报告、stop与新数据仍未获准；[授权记录](../../experiment_records/2026-10-13_ic_role_mechanism_authorization.md)固定最终验收的剩余预算监督。
- Verification Status：`SYNTHETIC_VERIFIED_REAL_UNVERIFIED`。最终60项新测试、33项旧纯合成回归通过；独立代码审核四项重要问题已一次TDD修复。一次原生计时保留其f933153源码身份，不重跑；实际角色效果及论文新颖性尚未确认。
- 旧负结果基线：`d34d853a0b2f6352f6b975cede91e68560057f83`；规格快照：`dfee7b592e274bb78fd4d4a3d8a1c3c283e1419e`。
- 当前没有独立子代理工具；建议原生逐任务实施，不能承诺尚不存在的独立评审。作者侧自审不是第三方数学验证。

## 文件与接口边界

以下文件已在获准的Tasks1--7实施中创建。另增一次性合成计时脚本`check_ic_role_implementation.py`，判断与证据见实施验收记录；历史代码、协议、结果及正式报告不改。

| 拟创建文件 | 责任 | 对应测试 |
| --- | --- | --- |
| `scripts/ic_role_data.py` | fit-only载入、既有分配及训练折预处理包装 | `tests/test_ic_role_data.py` |
| `scripts/ic_role_model.py` | 12单元、冻结参照、正则和概率输出 | `tests/test_ic_role_model.py` |
| `scripts/ic_role_audit.py` | P1/P2/P3、真实有限差分、同权重诊断 | `tests/test_ic_role_audit.py` |
| `scripts/ic_role_training.py` | 固定轨迹与完整梯度、无选模 | `tests/test_ic_role_training.py` |
| `scripts/run_ic_role_mechanism.py` | 合成/获准真实入口、范围与监督状态机 | `tests/test_run_ic_role_mechanism.py` |
| `scripts/analyze_ic_role_mechanism.py` | 推理重放、OOF配对计数与机制报告 | `tests/test_analyze_ic_role_mechanism.py` |
| `scripts/generate_ic_role_figure.py` | 公式/尺寸一致的网络结构图 | `tests/test_generate_ic_role_figure.py` |

仅复用既有公共纯函数：`lc_role_data.fit_preprocessing/transform_subset`、`lc_role_adapter.anchor_geometry/bounded_log_probabilities/pair_loss`、`analyze_lc_role_adaptation.summarize_predictions`。不调用旧 `load_cache_subset`、`select_oof`、`evaluate_locked`、`verify_delivery` 或旧训练入口。

安全复用限定为 `lc_execution_safety.WindowsAwakeGuard/stage_elapsed_seconds` 和 `run_lc_role_adaptation.run_supervised_job(job, protocol, remaining_seconds)`；只传入新入口明确构造的子进程命令，不把旧 fit/evaluate stage 改名重跑。

共同数据结构沿用 dict + CPU tensor：

- RawFit：`p[n,4]`、H[n,1280]、evidence[n,18]、contradiction[n,1]、labels[n]、records、hashes；浮点float64，labels long。
- Batch：h[n,64]、e[n,23]、log_anchor[n,4]、labels、records；输入上游冻结，无优化依赖。
- Forward：logq[n,4]、scores[n,4]、g[n,3]、b[n,1]、delta[n,3,64]、adapted_h[n,64]、u[n,1]、v[n,3]、alpha/beta[n,1]；可选aux_logq[n,3,4]。推理接口不接收labels。
- 参数布局：按 `named_parameters` 的固定名字顺序记录名字/shape/offset/numel；梯度展平只含requires_grad=True项，明确单位与phase。
- role参数为真实类别ID 1/2/3，对应分支索引role−1；auxiliary视图索引同为role−1。HC/FC无分支的Jacobian/核诊断标记NOT_APPLICABLE，不伪造零矩阵或有效秩。
- 合成数据由测试内存构造 `make_fixture(n=120, seed=7)`，四类均衡、唯一行/group、H1280/E18/正p/有限矛盾量；不读取正式tensor或图片。真实入口没有缩减行数或放宽SHA的生产开关。
- 下列命令中的 `python` 指工作区 `.venv-training\Scripts\python.exe`；在当前工作树根目录运行，不静默换Python或升级PyTorch。

## Task 1: 严格 fit-only 数据包装

执行记录：2026-10-12，8项新测试RED→GREEN，旧数据模块6项合成回归通过；最终合并14项通过（0.475秒测试时间），提交 `d9baa2b598271decef5abdfdd8726127570afc96`。未加载任何真实fit/stop包。Steps1--5完成；详见本计划独立执行ledger与测试日志。

**Files:** Create `scripts/ic_role_data.py`; Test `tests/test_ic_role_data.py`。

**Interfaces:**
- `load_fit_bundle(root: Path) -> tuple[dict,list[dict]]` 返回RawFit与既有分配；生产输入路径及SHA固定取自规格第3节。
- `validate_assignment(raw: dict, assignment: list[dict]) -> list[dict]` 检查逐行image/group/class/fold身份，不生成新划分。
- `prepare_inner_fold(raw: dict, assignment: list[dict], fold: int) -> dict` 返回state/train/validation与其训练/验证索引。

- [x] **Step1 写失败测试：** `test_loader_never_opens_stop_or_combined_cache`、`test_assignment_reorder_duplicate_missing_class_rejected`、`test_validation_changes_do_not_change_preprocessing`、`test_fullfit_state_cannot_be_reused`。用mock捕获打开路径；生产路径仅可为列明两个输入，假blob错误哈希先于反序列化失败。训练状态train_ids不得含验证ID；改验证数值/标签后状态逐元素不变。
- [x] **Step2 红灯：** `python -m unittest discover -s tests -p test_ic_role_data.py -v`；预期新模块缺失或断言失败，不接受fixture/依赖错误冒充红灯。
- [x] **Step3 实现接口：** 对 `fit_raw.pt` 的SHA `d1edbe1f87977ce0a91639bc8cb455708d4c2e17d53340c48d11d2e8b2a568b1`、assignment SHA `e603ba0b603bc903f82649dd19edaebb69f7e171be0dec2092d69048c5469927` 先核验，后 `torch.load(weights_only=True,map_location="cpu")` 只载入该fit包。assignment的JSON根对象为dict，取其 `rows` 列表，再校验有序的image_id/split_group_id/class_id/inner_fold字段。验证340行/分组、四类数量、形状、有限性及record身份；预处理公共纯函数只接该折训练索引。现有assignment原样沿用，不调用make_inner_assignment。首版统一重拟合3个内折状态，不读取已有fullfit状态；最多3次T0标定。
- [x] **Step4 绿灯与回归：** 上述新测试全部通过，既有 `test_lc_role_data.py` 合成测试通过；相关旧测试先静态确认无真实数据读取，不能运行可能读取stop的全库任务。保存/载入训练状态、变批量transform一致。
- [x] **Step5 提交：** 仅模块/对应测试；`feat: add isolated fit-only identity-centered data contracts`。

## Task 2: 中心化网络及12单元

执行记录：2026-10-12，先8项RED，再实现；补查角色重排/别名/随机流后11项新模型测试通过（警告作错误），旧适配器8项合成回归通过。最终IC数据+模型19项通过（0.705秒测试时间），提交 `58f03941234e01e6c25e3bb9d90d8a06afd2d0fc`。仅基础结构验收，不是Task3理论验收或真实效果。

**Files:** Create `scripts/ic_role_model.py`; Test `tests/test_ic_role_model.py`。

**Interfaces:**
- `unit_spec(unit: str) -> dict` 返回parameterization/penalty/routing/width/pair_weight等固定配置；只允许规格第8节12个名称。
- `ICRoleModel(unit: str, *, seed: int)` 为nn.Module；`forward(h, e, log_anchor, *, auxiliary: bool=False, adaptation: bool=True) -> dict`。
- `reference_identity(model: ICRoleModel) -> dict` 返回版本、初始化hash、当前/函数参照/正则快照计数及无别名证据。
- `objective(model: ICRoleModel, result: dict, labels: Tensor) -> dict` 返回 `nll/pair/data/regularization/total`，不使用model.parameters无差别惩罚参照。

- [x] **Step1 写失败测试：** `test_all_units_identity_parameter_and_storage_counts`、`test_reference_is_frozen_unaliased_and_reload_stable`、`test_same_seed_controls_have_same_initial_current_parameters`、`test_pair_loss_paths_and_regularization_centers`、`test_numeric_underflow_and_invalid_inputs_fail`。C核心3024/3024、HC1476/1476、FC3132/3132；Z各3024当前，不伪造函数参照。C主/辅初始logq等于anchor，atol1e-12；更新当前后参照hash不变且无grad。O初始正则可非零，S初始为0；noaux系数0。
- [x] **Step2 红灯：** `python -m unittest discover -s tests -p test_ic_role_model.py -v`。
- [x] **Step3 实现接口：** 按规格式(1)--(9)和初始化表。C差值必须在外层tanh输出相减后除16，Z旧差值分支除sqrt64；固定读出参照输入h/e，ref权重深复制requires_grad=False。S正则复用同身份固定参数，Z-S只另存初始参数buffer。HC/FC在主输出计算三对pair，C/Z在三个辅助视图计算。复用log-space公开概率函数但强制bounded=True；同样本output不依赖batch。
- [x] **Step4 绿灯：** 新测试及既有 `test_lc_role_adapter.py` 合成回归通过；单独/分批、保存重载和同时角色重排等价。逐项核对非零U/W2、fan-in范围和构造流，环境版本不符即失败，不事后重初始化。
- [x] **Step5 提交：** `feat: add identity-centered bounded adapters and matched controls`。

## Task 3: 三个命题的数值与反例验收

**Files:** Create `scripts/ic_role_audit.py`; Test `tests/test_ic_role_audit.py`。

**Interfaces:**
- `audit_bounds(model: ICRoleModel, batch: dict, result: dict) -> dict` 返回逐行界、实际漂移、最大残差与违规身份。
- `scalar_branch_jacobian(model: ICRoleModel, batch: dict, role: int, view: str) -> dict` 返回log(qT/qj)对该分支参数的Jacobian、layout和输入行序；view仅main/aux。
- `audit_initial_kernel(model: ICRoleModel, batch: dict, role: int) -> dict` 返回Jmain/Jaux、Kmain/Kaux、G、谱和恒等残差。
- `audit_directional_derivative(model: ICRoleModel, batch: dict, role: int, parameter: str, direction: Tensor) -> dict` 固定step1e-6，不优化模型，扰动后精确恢复状态。

- [x] **Step1 写失败测试：** `test_identity_has_nonzero_constructed_branch_jacobian`、`test_wrong_reference_and_detach_negative_fixture`、`test_zero_u_and_zero_w2_have_distinct_blocking_effects`、`test_single_row_bound_violation_is_not_averaged_away`、`test_gram_congruence_and_duplicate_input_counterexample`、`test_same_weight_off_is_not_retrained_control`。
  非退化fixture只在测试中明确设down=.03、up=.04、读出第一层h块=.02；读出第二层T行=.10、其余行=-.10，偏置0，h全1、e全0，p均匀，再复制初始参照。检查U/V方向Jacobians>1e-8及自动求导/固定有限差分一致；不筛选随机种子，不声称真实数据梯度下界。
- [x] **Step2 红灯：** `python -m unittest discover -s tests -p test_ic_role_audit.py -v`。
- [x] **Step3 实现P1/P2/P3：** 对冻结原始输入上的正确参照核对真导数。错误ref在htilde上接收输入时，只改适配参数的链式导数相消；加detach负例的autograd须与值函数差分不一致。P2记录主b²/辅b证书、单类及成对log-odds漂移、q_off保护覆盖，不能只由两个单类界推margin η。P3仅共同零适配点/标量分支块使用Kmain=G Kaux G；训练后不得套此恒等。用实际谱范数L_F，不引用旧模型常数。
- [x] **Step4 绿灯及合成证书：** 固定seed20261006的至少5000行输入，包含极端正锚点、近并列、预算0和重复输入；全部最大残差入档。重复输入导致奇异核是预期反例；构造正定条件不满足不报“可学习性保证”。调用旧通用输出包络仅限接口/定义完全相符的纯函数，不读旧真实汇总。
- [x] **Step5 提交：** `test: certify centered derivatives perturbation limits and initial kernels`。

## Task 4: 固定轨迹与完整梯度日志

**Files:** Create `scripts/ic_role_training.py`; Test `tests/test_ic_role_training.py`。

**Interfaces:**
- `gradient_components(model: ICRoleModel, result: dict, labels: Tensor) -> dict` 返回按layout对齐的data/regularization/total三个完整向量及每块范数。
- `fit_trajectory(train: dict, validation: dict, unit: str, seed: int, *, max_updates: int=400) -> dict` 返回checkpoints/history/gradient_vectors/identities/costs，不返回winner。
- `build_fit_matrix() -> list[dict]` 生成固定12单元×3折×3种子的108唯一任务，固定排列为规格表顺序、fold0/1/2、seed升序。

- [x] **Step1 写失败测试：** `test_gradients_are_vectors_and_sum_before_update`、`test_checkpoint_phase_and_reference_hashes`、`test_validation_does_not_choose_epoch_or_change_trajectory`、`test_fixed_matrix_has_108_without_refits`、`test_failure_returns_no_promoted_partial_result`。验证first5 pre_update_index=0..4，gradient_total=data+reg在容忍度内；改变验证标签不影响参数轨迹。0检查点无更新，100对应恰好100次更新；不能以第99步状态替代。
- [x] **Step2 红灯：** `python -m unittest discover -s tests -p test_ic_role_training.py -v`。
- [x] **Step3 实现接口：** 每一步用完整batch计算objective及Adam；初始采集仅train的3个标量分支Jacobians/Gram。前五步保留完整梯度向量，其他指定点保留梯度范数、参数状态、参照距离、NLL/pair/惩罚/总损失和分支范数/相似度。关闭单支和全适配只作同状态推理，不再训练。记录train/validation row身份。不存在选择、early stopping、scheduler、最佳种子、fullfit refit或optimization replay接口。
- [x] **Step4 绿灯及合成轨迹：** 对全部12单元的短合成轨迹检查日志字段、保存重载、有效正则中心和失败行为；构造不同验证标签不能改变同种子更新。保留400截断语义，不能让小合成成功被解释为真实收敛。
- [x] **Step5 提交：** `feat: add fixed centered-adaptation trajectories and gradient provenance`。

## Task 5: 有限执行器与授权隔离

**Files:** Create `scripts/run_ic_role_mechanism.py`; Test `tests/test_run_ic_role_mechanism.py`。

**Interfaces:**
- `validate_protocol(root: Path, protocol: dict, stage: str) -> None` 核对真实/合成类型、输入/源码/审批身份、预算与路径；不推断旧授权。
- `run_stage(root: Path, protocol_path: Path | None, output: Path, stage: str) -> dict` stage仅synthetic/fit-mechanism；真实必须单独已批准并已登记的协议。
- `worker(packet_path: Path) -> dict` 只接受新入口的prepare/trajectory/analysis packet；无evaluate-stop/refit/replay-training类型。

- [x] **Step1 写失败测试：** `test_draft_or_consumed_protocol_cannot_run_real`、`test_existing_output_and_path_escape_rejected`、`test_stop_or_combined_cache_path_rejected`、`test_timeout_interrupt_release_fail_closed`、`test_108_count_three_temperatures_and_whole_stage_budget`。所有真实路径用mock，短超时仅临时合成子进程；不再自动做360秒空闲验收。
- [x] **Step2 红灯：** `python -m unittest discover -s tests -p test_run_ic_role_mechanism.py -v`。
- [x] **Step3 实现新状态机：** DRAFT不运行真实；新独占输出依次PREPARING→RUNNING→MECHANISM_COMPLETE，另有FAILED/TIMED_OUT/SUPERVISION_INTERRUPTED/POWER_REQUEST_FAILED/POWER_RELEASE_FAILED不可续跑。使用既有run_supervised_job与WindowsAwakeGuard；heartbeat15秒、每job预算min(剩余累计预算,7200)，全部preprocess/诊断/export包含在累计计时，父进程监督不中断。输入范围在任何真实文件打开前校验；SHA在反序列化前校验。要求接电，不改永久电源方案、不主动休眠，worker退出释放请求、失败停止余下job。
- [x] **Step4 绿灯与合成全链路：** 新入口只接测试生成内存/临时合成packet，模拟108任务和≤3温度计数、失败后无下一任务。与既有 `test_lc_execution_safety.py` 合成安全回归一起通过；不将旧真实fit或stop用于速度估计。
- [x] **Step5 提交：** `feat: gate centered mechanism execution and reuse fail-closed supervision`。

## Task 6: 推理重放与角色增量分析

**Files:** Create `scripts/analyze_ic_role_mechanism.py`; Test `tests/test_analyze_ic_role_mechanism.py`。

**Interfaces:**
- `collect_oof(artifacts: list[dict], assignment: list[dict], *, checkpoint: int=100) -> dict` 返回每unit/seed pooled逐行logq/labels/records与T0，不进行训练或排名选模。
- `compare_units(left: dict, right: dict) -> dict` 返回配对新增/损失、目标净正确、Ti/M误入计数/分母、NLL差、完整混淆矩阵与逐行变化。
- `verify_mechanism_delivery(root: Path, output: Path) -> dict` 只重放已有108轨迹checkpoint的前向推理，核对hash/界/计数/参照身份，不优化。
- `write_mechanism_report(evidence: dict, output: Path) -> dict` 输出描述性Markdown/CSV/JSON，不修改正式报告。

- [x] **Step1 写失败测试：** `test_oof_missing_duplicate_or_reordered_rows_rejected`、`test_pooled_metrics_not_unweighted_fold_average`、`test_100_primary_no_best_epoch_selection`、`test_paired_new_lost_counts_and_false_target_denominators`、`test_inference_verifier_cannot_optimize_or_open_stop`。不等长fold、标签计数和人工指定小混淆表须能手算验证；三个seed不得充当独立样本t检验。
- [x] **Step2 红灯：** `python -m unittest discover -s tests -p test_analyze_ic_role_mechanism.py -v`。
- [x] **Step3 实现分析：** 复用summarize_predictions的四类argmax、Macro F1、Brier、15桶ECE和每类原始分母；OOF每行每unit/seed恰好一次。核心先比较C-S-R/C-S-U，再检查noaux、HC/FC及Z/O/S；不以完整模型对弱旧模型比较宣称角色收益。另报梯度、kernel数值秩阈值1e-10、分支移除变化和参照/缓存/无缓存计算成本。verify只载入已保存的各内折state后调用transform_subset，不能调用prepare_inner_fold/fit_preprocessing或重做PCA/温度拟合。
- [x] **Step4 绿灯与合成重放：** replay输出/hash一致、单行损坏能够失败；checkpoint0为锚点，训练后off不强称锚点。验证报告模板有“仅fit探索”“固定400预算不是收敛”“无独立确认”“中心化/L2-SP/Gram同余已有前例”等限制文字。
- [x] **Step5 提交：** `feat: report paired role gains and inference-only mechanism evidence`。

## Task 7: 结构图与一次合成集成交付

**Files:** Create `scripts/generate_ic_role_figure.py`; Test `tests/test_generate_ic_role_figure.py`; Outputs（批准后）`outputs/paper_figures_ic_rfa_v1/`、`outputs/ic_role_implementation_qa_v1/`；Modify README导航。

**Interfaces:** `render_architecture(output: Path) -> dict` 返回SVG/PDF/PNG路径、尺寸和元数据；从unit_spec读取实际维数，不读真实数据。

- [x] **Step1 写失败测试：** `test_diagram_matches_formula_inputs_and_dimensions`、`test_export_has_all_nodes_and_no_clipping`。应有H1280、h64、E23、64-4-64、87-16-4、A−A0除16、原始h/e参照箭头、训练辅助虚线、3024+3024，不得把固定参照画成共享可训练模块。
- [x] **Step2 红灯：** `python -m unittest discover -s tests -p test_generate_ic_role_figure.py -v`。
- [x] **Step3 用Python生成图：** 执行前读取nature-figure技能；输出可编辑SVG、矢量PDF、300dpi PNG及caption.md；绘图只读架构定义。图题标注“结构/机制验收，不表示真实效果”，不生成伪结果曲线。
- [x] **Step4 一次集成验收：** 新7个测试文件全部通过，静态确认的旧纯合成相关回归通过；短合成runner验证全部12单元、保存重载和安全失败。采用合成最大行数340、训练步数20、仅一个seed作计时上界估计（不是正式实验缩减版）；只运行预定合成轨迹，不扩规模或重复耗时试验寻找更快记录。估算取 `1.5*[3*t_prepare_max+108*(t_initial_kernel_max+400*t_update_max+9*t_checkpoint_max+t_artifact_max)+t_verify_export_max]`，各t均为这次合成运行测得的最大对应耗时，不能用最小/平均单元掩盖慢控制；固定20步计时中缺失的检查点成本用同次保存重载计时，verify/export若未覆盖则估算证据不足。7200秒包含全部成本；估算超限或监督证据不足则停在未就绪，不悄悄减对照。查看所有图PNG/PDF预览，修复布局，不嵌入正式报告。
- [x] **Step5 提交和交付：** 只提交代码/测试、轻量合成证据和结构图；原始tensor/权重不上传。输出实施验收记录、完整测试数/耗时/最大数学残差、数据访问审计、参照成本、合成计时估算和GitHub提交。即使通过，也只能称“实现符合规格”。

## Task 8: 另行登记真实机制协议与研究决策

本任务不在未来仅代码/合成审批范围中，须独立获得真实实验批准。

**Future Files:** 新独立协议与输出目录由后续登记时确定；本轮不创建批准协议或真实结果。不会续跑lc_role_adaptation_v1/v2。

**Interfaces:** 使用Task5的获准fit-mechanism入口和Task6的推理验证/分析，无新增模型或网格。

- [x] **Step1 前置审核：** 1b91d0e已在GitHub登记，随后取得明确单次批准；Tasks1--7验收及34源码/环境/输入字节身份核对通过。授权envelope重新登记后才执行；旧计时新增校验成本不确定性明确，累计7200秒与固定矩阵不变。
- [ ] **Step2 一次执行：** 只载入fit-only包和既有assignment；3次内折T0，108轨迹，禁止读取stop、全fit重训、扩网格、训练重放和失败自动重试。结束确认所有子进程终止和防待机释放。
- [ ] **Step3 一次推理重放：** 固定100主点以及其他轨迹描述，核对输入/hash/参照/全部理论界、实际拟合计数、累计时间和失败。合法的零梯度、奇异核、没有角色增量均保留，不换数据/种子。
- [ ] **Step4 审核研究结论：** 同修复C-S-U为首要因果边界；查看三seed方向一致的角色增量、目标新增/损失、Ti/M误入、宽头比较、noaux及真实分支作用。400未收敛或只改善NLL却损失目标时不能宣称选矿收益。若没有净角色机制证据，保留负结果并停止这条候选晋级。
- [ ] **Step5 后续单独决定：** 有探索证据也只支持另写完整效果协议，不把已暴露stop当独立测试。P/E/无界强对照、来源外/真正未暴露确认、独立数学审核与报告整合需后续单独范围；本计划不自动执行或承诺正结果。

## 理论命题到证据的映射

| 理论/方法问题 | 实施证据 | 效果证据及限制 |
| --- | --- | --- |
| P1 恒等函数值但参数微分可非零 | Task2身份；Task3固定方向真导数、零U/W2与错误ref反例 | Task4真实数据/正则梯度分离；只去掉特定阻断，不证明召回或普适新理论 |
| P2 双预算扰动保护 | Task3逐行主b²/辅b界、同权重off与成对margin证书 | Task6实际谱范数/覆盖率/成本；界成立不等于标签正确或更紧保护 |
| P3 路由标量分支核关系 | Task3起点Kmain=G Kaux G、重复输入奇异反例 | Task4/6真实核/方向权重与分支作用；初始核不证明全局收敛或性能 |
| 矿物角色机制的增量方法贡献 | Task2同参照/参数/初态C-S-U控制；Task6配对分析 | Task8探索与未来独立确认；中心化、SP、瓶颈、核同余本身不称首次 |

## 自审与交付边界

覆盖检查：规格1/2/11/12→本计划Passport、Task6/8；规格3→Task1；规格4/5/6/8→Task2/4；规格7/9→Task3；规格10→Task4/5/6/8。五项Review Focus分别有命名测试；各消费者使用上文定义的接口，不改旧公共模块。

授权状态已更新：Tasks1--7原生代码/预定合成验收获准，按各任务实际证据逐步标记；Task8真实fit、stop评价和报告整合仍未获准。文档检查不冒充网络验收，未执行任务不勾选完成。

用户已批准原生逐任务实施Tasks1--7；当前已完成Tasks1/2，继续执行Tasks3--7，不逐任务重复申请许可。Task8和报告修改继续单独授权。正式学术结论仍须真正效果证据和独立检查，不用自审代替。
