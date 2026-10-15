# MSB-RRE v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 原生实现间隔缩放角色证据模块及充分求解的普通强对照，通过合成验收检验数学合同，再单独申请真实机制实验；不以公式成立替代识别效果。

**Architecture:** 复用已有按折冻结h64/e23/log-anchor，三个独立rank4交互分支直接修正目标相对三类的log-odds。固定初始函数参照保证恒等起点；一次共同softmax输出四类。共享双线性、拼接MLP、线性及bias-only等对照使用明确匹配的输入、预算或损失。

**Tech Stack:** 当前Windows/PowerShell、`.venv-training` Python、PyTorch CPU float64、unittest、标准库JSON/CSV/hashlib、threadpoolctl及现有Python matplotlib；不安装或升级依赖。

**Spec:** [MSB-RRE v1规格](../specs/2026-10-14_margin_scaled_bilinear_role_evidence_design.md)，提交 `7d039733976534afec427f0f1332d8fac35ac73b`，SHA-256 `f656d2c73826deda4b850b30a94e9b78f5c3b9dabfdd873157bfcbff6681df42`。原规格保留待审历史文字，用户认可事件以下述Material Passport为准。

## Global Constraints

- 类序固定T/Ti/G/M，类ID 0/1/2/3；B为当前主线，C阶段动作、品位、回收率及送检经济代价不进入本实现。
- 输入h64、e23和四列log-anchor冻结；沿用已有三折PCA64/标准化/T0，不重拟合，不读取raw H1280/原图、stop、联合缓存或新分组。
- 每角色r=4；U为4x64、u为4、V为4x23、v为4、a为4、zeta为1。每角色361、MAIN共1083当前参数和1083固定参照标量。
- 保留规格式(1)--(21)，`R_ref=2*abs(m)+0.25`，`d=R_ref*tanh(f_current-f_initial)`；T的logit不变，其他三类减d，一次log-softmax。
- 主NLL不加类权重；pair三对各1/3，系数0.1；相对修正系数0.01、SP系数0.001/2。FIX1输出半径1但惩罚仍使用同一R_ref；NOAUX只去掉pair项。
- 三个seed为20261002/20261003/20261004；每角色随机流seed+200+(j-1)，U/V/a按规格顺序与fan-in各抽样一次，bias置零。固定参照无存储别名。同形消融复用相同初始化字节。
- SHARED-BIL采用rank12块嵌入、缩放1/sqrt4及24个可训练零混合权重；1107参数。CONCAT-MLP为87->12->3、1095参数；LIN-UNB为264参数。
- 所有生产输入/计算CPU float64；labels为CPU long，推理接口不接收labels。无BN、Dropout、调度、STE、反馈更新、隐藏epsilon/clamp或后处理温度。
- 数值非法、身份不符、缺类、实体概率下溢、溢出或参考变动即失败。浮点tanh端点饱和按闭预算记录，不能据此声称内部导数保证。
- 保留既有34文件执行源码及旧协议、负结果和用户PNG。正式DOCX两份SHA-256均保持 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。
- 本轮只编写计划；以下代码、测试、合成前向/拟合、原生监督任务、真实输入读取及图像导出均**尚未执行或授权**。

## Review Focus

1. 固定参照与当前参数存储别名、不同初始化字节或重载漏参照：必须在一步更新及save/load后仍恒等可核验（Task2）。
2. 逐图oracle可达被写成全样本可学习，或扩大预算只改善错误不损害正确：同输入冲突标签和高置信正确翻转反例必须保留（Task4）。
3. 共享双线性包含MAIN、bias-only也满足部分理论性质：必须有精确块嵌入及充分求解的凸子模型，不能只比较旧预算（Tasks2--3）。
4. 已保存prepared文件身份错位、验证标签混入优化、重拟合/读取raw作为fallback：严格白名单、train_ids及OOF身份验收（Tasks1、5、7）。
5. RUNNING未发布即派发、计时重置、失败后继续、最终验收脱离监督：状态先落盘，累计预算包括导出与最后验收，失败即终止并释放（Task6）。

## Material Passport

- 日期2026-10-14；2026-10-15执行状态 `NATIVE_SYNTHETIC_ACCEPTED_PENDING_INDEPENDENT_REVIEW`。
- 后续批准项 `call_978b15d460804fa2982cb750a1d4b4ad`：“认可计划，批准原生代码与合成验收（推荐）”；仅Tasks1--8，Task9仍未获准。
- 人类批准项 `call_53a58e64d5ee47fbaba8b7f127ec10b0`，回答“认可规格，编写实施计划（推荐）”。原问题明确仅编写计划，不开发、不做合成前向、不读真实输入、不训练、不改报告。
- 起点为规格归档提交7d039733；此前所有真实/计时/空闲验收许可已消费，本计划不恢复任何旧许可。
- 用户历次选择原生实施，拟继续原生逐任务TDD；当前不派子代理、不执行独立审查。计划作者自审不等于独立理论审核。
- 下一授权只能覆盖Tasks1--8的原生代码与合成验收；Task9预登记、真实机制实验、未暴露确认及正式报告修改各自单独授权。

## 文件、数据与接口

| 拟创建文件 | 职责 | 新测试 |
|---|---|---|
| `scripts/msb_role_data.py` | 已登记三折prepared只读载入与身份合同 | `tests/test_msb_role_data.py` |
| `scripts/msb_role_model.py` | 七个神经单元、LIN-UNB单元、预算/参照/decoder | `tests/test_msb_role_model.py` |
| `scripts/msb_role_controls.py` | 目标函数、四个凸对照及求解证书 | `tests/test_msb_role_controls.py` |
| `scripts/msb_role_audit.py` | 合成理论/梯度/重写验收、数值界 | `tests/test_msb_role_audit.py` |
| `scripts/msb_role_training.py` | 固定轨迹、75拟合矩阵、检查点与梯度来源 | `tests/test_msb_role_training.py` |
| `scripts/run_msb_role_mechanism.py` | 独占入口、授权检查、父子监督及最终验收 | `tests/test_run_msb_role_mechanism.py` |
| `scripts/analyze_msb_role_mechanism.py` | 保存结果核验、OOF配对及证伪门 | `tests/test_analyze_msb_role_mechanism.py` |
| `scripts/generate_msb_role_figure.py` | 公式/尺寸一致的独立方法图 | `tests/test_generate_msb_role_figure.py` |
| `tests/msb_synthetic_helpers.py` | 纯内存/临时目录fixture，不读项目数据 | 上述测试复用 |

不修改旧训练器或旧模型，不调用 `load_fit_bundle`、`prepare_inner_fold`、`fit_preprocessing`、`fit_global_temperature`、旧fit/evaluate入口。公共安全复用仅 `lc_execution_safety.WindowsAwakeGuard/stage_elapsed_seconds`、`run_lc_role_adaptation.run_supervised_job/terminal_status`及 `check_lc_native_environment.NativeEnvironmentAPI`；传入新入口的命令，不把旧任务重放。

共同合同沿用dict+tensor，不引入新schema框架：

- `Batch`：h[n,64]、e[n,23]、log_anchor[n,4]、labels[n]、records；前三者不requires_grad。records含image_id/split_group_id/four_class_id，拒绝projector_stop。
- `PreparedFold`：state、train、validation、train_indices、validation_indices、inner_fold；state.train_ids严格对应训练有序ID，components=64。P等旧冗余字段可保留但不作为新模型输入。
- `Forward`：logq[n,4]、scores[n,4]、m/R_ref/d[n,3]，s/c为相应单元有效值或明确`NOT_APPLICABLE`；饱和计数与unit身份。普通类偏置/REPAIR的s/c/d诊断不伪造为角色分支效果。
- `LossTerms`：nll、pair_raw、pair_weighted、relative_penalty、sp、data、regularization、total标量；data=nll+pair_weighted，regularization=relative_penalty+sp。
- `Trajectory`：unit、seed、fold、checkpoint列表、初始/最终参照哈希、参数布局、train/validation行身份与complete标志；不会根据验证表现挑轮次。
- `ConvexResult`：unit、fold、coordinates、objective、gradient、mu、gap_upper、iterations、evaluations、converged及模型状态。
- `make_prepared_fixture(n: int=340, seed: int=7) -> dict` 位于测试helper，生成合成h/e/正anchor与三个完整折。340样例使用66/130/97/47合成标签；更小模型测试不用生产载入器。fixture不是真实矿物或PCA/T0有效性证据。

以下命令的`python`指主项目 `.venv-training/Scripts/python.exe`，从现有 `.worktrees/theory-aware-report`运行。命令仅列为未来获准的执行步骤，本轮不运行。

## Task 1: prepared-only 数据边界

执行进度：2026-10-14，7项合成测试RED到GREEN，提交 `da08a764563060452e5d1b4b93fa2cad79246f60`；隔离载入/有序身份已验收，未读真实输入。以下Step1--5已完成；历史计划checkbox由任务执行记录覆盖。

**Files:** Create `scripts/msb_role_data.py`、`tests/msb_synthetic_helpers.py`; Test `tests/test_msb_role_data.py`。

**Interfaces:** `validate_prepared_fold(data: dict, fold: int) -> dict`; `validate_fold_collection(folds: list[dict]) -> dict`; `load_prepared_fold(root: Path, entry: dict, authorization: dict) -> dict`。entry含fold、relative_path、sha256、registered_train_ids_digest及registered_validation_ids_digest。

- [ ] **Step1 写失败测试：** `test_missing_or_reordered_rows_fail`、`test_train_validation_and_oof_groups_disjoint`、`test_wrong_fold_or_stop_record_fail`、`test_hash_checked_before_torch_load`、`test_no_raw_or_refit_fallback`、`test_validation_label_changes_do_not_change_input_state`。mock所有读入口及预处理函数；错误哈希时反序列化调用数0。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_msb_role_data.py -v`，预期新接口缺失造成有意义失败，不接受fixture/环境错误。
- [ ] **Step3 最小实现：** 真实数据载入器只接受 `outputs/training/ic_role_mechanism_v2/development_fold_0/runs/prepare_fold{0,1,2}/result.pt`，从独立的新协议读取登记SHA，核验后 `torch.load(weights_only=True,map_location="cpu")`。所有路径解析后必须位于当前root、禁止符号链接逃逸；该数据载入器不接受其他.pt。新模型自身保存状态的回读由Task7另受输出白名单和哈希约束，不是新数据入口。既有产物是否完整尚未在本轮检查，缺失即停止，不回读raw重新prepare。
- [ ] **Step4 绿灯：** 同一命令全部通过；三折验证合并恰好340唯一分组及66/130/97/47，训练/验证身份与train_ids一致。loader必须先验证新授权，而不是只见旧文件就读取。
- [ ] **Step5 提交：** 仅这三个新文件；`feat: add registered prepared-only MSB data contracts`。

## Task 2: 模型、固定参照及精确普通包含

执行进度：2026-10-15，9项合成测试通过，另7项数据回归通过。共享头偏置缩放故障已由新增测试RED到GREEN修复；恒等、参照无别名/重载、参数计数及共享非恒等包含得到限定验收。未读取真实输入。

**Files:** Create `scripts/msb_role_model.py`; Test `tests/test_msb_role_model.py`。

**Interfaces:** `unit_spec(unit: str) -> dict`; `decode_role(log_anchor: Tensor, d: Tensor) -> dict`; `RoleBranch.score(h: Tensor,e: Tensor) -> Tensor`; `MSBRoleModel(unit: str, *, seed: int)`；`forward(h,e,log_anchor,*,role_mask: tuple[bool,bool,bool]=(True,True,True)) -> dict`；`named_current() -> list[tuple[str,Tensor]]`; `reference_digest(model) -> str`。

- [ ] **Step1 写失败测试：** `test_initial_identity_all_neural_units`、`test_counts_and_seed_streams_exact`、`test_reference_no_alias_after_update_and_reload`、`test_matched_ablation_initial_bytes`、`test_shared_block_embedding_contains_main`、`test_flat_decoder_rewrite_equivalent`、`test_all_roles_off_returns_anchor`。MAIN1083、SHARED1107、MLP1095、LIN264，不把固定参照当可训练量。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_msb_role_model.py -v`，记录缺失接口/断言失败。
- [ ] **Step3 最小实现：** MAIN/ADD/FIX1/UNB-SCALE/NOAUX复用独立分支与同初始化；SHARED/MLP按认可规格；LIN保留87->3中心化仿射但以参数差坐标实现等价线性修正以免浮点相消。当前函数和冻结函数分别有明确state_dict名字；引用标量保存为buffers，不进入优化器。role_mask仅MAIN族推理消融使用，普通其他对照拒绝非默认mask。
- [ ] **Step4 绿灯：** 同一命令通过；恒等/重写atol=1e-12、rtol=1e-12，额外混合权重初始零但requires_grad。用非零随机参数变化测试SHARED块包含，不仅验证恒等点。
- [ ] **Step5 提交：** 仅模块和测试；`feat: add centered margin-scaled role models and matched heads`。

## Task 3: 损失及充分求解的四种凸对照

执行进度：2026-10-15，10项限定合成测试通过，四种凸对照均验收优化证书、bias-only目标等价和正ridge。最后更新后的证书漏检由回归RED到GREEN修复；未在真实输入拟合这些对照。

**Files:** Create `scripts/msb_role_controls.py`; Test `tests/test_msb_role_controls.py`。

**Interfaces:** `objective(model,result: dict,labels: Tensor) -> dict`; `make_convex_control(unit: str,batch: dict) -> object`; `control_forward(control,h,e,log_anchor) -> dict`; `convex_objective(control,coordinates: Tensor,batch: dict) -> Tensor`; `solve_convex(control,batch: dict,*,deadline_check: Callable[[],None]) -> dict`。

- [ ] **Step1 写失败测试：** `test_pair_uses_main_odds_and_original_denominators`、`test_fix1_regularizer_uses_main_reference_radius`、`test_bias_objective_is_main_submodel`、`test_bias_hessian_lower_bound_and_endpoint_rejection`、`test_lin_classwise_comp_positive_ridge`、`test_repair_features_sign_ties_counts_and_degeneracy`、`test_unsolved_control_fails_certificate`。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_msb_role_controls.py -v`，新公式/solver缺失应失败。
- [ ] **Step3 最小实现：** 按规格式(6)(7)(18)--(21)返回分项。LIN在264个参数差坐标、BIAS在3个c坐标、CLASSWISE在3个偏置、COMP在3偏置+4theta坐标求解；各自mu=.001。四个凸对照只用对应训练折，计数特征取训练组计数，不使用验证计数。
- [ ] **Step4 求解合同验收：** float64阻尼Newton，起点全零，最多200个外迭代；Cholesky解Hessian系统，不加未登记jitter。Armijo系数1e-4、初始步长1、每次乘.5、最多20候选；BIAS候选需严格abs(c)<1，非法候选不计算loss、更不clamp。每个拟合最多4201次目标评估；每次迭代/候选调用deadline_check。证书 `||grad||²/(2*.001)<=1e-8` 才converged；失败、不有限或耗尽则整机制失败，不重启/换求解器。用已知凸合成目标验收唯一解与证书；MAIN参数固定到参照且只改zeta=atanh(c)时目标一致。
- [ ] **Step5 绿灯及提交：** 上述模块全部通过；保存求解路径/certificate不是声称已在真实输入求出最优。仅两个文件；`feat: add MSB losses and certified convex controls`。

## Task 4: 理论性质、反例与梯度来源

执行进度：2026-10-15，13项限定合成测试通过，包含500个内部见证、边界有限差分、分项梯度、概率共同归一化、bias子模型及高置信正确可被损害的反例。验收不构成共同可学习、全局收敛、原创性或真实识别增益证明。

**Files:** Create `scripts/msb_role_audit.py`; Test `tests/test_msb_role_audit.py`。

**Interfaces:** `audit_forward_contract(batch,result) -> dict`; `audit_synthetic_theory(model,batch) -> dict`; `gradient_components(model,result,labels) -> dict`; `audit_control_equivalence(main,control,batch) -> dict`。

- [ ] **Step1 写失败测试：** `test_all_class_interior_witnesses`、`test_pair_boundary_bias_derivative`、`test_wrong_pair_initial_gradient_bound`、`test_root_odds_recover_correction_and_branch_cross_zero`、`test_common_normalizer_changes_other_probabilities`、`test_single_input_oracle_not_joint_fit`、`test_correct_high_confidence_can_be_harmed`、`test_mixed_hessian_not_claim_mlp_superiority`、`test_data_regularizer_gradient_decomposition`、`test_underflow_and_tanh_saturation_explicit`。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_msb_role_audit.py -v`，不能把没有模型的测试通过算验收。
- [ ] **Step3 最小实现：** 合成正/负/零间隔取[-50,-10,0,10,50]，bias见证±atanh(.5)，检查目标至少.125及非目标间至少.25。边界检查导数>=.1875；单样本pair损失导数>=.09375，恒等错pair>=.125。函数量回读及共同归一化漂移用规格界，不对绝对概率错误声称支路独立。混合Hessian仅内部score数学探针允许可微合成输入，公共推理Batch仍冻结。
- [ ] **Step4 绿灯：** atol=1e-10+1e-10*abs(bound)；中心有限差分epsilon=1e-6、导数atol=1e-7/rtol=1e-5。分别保存NLL、pair、relative、SP梯度及合计误差；同输入不同真值必须明确不可同时拟合。非法anchor零/不归一化/非有限及极端概率下溢显式失败。
- [ ] **Step5 提交：** 仅模块和测试；`test: verify MSB interior controls and preserve counterexamples`。

## Task 5: 固定矩阵、轨迹与存储合同

执行进度：2026-10-15，9项限定合成测试通过并提交 `7323dce5438b7edf59de0e97e9f08f0549e2cb58`。包括验证标签改动不影响训练状态、实际100步的角色关闭记录及空检查点边界修复；75任务只生成固定调度表，未执行真实矩阵。

**Files:** Create `scripts/msb_role_training.py`; Test `tests/test_msb_role_training.py`。

**Interfaces:** `build_fit_matrix() -> list[dict]`; `fit_trajectory(train,validation,unit,seed,*,max_updates: int=400,checkpoints: tuple=(0,1,2,5,20,50,100,200,400)) -> dict`；`fit_convex_fold(train,validation,unit,*,deadline_check) -> dict`。

- [ ] **Step1 写失败测试：** `test_63_neural_plus_12_convex_jobs`、`test_no_validation_grad_or_selection`、`test_checkpoint_is_completed_updates`、`test_reference_unchanged_all_steps`、`test_convex_not_repeated_as_three_seed_evidence`、`test_missing_or_unsolved_job_fails_complete`。生产矩阵不接受任意unit/seed/fold/步数网格。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_msb_role_training.py -v`，计数、阶段或身份错误要失败。
- [ ] **Step3 最小实现：** 七个神经单元×三折×三seed=63轨迹；全批Adam lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=0，CPU单线程/确定性。最多400步，固定主点100，不早停、不根据验证挑选。四个凸单元×三折=12拟合，确定性只解一次，不能复制成三次独立实验；LIN以零参数差坐标求解。T0使用已有锚点，新增温度拟合为0。
- [ ] **Step4 绿灯：** 测试用纯合成2步/检查点0/1/2及小凸目标，接口不得通过CLI把测试预算混入真实协议。保存每个固定点状态、train/validation logq、d/R/s/c、梯度分项及参照身份；MAIN主点另保存四次同状态角色mask(关闭各角色及全部)。每条轨迹无验证标签参与backward，数据完整性失败不返回COMPLETE。
- [ ] **Step5 提交：** 仅模块和测试；`feat: add fixed MSB trajectories and deterministic control schedule`。

## Task 6: 单次执行器及受监督最终验收

**Files:** Create `scripts/run_msb_role_mechanism.py`; Test `tests/test_run_msb_role_mechanism.py`。

**Interfaces:** `validate_protocol(root: Path,protocol: dict,stage: str) -> None`; `run_stage(root,protocol_path,output,stage) -> dict`; `worker(packet_path: Path) -> dict`; public stage仅`synthetic`及`fit-mechanism`，无独立不受监督的verify/evaluate命令。

- [ ] **Step1 写失败测试：** `test_draft_old_or_consumed_approval_opens_nothing`、`test_running_published_before_dispatch`、`test_remaining_budget_reaches_final_verification`、`test_timeout_power_release_interrupt_fail_closed`、`test_existing_output_symlink_and_invalid_worker_rejected`、`test_no_resume_stop_or_training_replay`。nativeAPI、guard、subprocess与计时均mock，避免重做真实空闲/计时验收。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_run_msb_role_mechanism.py -v`，针对旧RUNNING发布故障先有回归失败证据。
- [ ] **Step3 最小实现：** 入口先检查新MSB协议身份/授权/源码/路径，exclusive创建新目录，PREPARING后先把RUNNING原子写回，再派发任何worker。新协议不得含旧IC授权ID；真实单次授权与准备来源文件哈希都登记后才允许读三个prepared。状态与协议均不可用训练模型绕过。synthetic只构造内存/临时数据，不允许真实输入路径/授权字段。
- [ ] **Step4 绿灯：** 累计拟议7200秒从入口开始，包括载入/训练/凸证书/诊断/导出及最终推理验证；每job只传剩余时间，heartbeat15秒，per_fit上限同剩余总预算。接电、临时防待机、退出释放；任何失败只派发到失败job，不继续队列。计划最多80job=3载入+75拟合+1分析导出+1最终验证。终态有attempt=1、alive=false及父子释放证据才算执行完整；失败目录不可续跑，网络不自动改参。
- [ ] **Step5 提交：** 仅模块和测试；`feat: gate MSB stages with single-attempt supervised verification`。

## Task 7: 逐图核验、配对评价和证伪门

**Files:** Create `scripts/analyze_msb_role_mechanism.py`; Test `tests/test_analyze_msb_role_mechanism.py`。

**Interfaces:** `verify_saved_mechanism(root: Path,output: Path,protocol: dict) -> dict`; `summarize_oof(records: list[dict]) -> dict`; `paired_comparison(main: list[dict],other: list[dict]) -> dict`; `mechanism_gate(summaries: dict) -> dict`; `write_mechanism_report(evidence: dict,output: Path) -> dict`。

- [ ] **Step1 写失败测试：** `test_oof_count_row_order_and_duplicate_fail`、`test_same_state_off_returns_anchor_not_old_shared_head`、`test_nll_change_without_argmax_gain_fails_gate`、`test_bias_or_shared_equal_not_role_success`、`test_unsolved_convex_cannot_be_called_weak_winner`、`test_mixed_seed_or_false_target_tradeoff_not_promoted`、`test_tampered_state_or_prediction_fails`。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_analyze_msb_role_mechanism.py -v`，构造相同argmax但更低NLL仍必须输出未验证角色增量。
- [ ] **Step3 最小实现：** 只合并原三折验证行，不重训。生成逐seed目标P/R/F1、Ti/G/M各误入目标原分母、Macro F1/Accuracy/NLL/Brier及15等宽置信区间ECE（区间左闭右开，末区间含1）。bootstrap/p值/来源外结论不进入本开发阶段；seed标准差不当样本置信区间。训练与验证objective分开，MAIN训练值高于可靠BIAS最优需指出优化欠缺；只有低验证NLL不能证明视觉角色有效。
- [ ] **Step4 绿灯：** 主点100的探索性角色晋级门：MAIN对SHARED、MLP、BIAS、LIN、CLASSWISE、COMP及T0七个参照分别比较，三个seed每个目标净纠错至少+1/66，Ti/G/M各误入数不增加、Macro F1不下降，且所有凸证书通过。此严格描述性门不是功效设计或显著性证明；不满足即`NO_ROLE_PROMOTION`，不以400点补救。即使过门，显式乘性交互贡献还需MAIN相对ADD满足同样逐seed条件，否则不主张Hadamard是必要机制；预算/辅助项分别由FIX1、UNB-SCALE、NOAUX结果作条件性解释，不能合并成未检验的因果创新。七神经单元9点共567个checkpoint验证前向，四凸12个最终验证前向，MAIN主点四mask共36个验证前向，共615个固定最终回读；不重放训练，不读stop。预算不足导致不完整必须失败，不跳过点。
- [ ] **Step5 提交：** 仅模块和测试；`feat: audit paired MSB corrections and falsification gates`。

## Task 8: 方法图与合成交付闭环

**Files:** Create `scripts/generate_msb_role_figure.py`; Test `tests/test_generate_msb_role_figure.py`。仅另行获准后新增 `docs/experiment_records/2026-10-14_msb_role_native_synthetic_acceptance.md`。

**Interfaces:** `build_architecture_figure(output: Path,*,status_label: str) -> dict`; `figure_spec() -> dict`。图用Python/matplotlib，参数/边关系直接取unit_spec，不另写一套网络尺寸。

- [ ] **Step1 写失败测试：** `test_figure_matches_formulas_and_frozen_paths`、`test_new_figure_never_overwrites_old_assets_or_docx`、`test_exports_valid_svg_png_pdf_and_candidate_label`；检查U/V/Hadamard/固定参照/间隔预算/共同softmax齐全，无shared-readout残留。拒绝无状态标记的“已验证”图。
- [ ] **Step2 红灯：** `python -m unittest discover -s tests -p test_generate_msb_role_figure.py -v`，缺失绘图接口应失败；不能以只有文件名而没有输出内容通过。
- [ ] **Step3 最小实现：** 输出新目录 `outputs/msb_role_synthetic_v1/assets/` 的SVG/PNG/PDF；清晰区分冻结输入、当前/固定分支和训练损失。标“MSB-RRE v1：合成验收，真实效果未验证”，不嵌入正式DOCX。SVG保留可检查标签与矢量路径，300dpi PNG；在未来验收中实际查看导出图的布局/中文字体，不只查字节头。
- [ ] **Step4 获准后联合验收：** 依次运行八个上述限定unittest模块，以及仅mock控制面的新synthetic入口；不全库discover、不读取真实数据、不另开计时/空闲验收。每Task先RED再GREEN，必要修复按TDD记录；整体验收记录结果、源码闭包、保护文件哈希与未完成理论/效果项。不把计划或预期断言当已通过证据。
- [ ] **Step5 提交与审核：** 仅白名单新源码/测试/合成图/日志及README导航，排除.pt、原图、模型权重和正式DOCX。原生实现之后再单独请求一次独立只读工程/数学复查的范围，不在当前写计划许可内派评审。发现问题先登记，不擅自真实跑或改模型。

## Task 9: 后续预登记与真实实验，当前明确未获准

**Files (future only):** `docs/experiment_protocols/msb_role_mechanism_v1.json`、同名`.md`及独立登记/授权记录。

**Interfaces:** 复用Task6的严格schema；先`human_approved=false`登记源码身份、75拟合/0T0/7200秒/主点100/615最终回读及80任务上限。另一次人类明确批准后，才补授权与新独占输出路径。

- [ ] **Step1 单独获准编写协议：** 从旧v2归档元数据核对三个prepared的登记哈希及有序ID摘要；未核到原身份即停止，不猜hash、不载入张量补生成prepared、不重做PCA/T0。
- [ ] **Step2 合成验证并归档协议：** 对新schema只用fake文件/身份做拒绝旧许可与未授权运行回归；核对源码、矩阵和输出白名单。该阶段是否允许再跑合成测试须在该次授权中明确，不能沿用Task8的一次验收许可。
- [ ] **Step3 单独请求一次真实运行批准：** 明列340已暴露fit、75拟合（63神经+12凸）、0预处理拟合、7200秒累计预算、固定100主点及停止规则；不是独立测试，不读stop/原图，不修改网络/网格/报告。
- [ ] **Step4 按固定协议单次执行：** 要求接电；失败即停止、不重试、不续跑、不调参。完成后按预设证伪门报告正/负结果，保留全部对照，不挑模型或seed改名。
- [ ] **Step5 另议确认/报告：** 开发门通过也须新未暴露分组与来源身份核验及独立效果协议。更新正式报告/论文需再授权，明确区分已测基线、条件推导、合成验收、开发负结果和独立确认；当前没有任何报告写入任务。

## 作者计划自审与交接

规格16节均映射到任务：输入/身份Task1；分支/初始化/预算/结构包含Task2；损失/凸证书Task3；式(8)--(17)及反例Task4；固定开发矩阵Task5；监督/数值/隔离Task6；效果与停止Task7；公式图/交付Task8；未授权真实与报告边界Task9。

编写计划时检查了接口名、Forward/Loss/Trajectory/Convex字段的跨任务一致性；五项Review Focus各有指定失败测试。原计划checkbox保留历史状态，当前进度以各Task执行记录与原生ledger为准。数学原创性、独立复查、实际产物身份及真实效果尚未验证；上述真实计数/预算仍不是已消费运行许可。

本轮仅用标准库做文档静态检查：9任务/45未勾选步骤、链接及分隔符、规格快照SHA、63+12=75拟合/80任务/615最终回读的算术均通过；34旧源码及两份正式报告、六张用户PNG哈希未变化。确认没有创建拟议源码/测试、真实协议或输出目录。这个检查不是执行计划中的模型测试、前向或求解证书。

下一步只请求认可本计划并授权**原生代码与合成验收Tasks1--8**。若认可，不从Task9直接开始真实实验；若不认可，先修订计划，不实施。

## 2026-10-15执行补记

上段保留计划编写时的状态；此后用户批准Tasks1--8，八个模块已原生实现并完成83项限定合成联合验收。
逐任务RED/GREEN、修复和裁决见[原生合成验收记录](../../experiment_records/2026-10-14_msb_role_native_synthetic_acceptance.md)。
最终复算补充明确12次训练侧凸梯度证书核验，故为615验证前向+12训练证书检查，非615总计算；75拟合/0T0/7200秒拟议上限不变。
代码中分析与最终核验绑定既有活跃worker guard并仅消费一次；无独立不受监督的verify入口。
结构图已输出新目录并目视检查PNG及PDF预览；正式报告、六张用户PNG、34旧源码、旧IC规格及协议均未改。
独立只读复查仍需单独批准，作者自审不能替代。Task9仍未获准，未创建真实MSB协议，未读取真实输入、未训练、未更新报告。
