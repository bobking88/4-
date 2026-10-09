# LC-RFA-B v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并可否证地检验轻量角色条件特征适配，区分网络机制收益、通用输出保护与开发集偶然改进。

**Architecture:** 冻结已有 EfficientNet-B0 池化特征和后验，内层训练折独立拟合 PCA64、标准化与温度。三个角色适配器和共享读出产生有界修正，用同参数、同监督等权适配器及其他强对照验证净作用；理论检查和模型选择分别留痕。

**Tech Stack:** 现有 Windows / PowerShell、`.venv-training`、Python、PyTorch、NumPy、SciPy、scikit-learn、threadpoolctl、unittest、CSV/JSON、matplotlib；不新增框架或升级依赖。

**Spec:** [2026-10-06-lightweight-role-adaptation-design.md](../specs/2026-10-06-lightweight-role-adaptation-design.md)，来源提交 `7a2d6af4038f2c8a6e2e09f3122aef66561cee44`，SHA256 `aa9fcec06b6190fa6e56969da667c7adbcd07c5eb9ea56574031bd51dabefea0`。

## Global Constraints

- B 为当前四类矿物角色视觉代理主线；C 的选矿阶段、送检经济代价和品位/回收率验证继续后置。
- 类别顺序：`0 target_mineral`、`1 ti_bearing_negative`、`2 gangue_negative`、`3 metallic_hard_negative`。
- 仅现有 Fold 0 的 `gate_stop_projector_fit` / `projector_stop`，各 340 行；不读取图片、不更新骨干、不启用其他真实折。
- 原始后验只用 `oos_pre`；E23 为 E18 + 四列 log(p) + 矛盾量 L，是确定性重参数化，不是新增信息。
- 每内层训练折重新拟合 PCA64、列标准化、E23 标准化和 T0；总体标准差下限 `1e-8`，full SVD、规范符号、不 whitening。
- 三个 `64 -> 4 -> 64` 分支，U 无偏置、每支 516 参数；共享 `87 -> 16 -> 4`，1476 参数；完整网络 3024 参数，`epsilon_f=1`。
- 路由 `g_j=4*pbar_T*pbar_j`，`b=sum(g_j)`；逐图预算 `alpha=alpha_max*b`、`beta=beta_max*b`；不得增加可学习预算头。
- 主损失为不加权四类 NLL；角色对均权 `1/3`，辅助损失系数 `0.10`；全部可训练权重和偏置显式 L2。
- CPU float64、全批次 Adam、学习率 `0.001`、optimizer weight_decay=0；无调度、BatchNorm、Dropout、额外增强或后处理标定。
- 种子 `20261002/20261003/20261004`；lambda=`0.001/0.01/0.1`；预算对 `(0.25,0.25)/(0.5,0.5)`；检查点 `0/20/50/100/200/400`。
- 内层 `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261006)`；按 `split_group_id` 分组，不宣称控制摄影者或物理标本。
- 按 pooled OOF NLL 的三种子均值选模；相同分数依次选更小预算和、更大 lambda、更早检查点、参数元组字典序；stop 不参与。
- 最多 422 次拟合，另有 4 次 T0 标定；400 为预算上限，不是收敛证书；训练失败不自动重试或扩大范围。
- 数学容忍度 `1e-10 + 1e-10*abs(bound)`，逐图记录最大残差和违规；不得只核对均值。
- 正式报告保持 SHA256 `b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`；不新增第二正式版本。

## Review Focus

1. 行数相同但缓存行被重排：身份和逐行 feature digest 必须拒绝错配，不能仅检查形状（Task 1）。
2. 验证集被用于预处理或选模：改变内层验证数值/stop 标签不得改变已拟合状态或选择；stop 解析须晚于选择锁（Tasks 1、4、5）。
3. 极端正后验或高置信错误：log-space 避免 `1-t` 抵消；表示下溢须明确失败，不能静默裁剪，限制纠错需披露（Tasks 2、3、6）。
4. 零初始化与 L2 造成假学习信号：分别记录数据项和总梯度，不能将正则收缩当角色特征学习（Tasks 2、4）。
5. 中途失败、陈旧锁或目录已存在：保留失败记录且禁止覆盖、自动重试、部分结果晋级或读取其他折（Tasks 5、6）。

---

## Material Passport

- Origin Skill / Mode：writing-plans；academic-research-suite / experiment-agent plan。
- 日期：2026-10-06；版本：`lc_rfa_b_implementation_plan_v1`。
- 用户已认可规格，原话：“认可规格，编写实施计划”。该授权仅支持本计划，不等于实施或训练许可。
- 当前状态：`IMPLEMENTATION_IN_PROGRESS`。用户原话：“认可计划，原生实施代码与合成验收”；仅批准 Tasks 1-7，2026-10-08 继续实施。Task8 真实训练和 Task9 报告整合仍未获准。
- Verification Status：`UNVERIFIED`（计划已自审；新模型、测试、实验与结构图尚不存在）。
- 已认可规格保留提交时的原文/哈希；其中待审核状态是历史快照，新的认可事件以本计划记录为准。
- 推荐执行方式：当前会话原生逐任务实施，执行前读 executing-plans；当前工具没有独立子代理，不能承诺独立评审。后续正式结论仍需另行评审和独立确认。

## 文件边界与共同接口

以下均是**未来拟创建**，本轮不创建实现、协议或结果。各模块沿用 `scripts/` + 对应 `unittest` 的仓库模式，不改动已冻结旧模块。

| 文件 | 责任 | 对应测试 |
| --- | --- | --- |
| `scripts/lc_role_data.py` | 缓存身份、fit-only 预处理、内折分配 | `tests/test_lc_role_data.py` |
| `scripts/lc_role_adapter.py` | 适配器、各神经组、log-space 输出及损失 | `tests/test_lc_role_adapter.py` |
| `scripts/lc_role_audit.py` | 命题 A/B/C、重放与分支诊断 | `tests/test_lc_role_audit.py` |
| `scripts/lc_role_training.py` | 拟合、凸对照、OOF 选择及轨迹 | `tests/test_lc_role_training.py` |
| `scripts/run_lc_role_adaptation.py` | 范围/协议/状态机、串行监督执行 | `tests/test_run_lc_role_adaptation.py` |
| `scripts/analyze_lc_role_adaptation.py` | 指标、完整重放、筛查和结果汇总 | `tests/test_analyze_lc_role_adaptation.py` |
| `scripts/generate_lc_role_adaptation_figure.py` | 真实尺寸/模块名对应的 SVG/PDF/PNG 结构图 | `tests/test_generate_lc_role_adaptation_figure.py` |

接口统一使用现有 `dict` + CPU tensor，不引入通用实验框架：

- `RawSubset` 是 dict：`records`（有序行记录）、`p[n,4]`、`evidence[n,18]`、`contradiction[n,1]`、`H[n,1280]`、`labels[n]`、`hashes`；labels 为 long，其他 tensor 为 float64。
- `ProcessedBatch` 是 dict：`h[n,64]`、`e[n,23]`、`P[n,4]`（训练侧标准化 log(p)，仅凸 P 用）、`log_anchor[n,4]`、`labels`、`records`。
- `ForwardResult` 是 dict：`logq[n,4]`、`scores[n,4]`、`g[n,3]`、`b[n,1]`、`delta[n,3,64]`、`adapted_h[n,64]`、`alpha/beta/u[n,1]`、`v[n,3]`，训练辅助时另含 `aux_logq[n,3,4]`。
- 单元测试本地构造 `raw(n=120, seed=7)`：四类均衡、unique groups、H1280/E18/正 p/有限 L；不读取任何真实图像或 fold。真实入口仍严格要求 340 行，不设生产绕过参数。
- 在工作树根目录执行。命令中的 `python` 指 `D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe`；不得静默换环境。

## Task 1: 来源隔离与内折预处理

执行记录（2026-10-08）：6项新测试 RED -> GREEN；全套410项通过，44.030秒。步骤1-5已执行，缓存测试均为临时合成源；没有读取真实stop。

**Files:** Create `scripts/lc_role_data.py`; Test `tests/test_lc_role_data.py`。

**Interfaces:**
- `load_cache_subset(root: Path, protocol: dict, subset: str) -> dict` 返回 RawSubset；只接受两个指定名称。
- `make_inner_assignment(raw: dict) -> list[dict]` 返回有序 `image_id/split_group_id/inner_fold/class_id`。
- `fit_preprocessing(raw: dict, train_indices: Tensor) -> dict`；`transform_subset(raw: dict, state: dict, indices: Tensor | None = None) -> dict` 返回 ProcessedBatch。

- [ ] **Step 1: 写失败测试。** 断言要求如下；对真实加载使用临时文件和 mock，不读取正式数据。

```python
def test_preprocessing_uses_only_train_indices(self):
    state1 = data.fit_preprocessing(self.raw, self.train_indices)
    changed = self.change_validation_values_and_labels(self.raw)
    state2 = data.fit_preprocessing(changed, self.train_indices)
    self.assert_state_equal(state1, state2)
    self.assertEqual(state1["basis"].shape, (1280, 64))

def test_row_reorder_and_fold_one_fail(self):
    with self.assertRaises(ValueError): self.load_swapped_feature_rows()
    with self.assertRaises(ValueError): self.load_subset("fold_1")
```

另测 duplicate image ID、缓存 SHA 错、逐行 feature SHA 错、标签/路径/group 错配、分组跨内折、任一训练/验证折缺类及 PCA 不足 64 维时拒绝。常量列标准差为 `1e-8`，不输出 NaN；内折测试改标签只改验证部分。
- [ ] **Step 2: 运行红灯。** `python -m unittest discover -s tests -p test_lc_role_data.py -v`；应因新模块缺失失败，而非依赖或 fixture 错误。
- [ ] **Step 3: 实现上述接口。** 复用 `run_verifier_trust_development.load_subset`（单子集，不调用双子集 snapshot/prepare）、`frozen_visual_probe.fit_preprocessing/apply_preprocessing`、`anchor_linear_controls.fit_representation/representations`、`verifier_strong_controls.fit_global_temperature`。缓存行摘要先按原存储dtype核验，再转CPU float64；不得先转型再比原摘要。对每内折仅传其训练行；T0 使用既有 beta 范围 `[1e-6,100]` 与 80 次二分。保存训练 ID、状态和载荷符号规范；禁止读旧全-fit `preprocessing.pt` 用于 CV。
- [ ] **Step 4: 运行绿灯及旧回归。** 上述测试和 `test_anchor_linear_controls.py`、`test_verifier_strong_controls.py`、`test_frozen_visual_probe.py` 均通过。预处理重复调用和保存/载入重放一致；不将冻结专家视为已内层交叉拟合。
- [ ] **Step 5: 仅暂存该模块/测试并提交。** `feat: add fit-only role adaptation data preparation`。

## Task 2: 有效适配结构与成对监督

执行记录（2026-10-08）：8项新测试 RED -> GREEN；全套418项通过，44.393秒。步骤1-5已执行；3024参数/共享辅助视图/五步数据与总梯度/下溢失败均合成核验。

**Files:** Create `scripts/lc_role_adapter.py`; Test `tests/test_lc_role_adapter.py`。

**Interfaces:**
- `RoleResidualModel(arm: str, *, seed: int, alpha_max: float, beta_max: float)` 是 nn.Module，`forward(h: Tensor, e: Tensor, log_anchor: Tensor, *, auxiliary: bool = False) -> dict` 返回 ForwardResult；推理接口没有 labels。
- `bounded_log_probabilities(log_anchor: Tensor, scores: Tensor, b: Tensor, *, alpha_max: float, beta_max: float, bounded: bool = True) -> dict` 返回 logq/u/v/alpha/beta。
- `pair_loss(logq: Tensor, labels: Tensor) -> Tensor` 接受 `[n,3,4]`（各视图）或 `[n,4]`（F0 同输出）；`objective(model: nn.Module, result: dict, labels: Tensor, regularization: float) -> dict` 返回 `total/nll/pair/l2`。

- [ ] **Step 1: 写失败测试。**

```python
def test_parameter_counts_and_identity(self):
    expected = {"H0":1476, "F0":3132, "S0":3024,
                "S1":3024, "R0":3024, "R1":3024, "U1":3024}
    for arm, count in expected.items():
        model = self.build(arm)
        self.assertEqual(sum(p.numel() for p in model.parameters()), count)
        result = self.forward(model)
        torch.testing.assert_close(result["logq"], self.log_anchor,
                                   atol=1e-10, rtol=1e-10)
```

另测 norm(delta)<=1、等权 S1 的 g=b/3、R1 的 g=4*pbar_T*pbar_j、辅助视图为 h+delta、共享而非复制读出、F0 只在主输出算三对条件损失、S0/R0/H0 pair 项为零、每对平均后再等权而非样本池化。缺类、非正/NaN/未归一化 p 或非法预算拒绝；同样本单独/分批推理近似一致。
- [ ] **Step 2: 运行红灯。** `python -m unittest discover -s tests -p test_lc_role_adapter.py -v`；预期新模块缺失。
- [ ] **Step 3: 实现规格式 (2)-(8)。** 同形状组的读出初始化固定用 `seed+100`，各分支用 `seed+200+j`，避免构造顺序导致组间初态不同；F0 同规则但不同维数，不强称相同初态。U 和最后读出层置零；分支 U 无 bias。用 logsumexp/logsigmoid 计算锚点条件比例及 logq，以 logq 差算 BCE。U1 保留角色适配和辅助监督，但 u/v 不限幅；不计算其通用有界证书。
- [ ] **Step 4: 运行绿灯并检查可训练性。** 非退化合成例中读出数据梯度先出现，后续分支数据梯度出现；同时记录总梯度，不把 L2 的 V 梯度当专门化。接近 t=1 的正后验仍避免 1-t 相消；若 float64 实体概率 exp(logq) 下溢为 0，明确 `NUMERIC_RANGE_FAILURE` 并保留有限 logq 诊断，不裁剪、不接受为合法成功输出。
- [ ] **Step 5: 提交。** `feat: implement bounded role-conditioned feature adapters`。

## Task 3: 三个理论部分的实现证书

执行记录（2026-10-08）：7项新测试 RED -> GREEN；全套425项通过，43.064秒。步骤1-5已执行；两预算各5000合成行、总10000行零违规，单行破坏可检出；摘要为 `outputs/lc_role_implementation_qa/theory_synthetic_acceptance.json`。这不是精度/召回的实验证据。

**Files:** Create `scripts/lc_role_audit.py`; Test `tests/test_lc_role_audit.py`。

**Interfaces:** `audit_theory(model: RoleResidualModel, batch: dict, result: dict) -> dict` 返回逐行诊断和汇总；`replay_without_adaptation(model: RoleResidualModel, batch: dict) -> dict` 固定同权重、锚点、e、b，只改主视觉输入为 h；`check_flat_reconstruction(result: dict, log_anchor: Tensor) -> dict`。

- [ ] **Step 1: 写失败测试。**

```python
def test_bound_violations_are_not_averaged_away(self):
    audit = self.audit_valid_nonzero_model()
    self.assertEqual(audit["violation_count"], 0)
    broken = self.inject_one_row_norm_violation()
    self.assertGreater(broken["violation_count"], 0)
    self.assertEqual(len(broken["violating_image_ids"]), 1)
```

另测 SiLU 谱范数上界、eta 的 b^2 而非 b、逐类 logq 差和成对 log-odds 差<=eta、唯一 plain 胜者间隔>eta 时不变、并列时不作保护判断。验证普通 softmax 残差精确重建；同时重排非目标类/分支/监督后输出相应重排。辅助视图不套 b^2；关闭训练后适配器不强称回到锚点。
- [ ] **Step 2: 运行红灯。** `python -m unittest discover -s tests -p test_lc_role_audit.py -v`。
- [ ] **Step 3: 实现 A/B/C 证书。** 逐行导出 delta_norm、adaptation_norm、|u|、span(v)、L_F、eta、actual logq/log-odds drift、plain margin、M_T^*、实际目标 margin、NLL/KL 漂移及界。rho 用 log-space；L_F 用实际读出矩阵谱范数乘 `1+exp(-1)`。既有有限预算几何函数仅在接口和定义相符时复用，不复用旧常数预算的汇总计数。
- [ ] **Step 4: 运行绿灯及合成核验。** 固定 seed=20261006，至少 5000 行包含预算0、t近0/1、非目标比例极不均、target margin近0。任一式超出容忍度则 FAIL 并保存行。M_T^*<0 是不可能目标胜出的必要条件；非负不标为模型能够纠错。统计证书覆盖和过松比例，不用零违规宣称标签正确。
- [ ] **Step 5: 提交。** `test: certify feature and probability perturbation bounds`。

## Task 4: fit 内选模与收敛强对照

执行记录（2026-10-08）：9项新测试 RED -> GREEN；全套434项通过，49.733秒。步骤1-5已执行；九组均完成合成3折选择/重拟合/推理重放，未取得证书的凸OOF明确拒绝，400获选标为截断。未开展真实训练。

**Files:** Create `scripts/lc_role_training.py`; Test `tests/test_lc_role_training.py`。

**Interfaces:**
- `fit_neural(train: dict, validation: dict, arm: str, config: dict, seed: int) -> dict` 返回全部检查点状态、验证 logq、历史与梯度；validation 只能是内折。
- `fit_convex(train: dict, validation: dict, arm: str, regularization: float) -> dict` 只接受 P/E，零初值，返回驻点证书和预测。
- `select_oof(candidates: list[dict], assignment: list[dict]) -> dict` 返回仅含 fit 来源的选择锁；`refit_neural(fit: dict, arm: str, config: dict, seed: int, updates: int) -> dict` 无 stop 输入。

- [ ] **Step 1: 写失败测试。**

```python
def test_pooled_nll_and_tie_break(self):
    winner = training.select_oof(self.unequal_fold_candidates(), self.assignment)
    self.assertEqual(winner["pooled_row_count"], 120)
    self.assertEqual(winner["config"], self.expected_pooled_winner)
    tied = training.select_oof(self.exact_ties(), self.assignment)
    self.assertEqual(tied["config"], self.smaller_budget_larger_l2_earlier)
```

另测 OOF 每行每种子仅一次、重复/遗漏拒绝、不能用折均值代替 pooled、不能挑最佳种子、缺类拒绝、400 上界/0检查点、相同初态、L2含偏置、NaN失败及所选400的 `OPTIMIZATION_TRUNCATED` 标记。改变外部 stop 标签对 fit 选择没有影响；接口不传 stop。
- [ ] **Step 2: 运行红灯。** `python -m unittest discover -s tests -p test_lc_role_training.py -v`。
- [ ] **Step 3: 实现训练与选择。** 400更新的一条轨迹保留六检查点，不将每检查点算成一次新拟合。损失使用 Task 2；前5步保存每参数数据项梯度范数和总梯度范数，每检查点保存训练/验证 NLL、pair、L2、参数状态。P/E 复用 `anchor_linear_controls.fit` 和既有 SOLVER：L-BFGS-B 5000 后 trust-exact 100、gradient_l2<=1e-7、gap_upper<=1e-8；完整求解警告保留，未获证书不参与排序。
- [ ] **Step 4: 运行绿灯及合成端到端。** 小网格合成测试覆盖每组选择/重拟合/重放；0获选就保留锚点、不强制更新。有界组各自 fit 内选预算；另按 R1 所选预算为 S1 做 fit 内 lambda/轮次选择。完全相同协议/种子/预处理/配置的轨迹可复用，保存 provenance；不得以 stop 决定做不做此对照。
- [ ] **Step 5: 提交。** `feat: add fit-only grouped selection and strong controls`。

## Task 5: 执行状态机、上限与故障留痕

执行记录（2026-10-08）：13项新测试通过；全套447项通过，70.985秒。监督器真实合成进程、失败/超时/累计预算、完整选择锁和S1精确复用已验证。实际 preflight/benchmark 命令推迟到 Task7 后，以便快照包含全部七个模块；仍无真实训练授权。登记字节按Git过滤后的blob身份核对，运行时源文件SHA另行保留。

**Files:** Create `scripts/run_lc_role_adaptation.py`; Test `tests/test_run_lc_role_adaptation.py`; 后续经批准创建 `docs/experiment_protocols/lc_role_adaptation_v1.json`。

**Interfaces:** `validate_protocol(protocol: dict, stage: str) -> None`；`build_jobs(protocol: dict) -> list[dict]`；`maximum_fit_count(protocol: dict) -> int`；`temperature_fit_count(protocol: dict) -> int`；`run_supervised_job(job: dict, protocol: dict, remaining_seconds: float) -> dict`；`run_stage(root: Path, protocol_path: Path | None, output: Path, stage: str) -> dict`。CLI 阶段为 `preflight/benchmark/fit/evaluate/verify`，每次显式指定，无默认训练动作。

- [ ] **Step 1: 写失败测试。**

```python
def test_scope_count_and_failure_gates(self):
    self.assertEqual(runner.maximum_fit_count(self.protocol), 422)
    self.assertEqual(runner.temperature_fit_count(self.protocol), 4)
    with self.assertRaises(ValueError): self.evaluate_before_selection_lock()
    with self.assertRaises(FileExistsError): self.run_into_existing_stage_output()
    failure = self.run_fake_crashing_job()
    self.assertEqual(failure["attempts"], 1)
    self.assertEqual(failure["status"], "FAILED")
```

另测假时钟 hard timeout 杀死子进程且保留 PID/退出码/尾日志、总预算耗尽不排下一 job、原报告/源码/输入哈希变化拒绝、拒绝缺失/篡改/不完整 selection.lock、读其他折拒绝。记录何时首次 parse stop；测试只能在锁原子写入后发生。哈希检查和整体缓存反序列化可以涉及 stop bytes，但训练阶段不得索引 stop tensor 或解析 stop 表/标签。
- [ ] **Step 2: 运行红灯。** `python -m unittest discover -s tests -p test_run_lc_role_adaptation.py -v`。
- [ ] **Step 3: 实现有界串行监督器。** 每拟合为一个受监督 worker，限制1线程、每10秒记录 alive/elapsed/log，per-fit hard timeout=600秒，整个开发执行 hard timeout=21600秒。累计实际执行秒数写ledger，在fit/evaluate不同进程间延续，不在下一阶段重置6小时额度。状态为 `PREFLIGHT_OK/FITTING/FIT_LOCKED/EVALUATED/VERIFIED` 或 `FAILED/TIMED_OUT/NUMERIC_RANGE_FAILURE`。异常保留目录，不重试；已有阶段输出拒绝覆盖，下一阶段只读取已完成且哈希相符的前一阶段。精确重复任务在首次排程时去重，不是崩溃续跑。
- [ ] **Step 4: 运行绿灯和无真实训练 dry-run。** preflight/benchmark 可不传protocol，使用已认可规格的固定配置，输出 `draft_protocol.json`，状态 `DRAFT_NOT_AUTHORIZED`；它只能预检/合成估时，fit/evaluate必须拒绝。preflight只核验范围、身份、软件版本/依赖、折分配和排程；benchmark仅合成120行各神经结构20更新和合成P/E求解，不能查看stop选择计算预算。命令为 `python scripts/run_lc_role_adaptation.py --stage preflight --output-dir outputs/training/lc_role_adaptation_v1/preflight` 和独立 `--stage benchmark --output-dir outputs/training/lc_role_adaptation_v1/synthetic_benchmark`。预测422次拟合预计30-180分钟只是规划量级，未实测；benchmark写投影时间，若高于6小时则停止，申请修改预算而非静默缩网格。
- [ ] **Step 5: 提交。** `feat: guard role adaptation development execution`；协议文件只在训练授权后登记，不能将“认可规格”写成“批准训练”。

## Task 6: 指标、逐图证据与可重放交付

执行记录（2026-10-08）：9项缺模块失败 RED -> GREEN，随后可移植预处理引用回归 RED -> GREEN；11项新测试，全套458项通过，131.750秒。合成完整网格重建2124个OOF候选、26组最终状态及54份fit/stop逐图CSV；改动记录被拒绝，默认复验禁止优化器调用。训练重放代码单独授权并严格比对状态，差异留痕；没有开展真实训练或真实stop评价。

**Files:** Create `scripts/analyze_lc_role_adaptation.py`; Test `tests/test_analyze_lc_role_adaptation.py`。

**Interfaces:** `summarize_predictions(batch: dict, logq: Tensor, anchor_logq: Tensor) -> dict`；`screen_development(results: list[dict], controls: list[dict]) -> dict`；`verify_delivery(root: Path, output: Path, *, training_replay_protocol: Path | None = None) -> dict`；CLI `--output-dir PATH --verify`。训练重放另需 `--replay-training --replay-protocol PATH`，缺少独立授权文件直接拒绝。

- [ ] **Step 1: 写失败测试。**

```python
def test_each_seed_must_pass_and_tampering_fails(self):
    screen = analysis.screen_development(self.one_seed_fails(), self.controls)
    self.assertFalse(screen["passed"])
    with self.assertRaises(ValueError): self.verify_one_altered_prediction()
```

另测混淆矩阵方向、NLL两通道之和、Brier为四类平方误差和的行均值、15等宽ECE边界/空桶/置信度1、零预测目标时precision=0且声明分母0。测每干扰侵入率用该真类数量作分母、净正确目标=新增-损失；U1支配时否定限幅净效益，400截断/理论违规/缺组不作有效晋级。
- [ ] **Step 2: 运行红灯。** `python -m unittest discover -s tests -p test_analyze_lc_role_adaptation.py -v`。
- [ ] **Step 3: 实现全套证据输出。** 保留 fit/stop 全部最终组/种子 logq 与 q、ID、group、路径、真/预测标签、锚点、预算、理论诊断；严禁只留最好种子。记录 Accuracy/Macro F1、四类P/R/F1、完整混淆、目标新增/损失、Ti/G/M误入分母、NLL通道、Brier、ECE。高置信错判固定为 pbar最大值>=0.90且argmax错误；饱和固定为 |tanh|>=0.95，这两个仅描述，不选模。
- [ ] **Step 4: 运行绿灯与重放。** 重载状态重建所有OOF及最终预测、指标/筛查/理论表；同环境确定性训练重跑应得到相同参数/选择/tensor digest。若不一致保留差异，标 `REPLAY_MISMATCH`，不能泛用5%容差通过。正式目录验证重放本身会额外消耗最多相同拟合数，但不增加搜索机会；须另行获准并登记计算上限，不由 `--verify` 隐式启动训练。默认 verify 仅推理/预处理重放；训练重跑另用显式 `--replay-training`。
- [ ] **Step 5: 提交。** `feat: audit and replay role adaptation evidence`。

## Task 7: 与真实代码一致的网络结构图

执行记录（2026-10-09）：3项缺模块失败 RED -> GREEN；全套461项通过，139.237秒。真实模块确认3024参数、各支516、共享头1476；SVG/PDF/300dpi PNG/图源JSON已导出。PNG及单页PDF渲染已查看，文本不溢出；图仅标已实现、效果待真实实验验证。正式报告未修改。

**Files:** Create `scripts/generate_lc_role_adaptation_figure.py`; Test `tests/test_generate_lc_role_adaptation_figure.py`。

**Interfaces:** `architecture_manifest(model: RoleResidualModel) -> dict` 返回 `trainable_parameter_count/adapter_parameter_count/shared_head_parameter_count/nodes/edges/state` 及真实 named_modules/示例尺寸；`render_architecture(manifest: dict, output_dir: Path) -> dict` 输出 SVG/PDF/300dpi PNG 和图源 JSON。执行时读取 nature-figure 技能，优先沿用现有图表字号/字体与颜色。

- [ ] **Step 1: 写失败测试。** 从 R1 实例断言每支516、共享头1476、总3024；图源含冻结H1280、PCA64、E23、三支、共享头、有界层与训练辅助虚线；图中无“新骨干/LoRA/召回保证/已验证工业选矿”。

```python
def test_figure_counts_come_from_real_model(self):
    manifest = figures.architecture_manifest(self.r1_model)
    self.assertEqual(manifest["adapter_parameter_count"], [516, 516, 516])
    self.assertEqual(manifest["shared_head_parameter_count"], 1476)
    self.assertEqual(manifest["trainable_parameter_count"], 3024)
    self.assertEqual(manifest["state"], "IMPLEMENTED_EFFECT_UNVERIFIED")
```
- [ ] **Step 2: 运行红灯。** `python -m unittest discover -s tests -p test_generate_lc_role_adaptation_figure.py -v`。
- [ ] **Step 3: 实现公式驱动的图。** 实线推理、虚线训练，g与b、每图alpha/beta位置准确；辅助路径h+delta明确，不画三个独立分类头。附局部 `64-4-64` 模块细节和普通S1/R1对比。结构图标“已实现；效果待验证”或实际完成状态，不用草图冒充结果。
- [ ] **Step 4: 运行绿灯并视觉核验。** 各输出非空、尺寸稳定；渲染后查看PNG/PDF，不重叠、不裁字；以图源JSON/代码hash追溯。若训练失败也允许交付“实现态”图，图不能证明创新效果。
- [ ] **Step 5: 提交图代码与小型矢量/预览文件。** `docs: illustrate implemented role adaptation architecture`；不修改正式报告。

## Task 8: 预登记、训练授权与有限开发执行

**Files:** Create 训练获准后的 `docs/experiment_protocols/lc_role_adaptation_v1.json`、`docs/experiment_records/2026-10-06_lc_role_adaptation_registration.md`（日期按实际登记）；输出根 `outputs/training/lc_role_adaptation_v1/development_fold_0/`，禁止覆盖旧实验。

**Interfaces:** 消费 Tasks 1-7 的已通过测试实现、排程dry-run和合成benchmark；协议固定所有代码/规格/计划/来源哈希及用户训练原话，不能留待填占位符。无其他数据审批的入口。

- [ ] **Step 1: 实现阶段验收。** 运行 `python -m unittest discover -s tests -p 'test_lc_role*.py' -v`、执行器/分析器/图测试及旧回归；检查所有理论测试和来源隔离测试通过。提交并同步这一实现版本；不执行真实训练。
- [ ] **Step 2: 呈交 preflight/benchmark 与正式开发协议。** 重新hash冻结特征、原专家、两子集三源表、原probe summary/protocol、图源及软件版本；确认340fit unique groups和折分配hash。请求明确批准本协议的422拟合上限、6小时timeout和一次stop评价。不得把本计划审核扩展成自动独立确认或训练重放。
- [ ] **Step 3: 得到训练批准后，先提交并远端核验协议。** 把授权记录、源码/输入hash、排程摘要、failure policy冻结。远端提交可核验后才执行，网络失败则保留本地登记并说明未启动；不伪称已预注册。
- [ ] **Step 4: 执行 fit，锁定后执行 evaluate。**

```powershell
python scripts/run_lc_role_adaptation.py --protocol docs/experiment_protocols/lc_role_adaptation_v1.json --output-dir outputs/training/lc_role_adaptation_v1/development_fold_0 --stage fit
python scripts/run_lc_role_adaptation.py --protocol docs/experiment_protocols/lc_role_adaptation_v1.json --output-dir outputs/training/lc_role_adaptation_v1/development_fold_0 --stage evaluate
```

两命令分开：第一条成功并原子写入selection.lock才可执行第二条；所有最终状态hash先锁定，然后一次性载入stop计算全部组。400获选必须展示截断风险，不升级“已充分排除优化不足”。错误/超时立即结束当前开发批次并告知，不执行第二条。
- [ ] **Step 5: 验证并同步完整小型结果。** 只运行无训练的默认verify；保存失败项或完整结果，不开启训练重放。上传代码、协议、日志、CSV/JSON、图、结果记录；缓存/权重/原图仅本地档案及hash。不按阳性结果选择上传项。

## Task 9: 理论—实验对齐与报告候选段落

**Files:** Create `docs/experiment_records/YYYY-MM-DD_lc_role_adaptation_results.md` 和 `docs/report_candidates/lc_role_adaptation_method_and_limits.md`；Modify README/研究入口；**不改正式docx**。

**Interfaces:** 只消费已 `VERIFIED` 的默认推理重放结果，引用锁、hash、图和逐种子表。训练重放未批准/未完成需分别注明，不能称完全实验复现。

- [ ] **Step 1: 写结果口径验收表。** 分列“解析成立/实现检查/开发收益/独立确认”，每项有来源；对照失败、截断或证书过松必须原样保留，不将开发门写成统计显著性。
- [ ] **Step 2: 逐种子判定规格门槛。** R1 stop NLL比T0及H0/F0/S0/S1/R0/P/E最强合格对照低至少0.005；相对T0正确目标净增>=1，Ti/M假目标各最多+1；R1需低于匹配预算S1。未通过任何一项都不晋级；主强对照不可靠/截断时结论为“不足以排除优化混杂”，不能宣传稳定网络优势。
- [ ] **Step 3: 写候选方法章节。** 公式及结构对应规格A/B/C，报告界的最大残差/覆盖率、分支学习信号、辅助对损失、分支delta cosine相似度（零范数置undefined并单列计数）、禁用分支同权重诊断；这些不是独立重训练对照。标准理论工具与可能的领域方法贡献分开。
- [ ] **Step 4: 提出下一次确认范围。** 先审计数据暴露史，再请求真正未暴露的组隔离或来源外样本；不能仅凭Fold1/2名称判定独立。通过开发也不直接写“首次/保证召回/工业分选/品位预测”。失败则记录负结果和容量—风险启示，不继续自动堆模块。
- [ ] **Step 5: 提交候选文本与完整证据。** `docs: align role adaptation theory with development evidence`。用户另行批准后，才把合格理论/实现说明合并进唯一正式报告；独立确认前效果定位仍为开发观察。

## 实验排程与成本账本

| 拟合块 | 上限 | 说明 |
| --- | ---: | --- |
| H0/F0/S0/S1/R0/R1 CV | 324 | 6组 x 6配置 x 3折 x 3种子 |
| U1 CV | 27 | 3个lambda x 3折 x 3种子；不用无效预算复制试验 |
| 7神经组最终fit | 21 | 每组3种子；最多更新所选检查点 |
| S1与R1匹配预算补充 | 30 | 27 CV + 3 final；同协议配置可复用 |
| P/E CV和最终fit | 20 | 18 CV + 2 final，可靠驻点求解 |
| **拟合总上限** | **422** | 检查点同轨迹保留，不增加搜索次数 |
| T0标定 | 4 | 3内层训练折 + 全fit，所有组共享 |

- 合成benchmark仅20步/结构和合成凸求解，不属于真实开发拟合；估时需列PCA/SVD、神经与凸求解时间，不推测GPU加速。
- 默认verify只做状态载入、预测和数学重放，不自动再训练422次。完整训练重放若另批准：最多额外422拟合及4温度重放，单独6小时hard timeout，结果不得回流选模；注册实际去重后的任务数。
- 原始hash清单允许训练前计算两子集文件hash；“不读取stop”具体指锁前不解析stop源表/标签、不调用stop变换/指标、不用于选择。缓存文件同时含fit/stop，反序列化无法实现物理不可见；仅使用fit键并记录访问范围，不宣称盲法。
- 训练日志每10秒、关键状态立即输出；运行期间向用户约30秒一次简短进度，最终前确认没有所需worker仍运行。异常不会自动创建延时任务或后台无限训练。

## 预期交付与自审映射

| 交付 | 证据与验收 |
| --- | --- |
| `input_snapshot.json / inner_assignment.csv / preprocessing_*.pt` | 行身份、train-only状态、来源hash；pt仅本地 |
| `job_ledger.jsonl / runtime_projection.json / registered_protocol.json` | 次数、alive/timeout、失败/去重、版本与授权 |
| `cv_index.csv / oof_predictions.csv / selection.lock.json` | 全网格/三种子/每行OOF、fit-only选择及冻结时刻 |
| `runs/*/history.csv / gradients.csv / head.pt` | 检查点、数据/总梯度、实际更新数、截断；权重仅本地 |
| `predictions/*.csv / theory_rows.csv / theory_summary.json` | 每组每种子、逐图界/违规/保护条件/纠错容量 |
| `development_summary.json / verification.json` | 全指标、各门、来源重放；无训练重放不称全量复现 |
| `outputs/paper_figures_lc_rfa_v1/` | 实际架构图及图源、推理/训练路径和参数数 |
| `docs/report_candidates/` | 可审阅公式/方法/结果候选，正式报告不变 |

计划自审映射：规格1-3 -> Task1/5；规格4-5 -> Task2/4；规格6 -> Task3/6；规格7-8 -> Task4/5/8及排程表；规格9 -> Task6/9；规格10-11 -> Task7/9与授权门槛。此映射表示计划覆盖，不表示各测试或实验已经通过。评审是作者自审，不是独立同行评审。

## 待用户审核的执行边界

审核本计划后，推荐先批准**Tasks 1-7 的代码、合成验收与结构图**，采用当前会话原生执行。Task8真实训练在实现验收后再展示协议与估时、单独获准；Task9正式报告整合和新数据确认也分别审核。不将既往ABMP/强对照授权借用于新网络，不自动扩网格、读新图或修改正式报告。
