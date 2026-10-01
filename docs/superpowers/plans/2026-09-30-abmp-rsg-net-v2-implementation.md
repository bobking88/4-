# ABMP-RSG-Net v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现、审计和验证自适应预算与目标锚点边际保护网络，补充结构图、理论证据与技术报告。

**Architecture:** 复用冻结 HRGV 专家和 OOS 候选门生成验证器完整的 q0/qphi；18 维证据经过共享 MLP 输出路由和预算。解析双重投影约束最终凸融合，开发结果通过锁文件约束 Fold 1/2 确证。

**Tech Stack:** Python、PyTorch、unittest、CSV/JSON、matplotlib、python-docx；使用现有 `.venv-training`。

**Spec:** `docs/superpowers/specs/2026-09-30-abmp-rsg-net-v2-design.md`

## Global Constraints

- 18 -> 64 -> LayerNorm -> SiLU -> Dropout(0.10) -> 16 -> SiLU；两个 16 -> 1 头。
- tau_r=0.25、alpha=0.25、beta=0.10；预算上限 0.02/0.04/0.08；三种锚点配置；偏移 -0.5/0/0.5。
- Fold 0 是开发折；Fold 1/2 外层结果只在开发锁定后加载。
- 后验、锚点边际、凸分解与凸 NLL 审计零违规；不把软上界写成总体召回保证。
- 权重不进 Git；现有 v1 输出不覆盖；仅暂存本计划涉及文件。

## Review Focus

- d_post 为零或极小：避免除零并保持后验上界。
- 高置信假阳性锚点：必须保持且披露其代价。
- 预算头饱和或批次变化：推理结果独立于其他样本，梯度有限。
- 停止集没有目标或难负类：选择必须拒绝无定义指标。
- 输出目录存在或协议/哈希不符：不得覆写或打开未锁定外层。

## Task 1: 双头网络与解析双投影

**Files:** Create `scripts/abmp_rsg.py`; test `tests/test_abmp_rsg.py`.

**Interfaces:** `AdaptiveBudgetPolicy.forward(evidence, epsilon_max, lambda_cal=0) -> dict`; `apply_abmp_projection(q0, q_candidate, raw_route, epsilon, *, tau_p, tau_m, delta, target_index=0, mode='full') -> dict`; `risk_supervision_targets(q0, q_candidate, labels, epsilon_max, tau_r=.25) -> dict`; `audit_abmp_projection(result, q0, q_candidate, labels, epsilon_max, delta, target_index=0) -> dict`.

- [x] 写失败测试：预算头与路由头有梯度；推理批次不变；随机及零概率后验上界；锚点边际；非目标锚点包括假阳性；退化回退；错误输入。
- [x] 运行 `python -m unittest discover -s tests -p test_abmp_rsg.py -v`，确认缺少模块失败。
- [x] 复用 `build_projection_evidence` 和现有输入验证；实现双头、软收益/预算目标、双上限和逐图审计。
- [x] 运行上述测试及 TC 核心回归测试，确认通过。
- [x] 提交 `feat: add adaptive-budget margin-protected policy`（9a6dc11）。

## Task 2: Fold 0 开发执行器

**Files:** Create `scripts/run_abmp_rsg_development.py`; test `tests/test_run_abmp_rsg_development.py`.

**Interfaces:** 消费 Task 1；`train_policy(fit_cache, stop_cache, config, *, epochs=30, seed=20260930) -> dict`; `evaluate_policy(cache, trained, *, lambda_cal=0) -> dict`; `select_development_candidate(candidates, q0_metrics) -> dict | None`; CLI 输出 `development_summary.json` 和 `abmp_rsg_v2_lock.json`。

- [x] 写失败测试：27 个配置；无定义指标拒绝；投影未激活拒绝；确切安全/NLL选择顺序；合成训练参数更新；拒绝已有输出目录。
- [x] 运行测试并确认失败；实现仅打开 Fold 0 的缓存提取、训练、开发评估和审计。
- [x] 复用 v1 Fold 0 专家/候选门权重，两轮各9组训练、27个候选；r2排除解析冗余区间。
- [x] 对最低 NLL 诊断配置运行 A0-A8，记录逐图预测和哈希；r2没有晋级选中配置。
- [x] 测试通过后执行开发实验；r2开发门失败，停止确证并废止r1旧锁。
- [x] 提交开发执行器（aa41e03）和冗余条件修正（60fe32b）；r2结果随本次报告提交。

## Task 3: 确证执行器与统计

**Status:** 暂不执行。r2没有通过开发门，Fold 1/2未用于v2评估；不能用开发最低NLL配置启动确证。

**Files:** Create `scripts/run_abmp_rsg_confirmation.py`, `scripts/analyze_abmp_rsg.py`; corresponding tests.

**Interfaces:** 消费 Task 2 开发锁；复用 TC 专家训练和候选门接口；`analyze_confirmation(prediction_paths, lock_path, *, bootstrap_samples=10000) -> dict`。

- [ ] 测试锁缺失/篡改/开发未通过时拒绝；锁事件先于外层；分组重叠拒绝；配对统计同组重采样；单侧非劣界。
- [ ] 实现固定配置训练 A2-A6 和两折预测、理论审计；不根据外层重选。
- [ ] 开发门通过后运行 Fold 1/2；统计 NLL、目标召回、误入和晋级结果。
- [ ] 提交代码与两折结果；失败保留负证据。

## Task 4: 理论验证与正式结构图

**Files:** Create `scripts/generate_abmp_rsg_figure.py`; test `tests/test_generate_abmp_rsg_figure.py`; outputs `outputs/paper_figures_v5/fig_abmp_rsg_architecture.*`, `outputs/theory/abmp_rsg_invariants.json`.

- [x] 验证图源覆盖冻结双分支、18维证据、双头、双上限、最终凸融合和两条核心公式。
- [x] 用构造样本和25000组随机单纯形验证不变量、冗余条件和同argmax凸融合性质；保存审计计数。
- [x] 输出 SVG/PDF/300dpi PNG；查看实际图片，修正重叠和公式排版。
- [x] 提交脚本、图源与图表（cb676b6）。

## Task 5: 报告与复现材料

**Files:** New reproducible report updater and tests; update sole official DOCX; add experiment record and README entries.

- [x] 根据已完成证据补充方法、10个公式、结构推导、网络图、两轮开发结果与容量诊断。
- [x] 将结构性质与经验结果分开，报告锚点假阳性代价及开发/确证用途。
- [x] 更新研究分支唯一正式版，新增附录M；原有689个正文元素不变，新增第72至76页已逐页检查。
- [x] 运行30项专项和331项全套回归，均通过；独立审查及刷新保护修复完成；仅提交相关文件（cb676b6），已重试GitHub同步。

**Delivery note:** 两轮开发失败证据、报告附录M和结构理论已提交；开发门未通过，研究性能创新目标仍未达成。GitHub同步状态以终端推送与远端SHA核验为准，不将本地提交等同于上传成功。

## Execution

继续按用户已确认的 v2 和持续执行授权内联实施；必要的设计细节在 ledger 中记录。若开发门失败，Task 3 实验停止，Task 4/5 继续生成已有证据；该条件不等于研究目标完成。
