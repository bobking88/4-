# Task 4 自审与验证报告

## 状态

Task 4 已完成实现、DOCX 重建、结构验证和范围审计。已尝试本地可视化渲染，但当前环境未安装 LibreOffice，无法完成 PNG 页面级视觉检查；该限制不影响 DOCX 结构验证结果，但属于剩余验证缺口。

## 实现范围

- 更新 `scripts/build_technical_report.py`：直接读取两份理论验证 JSON 与已提交的组件消融 JSON；嵌入图 9、图 10；新增理论模型、符号表、命题与证明要点、受控逻辑验证、选择性识别及结论边界。
- 更新 `docs/experiment_records/2026-07-30_role_aware_hard_negative_learning.md`：补充公式、证据路径、关键数值及未来工作边界。
- 新增 `tests/test_build_technical_report.py`：覆盖输入契约、KL 方向、Windows 解释器别名容错、DOCX 结构与必需图像嵌入。
- 重建 `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（初稿）.docx`。

未修改或重新生成 Task 1–3 的 JSON、组件消融汇总、图 9 或图 10。工作树中原有的 `docs/superpowers/plans/2026-08-01-theory-aware-hierarchical-mineral-recognition.md` 修改未纳入 Task 4。

## 证据追踪

| 报告内容 | 直接来源 | 集成方式 |
| --- | --- | --- |
| 角色一致/冲突候选集计数与可识别率 | `outputs/theory_validation/role_identifiability/role_identifiability_summary.json` | 构建时读取并生成 5.9 节表格与结论 |
| 阈值、覆盖率、保留风险与保留样本数 | `outputs/theory_validation/selective_recognition/selective_recognition_summary.json` | 构建时读取并生成 5.10 节代表性阈值表与摘要 |
| 选择性识别曲线 | `outputs/paper_figures_v1/fig9_selective_recognition.png` | 作为报告图 9 原样嵌入 |
| 理论感知分层结构 | `outputs/paper_figures_v1/fig10_theory_aware_hierarchical_architecture_cn.png` | 作为报告图 2 原样嵌入 |
| 三组组件消融 | `outputs/business_metrics/hierarchical_component_ablation/hierarchical_component_ablation_summary.json` | 替换原硬编码行，构建时格式化均值与样本标准差 |

## 公式与方法核对

- 种类到角色采用概率聚合：`p_tilde_r = A p_s`。
- 一致性方向严格为：`L_cons = KL(p_r || p_tilde_r)`。
- 联合损失与训练代码一致：`L_role + 0.50 L_species + 0.10 L_cons + 0.25 L_binary + 0.10 L_hard`。
- 角色头、种类头、目标代理二分类头和投影头均从共享视觉特征并列导出；报告未写成头间串联。
- 公式由 Matplotlib MathText 生成 PNG，再由规定的 DOCX Python 运行时嵌入。顶层报告构建命令使用 `C:\Users\bob\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`；该运行时缺少 Matplotlib，因此仅 MathText 栅格化委托给已有项目 `.venv-training`，未安装新依赖。

## 结论边界审计

新增结论限定为：理论形式化、受控逻辑条件验证、固定测试划分上的选择性识别证据。报告明确否定以下当前结论：工业分选有效、品位或回收率改善、XRF 成本最优、外部验证完成、统计显著。

阶段条件化决策、真实动作/检查成本矩阵、XRF 多模态确认、来源留出测试、独立外部验证和真实矿石成像均只出现在局限性或未来工作语境中。`defer` 仅解释为建议后续检查。

## 测试与验证

1. TDD 红阶段：新增输入契约与 KL 方向测试，初次运行因缺少 `load_theory_report_inputs` 和 `theory_equation_specs` 按预期失败。
2. 报告专项测试：规定 DOCX Python 运行时执行 `python -m unittest tests.test_build_technical_report -v`，4 项全部通过。
3. 原有项目测试：项目 `.venv-training` 执行除报告专项外的 55 项测试，全部通过，耗时 7.750 秒。
4. DOCX 重建：规定 Python 运行时执行 `scripts/build_technical_report.py`，成功生成 1,778,704 字节的报告。
5. 结构验证：DOCX 含 20 个表格、15 个内联图像和 15 个媒体文件；Python 结构测试确认段落数大于 130、必需章节齐全、禁用正向声明不存在。
6. 图像一致性：结构测试以 SHA-256 核对 DOCX 媒体，确认图 9 与图 10 均为指定源文件的精确嵌入。
7. 代码检查：`py_compile` 通过；`git diff --check` 通过，无空白错误。
8. 本地渲染：已运行 bundled `render_docx.py --emit_pdf --verbose`。渲染器在启动 LibreOffice 时返回 `FileNotFoundError: [WinError 2]`，且 `soffice`/`libreoffice` 均不在 PATH。因此未生成页面 PNG/PDF，无法声明完成视觉版式检查。

## 基线与环境说明

规定 DOCX 运行时不包含 `pytest`、Matplotlib 或 PyTorch。初始 `unittest discover` 共运行 48 项，其中 19 项因缺少 Matplotlib/PyTorch 报环境错误；这是 Task 4 修改前的运行时能力限制。最终采用项目训练环境验证原有 55 项测试，并用规定 DOCX 运行时验证 4 项报告专项测试和实际报告构建。

## 自审结论与剩余风险

代码和文档改动均限定在 Task 4，数值由指定输出动态生成，公式方向与训练实现一致，必需图像已嵌入，禁止性声明审计通过。唯一未闭合项是 LibreOffice 缺失导致的页面级视觉 QA 不可用；可能仍存在只能通过 Word/LibreOffice 页面渲染发现的分页、表格跨页或图像缩放问题。提交中不包含原有计划文件修改和构建产生的未跟踪总体技术路线 PNG。

## Fix round 1/5 (2026-08-01)

- Added concise DOCX provenance references in conclusions 7-9: the controlled candidate-set validation now cites `outputs/theory_validation/role_identifiability/role_identifiability_summary.json` and §5.9; selective recognition now cites `outputs/theory_validation/selective_recognition/selective_recognition_summary.json`, Figure 9, and §5.10.
- Extended Appendix A.2 with the exact JSON and Figure 9 output paths, and added DOCX-text assertions for those references in `tests/test_build_technical_report.py`.
- Rebuilt the DOCX and ran `python -m unittest tests.test_build_technical_report -v`: 4 tests passed. No claims or boundaries were changed; no industrial, XRF, or external-validation assertions were added.
