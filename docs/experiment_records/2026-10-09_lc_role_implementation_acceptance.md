# LC-RFA-B v1 原生实现与合成验收

日期：2026-10-09。分支：`codex/theory-aware-report`。

## 1. 范围与状态

用户批准原话：“认可计划，原生实施代码与合成验收”。本轮落实实施计划 Tasks 1-7：预处理、网络、理论审计、强对照/选模、受控执行器、证据回放和结构图。真实 LC-RFA 训练、真实 stop 评价、训练重放、其他折/新数据及正式报告修改均未开展。

绑定规格：`docs/superpowers/specs/2026-10-06-lightweight-role-adaptation-design.md`，SHA-256：`aa9fcec06b6190fa6e56969da667c7adbcd07c5eb9ea56574031bd51dabefea0`。

正式报告保持不变，SHA-256：`b7319ccf6def029f5bbc9ae61a0a68ff867c2751433b4876d8f68f7a2ac82d0c`。

## 2. 网络实现与图

冻结 EfficientNet-B0 的 H1280 经训练折拟合的 PCA/标准化得到 h64，与标准化 e23 结合。三条 `64 -> 4 -> 64` 非线性适配分支，各516参数；共享 `87 -> 16 -> 4` SiLU读出1476参数，总可训练参数3024。辅助路径是无路由缩放的 `h+delta_j`，复用同一读出，不是三个独立分类头。

主要公式对应实际实现：

$$
g_j=4\bar p_T\bar p_j,\quad b=\sum_j g_j\le1,\qquad
\delta_j=\frac{\tanh[U_j\operatorname{SiLU}(V_jh+c_j)]}{8},\quad
\widetilde h=h+\sum_jg_j\delta_j.
$$

$$
\alpha=\bar\alpha b,\quad\beta=\bar\beta b,\quad
u=\alpha\tanh a_T,\quad v_j=\frac\beta2\tanh a_j,
\qquad q_T=\sigma(\operatorname{logit}\bar p_T+u),\quad
q_j=(1-q_T)\frac{w_je^{v_j}}{\sum_kw_ke^{v_k}}.
$$

`scripts/lc_role_adapter.py` 实现前向与主/辅助损失；标签不进入推理接口。零初始化保持锚点；前五次数据梯度及含正则总梯度分别记录，不能用正则梯度冒充分支学习信号。

结构图：`outputs/paper_figures_lc_rfa_v1/lc_rfa_b_architecture.svg`、`.pdf`、`.png`、`.json`。图源来自真实模块/参数量/示例尺寸，并含模型与绘图代码哈希。可编辑SVG、单页PDF和300dpi PNG已视觉核验；状态为 **IMPLEMENTED_EFFECT_UNVERIFIED**，不是效果证明。

## 3. 理论与合成证据

| 理论对象 | 已实现检查 | 不能据此宣称 |
| --- | --- | --- |
| A：恒等锚点与幅度界 | `||delta_j||<=1`、`||htilde-h||<=b`、`|u|<=alpha`、`span(v)<=beta`，概率正性/归一化 | 召回率或真标签安全保证 |
| B：同权重主路径二阶抑制 | `eta=(alpha_max+beta_max)L_F b^2`、logq/成对log-odds漂移、间隔保护 | 辅助视图具有b²界；普通独立重训对照与同权重去适配等价 |
| C：预算纠错包络与风险漂移 | 修正后的目标胜出上包络、逐样本NLL/KL界、数值重参数化一致性 | 包络允许即代表有限网络/优化可达；KL小即校准正确 |

2026-10-08的独立合成审计共10000行，预算(.5,.5)与(0,0)各5000行，记录0个违规行；浮点判定容差为 `1e-10+1e-10*abs(bound)`。最大绝对重参数化残差约 `1.42e-14`。证据在 `outputs/lc_role_implementation_qa/theory_synthetic_acceptance.json`。注入单条违反幅度界的样本可定位具体ID。覆盖率仅描述合成分布，不是分类正确率。

这些命题采用标准Lipschitz、间隔和概率风险工具。拟争取的领域方法贡献是“语义监督的角色条件特征适配及有限纠错机制”；是否构成有效网络创新，须通过真实同预算强对照与后续独立确认检验，不能将通用界本身宣称为新普适理论。

## 4. 选模、强对照与回放

实现T0、P/E凸线性控制、H0普通读出、F0近参数量MLP、S0等权无辅助、S1等权同辅助、R0角色无辅助、R1完整模型、U1不限幅及R1匹配预算S1。全部神经控制使用同一冻结信息；fit内部3折/3种子选超参数、检查点，锁定后才能一次性评价stop。400步被选中明确标记优化截断。

受控排程上限422拟合、另4次温度拟合、21600秒累计预算、单worker600秒、一次尝试，无失败后自动扩网格/重试。已测试超时杀进程、预算耗尽不启动、来源变更拒绝、原子锁及stop访问顺序。

合成冻结状态回放重建2124个OOF候选、26组最终状态、54份fit/stop逐图CSV，并严格重建选模、指标和理论表。默认verify不调用优化器；被改动的预测会拒绝。该合成夹具验证推理完整性，不证明优化器历史真实。完整训练重放须独立授权，不适用统一5%容差。

分支诊断输出：各无缩放辅助视图的T-j条件NLL及锚点/主路径比较、delta余弦相似度（零范数置null并单列有效数）、非零分支数、固定同权重逐支禁用后的logq漂移和获胜类变化。诊断仅描述，不用于另选stop超参数。

## 5. 测试与作者自审

2026-10-09，`python -m unittest discover -s tests -v`：**464/464通过，138.611秒，exit 0**；日志 `outputs/lc_role_implementation_qa/final_tests.log`。针对性27项通过，84.532秒。

全范围自审覆盖实施基点 `48a57c4aade63e18ec36742af7808e4541808c35` 至Tasks1-7完成提交 `b828418`；随后只作一次修复验收。

| 重要发现 | RED -> GREEN 回归 |
| --- | --- |
| 缺少分支专门化证据 | `test_specialization_reports_pair_views_and_each_branch_effect` |
| 数值worker失败被归为普通FAILED | `test_numeric_worker_failure_keeps_numeric_terminal_stage` |
| 运行环境版本记录不完整 | `test_registered_runtime_records_packages_and_rejects_version_drift` |

无延后小项或不予判断项。此处是作者自审，弱于独立评审；不称同行审查完成。

执行裁定：实际preflight/合成benchmark延后至Task7完成，因为源码快照须包含分析和绘图模块；代价是计时证据直到实现齐备才取得。未改变模型、实验网格或授权范围。

## 6. 实际预检与合成计时

完整源码提交冻结后执行一次，尚待记录实测结果。输出路径：

- `outputs/training/lc_role_adaptation_v1/preflight/`
- `outputs/training/lc_role_adaptation_v1/synthetic_benchmark/`

预检只解析既定fit子集，允许对既有stop文件计算hash，不解析stop源表/标签。计时仅使用120行合成数据（80训练/40验证），每神经组20步及两种凸控制。预算预测不是实测完整开发耗时；若预测超过6小时，不静默缩小候选网格。

## 7. 后续边界

完成实现验收并同步后，呈交锁定开发协议和实测估时，请求批准最多422拟合、4次温度拟合、6小时硬上限及一次stop评价。真实结果必须逐种子与最强合格控制、匹配预算S1比较，未通过也保留全部负结果。真实开发成功仍不等于独立确认；正式报告继续保持唯一版本，另行批准后再整合合格证据。
