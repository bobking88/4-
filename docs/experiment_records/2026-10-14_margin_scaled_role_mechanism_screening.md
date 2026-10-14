# 角色证据与间隔预算：机制筛选和限域查新

## Material Passport

- 日期/检索截止：2026-10-14，Asia/Shanghai。
- 授权：`call_b71499819f6c4a8f9f7d2c966238fe40`，用户回答“批准机制筛选、查新与规格编写（推荐）”。
- 起点：`7d58b29507231cfa6c6de175134c4625b78dbfec`。
- 目标：针对既有角色路径零改判及预算排除错误的证据，筛选可证伪的后端神经网络机制；不以更换名称、增加公式替代创新效果。
- 范围：既有源码/规格/审计文档和公开原始文献；不读取PT、原图或新增真实输入，不前向、不求梯度、不拟合、不开发网络、不改正式报告。
- 类型：定向机制查新，不是PRISMA系统综述、专业查新证明或独立同行评审。
- AI辅助检索、数学推导和自审；没有外部模型复核，不声称原创性或实验优越性已确认。

## 1. 已有证据约束设计

[离线审计](2026-10-14_ic_role_offline_theory_audit.md)已指出：当前340-fit锚点108个错误中92个不可在旧预算内纠正；所有保存检查点角色适配开/关无改判。这支持两项设计问题，但不能断言冻结视觉特征完全无用：

1. 输出预算随错误锚点的置信度收缩，可能在学习前就排除纠错。
2. 特征路由和输出预算重复抑制，角色残差还经过当前共享读出，实际决策贡献难以分离。

[既有残差等价核查](2026-10-06_role_residual_equivalence_and_novelty.md)和[图头核查](2026-10-06_role_graph_head_audit.md)继续有效：概率重参数化、固定星形图和可合并线性头不能单独作为新网络能力。

## 2. 检索与取舍规则

学术检索专用MCP在本轮工具清单中未挂载，使用公开检索定位arXiv、PMLR、CVF/OpenReview和出版商页面。第三方汇总仅用于寻找原始来源，不用于方法或发表状态结论。不安装工具、不上传本地研究数据。

纳入：与成对残差、间隔条件校准、实例相关概率约束、低秩视觉交互直接相关的原始论文，以及两篇指定矿物识别论文的出版商资料。时间以原始来源的首次公开日期、版本日期或正式出版元数据为准，不使用“几个月前”的检索标签。

原始全文无法访问时，明确区分“原文方法已核对”和“仅出版商索引/摘要已核对”；不会把未读全文的论文当作数学细节的证据。无匹配结果不能证明不存在同类工作。

### 七轮26条检索式

| 轮次 | 检索式，按实际顺序 |
|---|---|
| 1 | `"Probability Bounds" softmax Atarashi classification`；`"pairwise" "residual" "logits" classification neural network`；`"mineral" "hard negative" image classification`；`"confidence" "classifier correction" 2025 2026` |
| 2 | `"logit" "correction" "margin" "bounded" neural`；`"margin" "residual" "adaptive" "logit adjustment"`；`"classification" "margin proportional" neural` |
| 3 | `"Sample Margin-Aware Recalibration of Temperature Scaling"`；`"logit correction" "margin" "scaled"`；`"adaptive logit adjustment" classification`；`"low rank bilinear pooling" Hadamard product` |
| 4 | `"Beyond Logit Adjustment" "COLM" "2026"`；`"Sample Margin-Aware" "v306"`；`"margin" "bounded residual" classifier correction`；`"logit margin" "budget" "correction" neural` |
| 5 | `"Low-Intervention Boundary-Risk Graph Calibration" classification`；`"margin-scaled" "logit" classifier`；`"Iron Ore Image Recognition Through Multi-View Evolutionary Deep Fusion Method"`；`"Identifying Minerals from Image Using Out-of-Distribution"` |
| 6 | `"10.3390/s26154903"`；`"Identifying Minerals from Image" "2024" "June"`；`"Iron Ore Image Recognition" "2025" "November"`；`"bounded" "logit" "margin" "tanh" correction classification` |
| 7 | `"logit" "margin-scaled" "residual"`；`"residual correction" "absolute margin" classifier`；`"bounded correction" "decision boundary" classifier` |

核心技术检索优先限制到arXiv、OpenReview、PMLR、CVF；矿物应用检索优先出版商。少数宽检索用于避免只搜自己命名的候选词。对同一论文按arXiv ID/DOI合并，不把v1/v2计成不同研究。医学、金融、语言模型生成以及非分类残差的无关命中没有进入机制证据表。

## 3. 直接影响原创性判断的原始资料

| 工作与原始来源 | 可核对的时间/状态 | 本轮读取层级 | 对我们的限制或启发 |
|---|---|---|---|
| [Probability Bounding: Post-Hoc Calibration via Box-Constrained Softmax](https://arxiv.org/abs/2506.10572)，[v2方法](https://arxiv.org/html/2506.10572v2) | v1 2025-06-12；v2 2026-02-23；按预印本处理 | 原文§3.3、§4 | 已有神经网络预测实例相关边界及logit bounding。硬约束、可学习预算本身不能算首次。 |
| [Beyond Logit Adjustment: A Residual Decomposition Framework for Long-Tailed Reranking](https://arxiv.org/abs/2604.01506)，[REPAIR v2](https://arxiv.org/html/2604.01506v2) | v1 2026-04-02；v2 2026-07-14；arXiv作者注记COLM2026接收，本轮未核官方录用页 | 原文§4、§5 | 已有输入/竞争类别相关残差、成对特征与固定偏置失效条件。单纯成对残差不能作为我们的核心新意。 |
| [Sample Margin-Aware Recalibration of Temperature Scaling](https://proceedings.mlr.press/v306/guo26s.html)，[v2方法](https://arxiv.org/html/2506.23492v2) | 首次2025-06-30；ICML2026/PMLR306正式条目；v2 2026-08-26 | 原文§3.5及排序保持说明 | 已有间隔到样本温度映射；其正温度保持argmax，不能纠正类别错误。不能声称首次利用logit间隔。 |
| [Hadamard Product for Low-rank Bilinear Pooling](https://arxiv.org/abs/1610.04325) | 首次2016-10-14；原始索引PDF标注ICLR2017 | arXiv摘要/元数据及原始PDF索引；OpenReview直连被浏览器校验阻挡 | Hadamard低秩双线性融合是已知组件，不作为独立原创点。 |
| [Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition](https://arxiv.org/abs/2104.06094) | 首次2021-04-13；本轮仅核arXiv，不另推断正式发表场所 | 原始摘要/元数据 | 已有样本难度与数量因素驱动的logit调整；自适应难样本处理并不新。 |
| [Taming Softmax at the Limits of Muon: A Unified Approach through Hilbert Geometry](https://arxiv.org/abs/2610.17588)，[v1](https://arxiv.org/html/2610.17588v1) | 首次2026-10-13，早于本轮截止；预印本 | 原文§3及附录A.2 | softmax概率变化与相对logit跨度已有明确几何分析。不能把普通漂移界或共同平移不变性当新理论。 |
| [Low-Intervention Boundary-Risk Graph Calibration for Cross-Domain Few-Shot Hyperspectral Image Classification](https://www.mdpi.com/1424-8220/26/15/4903) | Sensors2026；出版商索引2026-08-03，DOI `10.3390/s26154903` | 出版商索引摘要/方法片段；正文直连Internal Error | 已有低间隔门控和有界图残差。只做“低置信纠错＋小残差”已有近邻研究；不据未读全文评价其完整理论。 |

这是机制层面的有限核对，不宣称七项覆盖了全部相关文献。PB/REPAIR/SMART是必须正面讨论的近邻；Hadamard融合属于明确的既有技术构件。

## 4. 两篇矿物论文怎样放入研究论证

| 工作 | 原始出版资料 | 本轮可使用的结论 | 不作的推断 |
|---|---|---|---|
| [Iron Ore Image Recognition Through Multi-View Evolutionary Deep Fusion Method](https://www.mdpi.com/1999-5903/17/12/553) | DOI `10.3390/fi17120553`；[出版商专题条目](https://www.mdpi.com/journal/futureinternet/special_issues/AM_NGVS)明确2025-11-30 | 出版商索引方法包含八个预训练特征视图及进化融合。可以借鉴总框图、模块分解、复杂度和外部评价的组织。 | 不把我们340-fit指标与其不同数据/任务准确率直接排名，不声称首次铁矿图像深度识别。 |
| [Identifying Minerals from Image Using Out-of-Distribution Artificial Intelligence-Based Model](https://www.mdpi.com/2075-163X/14/6/627) | DOI `10.3390/min14060627`；[出版商卷期条目](https://www.mdpi.com/2075-163X/14/6)明确2024-06-20 | 题名和出版资料已核对，说明未知矿物处理已是既有研究方向；目前候选明确限定闭集角色识别。 | 文章直连失败，卷期摘要存在明显跨主题混杂；不据该摘要重述模型细节或定量OOD成绩，不宣称重新精读了本地PDF。 |

这些资料支持研究边界，不替代真实矿石/现场图像和品位化验。C阶段条件决策、OOD与来源外验证仍需相应数据及单独协议。

## 5. 三个可选机制及筛选结论

| 机制 | 优点 | 关键反证/风险 | 本轮结论 |
|---|---|---|---|
| 给旧预算加常数下限、继续共享特征适配 | 修改小 | 仍可能容量不足；继续共享读出吸收，且已有实例预算/门控先例 | 不作主候选，不自动延长旧实验 |
| 无界非线性角色残差 | 任意有限anchor在输出层不再被硬半径排除 | 普通残差本身已知，缺少幅度控制，可能提高假目标 | 保留为强对照，不凭“能跨界”宣称优越 |
| 间隔缩放预算＋角色双线性证据 | 可以显式构造远离tanh饱和区的全类别纠错见证，角色log-odds路径直接可读 | 与普通残差/低秩融合仍有大量重合；扩大预算也允许损害正确锚点 | **选择为待审候选**，必须分离预算贡献与角色表征贡献 |

候选定名仅用于版本识别：**MSB-RRE v1（Margin-Scaled Budget Role-Residual Evidence）**。它不是已验证的新算法，详见[书面规格](../superpowers/specs/2026-10-14_margin_scaled_bilinear_role_evidence_design.md)。

## 6. 当前可辩护的理论问题，而非首次宣称

从相对margin $m_j=\log(p_T/p_j)$ 构造 $R_j=\gamma|m_j|+\tau$，$\gamma>1,\tau>0$；令纠错量为 $d_j=R_j\tanh s_j$。待审推导给出：

1. 全部类别都有 $|\tanh s_j|=1/\gamma$ 的有限内部见证，消除旧预算的逐图不可达性；不是共同网络能实现所有真值的证明。
2. 成对边界的控制导数下界不因大anchor间隔变为零，错判成对样本在恒等起点也有非零单样本bias梯度；不是全批梯度、优化收敛或泛化保证。
3. 角色修正量可由预测log-odds差唯一读出，避免把特征残差非零等同于分类贡献；这是功能量可识别，不是参数唯一或角色优越。

本轮定向检索未核到与以上**整套具体参数化及性质**完全相同的论文，但相邻原理已经充分存在。不能把“未核到”写成“首次”；精确数学结果可能是已有一般原理的直接推论。理论原创强度评为**待外部复查、中等至高重合风险**，网络效果仍为**零新实验、完全未验证**。

## 7. 最强反方意见与处置

“这仍是已知bilinear pooling、普通残差、动态预算与辅助监督的组合，普通头可以表达同样的函数；全类别oracle可达是通过增大半径人为保证的，不说明真实图像识别更好。”

这一意见不能靠图示或名称反驳。规格要求相同输入/预算/监督的共享双线性和普通MLP强对照、无界对照、固定半径消融、无辅助监督消融；若没有角色独立净纠错或只有NLL微动，就否决角色创新效果，不改名晋级。若只证明预算作用，论文只能按预算参数化贡献表述。

## 8. 交付与限制

本轮不修改旧公式或已归档结果，不跑合成神经前向，不拟合文献方法，也不查看新的样本。方法细节只冻结为待审核规格；下一步须先审核规格，再编写实施计划。

本查新缺少Scopus/WoS/CNKI完整覆盖，部分出版商正文不可直连，并且没有独立数学/原创性审核。论文投稿前应对实际冻结的方法再次检索、补全原文核对和引用。正式技术报告现在保持原版本与原哈希，不写入“创新效果已验证”。
