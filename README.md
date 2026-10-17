# 钒钛矿相关矿物图像识别

本仓库保存“人工智能选矿”项目中，面向钒钛矿相关成分矿物图像识别的数据治理记录、训练代码、人工复核结果和正式实验输出。研究任务采用四分类设置：目标矿物、含钛干扰矿物、脉石/废石和金属光泽干扰矿物。

## 当前版本

- 最终数据版本：`dataset_final_v1`
- 图像总数：8,529 张
- 固定划分：训练集 5,961 张，验证集 1,284 张，测试集 1,284 张
- 人工复核：分层抽检 823 张，排除 32 张，2 张不确定样本未进入训练
- 正式对照：ResNet50、EfficientNet-B0、Focal Loss 与角色感知困难负样本学习，均使用 3 个随机种子
- 分层模型：矿物种类—选矿角色分层一致性 EfficientNet-B0，使用 17 类细粒度矿物标签与四类角色标签联合训练
- 分层模型结果：Macro F1 为 `73.41% ± 2.40%`；目标代理召回提高，但两类困难干扰误入目标率也上升，当前仅作为召回—风险取舍原型，不宣称稳定优于基线
- 分层组件消融：分别移除困难负样本约束和层级一致性约束，各完成 3 个随机种子。完整模型的总体 Macro F1 与两种删减配置接近，但目标代理 F1 更高、含钛和金属光泽干扰误入目标的比例呈更低方向；均值差与种子间波动相近，作为风险趋势报告而不作显著性宣称。
- 成对统计推断：10,000 次两阶段分层簇 Bootstrap 显示目标召回提高 `9.16` 个百分点，95%区间为 `[5.18, 13.28]`；但含钛和金属光泽干扰误入目标分别增加 `3.19` 和 `4.73` 个百分点。Accuracy 与 Macro F1 区间跨 0，三个种子的 Holm 校正 McNemar p 值均大于 0.05，因此结论定位为风险重分配而非总体稳定优势。

完整实验汇总见 [outputs/training/formal_experiment_summary_v1.md](outputs/training/formal_experiment_summary_v1.md)。

## 仓库结构

```text
scripts/                    数据集审计、人工复核、训练与验证脚本
tests/                      数据处理与训练辅助函数测试
数据集/dataset_audit/        初始数据审计、质量问题与固定划分清单
数据集/dataset_final_v1/     人工复核后的最终训练清单和排除记录
数据集/dataset_review_20260727/
                            人工复核队列、决策汇总与复核说明
outputs/training/            正式实验的指标、混淆矩阵与逐图预测结果
outputs/business_metrics/    目标代理风险指标与分层模型三随机种子汇总
outputs/paper_figures_v1/    技术报告与论文图表、图表源数据及结构图
outputs/paper_figures_v3/    成对簇 Bootstrap 论文图及矢量/高分辨率版本
outputs/paper_experiments_v3/
                            成对统计推断、逐种子指标和重采样分布
docs/                       项目过程文档
训练说明.md                  本地训练操作说明
requirements-training.txt    训练环境依赖
```

## 复现步骤

1. 按 `requirements-training.txt` 创建并安装 Python 训练环境。
2. 准备与 `数据集/dataset_final_v1/dataset_split_manifest_v1_0.csv` 中 `relative_path` 对应的、具有合法来源的图像文件。
3. 运行训练，例如：

```powershell
python .\scripts\train_mineral_classifier.py `
  --model efficientnet_b0 `
  --epochs 30 `
  --batch-size 16 `
  --num-workers 2 `
  --seed 20260727 `
  --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv `
  --dataset-root .\数据集\mindat_manual_positive_v1 `
  --output-dir .\outputs\training\formal_efficientnet_b0_seed20260727
```

分层模型复现示例：

```powershell
python .\scripts\train_hierarchical_mineral_classifier.py `
  --epochs 30 `
  --batch-size 16 `
  --num-workers 2 `
  --seed 20260727 `
  --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv `
  --dataset-root .\数据集\mindat_manual_positive_v1 `
  --output-dir .\outputs\training\formal_hierarchical_efficientnet_b0_seed20260727
```

组件消融示例（移除困难负样本约束）：

```powershell
python .\scripts\train_hierarchical_mineral_classifier.py `
  --epochs 30 `
  --batch-size 16 `
  --num-workers 2 `
  --seed 20260727 `
  --lambda-species 0.50 `
  --lambda-consistency 0.10 `
  --lambda-binary 0.25 `
  --lambda-contrast 0.0 `
  --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv `
  --dataset-root .\数据集\mindat_manual_positive_v1 `
  --output-dir .\outputs\training\formal_hierarchical_no_contrast_seed20260727
```

两项组件消融的汇总结果位于 [outputs/business_metrics/hierarchical_component_ablation/hierarchical_component_ablation.md](outputs/business_metrics/hierarchical_component_ablation/hierarchical_component_ablation.md)。

成对簇 Bootstrap 与 McNemar 推断复现：

```powershell
python .\scripts\analyze_paired_cluster_statistics.py `
  --training-root .\outputs\training `
  --output-dir .\outputs\paper_experiments_v3\statistical_inference `
  --figure-prefix .\outputs\paper_figures_v3\fig_paired_cluster_effects `
  --bootstrap-replicates 10000 `
  --rng-seed 20260819
```

方法、统计假设和解释边界见 [docs/experiment_records/2026-08-19_paired_cluster_inference.md](docs/experiment_records/2026-08-19_paired_cluster_inference.md)。

开放集评价工具位于 `scripts/evaluate_open_set_protocol.py`。它需要独立、经核验的未知矿物图像预测表；当前仓库不以闭集四分类测试集伪造未知矿物结果。

## 数据说明与边界

本仓库不分发原始矿物图片、模型权重、虚拟环境或下载缓存。原始图像来自公开矿物图像页面，仍须遵守各图片页面的署名、许可和使用条件。仓库保留来源元数据、质量控制记录和最终数据清单，以支持研究过程审计和合法复现。

当前结果是基于公开矿物标本图像的“钒钛矿相关矿物识别”基线，不应直接解释为工业传送带场景下的实际分选性能。

## ABMP-RSG-Net v2 开发证据

双头自适应预算与目标间隔保护已实现，网络图、数学推导、两轮 Fold 0
开发结果与几何诊断均可追溯。第二轮的 27 个候选未通过独立间隔作用判据，
没有启动 Fold 1/2；初轮锁已废止，不能将旧通过状态当作确证依据。
当前预算均值重放与完整模型完全相同，因此尚不支持自适应预算改善性能的结论。

入口：[实验与理论诊断](docs/experiment_records/2026-10-01_abmp_rsg_v2_development.md)、
[协议状态](outputs/training/abmp_rsg_v2/protocol_status.json)、
[开发结果](outputs/training/abmp_rsg_v2/development_fold_0_r2/development_summary.json)、
[结构图 SVG](outputs/paper_figures_v5/fig_abmp_rsg_architecture.svg)。
唯一正式报告新增附录 M，保持现有正文结果和经验结论边界。

2026-10-02 新增 [候选容量与校验约束审计](docs/experiment_records/2026-10-02_abmp_candidate_capacity.md)。
只提取两个 Fold 0 内层子集，共 680 张，不读取本轮 outer_eval 或 Fold 1/2。
完整校验区间在停止集没有额外目标恢复容量，校验前完整区间有 2 张机会；
这些是标签辅助的可达上界，不是新网络精度。正式报告附录 N 补充 6 个公式、
数学证明、容量图及下一阶段强对照计划。v2 性能创新仍未成立。

入口：[审计 JSON](outputs/theory/abmp_candidate_capacity_v1/audit_summary.json)、
[公式检查](outputs/theory/abmp_candidate_capacity_v1/geometry_properties.json)、
[容量图 SVG](outputs/paper_figures_v5/fig_abmp_candidate_capacity.svg)、
[后续研究计划](docs/superpowers/plans/2026-10-02-abmp-verifier-trust-research.md)。

~~~powershell
python scripts/analyze_abmp_development.py
python scripts/generate_abmp_rsg_figure.py
python tools/append_abmp_to_official_report.py
python -m unittest discover -s tests -p "*abmp*.py" -v
~~~

## Theory-aware evidence reproduction

2026-10-05 已完成 [后验锚点收敛强对照](docs/experiment_records/2026-10-05_anchor_linear_controls_results.md)。
协议/代码在真实拟合前以 `11af17f` 推送；15 次凸求解与 32 张预测表精确重放。
收敛检查通过，但视觉联合残差未超过 T0，角色约束也未通过，不能宣称网络创新有效。
入口：[预注册](docs/experiment_records/2026-10-05_anchor_linear_controls_registration.md)、
[结果 JSON](outputs/training/abmp_anchor_linear_controls_v1/development_fold_0/development_summary.json)、
[交付验证](outputs/training/abmp_anchor_linear_controls_v1/development_fold_0/delivery_verification.json)。
本轮只读既有缓存，不打开新图片或其他真实折，正式报告未修改；系数/缓存不公开分发。

```powershell
python -m unittest discover -s tests -p "test_anchor_linear_controls.py" -v
python scripts/run_anchor_linear_controls.py --verify
```

2026-10-05 后续 [通道诊断与有限预算理论](docs/experiment_records/2026-10-05_bounded_role_residual_theory.md)
给出目标 odds 与非目标条件分配的精确损失分解，以及有限预算下的目标可达条件、
逐标签损失/KL 漂移界与间隔保护反例。EH stop 的净损失增量约 72.27% 来自条件分配。
768 个合成 LP 对照与 5000 个概率检查通过，两个
[分析 JSON](outputs/theory/abmp_bounded_role_capacity_v1/reproduction_verification.json)
精确重放；[完整只读代码](docs/experiment_records/2026-10-05_bounded_role_reproduction.md)
不依赖图片或训练权重。该分析不是新网络效果验证；双通道方向仍待规格批准，
不把通用概率分解/校准性质称为首次理论创新，正式报告未修改。

2026-10-06 [精确等价与约束几何核查](docs/experiment_records/2026-10-06_role_residual_equivalence_and_novelty.md)
证明双通道与普通 softmax 残差逐点精确互换：同潜变量、同损失的输出和梯度相同。
既有 3400 行预测重建后类别不变；两类预算形成耦合凸多面体，但不证明网络训练为凸。
同数值的双预算与普通残差 span 不等容量，后续对照须分离坐标、可行域和实际学习机制。
PB 与 REPAIR 已覆盖通用边界及成对残差思路；本轮不实施新网络或宣称性能收益。
[重放验证](outputs/theory/abmp_role_residual_equivalence_v1/reproduction_verification.json)
保留输入和代码哈希，正式报告保持不变。

同日 [角色图头核查](docs/experiment_records/2026-10-06_role_graph_head_audit.md)
进一步排除固定图线性聚合作为独立输出能力的主张：3072 行合成输出可精确合并为普通头。
无约束树图的正边权自适应不改变输出；遗漏脉石会降低输出秩。
这些是标准代数结构的设计前核查，不是矿物测试结果。下一候选应验证角色条件表示学习的
独立作用，不能只改变输出图名称；规格与训练范围仍待确认。

2026-10-06 用户同意先写轻量角色适配规格，已形成
[LC-RFA-B v1 书面设计](docs/superpowers/specs/2026-10-06-lightweight-role-adaptation-design.md)。
候选使用冻结池化特征后的三个非线性瓶颈适配器及共享读出，拟训练参数共 3024；
包含网络草图、双层幅度/局部稳定性推导、有限纠错包络和精确同参数 S1 强对照。
公式中的输出限制不是召回或泛化保证。用户已认可规格，现已形成
[LC-RFA-B v1 实施计划](docs/superpowers/plans/2026-10-06-lightweight-role-adaptation-implementation.md)，
状态为待计划审核；推荐先进行代码、合成验收与结构图，真实开发训练单独登记获准后执行。
该条目是 2026-10-06 设计时点的状态；后续实际实施与研究状态见下方更新。

2026-10-12 更新：LC-RFA 已完成代码、合成验收及一次获准的完整开发重跑，
[v2 负结果记录](docs/experiment_records/2026-10-12_lc_role_v2_results.md)
明确注册筛选未通过，R1/S1 三种子的开发 stop argmax 完全相同。
[全量梯度诊断与查新](docs/experiment_records/2026-10-12_lc_role_gradient_mechanism_diagnosis.md)
分析了串联零初始化、正则化收缩与条件性核关系；没有证明角色收益。

用户现已同意先写新网络规格，形成
[IC-RFA-B v1 恒等中心化书面设计](docs/superpowers/specs/2026-10-12-identity-centered-role-adaptation-design.md)。
新候选固定原始输入上的函数参照，拟训练3024参数、另存3024固定函数参照参数，
通过参数化/正则化/路由的交叉控制及同修复普通路由检验角色净作用。
用户已确认书面规格，现形成
[IC-RFA-B v1 实施计划](docs/superpowers/plans/2026-10-12-identity-centered-role-adaptation-implementation.md)，
用户已批准原生代码与合成验收，Tasks1--7已完成：最终60项新测试、33项旧纯合成回归通过，
一次340行、12单元、20步的原生合成计时与安全清理通过。未开展真实训练，也未修改正式报告。
详见[IC-RFA实施验收记录](docs/experiment_records/2026-10-13_ic_role_implementation_acceptance.md)、
[公式结构图PDF](outputs/paper_figures_ic_rfa_v1/final/ic_rfa_architecture.pdf)及
[一次合成计时证据](outputs/ic_role_implementation_qa_v1/acceptance_summary.json)。
独立代码审核发现的四项重要问题已通过一次TDD修复处理，详见
[审核与修复记录](docs/experiment_records/2026-10-13_ic_role_final_review.md)。合成计时未重跑，原记录保留其源码身份。
首轮只拟做现有fit分组的有限机制探查，stop评价与新数据确认另行设计批准。
2026-10-13 用户仅批准协议编写与GitHub登记，现形成
[IC-RFA fit-only机制预登记](docs/experiment_protocols/ic_role_mechanism_v1.md)和
[机器协议草稿](docs/experiment_protocols/ic_role_mechanism_v1.json)：固定340个fit分组、108条最多400步轨迹、
另最多3次温度拟合、累计7200秒。协议状态为DRAFT_NOT_AUTHORIZED，不能运行；
源文件、环境、对照、100步主点及失败停止条件均登记，运行仍需新的明确批准。
本次未读取真实输入内容、未训练、未创建真实输出目录、未修改正式报告。
身份与授权门核验见[机制协议登记记录](docs/experiment_records/2026-10-13_ic_role_mechanism_preregistration.md)。
用户随后明确回答“批准一次真实机制实验”，现仅补录
[单次运行授权](docs/experiment_records/2026-10-13_ic_role_mechanism_authorization.md)并重新登记机器协议。
运行仍限上述固定fit-only范围；准备、训练、诊断及最终推理验收共同受7200秒上限约束，
网络/损失/网格不变；登记和授权不等于真实效果已验证。
该单次运行在首个prepare任务因RUNNING状态未写回而失败，0条训练轨迹、0次温度拟合，
未反序列化特征；许可已消费，不续跑或重试。见[执行失败与状态同步诊断](docs/experiment_records/2026-10-13_ic_role_mechanism_v1_failure.md)。
失败不构成模型效果负结果；执行器修复/合成验收及新的真实运行需另行批准。
用户随后批准仅修复状态发布并做合成验收；生产代码仅增加派发前的RUNNING落盘，
三项回归已RED到GREEN，63项IC-RFA与33项旧纯合成回归通过，见
[状态发布修复与合成验收](docs/experiment_records/2026-10-13_ic_role_state_publication_repair.md)。
旧真实协议/失败目录保留且不复用；修复不是新模型效果，下一次真实实验须重新登记并单独批准。
独立工程复查因调用额度未返回结论，仅完成作者自审，不宣称独立审核通过。
用户已批准“新协议登记，暂不运行”，现新增
[修复后v2执行预登记](docs/experiment_protocols/ic_role_mechanism_v2.md)及
[v2未授权机器协议](docs/experiment_protocols/ic_role_mechanism_v2.json)。模型仍为IC-RFA-B v1，
只更新修复后34源码身份和独立输出目录；340fit/108轨迹/3温度/7200秒范围不变。
机器状态DRAFT_NOT_AUTHORIZED，独立复查尚未完成；未读真实输入、未训练、未重计时、未改正式报告。
登记与静态授权门核验见[v2登记记录](docs/experiment_records/2026-10-13_ic_role_mechanism_v2_preregistration.md)。
真实运行仍需单独明确批准，旧失败目录和已消费许可不复用，Task8整体未完成。
随后按新批准完成一次[独立只读工程复查](docs/experiment_records/2026-10-13_ic_role_state_publication_independent_review_v2.md)，
未发现新增的范围内工程问题；[审阅原文](outputs/ic_role_state_publication_review_v2/reviewer_response.md)保留全部未覆盖项。
此结果只更新限定静态复查状态，未运行测试/实验、未改源码或协议，v2仍未授权真实运行。
最终公开验收的剩余7200秒监督前提仍必须落实，理论新颖性和实际角色净纠错仍待验证。
2026-10-14用户明确批准一次固定v2协议的真实fit-only机制实验，
[单次授权记录](docs/experiment_records/2026-10-14_ic_role_mechanism_v2_authorization.md)仅补真实授权envelope并重新登记，
范围仍为340fit/108轨迹/3温度/7200秒，最终公开推理验收受剩余累计预算监督。
登记和许可不是效果结论；不续跑v1、不读stop/原图、不改网络或正式报告，失败即停、不重试。
该单次实验已完整完成，108轨迹/3T0/972检查点，含监督最终公开验收累计2299秒，
见[v2真实机制结果及负结论](docs/experiment_records/2026-10-14_ic_role_mechanism_v2_results.md)。
固定100步的C-S-R/C-S-U三个种子各340张逐图预测完全相同，目标净纠错均0；
平均NLL差仅-0.000015114652，不称选矿收益。结构数值核验通过不等于理论原创或效果成立。
已按登记规则停止该候选效果晋级，不读stop、不扩网格、不重训、不改正式报告；许可已消费。
[轻量摘要](outputs/training/ic_role_mechanism_v2/development_fold_0/summary/result_summary.json)和完整JSON证据归档，原始特征/权重仍仅本地保留。

2026-10-14经另行批准完成[离线理论审计](docs/experiment_records/2026-10-14_ic_role_offline_theory_audit.md)：
仅保存结果算术显示108个锚点错误中92个在现预算内不可纠正，全部预设检查点角色适配开/关无改判。
见[轻量诊断数据](outputs/ic_role_offline_theory_audit_v1/diagnosis.json)和
[下一版角色间隔网络建议](docs/experiment_records/2026-10-14_margin_budget_role_network_recommendation.md)。
建议未构成新网络开发/真实运行授权，正式报告未修改，原创性和独立确认均待验证。

另行批准后的[机制筛选与限域查新](docs/experiment_records/2026-10-14_margin_scaled_role_mechanism_screening.md)
形成[MSB-RRE v1待审规格](docs/superpowers/specs/2026-10-14_margin_scaled_bilinear_role_evidence_design.md)：
间隔缩放预算、角色双线性证据、内部可达/边界控制推导和普通/凸强对照，附公式化方法图。
[作者自审](docs/experiment_records/2026-10-14_msb_role_spec_self_review.md)明确bias-only也可满足部分理论性质，
不能把公式或图示当网络效果；本轮无真实数据读取、前向、拟合或实现，待规格审核后再写计划。
随后用户认可规格并限定编写[MSB-RRE v1原生实施计划](docs/superpowers/plans/2026-10-14-margin-scaled-role-evidence-implementation.md)。
计划区分代码/合成验收、真实协议登记、真实机制实验及独立确认；当前只有计划，没有实施或新的运行授权。
C阶段条件化路线继续后置；中心化、L2-SP与核缩放不能单独当作原创理论。

2026-10-15在另行批准Tasks1--8之后，已完成[MSB-RRE v1原生实现与合成验收](docs/experiment_records/2026-10-14_msb_role_native_synthetic_acceptance.md)：
八个限定合成模块83项通过；[条件数值证据](outputs/msb_role_synthetic_v1/conditional_theory_checks.json)及
[候选公式结构图](outputs/msb_role_synthetic_v1/assets/msb_rre_v1_architecture.svg)单独归档。
网络为1083个可训练参数+1083个固定初始参照参数的独立角色适配头，不改CNN主干；真实效果和原创性尚未验证。
新实现保留七参照逐seed证伪门及全部反例；最终615次验证回读之外还需12次训练侧凸证书核验。
没有读取真实prepared/fit/stop或原图、没有真实拟合、没有更改正式报告；旧34源码和保护文件哈希保持不变。
当前状态为等待单独批准的独立只读复查。真实MSB协议登记、真实机制实验与报告更新均未获准，不沿用旧IC许可。

随后获准的一次[独立只读工程与数学复查](docs/experiment_records/2026-10-15_msb_role_independent_review.md)
于2026-10-16因代理额度限制终止，未返回审阅结论，状态为`REVIEW_UNAVAILABLE_NO_VERDICT`。
[工具终止原文](outputs/msb_role_independent_review_v1/reviewer_response.md)不是审核报告；不能写成“独立复查通过”或“未发现问题”。
已按单次边界停止并关闭代理，没有重派、重跑测试或真实实验，也未修改代码和正式报告。新的独立复查须重新授权；Task9仍未获准。

2026-10-16获新的单次授权后，完成[第二次独立只读复查](docs/experiment_records/2026-10-16_msb_role_independent_review_v2.md)，
[完整原文](outputs/msb_role_independent_review_v2/reviewer_response.md)单独归档。核心条件数学及网络/损失基本一致，
但有1项P1及2项P2工程/身份合同缺口：陈旧RUNNING可脱离父监督、非fit训练scope未严格拒绝、联合三折验收晚于拟合。
状态为`REVIEW_COMPLETED_CHANGES_REQUIRED`，不作为工程审核通过或真实协议冻结依据。
本轮只记录并核对引用，未修代码、重跑测试、读真实输入、开展训练或改正式报告；修复规格/实施计划及后续工作另需批准。

随后用户仅批准三项修复的文档阶段，已形成[执行合同修复规格](docs/superpowers/specs/2026-10-16-msb-execution-contract-repair-design.md)与
[配套实施计划](docs/superpowers/plans/2026-10-16-msb-execution-contract-repair.md)，[授权与作者自审记录](docs/experiment_records/2026-10-16_msb_execution_contract_repair_planning.md)单独保存。
方案保持网络/损失/公式及80任务、75拟合不变，将联合三折验收前移到首次拟合之前，并设计活跃父监督及继承累计期限。
当前为`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`，只编写与归档文档；未修源码、运行测试或访问真实数据，开发和合成验收仍待用户审核及另行批准。

2026-10-16随后获准原生实施上述修复，已完成[执行合同修复与合成验收记录](docs/experiment_records/2026-10-16_msb_execution_contract_repair_acceptance.md)：
[最终v3守卫验收](outputs/msb_execution_repair_synthetic_v3/acceptance_summary.json)九模块121项通过，受限接口未模拟调用为0，旧34源码及10项保护文件不变。
原网络/损失/公式、80任务/75拟合矩阵及正式报告未改；新增安全模块后未来执行闭包为43项。
早期非法入口回归的只读系统时钟采样、v1验收器失败及修正均已单独披露，不反向标成完全模拟或模型效果。
当前为`AUTHOR_REPAIR_SYNTHETIC_ACCEPTED_INDEPENDENT_REVIEW_PENDING`：尚无新独立复查、原生监督验收或真实MSB结果，不据此宣称网络改进有效。

随后获新的单次授权，对`8de3804..0bb95a4`完成[执行合同修复独立只读复查](docs/experiment_records/2026-10-16_msb_execution_contract_repair_independent_review.md)，
[完整原文](outputs/msb_execution_repair_independent_review_v1/reviewer_response.md)保留四项P2：最终成功写入跨期、启动后异常进程所有权丢失、退出早于ACK/DONE读完、RENEW/DONE终结竞争。
当前代码裁决为`CHANGES_REQUIRED`，原生监督为`INSUFFICIENT_EVIDENCE`；精确fit-only及拟合前联合三折校验未发现新增问题，但整体不能据此冻结真实运行。
本轮未重跑测试、前向、拟合或原生验收，未读真实输入，未改代码和正式报告。121项仍是既有作者合成记录；v1重建验收器快照及早期时钟模拟偏差保留来源限定。
下一步先另批上述四项的修复规格与实施计划；实际修复、原生无数据验收及真实实验仍分别另行批准，不把工程进展写成理论或网络效果提升。

2026-10-17用户批准仅编写上述四项的文档，已形成[终结边界修复规格](docs/superpowers/specs/2026-10-17-msb-terminal-boundary-repair-design.md)和
[原生实施计划](docs/superpowers/plans/2026-10-17-msb-terminal-boundary-repair.md)，[授权及作者自审](docs/experiment_records/2026-10-17_msb_terminal_boundary_repair_planning.md)单独归档。
方案为唯一成功发布、启动所有权记录、显式CLOSE_REQUEST/CLOSE_GRANT/DONE握手及有界退出排空；不宽松接受旧序号，不增加等待额度。
状态`DOCUMENTED_NOT_IMPLEMENTED_NOT_RUN`：未改代码/测试/网络/真实协议/正式报告，未读取真实输入或执行任何实验；原工程裁决仍需修复。
书面文件待用户审核，开发及合成验收另批。真实强对照效果与理论创新尚未验证，不因文档完成改变网络成果或真实运行许可。

Run these commands from the repository root with the fixed manifest and a locally
authorized image directory. The analyses read existing data and prediction outputs;
they do not alter the frozen split.

For the candidate-set command, replace `"<外部原始图片数据根目录>"` with an
external, license-compliant source-image root. It must contain every image at the
path specified by that record's `relative_path` in the fixed manifest; image files
are not distributed by this repository.

```powershell
.\.venv-training\Scripts\python.exe .\scripts\analyze_role_identifiability.py `
  --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv `
  --dataset-root "<外部原始图片数据根目录>" `
  --output-dir .\outputs\theory_validation\role_identifiability `
  --candidate-sizes 2 3 4 `
  --seed 20260801

.\.venv-training\Scripts\python.exe .\scripts\analyze_selective_recognition.py `
  --input-glob "outputs\training\formal_hierarchical_efficientnet_b0_seed*\test_predictions.csv" `
  --output-dir .\outputs\theory_validation\selective_recognition `
  --figure .\outputs\paper_figures_v1\fig9_selective_recognition.png

.\.venv-training\Scripts\python.exe .\scripts\build_technical_report.py
```

Outputs: [candidate-set JSON summary](outputs/theory_validation/role_identifiability/role_identifiability_summary.json), [candidate-set Markdown summary](outputs/theory_validation/role_identifiability/role_identifiability_summary.md), [selective-recognition JSON summary](outputs/theory_validation/selective_recognition/selective_recognition_summary.json), [selective-recognition Markdown summary](outputs/theory_validation/selective_recognition/selective_recognition_summary.md), [Figure 9](outputs/paper_figures_v1/fig9_selective_recognition.png), [Figure 10](outputs/paper_figures_v1/fig10_theory_aware_hierarchical_architecture_cn.png), and the [technical report](结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（初稿）.docx).

Source images and trained model weights are not redistributed. The manifest and metadata support auditability, but reproduction requires separately obtained images whose licenses permit their use. Candidate-set results are controlled logical-condition validation; selective-recognition results describe the fixed test split and are not claims of industrial sorting, XRF, or cost optimization.
