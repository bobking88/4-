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
