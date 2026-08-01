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

开放集评价工具位于 `scripts/evaluate_open_set_protocol.py`。它需要独立、经核验的未知矿物图像预测表；当前仓库不以闭集四分类测试集伪造未知矿物结果。

## 数据说明与边界

本仓库不分发原始矿物图片、模型权重、虚拟环境或下载缓存。原始图像来自公开矿物图像页面，仍须遵守各图片页面的署名、许可和使用条件。仓库保留来源元数据、质量控制记录和最终数据清单，以支持研究过程审计和合法复现。

当前结果是基于公开矿物标本图像的“钒钛矿相关矿物识别”基线，不应直接解释为工业传送带场景下的实际分选性能。
## Theory-aware evidence reproduction

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
