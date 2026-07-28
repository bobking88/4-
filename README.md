# 钒钛矿相关矿物图像识别

本仓库保存“人工智能选矿”项目中，面向钒钛矿相关成分矿物图像识别的数据治理记录、训练代码、人工复核结果和正式实验输出。研究任务采用四分类设置：目标矿物、含钛干扰矿物、脉石/废石和金属光泽干扰矿物。

## 当前版本

- 最终数据版本：`dataset_final_v1`
- 图像总数：8,529 张
- 固定划分：训练集 5,961 张，验证集 1,284 张，测试集 1,284 张
- 人工复核：分层抽检 823 张，排除 32 张，2 张不确定样本未进入训练
- 正式模型：ResNet50 与 EfficientNet-B0，各 3 个随机种子
- 最佳平均结果：EfficientNet-B0 的 Macro F1 为 `72.87% ± 1.21%`

完整实验汇总见 [outputs/training/formal_experiment_summary_v1.md](outputs/training/formal_experiment_summary_v1.md)。

## 仓库结构

```text
scripts/                    数据集审计、人工复核、训练与验证脚本
tests/                      数据处理与训练辅助函数测试
数据集/dataset_audit/        初始数据审计、质量问题与固定划分清单
数据集/dataset_final_v1/     人工复核后的最终训练清单和排除记录
数据集/dataset_review_20260727/
                            人工复核队列、决策汇总与复核说明
outputs/training/            六组正式实验的指标、混淆矩阵与汇总
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

## 数据说明与边界

本仓库不分发原始矿物图片、模型权重、虚拟环境或下载缓存。原始图像来自公开矿物图像页面，仍须遵守各图片页面的署名、许可和使用条件。仓库保留来源元数据、质量控制记录和最终数据清单，以支持研究过程审计和合法复现。

当前结果是基于公开矿物标本图像的“钒钛矿相关矿物识别”基线，不应直接解释为工业传送带场景下的实际分选性能。
