# RSG-HRGV 高精度检查点重放与定理数值一致性（2026-08-31）

## 目的

原始 `test_predictions.csv` 为便于人工审阅，概率字段只保存六位小数。极小的真值类概率可能在该导出层被写为 `0.000000`，因而不能用该文件直接对 RSG-HRGV 的理论不等式进行浮点数值核验。本记录使用已保存的正式模型检查点，以只读方式重新执行测试推理，并导出 15 位小数的门控和真值类概率。

本实验验证的是实现一致性，不是新增的模型性能比较。

## 冻结输入

- 固定测试：`rsg_controlled/formal_rsg_complete` 的种子 20260727、20260728、20260729。
- 摄影者留出：`rsg_source_holdout/formal_rsg_complete` 的种子 20260727、20260729、20260730。
- ResNet50 主干替换确认：`rsg_resnet50_portability/resnet50/formal_rsg_complete` 的种子 20260727、20260728、20260729。
- 每次重放均读取对应 `environment.json`、`best_model.pt`、冻结清单和数据根目录；不更新权重、不进行反向传播、不覆盖原始训练输出。
- 输出位置：`outputs/paper_experiments_v3/rsg_theory_replay/`。

## 数值检查

对每张测试图像，使用高精度导出的 \(a=p_d(y\mid x)\)、\(b=p_m(y\mid x)\)、\(g\)、\(g_o\)、\(g^*\) 和路由后悔项，检查：

\[
-\log p_f(y\mid x)+\log\max(a,b)
\leq\frac{|g-g_o|\,|a-b|}{\epsilon},
\]

\[
|g^*-g_o|\leq\exp\left(-\frac{|\log a-\log b|}{T_r}\right).
\]

其中 \(\epsilon=\text{float32 eps}=1.1920928955078125\times10^{-7}\)，\(T_r=0.2\)，数值容差为 \(2\times10^{-6}\)。

## 结果

| 项目 | 结果 |
| --- | ---: |
| 重放次数 | 9 |
| 覆盖测试图像 | 10,242 |
| 最小导出真值概率 | \(1.19\times10^{-7}\) |
| B.1 最大残差 | \(0\) |
| B.1 违反数 | 0 |
| B.2 最大残差 | \(5.94\times10^{-8}\) |
| B.2 违反数 | 0 |

结果表明，定理 B.1 与 B.2 所采用的概率、软目标、门控和路由后悔关系，与 9 个已训练 RSG-HRGV 检查点的高精度测试输出数值一致。其中 ResNet50 三检查点覆盖 3,852 张图像，B.1 与 B.2 的违反数也均为 0；这补强了主干替换不变性命题的实现证据，但不把该命题写成任意主干的分类性能保证。

## 解释边界

本结果只证明已实现计算图在浮点容差下满足所述数值关系。它不证明 Accuracy、Macro F1、目标类召回或误入目标比例必然提升，也不构成工业分选、品位、回收率、元素含量、真实工况泛化或未知矿物拒识结论。

## 复现

单个检查点重放：

```powershell
.\.venv-training\Scripts\python.exe scripts\replay_hrgv_routing_diagnostics.py `
  --environment outputs\training\rsg_controlled\formal_rsg_complete_seed20260727\environment.json `
  --checkpoint outputs\training\rsg_controlled\formal_rsg_complete_seed20260727\best_model.pt `
  --output-csv outputs\paper_experiments_v3\rsg_theory_replay\fixed\seed20260727.csv `
  --device cuda
```

汇总：

```powershell
.\.venv-training\Scripts\python.exe scripts\analyze_rsg_theory_replay.py `
  --replay-root outputs\paper_experiments_v3\rsg_theory_replay `
  --output-dir outputs\business_metrics\rsg_hrgv\theory_replay_portability
```
