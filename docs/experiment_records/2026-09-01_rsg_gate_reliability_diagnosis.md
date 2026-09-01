# RSG-HRGV 门控可靠性分层诊断（2026-09-01）

## 目的

在已经完成 B.1/B.2 高精度数值一致性验证的基础上，本记录进一步检验两条公式如何解释不同专家分歧样本的路由难度。分析仅读取保存的 9 个 RSG-HRGV 最佳检查点重放输出，不训练、不微调、不覆盖既有分类结果。

## 局部 B.1 推论

对第 \(i\) 张图像定义：

\[
\epsilon_i=\min\{a_i,b_i\},\quad
a_i=p_d(y_i\mid x_i),\quad b_i=p_m(y_i\mid x_i).
\]

将 B.1 的条件 \(a_i,b_i\geq\epsilon_i\) 逐图代入，得到：

\[
r_i\leq B_i^{\mathrm{loc}}
=\frac{|g_i-g_{o,i}|\,|a_i-b_i|}{\epsilon_i}.
\]

这是一条直接推论，用于提供比统一 float32 下界更紧的诊断尺度；不是新的网络结构、训练目标或泛化理论。

## 冻结输入

- 固定测试：3 个 EfficientNet-B0 RSG-HRGV 检查点，3,852 张图像；
- 摄影者留出：3 个 EfficientNet-B0 RSG-HRGV 检查点，2,538 张图像；
- ResNet50 主干替换：3 个 RSG-HRGV 检查点，3,852 张图像；
- 合计：9 次只读高精度重放，10,242 张图像。

每个协议内，分别按局部 B.1 上界 \(B_i^{\mathrm{loc}}\) 和 B.2 的专家对数差 \(|\log a_i-\log b_i|\) 排序为三个等量层。由于这些层来自固定检查点输出，本记录将其定位为描述性机制诊断，不进行新的性能显著性推断。

## 结果

| 协议 | B.1 T1 平均后悔 | B.1 T3 平均后悔 | B.2 T1 软硬偏差 | B.2 T3 软硬偏差 | B.1 / B.2 违反数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定测试 | 0.00038 | 0.22702 | 0.49895 | 0.13979 | 0 / 0 |
| 摄影者留出 | 0.00141 | 0.22066 | 0.49622 | 0.13381 | 0 / 0 |
| ResNet50 跨主干 | 0.00013 | 0.24624 | 0.49967 | 0.14212 | 0 / 0 |

在三种协议中，局部 B.1 上界较高的样本层同时具有更高的实际平均路由后悔；专家对数差较大的样本层同时具有更小的软目标--硬最优门控偏差。所有高精度样本均满足 B.1 局部推论和 B.2 指数界的数值容差要求。

## 解释边界

本结果支持“当前 RSG-HRGV 计算图的门控难度与 B.1/B.2 量之间存在一致的分层关系”。它不证明 RSG 必然提高 Accuracy、Macro F1、目标类召回或两类误入目标比例，也不构成工业分选、品位、回收率、元素含量、真实外部泛化或未知矿物拒识结论。

## 复现

```powershell
.\.venv-training\Scripts\python.exe scripts\analyze_rsg_gate_reliability.py `
  --replay-root outputs\paper_experiments_v3\rsg_theory_replay `
  --output-dir outputs\business_metrics\rsg_hrgv\gate_reliability `
  --strata-count 3

.\.venv-training\Scripts\python.exe scripts\generate_rsg_gate_reliability_figure.py `
  --b1-strata-csv outputs\business_metrics\rsg_hrgv\gate_reliability\b1_local_bound_strata.csv `
  --b2-strata-csv outputs\business_metrics\rsg_hrgv\gate_reliability\b2_margin_strata.csv `
  --summary-json outputs\business_metrics\rsg_hrgv\gate_reliability\gate_reliability_summary.json `
  --output-prefix outputs\paper_figures_v3\fig_rsg_gate_reliability
```
