# 目标代理类/非目标类业务指标汇总

## 指标定义

将 `target_mineral` 折叠为目标代理类，其余三类折叠为非目标类。“误入目标比例”表示某个非目标类别被预测为 `target_mineral` 的比例；“漏选率”表示真实目标代理类未被预测为 `target_mineral` 的比例。

## 各运行结果

| 运行 | 目标 Precision | 目标 Recall | 目标 F1 | 漏选率 | 含钛干扰误入目标 | 金属光泽干扰误入目标 | 脉石误入目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| formal_hierarchical_no_contrast_seed20260727 | 61.85% | 80.08% | 69.79% | 19.92% | 14.46% | 17.61% | 6.01% |
| formal_hierarchical_no_contrast_seed20260728 | 69.23% | 75.30% | 72.14% | 24.70% | 10.18% | 12.50% | 3.28% |
| formal_hierarchical_no_contrast_seed20260729 | 69.04% | 77.29% | 72.93% | 22.71% | 9.78% | 14.77% | 3.55% |

## 三随机种子统计

| 指标 | 均值 | 样本标准差 |
|---|---:|---:|
| target_precision | 66.71% | 4.21% |
| target_recall | 77.56% | 2.40% |
| target_f1 | 71.62% | 1.63% |
| target_miss_rate | 22.44% | 2.40% |
| ti_bearing_intrusion_rate | 11.47% | 2.59% |
| metallic_intrusion_rate | 14.96% | 2.56% |
| gangue_intrusion_rate | 4.28% | 1.50% |

## 使用边界

上述指标衡量公开矿物标本图像上的类别层面代理识别风险，不等同于工业分选回收率、精矿品位或生产线实际漏选率。
