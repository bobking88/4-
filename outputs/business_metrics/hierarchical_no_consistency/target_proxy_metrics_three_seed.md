# 目标代理类/非目标类业务指标汇总

## 指标定义

将 `target_mineral` 折叠为目标代理类，其余三类折叠为非目标类。“误入目标比例”表示某个非目标类别被预测为 `target_mineral` 的比例；“漏选率”表示真实目标代理类未被预测为 `target_mineral` 的比例。

## 各运行结果

| 运行 | 目标 Precision | 目标 Recall | 目标 F1 | 漏选率 | 含钛干扰误入目标 | 金属光泽干扰误入目标 | 脉石误入目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| formal_hierarchical_no_consistency_seed20260727 | 61.82% | 81.27% | 70.22% | 18.73% | 14.46% | 18.18% | 6.28% |
| formal_hierarchical_no_consistency_seed20260728 | 68.63% | 74.10% | 71.26% | 25.90% | 10.39% | 11.36% | 3.83% |
| formal_hierarchical_no_consistency_seed20260729 | 67.02% | 76.10% | 71.27% | 23.90% | 11.20% | 14.77% | 3.55% |

## 三随机种子统计

| 指标 | 均值 | 样本标准差 |
|---|---:|---:|
| target_precision | 65.82% | 3.56% |
| target_recall | 77.16% | 3.70% |
| target_f1 | 70.92% | 0.60% |
| target_miss_rate | 22.84% | 3.70% |
| ti_bearing_intrusion_rate | 12.02% | 2.16% |
| metallic_intrusion_rate | 14.77% | 3.41% |
| gangue_intrusion_rate | 4.55% | 1.50% |

## 使用边界

上述指标衡量公开矿物标本图像上的类别层面代理识别风险，不等同于工业分选回收率、精矿品位或生产线实际漏选率。
