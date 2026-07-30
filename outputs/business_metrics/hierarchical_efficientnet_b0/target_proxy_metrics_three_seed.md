# 目标代理类/非目标类业务指标汇总

## 指标定义

将 `target_mineral` 折叠为目标代理类，其余三类折叠为非目标类。“误入目标比例”表示某个非目标类别被预测为 `target_mineral` 的比例；“漏选率”表示真实目标代理类未被预测为 `target_mineral` 的比例。

## 各运行结果

| 运行 | 目标 Precision | 目标 Recall | 目标 F1 | 漏选率 | 含钛干扰误入目标 | 金属光泽干扰误入目标 | 脉石误入目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| formal_hierarchical_efficientnet_b0_seed20260727 | 62.23% | 80.08% | 70.03% | 19.92% | 14.46% | 17.61% | 5.46% |
| formal_hierarchical_efficientnet_b0_seed20260728 | 71.91% | 76.49% | 74.13% | 23.51% | 7.74% | 11.93% | 4.37% |
| formal_hierarchical_efficientnet_b0_seed20260729 | 68.21% | 76.10% | 71.94% | 23.90% | 10.59% | 13.64% | 3.55% |

## 三随机种子统计

| 指标 | 均值 | 样本标准差 |
|---|---:|---:|
| target_precision | 67.45% | 4.89% |
| target_recall | 77.56% | 2.19% |
| target_f1 | 72.04% | 2.05% |
| target_miss_rate | 22.44% | 2.19% |
| ti_bearing_intrusion_rate | 10.93% | 3.37% |
| metallic_intrusion_rate | 14.39% | 2.92% |
| gangue_intrusion_rate | 4.46% | 0.96% |

## 使用边界

上述指标衡量公开矿物标本图像上的类别层面代理识别风险，不等同于工业分选回收率、精矿品位或生产线实际漏选率。
