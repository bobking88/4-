# 目标代理类/非目标类业务指标汇总

## 指标定义

将 `target_mineral` 折叠为目标代理类，其余三类折叠为非目标类。“误入目标比例”表示某个非目标类别被预测为 `target_mineral` 的比例；“漏选率”表示真实目标代理类未被预测为 `target_mineral` 的比例。

## 各运行结果

| 运行 | 目标 Precision | 目标 Recall | 目标 F1 | 漏选率 | 含钛干扰误入目标 | 金属光泽干扰误入目标 | 脉石误入目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| formal_role_aware_efficientnet_b0_seed20260727 | 73.03% | 70.12% | 71.54% | 29.88% | 7.54% | 10.80% | 2.46% |
| formal_role_aware_efficientnet_b0_seed20260728 | 75.00% | 70.52% | 72.69% | 29.48% | 7.54% | 9.09% | 1.64% |
| formal_role_aware_efficientnet_b0_seed20260729 | 71.37% | 64.54% | 67.78% | 35.46% | 8.15% | 9.09% | 2.46% |

## 三随机种子统计

| 指标 | 均值 | 样本标准差 |
|---|---:|---:|
| target_precision | 73.13% | 1.82% |
| target_recall | 68.39% | 3.34% |
| target_f1 | 70.67% | 2.57% |
| target_miss_rate | 31.61% | 3.34% |
| ti_bearing_intrusion_rate | 7.74% | 0.35% |
| metallic_intrusion_rate | 9.66% | 0.98% |
| gangue_intrusion_rate | 2.19% | 0.47% |

## 使用边界

上述指标衡量公开矿物标本图像上的类别层面代理识别风险，不等同于工业分选回收率、精矿品位或生产线实际漏选率。
