# 2026-08-25 未知矿物拒识（OOD）数据就绪性审计

## 审计目的

确认项目是否已具备可报告的未知矿物拒识实验条件。原则是：未知集必须由不属于当前 17 个细粒度矿物种类的、可追溯且经筛选的图像组成；不得从闭集四分类测试集、错误样本或低置信度样本中人为拼接“未知类”。

## 已检查材料

- 评估脚本：`scripts/evaluate_open_set_protocol.py`。
- 补充下载清单：`D:\成信工科研\人工智能选矿\数据集\open_source_supplement_v1\metadata\open_source_images_manifest.csv`。
- 当前最终清单：17 个细粒度矿物种类，属于四类视觉角色任务。

## 审计结果

1. `evaluate_open_set_protocol.py` 仅接收独立的已知与未知预测分数表，并显式拒绝用闭集四分类结果伪造 OOD 指标；该设计可保留。
2. `open_source_supplement_v1` 共记录 20 条：`magnetite_proxy` 10 条、`ilmenite_ti_mineral` 10 条。两者都属于本项目已经使用的目标矿物来源，不能作为未知矿物。
3. 该补充清单中仅 2 条为 `keep_prelim`，6 条为 `rejected_reference`，12 条为 `download_failed`；数量和质量均不足以形成独立测试集。
4. 因此，截至本审计日期，项目**没有可用于报告 AUROC、FPR95 或拒识覆盖率的独立未知矿物测试集**。技术报告和论文只能把 OOD 写为后续研究，不得发布数值结果。

## 后续采集规范

建立独立 OOD 集时，每条图像至少保留：`mineral_label`、`detail_page_url`、`source_record_id`、`screening_decision`、`local_path`、`sha256`、`source_group_id`、`reviewer` 和 `review_date`。未知矿物种类不得与现有 17 类重合；同一图片编号、近重复组、摄影者或来源组不得同时出现在已知训练数据与未知测试数据中。

建议先由矿物加工或地质专业人员确认候选未知种类，再按照与主数据集一致的主体面积、清晰度、宏观成像和混合矿物筛选规则采集。只有在独立未知集冻结后，才可比较最大 Softmax 概率、预测熵、Energy Score 等拒识基线，并在固定已知测试集与独立未知集上共同报告指标。

## 结论

当前可完成的论文主实验仍是 RSG-HRGV 的固定划分与摄影者留出机制验证；OOD 不是本轮实验的缺失补丁，而是需要先满足独立数据条件的后续扩展。
