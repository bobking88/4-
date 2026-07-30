# 钒钛矿相关矿物图像识别技术报告 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份忠实呈现现有数据与实验结论、可用于项目结题的中文技术报告 DOCX 初稿。

**Architecture:** 从现有训练汇总、数据清单、质量审计、错误分析和论文级图表中读取已验证证据；通过一个可重复运行的文档生成脚本输出 DOCX；使用统一的正式报告样式、插图和表格；最终将文档渲染为页面图像进行人工版式复核。

**Tech Stack:** Python 3、python-docx、Pillow、现有 PNG/PDF 图表、LibreOffice/文档渲染工具。

## Global Constraints

- 仅引用工作区中已验证的实验数据，不推断未完成的工业应用指标。
- 公开标本图像定位为方法验证数据，禁止声称实现工业现场分选或精矿品位预测。
- 项目编号、单位、负责人、经费等未经核验的字段必须明确标为“待填”。
- 保留原模板不改动，最终文件单独输出到 `结题` 目录。

---

### Task 1: 汇集并校验报告证据

**Files:**
- Read: `outputs/training/formal_experiment_summary_v1.md`
- Read: `outputs/training/focal_loss_ablation_seed20260728.md`
- Read: `outputs/error_analysis/efficientnet_b0_seed20260728/`
- Read: `数据集/dataset_final_v1/dataset_split_manifest_v1_0.csv`
- Read: `outputs/paper_figures_v1/figure_legends.md`

- [ ] 提取数据集总量、类别分布和训练/验证/测试划分。
- [ ] 提取 ResNet50、EfficientNet-B0 三随机种子指标和 Focal Loss 单种子消融结论。
- [ ] 提取关键错误对和数据质量审计结论。
- [ ] 逐项比对图表文件是否可用，建立报告插图清单。

### Task 2: 生成技术报告 DOCX 初稿

**Files:**
- Create: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（初稿）.docx`
- Create: `scripts/build_technical_report.py`

- [ ] 使用正式中文报告样式建立封面、目录、正文、图表与附录。
- [ ] 写入设计说明列出的十一个章节，并在适当位置插入五张现有图表。
- [ ] 写明实验可复现入口、数据来源与使用边界。
- [ ] 将所有未知项以“待填”保留，不填入假定事实。

### Task 3: 文档质量检查与交付

**Files:**
- Create: `结题/技术报告_版式检查/`
- Verify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（初稿）.docx`

- [ ] 渲染 DOCX 为逐页 PNG，并逐页查看文字、图片、表格和分页。
- [ ] 修复可见的裁切、重叠、孤行或图文脱节问题并重新渲染。
- [ ] 运行文档结构检查，确认全部章节、图表和引用文件存在。
- [ ] 输出可编辑 DOCX 初稿及简短的待补信息清单。
