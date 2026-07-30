# 角色感知困难负样本学习 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定四分类数据集上补齐 Focal Loss 三随机种子、建立目标代理类业务指标，并实现可复现的角色感知困难负样本学习实验与结题材料更新。

**Architecture:** 新增独立的业务指标分析脚本，从逐图预测折叠出目标代理/非目标二分类结果。新增独立训练脚本复用既有数据清单、数据增强和训练依赖，以 EfficientNet-B0 主干输出四分类、目标二分类和标准化嵌入；训练时以主分类损失、二分类损失和重点类别对比损失联合优化。所有正式运行保持相同清单和三个随机种子。

**Tech Stack:** Python 3.11、PyTorch 2.12、torchvision、scikit-learn、CSV/JSON、现有训练脚本与 unittest。

## Global Constraints

- 仅使用 `数据集/dataset_final_v1/dataset_split_manifest_v1_0.csv` 的固定划分。
- 不把公开标本图像结果解释为工业分选、精矿品位或回收率结果。
- `titanomagnetite` 仅作为合并目标代理类的少量样本，不单列性能。
- Focal Loss 与角色感知模型都必须在 `20260727`、`20260728`、`20260729` 三个随机种子上运行。
- GitHub 仅提交脚本、测试、汇总指标、图表源数据和报告文字；不提交原始图片、模型权重、缓存、论文原件或财务文件。

---

### Task 1: 目标代理类业务指标

**Files:**
- Create: `scripts/analyze_target_proxy_metrics.py`
- Create: `tests/test_analyze_target_proxy_metrics.py`
- Create: `outputs/business_metrics/`

**Interfaces:**
- Consumes: prediction CSV with `true_label` and `predicted_label` columns.
- Produces: `calculate_target_proxy_metrics(rows: list[dict[str, str]]) -> dict[str, float]` and a summary CSV/Markdown file.

- [ ] **Step 1: 写入失败测试，验证四分类折叠后的目标代理指标**

```python
def test_target_proxy_metrics_count_target_missed_and_negative_intrusions():
    rows = [
        {"true_label": "target_mineral", "predicted_label": "target_mineral"},
        {"true_label": "target_mineral", "predicted_label": "ti_bearing_negative"},
        {"true_label": "ti_bearing_negative", "predicted_label": "target_mineral"},
        {"true_label": "metallic_hard_negative", "predicted_label": "target_mineral"},
    ]
    result = calculate_target_proxy_metrics(rows)
    assert result["target_recall"] == 0.5
    assert result["target_miss_rate"] == 0.5
    assert result["ti_bearing_intrusion_rate"] == 1.0
    assert result["metallic_intrusion_rate"] == 1.0
```

- [ ] **Step 2: 运行测试并确认因模块缺失失败**

Run: `python -m unittest tests.test_analyze_target_proxy_metrics -v`

Expected: `ModuleNotFoundError` for `analyze_target_proxy_metrics`.

- [ ] **Step 3: 实现最小的折叠指标计算与文件汇总**

```python
TARGET_LABEL = "target_mineral"

def calculate_target_proxy_metrics(rows):
    target_truth = [row["true_label"] == TARGET_LABEL for row in rows]
    target_pred = [row["predicted_label"] == TARGET_LABEL for row in rows]
    tp = sum(t and p for t, p in zip(target_truth, target_pred))
    fn = sum(t and not p for t, p in zip(target_truth, target_pred))
    fp = sum(not t and p for t, p in zip(target_truth, target_pred))
    return {"target_recall": tp / (tp + fn), "target_miss_rate": fn / (tp + fn), "false_positive_count": fp}
```

- [ ] **Step 4: 运行单元测试和三个已有 EfficientNet-B0 预测结果的汇总**

Run: `python -m unittest tests.test_analyze_target_proxy_metrics -v`

Expected: PASS.

### Task 2: 补齐 Focal Loss 三随机种子实验

**Files:**
- Create: `outputs/training/formal_efficientnet_b0_focal_seed20260727/`
- Create: `outputs/training/formal_efficientnet_b0_focal_seed20260729/`
- Create: `outputs/training/focal_loss_ablation_three_seed_summary.md`
- Modify: `scripts/generate_paper_figures.py`
- Modify: `tests/test_generate_paper_figures.py`

**Interfaces:**
- Consumes: fixed manifest, dataset root, current `train_mineral_classifier.py` CLI.
- Produces: per-seed `test_metrics.json`, `confusion_matrix.csv`, `metrics_history.csv` and a three-seed loss summary.

- [ ] **Step 1: 写入失败测试，要求图表汇总能读取三种子 Focal 路径**

```python
def test_focal_seed_paths_include_all_three_formal_seeds():
    assert focal_seed_paths((20260727, 20260728, 20260729)) == [
        TRAINING_DIR / "formal_efficientnet_b0_focal_seed20260727" / "test_metrics.json",
        TRAINING_DIR / "formal_efficientnet_b0_focal_seed20260728" / "test_metrics.json",
        TRAINING_DIR / "formal_efficientnet_b0_focal_seed20260729" / "test_metrics.json",
    ]
```

- [ ] **Step 2: 运行测试并确认因路径函数缺失失败**

Run: `python -m unittest tests.test_generate_paper_figures -v`

Expected: import failure for `focal_seed_paths`.

- [ ] **Step 3: 实现路径函数，运行种子 20260727 和 20260729 的 Focal Loss 训练**

```powershell
python .\scripts\train_mineral_classifier.py --model efficientnet_b0 --loss focal --focal-gamma 2.0 --seed 20260727 --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv --dataset-root .\数据集\mindat_manual_positive_v1 --output-dir .\outputs\training\formal_efficientnet_b0_focal_seed20260727
```

- [ ] **Step 4: 生成三种子均值、标准差与谨慎结论**

Run: `python .\scripts\generate_paper_figures.py`

Expected: Focal 图表与 Markdown 汇总在三随机种子下明确显示均值和样本标准差。

### Task 3: 角色感知困难负样本模型

**Files:**
- Create: `scripts/train_role_aware_mineral_classifier.py`
- Create: `tests/test_role_aware_mineral_classifier.py`
- Create: `outputs/training/role_aware_efficientnet_b0_seed20260727/`
- Create: `outputs/training/role_aware_efficientnet_b0_seed20260728/`
- Create: `outputs/training/role_aware_efficientnet_b0_seed20260729/`

**Interfaces:**
- Consumes: `ManifestRecord`, `create_dataloaders`, `compute_class_weights`, `create_transforms` from `train_mineral_classifier.py`.
- Produces: `RoleAwareEfficientNet`, `target_binary_labels(labels)`, `compute_role_aware_contrastive_loss(embeddings, labels, temperature, torch)` and training outputs compatible with existing JSON/CSV conventions.

- [ ] **Step 1: 写入失败测试，验证目标二分类映射和重点对比损失**

```python
def test_target_binary_labels_map_only_target_class_to_one():
    labels = torch.tensor([0, 1, 2, 3])
    assert target_binary_labels(labels).tolist() == [1, 0, 0, 0]

def test_role_aware_contrastive_loss_is_smaller_for_separated_target_and_hard_negatives():
    labels = torch.tensor([0, 0, 1, 1, 3, 3])
    separated = torch.tensor([[1,0], [0.9,0.1], [-1,0], [-0.9,0.1], [0,-1], [0.1,-0.9]])
    mixed = torch.tensor([[1,0], [0.9,0.1], [0.8,0.2], [0.7,0.3], [0.6,0.4], [0.5,0.5]])
    assert compute_role_aware_contrastive_loss(separated, labels, 0.1, torch) < compute_role_aware_contrastive_loss(mixed, labels, 0.1, torch)
```

- [ ] **Step 2: 运行测试并确认模块缺失失败**

Run: `python -m unittest tests.test_role_aware_mineral_classifier -v`

Expected: `ModuleNotFoundError` for `train_role_aware_mineral_classifier`.

- [ ] **Step 3: 实现双头 EfficientNet-B0 与重点类别对比损失**

```python
class RoleAwareEfficientNet(nn.Module):
    def forward(self, images):
        return four_class_logits, binary_logits, normalized_embedding

total_loss = four_class_loss + lambda_binary * binary_loss + lambda_contrast * contrast_loss
```

- [ ] **Step 4: 运行 smoke test、完整单元测试和三种子正式训练**

Run: `python .\scripts\train_role_aware_mineral_classifier.py --smoke-run --seed 20260727 --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv --dataset-root .\数据集\mindat_manual_positive_v1 --output-dir .\outputs\training\role_aware_smoke_seed20260727`

Expected: smoke test produces test metrics and all role-aware unit tests pass before full runs.

### Task 4: 汇总、报告更新与发布

**Files:**
- Create: `outputs/business_metrics/target_proxy_metrics_three_seed.md`
- Create: `outputs/training/role_aware_experiment_summary.md`
- Create: `outputs/paper_figures_v2/`
- Create: `数据集/dataset_review_20260727/复核责任补录说明.md`
- Modify: `scripts/build_technical_report.py`
- Modify: `README.md`
- Test: existing tests plus new tests

**Interfaces:**
- Consumes: all seed-level metric JSON/CSV files from tasks 1-3.
- Produces: report-ready tables/figures, technical-report text and GitHub-safe reproducibility instructions.

- [ ] **Step 1: 汇总四种设置的三随机种子结果，生成模型对比和目标代理业务指标图表**

Run: `python .\scripts\generate_paper_figures.py`

Expected: figures use explicit `n=3` labels and distinguish exploratory/complete experiments.

- [ ] **Step 2: 更新技术报告与质控说明**

Report changes: add binary business metrics, three-seed Focal conclusion, role-aware method/results, and a factual reviewer/date completion checklist.

- [ ] **Step 3: 运行完整测试与数据一致性检查**

Run: `python -m unittest discover -s tests -v`

Run: `python .\scripts\validate_dataset_split.py --manifest .\数据集\dataset_final_v1\dataset_split_manifest_v1_0.csv`

Expected: all tests pass and fixed split validation reports no cross-split group leakage.

- [ ] **Step 4: 精确暂存并推送本轮成果**

Stage only: new scripts/tests, `outputs/business_metrics`, new formal summary/figures, updated report builder/README, and design/plan documentation. Do not stage raw images, model weights, local papers, finance files, temporary render folders or unrelated user files.
