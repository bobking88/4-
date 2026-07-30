# Hierarchical Mineral Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible EfficientNet-B0 mineral-species and beneficiation-role consistency experiment, prepare its result pipeline, and integrate verified findings into the technical report.

**Architecture:** A fixed 17-species-to-four-role mapping is validated from the frozen manifest. A shared EfficientNet-B0 backbone emits species, role, binary, and projection outputs. The trainer combines weighted role/species losses, a species-to-role KL consistency loss, binary auxiliary loss, and the existing hard-negative contrastive loss.

**Tech Stack:** Python 3.11, PyTorch/Torchvision, scikit-learn, CSV/JSON, python-docx, unittest.

## Global Constraints

- Use only the fixed `dataset_final_v1/dataset_split_manifest_v1_0.csv` for closed-set training.
- Preserve the existing four-class mapping and split membership.
- Do not claim grade prediction, recovery prediction, industrial deployment, or stage-conditioned decision validation.
- Run the formal experiment with seeds `20260727`, `20260728`, and `20260729`.
- Add a regression test before each new production behavior.
- Update the report only with results present in checked output files.

---

### Task 1: Hierarchy Mapping Utilities

**Files:**
- Create: `scripts/mineral_hierarchy.py`
- Create: `tests/test_mineral_hierarchy.py`

**Interfaces:**
- Produces `build_species_mapping(records)`, `aggregate_role_probabilities(species_probabilities, mapping, torch)`, and `validate_species_role_mapping(records)`.

- [ ] Write tests for stable sorted species labels, one role per species, and role-probability aggregation.
- [ ] Run the tests and confirm failure because the module does not exist.
- [ ] Implement only the mapping and aggregation functions.
- [ ] Run the tests and confirm pass.

### Task 2: Hierarchical Trainer

**Files:**
- Create: `scripts/train_hierarchical_mineral_classifier.py`
- Create: `tests/test_hierarchical_mineral_classifier.py`

**Interfaces:**
- Consumes the hierarchy utilities and the existing fixed manifest.
- Produces a CLI training command that writes `environment.json`, `metrics_history.csv`, `test_metrics.json`, `confusion_matrix.csv`, and `test_predictions.csv`.

- [ ] Write tests for the consistency loss and hierarchical labels before implementing the trainer.
- [ ] Run the tests and confirm failure because the trainer helpers do not exist.
- [ ] Implement a shared EfficientNet-B0 model with species, role, binary, and projection heads; reuse the existing hard-negative loss.
- [ ] Run the tests and a CPU smoke run; confirm fixed split sizes and output file schemas.

### Task 3: Formal Experiment and Analysis

**Files:**
- Create: `scripts/analyze_hierarchical_experiment.py`
- Create: `tests/test_analyze_hierarchical_experiment.py`
- Create: `outputs/training/formal_hierarchical_efficientnet_b0_seed20260727/`
- Create: `outputs/training/formal_hierarchical_efficientnet_b0_seed20260728/`
- Create: `outputs/training/formal_hierarchical_efficientnet_b0_seed20260729/`
- Create: `outputs/business_metrics/hierarchical_efficientnet_b0/`

**Interfaces:**
- Consumes three trainer output folders and emits a three-seed summary table and JSON with overall and target-proxy risk metrics.

- [ ] Write tests for missing-run validation and mean/standard-deviation aggregation.
- [ ] Run the tests and confirm failure because the analyzer does not exist.
- [ ] Implement the analyzer and run it against synthetic fixture outputs.
- [ ] Run the three formal commands and analyze the resulting runs.
- [ ] Confirm all three manifests and class labels match the frozen experiment configuration.

### Task 4: Figures and Technical Report Update

**Files:**
- Modify: `scripts/generate_paper_figures.py`
- Modify: `scripts/build_technical_report.py`
- Modify: `结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（初稿）.docx`

**Interfaces:**
- Consumes the verified hierarchical experiment summary and writes updated figure source data plus a rendered technical report.

- [ ] Add a test that refuses to create a hierarchical result figure when the required three-seed summary is missing.
- [ ] Run the test and confirm failure before implementation.
- [ ] Add a model architecture diagram, method equation, component-comparison figure, and bounded innovation/limitation text.
- [ ] Rebuild the report and render every page to PNG for visual review.

### Task 5: Open-Set Evaluation Interface

**Files:**
- Create: `scripts/evaluate_open_set_minerals.py`
- Create: `tests/test_evaluate_open_set_minerals.py`
- Create: `docs/open_set_unknown_dataset_protocol.md`

**Interfaces:**
- Consumes a known prediction table and a future unknown-image prediction table.
- Produces thresholded coverage, known accuracy, unknown rejection rate, and risk-coverage data for MSP and energy scores.

- [ ] Write tests for threshold selection and metric calculation.
- [ ] Run the tests and confirm failure because the evaluator does not exist.
- [ ] Implement the evaluator without downloading or fabricating unknown samples.
- [ ] Document the independent unknown-mineral data requirements and run the tests.
