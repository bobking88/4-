# Theory-Aware Hierarchical Mineral Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Produce reproducible evidence for the theory-aware hierarchical mineral-recognition line, add a Chinese architecture figure, and incorporate verified results into the technical report.

**Architecture:** The existing EfficientNet-B0 model retains one shared backbone and four sibling outputs: four-class role, 17-species, target-proxy binary, and hard-negative projection. Deterministic scripts will validate role-level identifiability under controlled candidate-label ambiguity and calculate confidence-threshold selective-recognition curves from saved predictions. Stage-conditioned process decisions remain future work.

**Tech Stack:** Python 3.11, PyTorch, scikit-learn, matplotlib, Pillow, python-docx, unittest.

## Global Constraints

- Preserve the fixed data manifest and splits; do not redistribute raw source images or weights.
- Treat Mindat-title ambiguity as a controlled candidate-set experiment, not fully annotated real multi-label ground truth.
- Interpret \`defer\` only as recommendation for subsequent inspection, never as verified industrial XRF-cost optimization.
- Do not claim industrial sorting, concentrate grade prediction, recovery-rate validation, statistical significance, or a first-of-its-kind method.
- Keep the architecture figure faithful: all four heads branch from shared features; no head feeds another head.
- Use fixed seeds and write every reported value to a tracked output file.

---

### Task 1: Add controlled candidate-set identifiability analysis

**Files:**
- Create: \`scripts/analyze_role_identifiability.py\`
- Create: \`tests/test_analyze_role_identifiability.py\`
- Create: \`outputs/theory_validation/role_identifiability/role_identifiability_summary.json\`
- Create: \`outputs/theory_validation/role_identifiability/role_identifiability_summary.md\`

**Interfaces:**
- Consumes: \`SpeciesRoleMapping\` from \`scripts/mineral_hierarchy.py\`.
- Produces: \`build_candidate_set_rows(mapping, candidate_sizes, seed) -> list[dict[str, object]]\`.
- Produces: \`summarize_candidate_sets(rows) -> dict[str, object]\`.

- [ ] **Step 1: Write the failing test**

\`\`\`python
def test_role_consistent_candidates_preserve_role_identifiability(self):
    mapping = make_mapping()
    rows = build_candidate_set_rows(mapping, candidate_sizes=(2, 3), seed=7)
    summary = summarize_candidate_sets(rows)
    self.assertEqual(summary["role_consistent"]["role_unique_rate"], 1.0)
    self.assertLess(summary["role_consistent"]["species_unique_rate"], 1.0)
\`\`\`

- [ ] **Step 2: Run the focused test and verify it fails**

Run: \`.venv-training\\Scripts\\python.exe -m unittest tests.test_analyze_role_identifiability -v\`

Expected: FAIL because \`analyze_role_identifiability\` does not exist.

- [ ] **Step 3: Implement deterministic candidate-set construction**

\`\`\`python
def build_candidate_set_rows(mapping, candidate_sizes=(2, 3, 4), seed=20260801):
    rng = random.Random(seed)
    rows = []
    for size in candidate_sizes:
        for species_index, role_id in enumerate(mapping.species_role_ids):
            same_role = [i for i, r in enumerate(mapping.species_role_ids) if r == role_id and i != species_index]
            if len(same_role) >= size - 1:
                rows.append(make_row("role_consistent", [species_index, *rng.sample(same_role, size - 1)], mapping))
            other_role = [i for i, r in enumerate(mapping.species_role_ids) if r != role_id]
            rows.append(make_row("role_conflict", [species_index, *rng.sample(other_role, size - 1)], mapping))
    return rows
\`\`\`

\`make_row\` reports \`candidate_size\`, \`scenario\`, candidate labels, \`species_unique\`, and \`role_unique\`. \`summarize_candidate_sets\` calculates rates, counts, and size breakdowns without inferring visual labels.

- [ ] **Step 4: Add command-line output**

Implement arguments \`--manifest\`, \`--dataset-root\`, \`--output-dir\`, \`--candidate-sizes\`, and \`--seed\`. Load records with existing helpers, validate the mapping, write UTF-8 JSON and Markdown, and state that this is a controlled logical validation.

- [ ] **Step 5: Run focused tests and analysis**

\`\`\`powershell
.venv-training\Scripts\python.exe -m unittest tests.test_analyze_role_identifiability -v
.venv-training\Scripts\python.exe scripts\analyze_role_identifiability.py --manifest 数据集\dataset_split_manifest_v1_0.csv --dataset-root 数据集 --output-dir outputs\theory_validation\role_identifiability --candidate-sizes 2 3 4 --seed 20260801
\`\`\`

Expected: PASS; every role-consistent row has \`role_unique=True\`, while role-conflict rows have \`role_unique=False\`.

- [ ] **Step 6: Commit**

\`\`\`bash
git add scripts/analyze_role_identifiability.py tests/test_analyze_role_identifiability.py outputs/theory_validation/role_identifiability
git commit -m "feat: validate role identifiability under candidate ambiguity"
\`\`\`

### Task 2: Add selective-recognition risk analysis

**Files:**
- Create: \`scripts/analyze_selective_recognition.py\`
- Create: \`tests/test_analyze_selective_recognition.py\`
- Create: \`outputs/theory_validation/selective_recognition/selective_recognition_summary.json\`
- Create: \`outputs/theory_validation/selective_recognition/selective_recognition_summary.md\`
- Create: \`outputs/paper_figures_v1/fig9_selective_recognition.png\`

**Interfaces:**
- Consumes: CSV prediction records with \`true_label\`, \`predicted_label\`, and \`confidence\`.
- Produces: \`calculate_selective_metrics(rows, thresholds) -> list[dict[str, float]]\`.

- [ ] **Step 1: Write the failing test**

\`\`\`python
def test_threshold_defers_low_confidence_errors(self):
    rows = [
        {"true_label": "target_mineral", "predicted_label": "target_mineral", "confidence": "0.95"},
        {"true_label": "ti_bearing_negative", "predicted_label": "target_mineral", "confidence": "0.55"},
    ]
    values = calculate_selective_metrics(rows, thresholds=(0.0, 0.8))
    self.assertEqual(values[0]["coverage"], 1.0)
    self.assertEqual(values[0]["risk"], 0.5)
    self.assertEqual(values[1]["coverage"], 0.5)
    self.assertEqual(values[1]["risk"], 0.0)
\`\`\`

- [ ] **Step 2: Run the focused test and verify it fails**

Run: \`.venv-training\\Scripts\\python.exe -m unittest tests.test_analyze_selective_recognition -v\`

Expected: FAIL because \`analyze_selective_recognition\` does not exist.

- [ ] **Step 3: Implement threshold metrics**

\`\`\`python
def calculate_selective_metrics(rows, thresholds):
    values = []
    for threshold in thresholds:
        retained = [row for row in rows if float(row["confidence"]) >= threshold]
        coverage = len(retained) / len(rows)
        errors = sum(row["true_label"] != row["predicted_label"] for row in retained)
        values.append({"threshold": threshold, "coverage": coverage, "risk": errors / len(retained) if retained else 0.0})
    return values
\`\`\`

Return target-proxy miss rate, titanium-interference intrusion rate, and metallic-hard-negative intrusion rate for retained records. Return \`None\` for an undefined denominator so zero is never fabricated.

- [ ] **Step 4: Implement aggregation and figure**

For the three full-hierarchical seeds, calculate thresholds \`0.00\` to \`0.95\` by \`0.05\`, retain seed-level values, and write mean and standard deviation. Plot a two-panel Chinese figure: coverage versus threshold and retained risk versus coverage. State below the figure that lower coverage means more samples are deferred for later inspection.

- [ ] **Step 5: Run focused tests and real analysis**

\`\`\`powershell
.venv-training\Scripts\python.exe -m unittest tests.test_analyze_selective_recognition -v
.venv-training\Scripts\python.exe scripts\analyze_selective_recognition.py --input-glob outputs\hierarchical_role_aware_efficientnet_b0_seed*\test_predictions.csv --output-dir outputs\theory_validation\selective_recognition --figure outputs\paper_figures_v1\fig9_selective_recognition.png
\`\`\`

Expected: PASS; every threshold is recorded and the Markdown does not call the method industrial cost-optimal.

- [ ] **Step 6: Commit**

\`\`\`bash
git add scripts/analyze_selective_recognition.py tests/test_analyze_selective_recognition.py outputs/theory_validation/selective_recognition outputs/paper_figures_v1/fig9_selective_recognition.png
git commit -m "feat: add selective mineral recognition analysis"
\`\`\`

### Task 3: Generate a faithful Chinese theory and network figure

**Files:**
- Modify: \`scripts/generate_paper_figures.py\`
- Modify: \`tests/test_generate_paper_figures.py\`
- Create: \`outputs/paper_figures_v1/fig10_theory_aware_hierarchical_architecture_cn.png\`

**Interfaces:**
- Produces: \`plot_theory_aware_hierarchical_architecture(output_path: Path) -> None\`.

- [ ] **Step 1: Write the failing test**

\`\`\`python
def test_theory_aware_architecture_figure_is_created(self):
    output = self.temp_dir / "architecture_cn.png"
    plot_theory_aware_hierarchical_architecture(output)
    self.assertTrue(output.exists())
    self.assertGreater(output.stat().st_size, 10_000)
\`\`\`

- [ ] **Step 2: Run the focused test and verify it fails**

Run: \`.venv-training\\Scripts\\python.exe -m unittest tests.test_generate_paper_figures.PaperFigureTests.test_theory_aware_architecture_figure_is_created -v\`

Expected: FAIL because the new figure function is absent.

- [ ] **Step 3: Implement the architecture diagram**

Draw \`矿物图像 -> EfficientNet-B0共享主干 -> 共享特征 h\`, then four independent sibling branches: \`角色头 p_r\`, \`种类头 p_s\`, \`目标代理二分类头 p_b\`, and \`投影头 e\`. Draw \`A p_s = p_tilde_r\` from the species head to the consistency-loss box and role output into the same box. Draw the footer \`L = L_role + alpha L_species + beta L_cons + gamma L_binary + eta L_hard\`. Annotate that hard-negative constraint only targets the two predefined high-risk pair families. No sibling head arrow is permitted.

- [ ] **Step 4: Run focused test and regenerate figures**

\`\`\`powershell
.venv-training\Scripts\python.exe -m unittest tests.test_generate_paper_figures -v
.venv-training\Scripts\python.exe scripts\generate_paper_figures.py
\`\`\`

Expected: PASS; the PNG is nonempty and existing figures remain generated.

- [ ] **Step 5: Visually inspect the PNG**

Open \`outputs/paper_figures_v1/fig10_theory_aware_hierarchical_architecture_cn.png\`; verify readable Chinese labels, correct four sibling branches, and only the species-to-role aggregation arrow.

- [ ] **Step 6: Commit**

\`\`\`bash
git add scripts/generate_paper_figures.py tests/test_generate_paper_figures.py outputs/paper_figures_v1/fig10_theory_aware_hierarchical_architecture_cn.png
git commit -m "docs: add Chinese hierarchical network architecture figure"
\`\`\`

### Task 4: Incorporate theory and verified outputs into the technical report

**Files:**
- Modify: \`scripts/build_technical_report.py\`
- Modify: \`结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（初稿）.docx\`
- Modify: \`docs/experiment_records/2026-07-30_role_aware_hard_negative_learning.md\`

**Interfaces:**
- Consumes: component ablations, both analysis JSON files, \`fig9_selective_recognition.png\`, and \`fig10_theory_aware_hierarchical_architecture_cn.png\`.
- Produces: an updated DOCX whose added conclusions each point to a reproducible output.

- [ ] **Step 1: Add theoretical-model content**

Insert a Chapter 4 subsection with a notation table for \`x\`, \`S\`, \`R\`, \`A\`, \`p_s\`, \`p_r\`, \`p_tilde_r\`, and \`q(x)\`; aggregation \`p_tilde_r=A p_s\`; joint loss; KL loss; and role-identifiability proposition with proof outline. Render equations as matplotlib MathText images where Word has no robust equation API.

- [ ] **Step 2: Add two verification subsections**

Report candidate-set counts and rates from JSON, label it controlled logical-condition verification, add the threshold table and selective-recognition figure, and explain coverage-risk tradeoff without a real XRF-cost claim.

- [ ] **Step 3: Insert structure figure and revise wording**

Embed \`fig10_theory_aware_hierarchical_architecture_cn.png\`, explain all heads share visual features, and say species-to-role mapping is probability aggregation. Define current contributions as role-level formalization, species-role consistency plus hard-negative learning, and selective-recognition evaluation. Put stage-conditioned decisions, true cost matrices, XRF, source-held-out testing, and real ore imaging only in future work.

- [ ] **Step 4: Rebuild and structurally verify DOCX**

\`\`\`powershell
$docPython='C:\Users\bob\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $docPython scripts\build_technical_report.py
@'
import os
from docx import Document
doc = Document(os.environ["DOCX_PATH"])
assert len(doc.paragraphs) > 130
assert len(doc.inline_shapes) >= 10
assert any("理论模型" in p.text for p in doc.paragraphs)
assert any("选择性识别" in p.text for p in doc.paragraphs)
print("docx structural verification passed")
'@ | & $docPython -
\`\`\`

Set \`DOCX_PATH\` to the generated report before the inline verification. Attempt the local document renderer; if LibreOffice is unavailable, record that structural verification passed and visual rendering could not be automated.

- [ ] **Step 5: Commit**

\`\`\`bash
git add scripts/build_technical_report.py docs/experiment_records/2026-07-30_role_aware_hard_negative_learning.md 结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（初稿）.docx
git commit -m "docs: add theory validation to technical report"
\`\`\`

### Task 5: Run full verification and publish reproducibility artifacts

**Files:**
- Modify: \`README.md\`
- Modify: \`docs/superpowers/plans/2026-08-01-theory-aware-hierarchical-mineral-recognition.md\`

**Interfaces:**
- Consumes: all outputs from Tasks 1 through 4.
- Produces: exact reproduction commands and clear evidentiary boundaries.

- [ ] **Step 1: Document reproduction commands**

Add candidate-set analysis, selective-recognition analysis, and report-generation commands. Link JSON summaries, figures, and report; state images and weights are not redistributed.

- [ ] **Step 2: Run all tests**

Run: \`.venv-training\\Scripts\\python.exe -m unittest discover -s tests -v\`

Expected: every existing test and both new analysis modules pass.

- [ ] **Step 3: Verify tracked scope**

\`\`\`bash
git status --short
git diff --check
git ls-files outputs/theory_validation outputs/paper_figures_v1
\`\`\`

Expected: no raw images, weights, caches, papers, or unrelated user files are staged.

- [ ] **Step 4: Update checklist, commit, and push**

Mark checkboxes complete only after commands pass.

\`\`\`bash
git add README.md docs/superpowers/plans/2026-08-01-theory-aware-hierarchical-mineral-recognition.md
git commit -m "docs: document theory-aware experiment reproduction"
git push origin main
\`\`\`

## Self-Review

**Spec coverage:** Task 1 validates the role-identifiability condition. Task 2 implements the supportable confidence-based deferral study. Task 3 creates the faithful Chinese neural-network diagram. Task 4 places formulas, proposition, figure, and outputs in the report while retaining industrial stage claims as future work. Task 5 verifies tests and reproducibility scope.

**Placeholder scan:** No \`TODO\`, \`TBD\`, \`implement later\`, or unspecified test instruction is used. Each implementation task specifies interface, command, expected behavior, and output location.

**Type consistency:** \`SpeciesRoleMapping\` is the common input to Task 1. Task 2 uses existing prediction fields. Task 3 reflects the existing four-head forward tuple. Task 4 consumes the JSON and figure paths from Tasks 1 to 3. Task 5 references those same paths.
