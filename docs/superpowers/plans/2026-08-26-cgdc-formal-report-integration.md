# CGDC Formal Report Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Append a reproducible CGDC-RSG-HRGV appendix to the sole formal technical report after the registered five-configuration, three-seed experiment matrix has completed.

**Architecture:** The report updater reads only formal analysis artifacts, validates all fifteen registered runs, appends a single idempotent appendix, and embeds the committed CGDC architecture figure. A structural DOCX test verifies headings, embedded figure relationships, table captions, and claim-boundary language without reinstalling a Word renderer.

**Tech Stack:** Python 3.11, python-docx, JSON/CSV artifacts, unittest, PNG figure bundle.

**Spec:** docs/superpowers/specs/2026-08-25-cgdc-rsg-hrgv-design.md

## Global Constraints

- Use only results produced by the formal CGDC runner for report tables.
- Require registered seeds 20260727, 20260728, and 20260729 for every CGDC configuration.
- Do not claim grade prediction, industrial sorting, recovery, deployment, or OOD performance.
- State P1/P2/P3 as formula-level properties; treat Accuracy, Macro F1, target recall, intrusion, Brier, and ECE as empirical evidence with paired Bootstrap intervals.
- Preserve the sole official report at 结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx.
- Do not install LibreOffice or another renderer; run structural DOCX checks only.

---

### Task 1: Validate the formal CGDC evidence contract

**Files:**
- Create: tests/test_update_formal_report_v9.py
- Create: scripts/update_formal_report_v9.py

**Interfaces:**
- Consumes: outputs/business_metrics/cgdc_rsg_hrgv/formal/cgdc_three_seed_summary.json and paired comparison JSON artifacts.
- Produces: load_formal_cgdc_evidence(analysis_dir: Path) -> dict[str, object].

- [ ] **Step 1: Write the failing test**

    def test_requires_all_formal_configurations_and_three_seeds(self):
        from update_formal_report_v9 import load_formal_cgdc_evidence
        with self.assertRaisesRegex(ValueError, "three registered seeds"):
            load_formal_cgdc_evidence(Path(temp_dir))

- [ ] **Step 2: Run test to verify it fails**

Run: D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_update_formal_report_v9 -v

Expected: FAIL because update_formal_report_v9 does not exist.

- [ ] **Step 3: Write minimal implementation**

    FORMAL_CONFIGURATIONS = (
        "rsg_complete", "cgdc_complete", "cgdc_shared_features",
        "cgdc_unconditional", "cgdc_no_decomposition_loss",
    )
    FORMAL_SEEDS = ("20260727", "20260728", "20260729")

    def load_formal_cgdc_evidence(analysis_dir: Path) -> dict[str, object]:
        summary = json.loads((analysis_dir / "cgdc_three_seed_summary.json").read_text(encoding="utf-8"))
        if set(summary) != set(FORMAL_CONFIGURATIONS):
            raise ValueError("Formal CGDC evidence must contain all five configurations.")
        for configuration in FORMAL_CONFIGURATIONS:
            if len(summary[configuration]["macro_f1"]["values"]) != len(FORMAL_SEEDS):
                raise ValueError("Formal CGDC evidence must contain three registered seeds.")
        return summary

- [ ] **Step 4: Run test to verify it passes**

Run: D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_update_formal_report_v9 -v

Expected: PASS.

- [ ] **Step 5: Commit**

Run: git add tests/test_update_formal_report_v9.py scripts/update_formal_report_v9.py
Run: git commit -m "report: validate CGDC formal evidence"

### Task 2: Append the idempotent CGDC appendix

**Files:**
- Modify: scripts/update_formal_report_v9.py
- Modify: tests/test_update_formal_report_v9.py

**Interfaces:**
- Consumes: load_formal_cgdc_evidence, the report DOCX, and outputs/paper_figures/cgdc_rsg_hrgv_architecture.png.
- Produces: update_report(input_path: Path, output_path: Path, analysis_dir: Path) -> Path.

- [ ] **Step 1: Write the failing test**

    def test_appends_cgdc_appendix_once_with_architecture_and_boundaries(self):
        update_report(source, output, analysis_dir)
        update_report(output, output, analysis_dir)
        document = Document(output)
        text = "\n".join(item.text for item in document.paragraphs)
        self.assertEqual(text.count("附录 D CGDC-RSG-HRGV 网络理论与实验"), 1)
        self.assertIn("命题 P1", text)
        self.assertIn("Brier", text)
        self.assertIn("不等同于工业分选", text)

- [ ] **Step 2: Run test to verify it fails**

Run: D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_update_formal_report_v9 -v

Expected: FAIL because update_report is absent.

- [ ] **Step 3: Write minimal implementation**

    def update_report(input_path: Path, output_path: Path, analysis_dir: Path) -> Path:
        document = Document(input_path)
        if any(p.text.strip() == APPENDIX_HEADING for p in document.paragraphs):
            document.save(output_path)
            return output_path
        evidence = load_formal_cgdc_evidence(analysis_dir)
        document.add_heading(APPENDIX_HEADING, level=1)
        add_theory_statement(document)
        add_architecture_figure(document, FIGURE_PATH)
        add_three_seed_table(document, evidence)
        add_paired_bootstrap_table(document, analysis_dir)
        add_claim_boundary(document)
        document.save(output_path)
        return output_path

- [ ] **Step 4: Run test to verify it passes**

Run: D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_update_formal_report_v9 -v

Expected: PASS.

- [ ] **Step 5: Commit**

Run: git add tests/test_update_formal_report_v9.py scripts/update_formal_report_v9.py
Run: git commit -m "report: append CGDC theory appendix"

### Task 3: Generate and structurally verify the official report

**Files:**
- Modify: 结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx
- Modify: tests/test_update_formal_report_v9.py

**Interfaces:**
- Consumes: completed formal evidence, official report, and updater.
- Produces: an idempotently updated official report and passing structural checks.

- [ ] **Step 1: Write the failing test**

    def test_official_report_has_valid_figure_relationships_and_unique_captions(self):
        self.assertTrue(report_has_valid_media_relationships(FORMAL_REPORT))
        self.assertEqual(len(caption_numbers(FORMAL_REPORT)), len(set(caption_numbers(FORMAL_REPORT))))

- [ ] **Step 2: Run test to verify it fails**

Run: D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest tests.test_update_formal_report_v9 -v

Expected: FAIL before regeneration because the appendix is absent.

- [ ] **Step 3: Generate from finished formal artifacts**

Run: D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe scripts\update_formal_report_v9.py --input "结题\基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx" --output "结题\基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx" --analysis-dir outputs\business_metrics\cgdc_rsg_hrgv\formal

- [ ] **Step 4: Run structural checks**

Run: D:\成信工科研\人工智能选矿\.venv-training\Scripts\python.exe -m unittest discover -s tests -v

Expected: PASS. Document rendering is intentionally skipped because no renderer is installed.

- [ ] **Step 5: Commit and push**

Run: git add scripts/update_formal_report_v9.py tests/test_update_formal_report_v9.py 结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx
Run: git commit -m "report: integrate CGDC formal evidence"
Run: git push origin codex/theory-aware-report

## Self-Review

- Spec coverage: tasks validate the formal experiment contract, add the CGDC architecture/propositions/empirical evidence, preserve claim boundaries, and structurally verify the report.
- Placeholder scan: each task includes test, expected failure, implementation shape, verification, and commit action.
- Type consistency: the evidence loader returns the dictionary used by update_report; the generated output path remains the formal report path used by the validation task.

## Execution Handoff

This plan is blocked until the formal matrix has produced all fifteen registered output directories and the analysis script has generated formal JSON evidence. The user selected inline continuation; execute Tasks 1-3 in this session once that evidence exists.
