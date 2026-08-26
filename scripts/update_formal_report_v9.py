from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from update_formal_report_v2 import FORMAL_REPORT, _set_run_font


FORMAL_CONFIGURATIONS = (
    "rsg_complete",
    "cgdc_complete",
    "cgdc_shared_features",
    "cgdc_unconditional",
    "cgdc_no_decomposition_loss",
)
FORMAL_SEEDS = ("20260727", "20260728", "20260729")
APPENDIX_HEADING = "附录 D CGDC-RSG-HRGV 网络理论与实验"
ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "outputs" / "paper_figures" / "cgdc_rsg_hrgv_architecture.png"
DEFAULT_ANALYSIS_DIR = ROOT / "outputs" / "business_metrics" / "cgdc_rsg_hrgv" / "formal"
FORMULA_DIR = ROOT / "outputs" / "report_assets_v9"


def load_formal_cgdc_evidence(analysis_dir: Path) -> dict[str, object]:
    """Load a complete five-configuration, three-seed CGDC analysis summary."""
    summary_path = analysis_dir / "cgdc_three_seed_summary.json"
    if not summary_path.is_file():
        raise ValueError("Formal CGDC evidence summary is missing.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(summary) != set(FORMAL_CONFIGURATIONS):
        raise ValueError("Formal CGDC evidence must contain all five configurations.")
    for configuration in FORMAL_CONFIGURATIONS:
        values = summary[configuration].get("macro_f1", {}).get("values", [])
        if len(values) != len(FORMAL_SEEDS):
            raise ValueError("Formal CGDC evidence must contain three registered seeds.")
    return summary


def _add_body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1.5
    run = paragraph.add_run(text)
    _set_run_font(run)


def _add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for index, value in enumerate(headers):
        table.rows[0].cells[index].text = value
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = value
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    _set_run_font(run)
                    run.font.size = Pt(8.5)
                    run.bold = row_index == 0
    document.add_paragraph()


def _format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _load_paired_comparison(analysis_dir: Path, configuration: str) -> dict[str, object]:
    path = analysis_dir / f"paired_{configuration}_vs_rsg_complete.json"
    if not path.is_file():
        raise ValueError(f"Formal paired comparison is missing: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _add_architecture_figure(document: Document) -> None:
    if not FIGURE_PATH.is_file():
        raise ValueError(f"CGDC architecture figure is missing: {FIGURE_PATH}")
    picture = document.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(FIGURE_PATH), width=Cm(15.4))
    caption = document.add_paragraph(style="Caption")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_run_font(
        caption.add_run("图 19 CGDC-RSG-HRGV-Net 的跨粒度证据分解与分歧校正结构"),
        size=9.5,
    )


def render_formulas() -> dict[str, Path]:
    """Render the report's CGDC equations as stable bitmap formula assets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    formulas = {
        "decomposition": (
            r"$\mathbf{u}_d=\mathbf{h}+A_d(\mathbf{h}),\quad "
            r"\mathbf{u}_s=\mathbf{h}+A_s(\mathbf{h}),\quad "
            r"\mathcal{L}_{dec}=\operatorname{mean}\!\left["
            r"\cos^2(A_d(\mathbf{h}),A_s(\mathbf{h}))\right]$"
        ),
        "calibration": (
            r"$\rho=1-\exp[-D_{JS}(\mathbf{p}_d\Vert\mathbf{p}_m)],\quad "
            r"\mathbf{p}_c=\operatorname{softmax}(\log\mathbf{p}_f+\rho\tanh("
            r"\operatorname{MLP}(\mathbf{z}_c)))$"
        ),
        "bound": (
            r"$\mathbf{p}_d=\mathbf{p}_m\Rightarrow\rho=0\Rightarrow"
            r"\mathbf{p}_c=\mathbf{p}_f;\qquad "
            r"\left|\log\frac{p_{c,j}}{p_{c,k}}-\log\frac{p_{f,j}}{p_{f,k}}\right|"
            r"<2\rho$"
        ),
    }
    paths: dict[str, Path] = {}
    for name, formula in formulas.items():
        figure = plt.figure(figsize=(14, 1.05), dpi=220, facecolor="white")
        figure.text(0.5, 0.5, formula, ha="center", va="center", fontsize=16, color="#172033")
        path = FORMULA_DIR / f"cgdc_{name}.png"
        figure.savefig(path, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        plt.close(figure)
        paths[name] = path
    return paths


def _add_formula(document: Document, path: Path, caption: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Cm(15.3))
    _add_body(document, caption)


def _add_theory_statement(document: Document) -> None:
    formulas = render_formulas()
    document.add_heading("D.1 网络结构与理论定位", level=2)
    _add_body(
        document,
        "CGDC-RSG-HRGV-Net 以 EfficientNet-B0 为共享视觉主干，将共享特征 h 通过直接角色残差适配器和矿物种类残差适配器分解为 u_d=h+A_d(h) 与 u_s=h+A_s(h)。直接角色专家给出 p_d，种类专家经固定种类-角色映射聚合为 p_m；既有 RSG 门控给出融合后验 p_f=g p_d+(1-g)p_m。仅在两路证据不一致时，CGDC 校正器再执行受界后验修正。",
    )
    _add_formula(document, formulas["decomposition"], "式（D-1） 跨粒度残差证据分解与适配器去相关约束")
    _add_body(
        document,
        "令 rho=1-exp[-D_JS(p_d||p_m)]，s=tanh(MLP[z_c])，其中 z_c 包含两路分解特征、逐元素乘积、绝对差、对数后验差、熵与 JS 分歧。最终校正后验为 p_c=softmax(log p_f+rho s)，总损失为 L_CGDC=L_HRGV+lambda_dec L_dec+lambda_cal L_cal，其中 L_dec=mean cos^2(A_d(h),A_s(h))，L_cal=-mean log p_c(y)。",
    )
    _add_formula(document, formulas["calibration"], "式（D-2） 分歧触发的受界后验校正")
    _add_body(
        document,
        "命题 P1（协议一致性）：当 p_d=p_m 时 rho=0，故 p_c=p_f=p_d=p_m。命题 P2（概率有效性）：p_c 为 softmax 输出，非负且四类概率和为 1。命题 P3（有界对数几率修正）：因每个校正残差满足 |s_j|<1，任意类别 j、k 有 |log[p_c(j)/p_c(k)]-log[p_f(j)/p_f(k)]|<2rho。P1-P3 是当前公式与计算图的确定性性质；适配器分解和校正器是否改善分类或校准，必须由以下三随机种子实验检验。",
    )
    _add_formula(document, formulas["bound"], "式（D-3） 协议一致性与分歧相关的有界对数几率修正")
    _add_architecture_figure(document)


def _add_three_seed_summary(document: Document, evidence: dict[str, object]) -> None:
    document.add_heading("D.2 三随机种子受控实验", level=2)
    labels = {
        "rsg_complete": "RSG 完整模型",
        "cgdc_complete": "CGDC 完整模型",
        "cgdc_shared_features": "共享特征消融",
        "cgdc_unconditional": "无条件校正消融",
        "cgdc_no_decomposition_loss": "取消分解损失",
    }
    rows: list[list[str]] = []
    for configuration in FORMAL_CONFIGURATIONS:
        summary = evidence[configuration]
        rows.append(
            [
                labels[configuration],
                _format_percent(summary["accuracy"]["mean"]),
                _format_percent(summary["macro_f1"]["mean"]),
                _format_percent(summary["target_recall"]["mean"]),
                _format_percent(summary["brier_score"]["mean"]),
                _format_percent(summary["expected_calibration_error"]["mean"]),
            ]
        )
    _add_table(
        document,
        ["配置", "Accuracy", "Macro F1", "目标类召回", "Brier", "ECE"],
        rows,
    )
    _add_body(
        document,
        "表中为三随机种子均值；完整配置与四个控制变量消融共享固定数据划分、图像增强、主干、训练预算和测试协议。Brier 与 ECE 均为越低越好的后验质量指标，不能替代真实选矿回收率或品位指标。",
    )


def _add_paired_evidence(document: Document, analysis_dir: Path) -> None:
    document.add_heading("D.3 成对 Bootstrap 证据与结论边界", level=2)
    rows: list[list[str]] = []
    for configuration, label in (
        ("cgdc_complete", "CGDC 完整模型"),
        ("cgdc_shared_features", "共享特征消融"),
        ("cgdc_unconditional", "无条件校正消融"),
        ("cgdc_no_decomposition_loss", "取消分解损失"),
    ):
        paired = _load_paired_comparison(analysis_dir, configuration)
        macro = paired["classification"]["macro_f1"]
        brier = paired["calibration"]["brier_score"]
        ece = paired["calibration"]["expected_calibration_error"]
        rows.append(
            [
                label,
                f"{_format_percent(macro['difference'])} [{_format_percent(macro['ci_low'])}, {_format_percent(macro['ci_high'])}]",
                f"{_format_percent(brier['difference'])} [{_format_percent(brier['ci_low'])}, {_format_percent(brier['ci_high'])}]",
                f"{_format_percent(ece['difference'])} [{_format_percent(ece['ci_low'])}, {_format_percent(ece['ci_high'])}]",
            ]
        )
    _add_table(document, ["相对 RSG", "Macro F1 差值 [95% CI]", "Brier 差值 [95% CI]", "ECE 差值 [95% CI]"], rows)
    _add_body(
        document,
        "每个区间以图片 split_group_id 聚类、随机种子与组两阶段重采样的 2,000 次 Bootstrap 得到。校正相关的经验结论仅以 Brier/ECE 差值及其区间为准；若区间跨零，报告应保留为未观察到稳定增益，而不能根据单个种子或单项指标宣称优势。",
    )
    _add_body(
        document,
        "结论边界：本附录验证的是公开矿物标本图像上的四角色闭集识别和后验校准行为，不等同于工业分选、精矿品位预测、回收率提升、钒钛元素含量检测或未知矿物拒识性能。阶段条件化选矿决策图仍作为下一篇工作与后续研究方向。",
    )


def update_report(input_path: Path, output_path: Path, analysis_dir: Path = DEFAULT_ANALYSIS_DIR) -> Path:
    document = Document(input_path)
    if any(paragraph.text.strip() == APPENDIX_HEADING for paragraph in document.paragraphs):
        document.save(output_path)
        return output_path
    evidence = load_formal_cgdc_evidence(analysis_dir)
    document.add_heading(APPENDIX_HEADING, level=1)
    _add_theory_statement(document)
    _add_three_seed_summary(document, evidence)
    _add_paired_evidence(document, analysis_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append CGDC formal evidence to the technical report.")
    parser.add_argument("--input", type=Path, default=FORMAL_REPORT)
    parser.add_argument("--output", type=Path, default=FORMAL_REPORT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()
    print(update_report(args.input.resolve(), args.output.resolve(), args.analysis_dir.resolve()))


if __name__ == "__main__":
    main()
