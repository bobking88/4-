"""Apply bounded report errata from archived manuscript evidence, without model imports."""

import hashlib
import json
from pathlib import Path
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Mm
from lxml import etree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/report_errata_20261111_v2"
REL = Path("结题/基于深度学习的钒钛矿相关矿物图像识别方法研究_技术报告（正式版）.docx")
SOURCE_HASH = "4faaa82752f72a37d5784f5118f47f3a7be49bb24ef5214704f59681be1f3893"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            assert fig.bbox.padded(1).contains(box.x0, box.y0)
            assert fig.bbox.padded(1).contains(box.x1, box.y1)
    fig.savefig(HERE / f"{name}.png", dpi=300)
    fig.savefig(HERE / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def figures(tables):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    values = []
    for row in tables[29].rows[1:]:
        values.append([tuple(float(x.strip()) for x in c.text.replace("%", "").split("±"))
                       for c in row.cells[1:]])
    assert values == [
        [(71.20, .69), (68.99, 1.17), (71.25, 6.31), (12.69, 1.83),
         (11.88, 3.96), (45.19, 8.34), (.1177, .0063)],
        [(71.47, 1.14), (69.25, .99), (72.04, 2.64), (11.40, 2.10),
         (14.19, 3.02), (53.69, 4.02), (.0824, .0080)],
    ]
    fig = plt.figure(figsize=(183 / 25.4, 89 / 25.4))
    ax1 = fig.add_axes([.08, .31, .46, .46])
    ax2 = fig.add_axes([.64, .31, .12, .46])
    ax3 = fig.add_axes([.86, .31, .12, .46])
    for idx, (name, color) in enumerate([("HRGV", "#829393"), ("RSG-HRGV", "#238679")]):
        offset = (idx - .5) * .34
        ax1.bar([i + offset for i in range(4)], [values[idx][j][0] for j in range(1, 5)],
                yerr=[values[idx][j][1] for j in range(1, 5)], width=.32,
                color=color, label=name, capsize=2, error_kw={"lw": .7})
        for ax, col in [(ax2, 5), (ax3, 6)]:
            ax.bar(idx, values[idx][col][0], yerr=values[idx][col][1], color=color,
                   width=.65, capsize=2, error_kw={"lw": .7})
            ax.set_xticks([0, 1], ["H", "R"])
    ax1.set_xticks(range(4), ["Macro F1", "Target\nrecall", "Ti FPR", "Metallic\nFPR"])
    ax1.set_ylabel("Final-prediction metric (%)")
    ax1.set_ylim(0, 90)
    ax2.set_ylabel("Conditional agreement (%)")
    ax2.set_ylim(0, 90)
    ax3.set_ylabel("Mean routing regret (nat)")
    ax3.set_ylim(0, .15)
    for ax, title in [(ax1, "a  Role classification"), (ax2, "b  Gate"), (ax3, "c  Regret")]:
        ax.set_title(title, loc="left", size=8, weight="bold", pad=8)
        ax.grid(axis="y", color="#DDDDDD", linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.51, .99), frameon=False, fontsize=8)
    fig.text(.04, .16, "Archived photographer-holdout means ± SD across 3 seeds; H = HRGV, R = RSG-HRGV.", size=7)
    fig.text(.04, .095, "Agreement is with the true-class-probability oracle, conditional on exactly one expert being correct.", size=7)
    fig.text(.04, .035, "Regret uses pre-verifier fusion. Classification superiority is not established.", size=7)
    save_figure(fig, "figure18_photographer_metrics")

    fig = plt.figure(figsize=(183 / 25.4, 65 / 25.4))
    fig.text(.04, .91, "Conditional properties of the existing RSG routing graph", weight="bold", size=10)
    for x, title, formula, note in [
        (.04, "Exact decomposition", r"$r=-\log(1-\delta d/M)$", "Positive expert probabilities;\nconvex fusion; sensitivity at fixed M."),
        (.37, "Soft oracle target", r"$g^*=\sigma(\Delta/T_r)$", "Entropy-regularized target;\nBCE is not the fused NLL."),
        (.70, "Local gradient isolation", r"$\nabla_{\theta_e}L_{reg}=0$", "Input, target and weight detached;\nno nonzero-step guarantee."),
    ]:
        fig.text(x, .72, title, weight="bold", size=8)
        fig.text(x, .52, formula, size=12)
        fig.text(x, .28, note, size=7, linespacing=1.5)
    fig.text(.04, .08, "These properties do not establish architectural novelty or classification superiority.", size=8)
    save_figure(fig, "figure22_conditions")

    source = ROOT / "outputs/paper_figures_v3/fig_rsg_theory_evidence_portability_source.json"
    assert sha(source) == "4f77a886a50a808125d4ca054005b30f71a6a7e4a3846c9ae20868ef22405b34"
    data = json.loads(source.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(183 / 25.4, 100 / 25.4))
    fig.text(.04, .94, "Archived paired effects: RSG-HRGV minus HRGV", weight="bold", size=10)
    ax = fig.add_axes([.30, .26, .64, .52])
    for y, key in zip([2, 1, 0], ["fixed_test", "photographer_holdout", "resnet50_portability"]):
        row = data[key]
        v, lo, hi = row["difference"], row["ci_low"], row["ci_high"]
        ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt="o", color="#238679", capsize=3)
        ax.text(v, y+.22, f"{v:.4f}  [{lo:.4f}, {hi:.4f}]", ha="center", size=8)
    ax.set_yticks([2, 1, 0], ["Fixed split", "Photographer\nholdout", "ResNet50"])
    ax.set_ylim(-.45, 2.55)
    ax.set_xlim(-.056, .006)
    ax.set_xlabel("Difference in mean routing regret (nat)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=.7)
    ax.grid(axis="x", color="#DDDDDD", linewidth=.5)
    ax.set_axisbelow(True)
    fig.text(.04, .12, "Points: archived estimates; bars: paired cluster-bootstrap 95% intervals; 3 seeds per protocol.", size=7)
    fig.text(.04, .055, "Routing-risk evidence only; overall classification superiority has not been established.", size=8)
    save_figure(fig, "figure23_routing_nat")


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = ROOT / REL
    assert sha(source) == SOURCE_HASH, "Formal report differs from reviewed input"
    backup = ROOT / "结题/历史版本/技术报告_正式版_20261111_指标勘误前.docx"
    if backup.exists():
        assert sha(backup) == SOURCE_HASH
    else:
        shutil.copy2(source, backup)
    d = Document(source)
    p, tables = list(d.paragraphs), list(d.tables)
    assert len(p) == 896 and len(tables) == 65
    original_math = [etree.tostring(x) for x in d._element.xpath("//m:oMath")]
    original_bodies = [[ [c.text for c in r.cells] for r in t.rows[1:]] for t in tables]
    changes = []

    def change(i, expected, value):
        assert expected in p[i].text, (i, p[i].text)
        assert not p[i]._p.xpath(".//m:oMath|.//w:drawing")
        changes.append({"paragraph": i, "old": p[i].text, "new": value})
        runs = p[i].runs
        if runs:
            runs[0].text = value
            for run in runs[1:]:
                run.text = ""
        else:
            p[i].add_run(value)

    change(18, "该网络创新", p[18].text.replace("该网络创新的实证支撑来自路由机制，而非对总体分类精度的过度宣传。", "现有证据支持完整RSG训练策略的预定义路由风险改善，不证明新的卷积结构、更强表达能力或总体分类优势。"))
    change(430, "门控选择正确专家", "RSG在恰有一位专家argmax分类正确的子集上，门控与真值类概率oracle的一致率从45.99% ± 4.82%升至58.01% ± 1.76%，均值相差12.02个百分点；它不必等于选中分类正确专家的比例。平均路由后悔从0.0986 ± 0.0096 nat降至0.0809 ± 0.0017 nat，成对差异（RSG−HRGV）为−0.0177 nat，95%簇Bootstrap区间[−0.0286, −0.0069] nat，方向性重采样比例为1.0000。该结果支持本协议下完整训练策略的路由风险差异，不是单一损失项的因果归因。路由后悔基于验证前融合；总体Accuracy与Macro F1使用验证后最终输出。表中“条件一致”均采用上述子集及概率oracle定义。")
    change(431, "所设计的局部监督稳定改善了该路由目标", p[431].text.replace("所设计的局部监督稳定改善了该路由目标", "完整RSG策略在既定协议中降低了该路由后悔；该比较同时涉及监督目标与门控输入梯度隔离，不作单一因素的因果归因"))
    change(440, "门控选择正确专家", p[440].text.replace("门控选择正确专家的比例", "门控与真值类概率oracle的一致率").replace("这为“后悔监督改善跨粒度证据路由”", "该一致率不等同于选择分类正确专家的比例。上述路由风险差异为完整RSG训练策略"))
    change(477, "可证明、可测量", p[477].text.replace("可证明、可测量的证据路由创新", "具有条件分析与机制证据的跨粒度路由方法"))
    change(561, "门控选择正确率", "主要机制指标为平均路由后悔（nat，越低越好），以及在恰有一位专家分类正确的子集上，门控硬选择与真值类概率oracle的一致率。后者在表中简称“条件一致”，不等于选择分类正确专家的比例。另报告验证后最终输出的Accuracy、Macro F1、目标召回及两类误入率，分别检查路由风险与主任务风险。")
    change(565, "增加1.94%", p[565].text.replace("增加1.94%", "增加0.0194 nat").replace("[1.30%, 2.60%]", "[0.0130, 0.0260] nat").replace("连续软目标优于将近似等价专家强制二值化", "软目标配置相对硬目标配置的预定义路由后悔较低"))
    change(566, "差异为-0.20%", p[566].text.replace("差异为-0.20%", "差异为−0.0020 nat").replace("[-0.62%, 0.12%]", "[−0.0062, 0.0012] nat").replace("由式(B-1)风险上界导出的", "受式(B-1)风险上界启发的"))
    change(567, "变化为-0.51%", p[567].text.replace("变化为-0.51%", "变化为−0.0051 nat").replace("[-0.85%, -0.16%]", "[−0.0085, −0.0016] nat").replace("变化为-1.58%", "变化为−1.58个百分点").replace("[-4.10%, 0.36%]", "[−4.10, 0.36]个百分点"))
    change(644, "三条可证明性质", "图22概括现有RSG计算图的精确分解、软目标与局部隔离条件；完整证明见附录B。精确分解要求两专家正概率和凸融合，敏感性讨论固定M；熵正则化oracle目标不等价于最终融合NLL；局部零梯度不保证门控每步参数梯度非零。这些是计算图的条件性质，不是总体分类优势或网络结构首创的证明。")
    change(646, "图 22", "图 22 RSG路由计算图的条件性质与适用边界")
    change(647, "图22及图23保留历史图面", "固定测试中RSG−HRGV的路由后悔差为−0.0177 nat，95%区间[−0.0286, −0.0069] nat；摄影者留出对应差为−0.0353 nat，区间[−0.0475, −0.0217] nat。两个预定义区间均低于零，支持本协议下的路由后悔下降，不代表Accuracy、Macro F1或目标类风险均改善。图23统一以nat展示三协议的既有成对差异；本次仅更正显示单位，没有重算区间或模型预测。旧图面保存在勘误前备份及历史归档中。")
    change(657, "图 23", "图23将固定EfficientNet-B0、摄影者留出与ResNet50三协议的平均路由后悔差及既有95%成对簇Bootstrap区间并列。每协议均为三个种子，重采样次数依次为2000、5000、2000。固定划分与主干替换使用同一测试图像队列，不是两套独立图像数据。总体Accuracy和Macro F1的区间未显示稳定差异，不据此主张分类优势。")
    change(659, "图 23", "图 23 三协议下RSG−HRGV的平均路由后悔差及既有区间（nat）")
    change(685, "理论上限", "本节沿用完整RSG-HRGV的三随机种子汇总，分别报告双专家Accuracy、验证后最终预测Accuracy、预测分歧、一对一错子集比例和条件门控一致率。表中“最终预测”对应历史字段fused_accuracy，其源码实际读取验证后最终预测标签，不是验证前凸融合Accuracy。“硬选择oracle”按两专家分类正确事件的并集计算，是不可部署的诊断参照，不是连续融合准确率的通用上界。“条件一致”则是在恰有一位专家argmax正确的子集内，比较门控硬选择（g≥0.5）与真值类概率oracle（pd(y)≥pm(y)）是否一致；这两个oracle不可混同。")
    change(688, "图 26", "图 26 跨粒度专家、验证后最终预测及门控条件一致性")
    change(689, "选对率", "三个协议中，两位专家存在约4%–6%的预测分歧，约4%–5%的样本恰有一位专家分类正确，说明两路证据并非完全冗余。门控在后一子集上与概率oracle的一致率为51.98%–58.01%，它不必意味着选中分类正确的专家，不能直接解释为分类纠错能力。图26的a面板为专家与最终预测的三种子均值，b面板分母为全部测试图像，c面板分母为一对一错子集；这些摘要没有提供对应不确定性区间，不补造误差条，也不将oracle差距视为可部署的增益。")

    for i in [28, 29]:
        tables[i].rows[0].cells[6].text = "条件一致\n（%）"
    tables[38].rows[0].cells[2].text = "条件一致率"
    for i, value in [(3, "最终预测"), (4, "硬选择\noracle"), (7, "条件一致")]:
        tables[53].rows[0].cells[i].text = value

    figures(tables)
    expert = ROOT / "docs/paper_v1/figures_2026-11-08/fig_rsg_expert_diagnostics.png"
    assert sha(expert) == "38ead3c7e5bdacf78f542a5dcb2c18211f572a2902f3aa4b9c400d5279131b8b"
    for i, image, alt in [
        (437, HERE / "figure18_photographer_metrics.png", "摄影者留出的既有均值和标准差；分类指标与条件一致率用百分比，路由后悔用nat"),
        (645, HERE / "figure22_conditions.png", "RSG条件性质及限制，不含新的实验结果"),
        (658, HERE / "figure23_routing_nat.png", "三个协议的既有路由后悔差和区间，单位nat"),
        (687, expert, "专家准确率、最终预测、子集比例及门控与概率oracle条件一致率"),
    ]:
        assert p[i]._p.xpath(".//w:drawing")
        p[i].clear()
        p[i].alignment = WD_ALIGN_PARAGRAPH.CENTER
        p[i].paragraph_format.keep_with_next = True
        shape = p[i].add_run().add_picture(str(image), width=Mm(160))
        shape._inline.docPr.set("descr", alt)

    assert original_math == [etree.tostring(x) for x in d._element.xpath("//m:oMath")]
    assert original_bodies == [[[c.text for c in r.cells] for r in t.rows[1:]] for t in tables]
    candidate = OUT / "report_candidate.docx"
    d.save(candidate)
    (HERE / "change_manifest.json").write_text(json.dumps({
        "date": "2026-11-11", "source_commit": "2f3920d5e9fb869b01e18d887ffa18747d3187f6",
        "source_sha256": SOURCE_HASH, "candidate_before_word_sha256": sha(candidate),
        "paragraph_changes": changes, "table_headers_changed": [28, 29, 38, 53],
        "figures_replaced": [18, 22, 23, 26], "all_65_table_bodies_unchanged": True,
        "all_29_omml_objects_unchanged_before_word": True,
        "scope": "Document errata only. No model/data execution or new scientific claims.",
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(candidate)


if __name__ == "__main__":
    main()
