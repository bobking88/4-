from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "fixed": "#237A8B",
    "photographer_holdout": "#D46A4C",
    "resnet50_portability": "#6857A5",
    "ink": "#20303C",
    "muted": "#647681",
    "grid": "#D8E0E4",
    "note": "#F6E7B8",
}
PROTOCOL_LABELS = {
    "fixed": "Fixed test",
    "photographer_holdout": "Photographer holdout",
    "resnet50_portability": "ResNet50 portability",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Diagnostic CSV is empty: {path}")
    return rows


def _rows_by_protocol(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        output.setdefault(row["protocol"], []).append(row)
    for protocol in output:
        output[protocol].sort(key=lambda row: int(str(row["stratum"]).replace("T", "")))
    return output


def _protocol_color(protocol: str) -> str:
    return COLORS.get(protocol, "#4C78A8")


def _protocol_label(protocol: str) -> str:
    return PROTOCOL_LABELS.get(protocol, protocol.replace("_", " "))


def generate_rsg_gate_reliability_figure(
    b1_strata_csv: Path,
    b2_strata_csv: Path,
    summary_json: Path,
    prefix: Path,
) -> dict[str, Path]:
    """Render descriptive theorem diagnostics from high-precision RSG replays."""
    b1_by_protocol = _rows_by_protocol(_read_csv(b1_strata_csv))
    b2_by_protocol = _rows_by_protocol(_read_csv(b2_strata_csv))
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    if set(b1_by_protocol) != set(b2_by_protocol):
        raise ValueError("B.1 and B.2 strata must cover the same protocols.")
    overall = summary["overall"]
    if int(overall["b1_local_violation_count"]) != 0 or int(overall["b2_violation_count"]) != 0:
        raise ValueError("Figure is reserved for zero-violation high-precision diagnostics.")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), constrained_layout=False)
    figure.patch.set_facecolor("white")
    b1_axis, b2_axis = axes

    for protocol, rows in b1_by_protocol.items():
        x_values = list(range(1, len(rows) + 1))
        color = _protocol_color(protocol)
        label = _protocol_label(protocol)
        b1_axis.plot(
            x_values,
            [float(row["mean_routing_regret_nll"]) for row in rows],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{label}: regret",
        )
        b1_axis.plot(
            x_values,
            [float(row["mean_b1_local_bound"]) for row in rows],
            marker="s",
            linewidth=1.5,
            linestyle="--",
            color=color,
            alpha=0.86,
            label=f"{label}: local bound",
        )
    b1_axis.set_yscale("log")
    b1_axis.set_xticks((1, 2, 3), ("T1", "T2", "T3"))
    b1_axis.set_title("Theorem B.1: local B.1 bound strata", fontsize=11, fontweight="bold", color=COLORS["ink"])
    b1_axis.set_xlabel("Tertiles ranked by local B.1 bound", fontsize=8)
    b1_axis.set_ylabel("Mean NLL regret or upper bound (log scale)", fontsize=8)
    b1_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    b1_axis.spines[["top", "right"]].set_visible(False)
    b1_axis.tick_params(labelsize=7)
    b1_axis.legend(fontsize=6.2, frameon=False, ncol=2, loc="upper left")

    for protocol, rows in b2_by_protocol.items():
        x_values = list(range(1, len(rows) + 1))
        color = _protocol_color(protocol)
        label = _protocol_label(protocol)
        b2_axis.plot(
            x_values,
            [float(row["mean_soft_hard_deviation"]) for row in rows],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{label}: deviation",
        )
        b2_axis.plot(
            x_values,
            [float(row["mean_b2_bound"]) for row in rows],
            marker="s",
            linewidth=1.5,
            linestyle="--",
            color=color,
            alpha=0.86,
            label=f"{label}: B.2 bound",
        )
    b2_axis.set_xticks((1, 2, 3), ("T1", "T2", "T3"))
    b2_axis.set_ylim(-0.02, 1.07)
    b2_axis.set_title("Theorem B.2: margin-stratified soft-target deviation", fontsize=11, fontweight="bold", color=COLORS["ink"])
    b2_axis.set_xlabel("Tertiles ranked by oracle margin |log a - log b|", fontsize=8)
    b2_axis.set_ylabel("Mean deviation or exponential bound", fontsize=8)
    b2_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    b2_axis.spines[["top", "right"]].set_visible(False)
    b2_axis.tick_params(labelsize=7)
    b2_axis.legend(fontsize=6.2, frameon=False, ncol=2, loc="upper right")

    figure.text(
        0.5,
        0.045,
        "High-precision checkpoint replay: "
        f"{int(overall['run_count'])} runs / {int(overall['sample_count']):,} image evaluations; "
        "B.1 local and B.2 violation counts = 0.  "
        "mechanism diagnosis only, not a classification comparison.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["note"], "edgecolor": "#C7A965", "boxstyle": "round,pad=0.45"},
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.18, wspace=0.25)

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": prefix.with_suffix(".png"),
        "svg": prefix.with_suffix(".svg"),
        "pdf": prefix.with_suffix(".pdf"),
        "tiff": prefix.with_suffix(".tiff"),
        "source_description": prefix.with_name(prefix.name + "_source.json"),
    }
    figure.savefig(outputs["png"], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    figure.savefig(
        outputs["tiff"],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    outputs["source_description"].write_text(
        json.dumps(
            {
                "figure_type": "rsg-gate-reliability-mechanism-diagnosis",
                "theorems": ["Theorem B.1", "Theorem B.2"],
                "b1_metric": "pointwise local B.1 bound with epsilon_i=min(a_i,b_i)",
                "b2_metric": "soft-target deviation and exp(-|log a-log b|/T_r)",
                "overall": overall,
                "claim_boundary": summary["claim_boundary"],
                "data_sources": [str(b1_strata_csv), str(b2_strata_csv), str(summary_json)],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RSG gate-reliability theorem figure.")
    parser.add_argument("--b1-strata-csv", type=Path, required=True)
    parser.add_argument("--b2-strata-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate_rsg_gate_reliability_figure(
        args.b1_strata_csv, args.b2_strata_csv, args.summary_json, args.output_prefix
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
