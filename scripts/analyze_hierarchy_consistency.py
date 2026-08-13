from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from train_mineral_classifier import CLASS_LABELS


EPSILON = 1e-12


def _validated_distribution(values) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    if not probabilities or any(value < 0.0 for value in probabilities):
        raise ValueError("Probabilities must be non-negative and non-empty.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Probability values must sum to one.")
    return probabilities


def total_variation(first, second) -> float:
    first_values = _validated_distribution(first)
    second_values = _validated_distribution(second)
    if len(first_values) != len(second_values):
        raise ValueError("Probability distributions must have equal length.")
    return 0.5 * sum(abs(left - right) for left, right in zip(first_values, second_values))


def kl_divergence(mapped, direct) -> float:
    mapped_values = _validated_distribution(mapped)
    direct_values = _validated_distribution(direct)
    if len(mapped_values) != len(direct_values):
        raise ValueError("Probability distributions must have equal length.")
    value = sum(
        probability * math.log(max(probability, EPSILON) / max(reference, EPSILON))
        for probability, reference in zip(mapped_values, direct_values)
        if probability > 0.0
    )
    if value < -1e-6:
        raise ValueError("KL divergence is negative beyond floating-point tolerance.")
    return max(value, 0.0)


def _probabilities(row: dict[str, str], prefix: str) -> tuple[float, ...]:
    try:
        return tuple(float(row[f"{prefix}{label}"]) for label in CLASS_LABELS)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Row is missing valid {prefix} probability columns.") from error


def _argmax_label(probabilities: tuple[float, ...]) -> str:
    return CLASS_LABELS[max(range(len(probabilities)), key=probabilities.__getitem__)]


def calculate_hierarchy_metrics(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        raise ValueError("At least one probability row is required.")
    species_correct = 0
    direct_role_correct = 0
    mapped_role_correct = 0
    species_wrong_role_correct = 0
    head_disagreement = 0
    kl_values = []
    tv_values = []
    pinsker_slacks = []
    confidence_gaps = []
    detail_rows = []
    for row in rows:
        direct = _validated_distribution(_probabilities(row, "role_probability_"))
        mapped = _validated_distribution(_probabilities(row, "mapped_role_probability_"))
        true_role = row["four_class_label"]
        direct_role = _argmax_label(direct)
        mapped_role = _argmax_label(mapped)
        predicted_species_role = row.get("predicted_species_role") or mapped_role
        is_species_correct = row["mineral_label"] == row["predicted_species"]
        is_predicted_species_role_correct = predicted_species_role == true_role
        kl = kl_divergence(mapped, direct)
        tv = total_variation(mapped, direct)
        bound = math.sqrt(0.5 * kl)
        species_correct += is_species_correct
        direct_role_correct += direct_role == true_role
        mapped_role_correct += mapped_role == true_role
        species_wrong_role_correct += (not is_species_correct) and is_predicted_species_role_correct
        head_disagreement += direct_role != mapped_role
        kl_values.append(kl)
        tv_values.append(tv)
        pinsker_slacks.append(bound - tv)
        confidence_gaps.append(max(direct) - max(mapped))
        detail_rows.append(
            {
                "kl": kl,
                "tv": tv,
                "head_disagreement": int(direct_role != mapped_role),
            }
        )
    count = len(rows)
    species_error_count = count - species_correct
    return {
        "row_count": count,
        "species_accuracy": species_correct / count,
        "direct_role_accuracy": direct_role_correct / count,
        "mapped_role_accuracy": mapped_role_correct / count,
        "species_error_count": species_error_count,
        "species_wrong_role_correct_count": species_wrong_role_correct,
        "species_wrong_role_correct_rate_all": species_wrong_role_correct / count,
        "species_wrong_role_correct_rate_among_species_errors": (
            species_wrong_role_correct / species_error_count if species_error_count else 0.0
        ),
        "hard_species_role_accuracy": (species_correct + species_wrong_role_correct) / count,
        "hard_species_role_error_rate": (
            species_error_count - species_wrong_role_correct
        ) / count,
        "empirical_hard_role_risk_contraction": species_wrong_role_correct / count,
        "mean_kl_mapped_to_direct": mean(kl_values),
        "mean_total_variation": mean(tv_values),
        "mean_pinsker_slack": mean(pinsker_slacks),
        "minimum_pinsker_slack": min(pinsker_slacks),
        "pinsker_violation_count": sum(slack < -1e-10 for slack in pinsker_slacks),
        "head_disagreement_rate": head_disagreement / count,
        "mean_direct_minus_mapped_confidence": mean(confidence_gaps),
        "kl_deciles": calculate_kl_deciles(detail_rows),
    }


def calculate_kl_deciles(rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    ordered = sorted(rows, key=lambda row: float(row["kl"]))
    result = []
    for decile in range(10):
        start = len(ordered) * decile // 10
        end = len(ordered) * (decile + 1) // 10
        subset = ordered[start:end]
        if not subset:
            continue
        result.append(
            {
                "decile": decile + 1,
                "count": len(subset),
                "mean_kl": mean(float(row["kl"]) for row in subset),
                "mean_tv": mean(float(row["tv"]) for row in subset),
                "head_disagreement_rate": mean(
                    int(row["head_disagreement"]) for row in subset
                ),
            }
        )
    return result


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else 0.0}


def aggregate_runs(run_metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    scalar_names = (
        "species_accuracy",
        "direct_role_accuracy",
        "mapped_role_accuracy",
        "species_wrong_role_correct_rate_all",
        "species_wrong_role_correct_rate_among_species_errors",
        "hard_species_role_accuracy",
        "hard_species_role_error_rate",
        "empirical_hard_role_risk_contraction",
        "mean_kl_mapped_to_direct",
        "mean_total_variation",
        "head_disagreement_rate",
    )
    return {
        name: _mean_std([float(metrics[name]) for metrics in run_metrics.values()])
        for name in scalar_names
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_deciles(decile_rows: list[dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for variant, color, marker in (
        ("full", "#1768AC", "o"),
        ("no_consistency", "#D1495B", "s"),
    ):
        subset = []
        for decile in range(1, 11):
            matches = [
                row
                for row in decile_rows
                if row["variant"] == variant and int(row["decile"]) == decile
            ]
            subset.append(
                {
                    "mean_kl": mean(float(row["mean_kl"]) for row in matches),
                    "head_disagreement_rate": mean(
                        float(row["head_disagreement_rate"]) for row in matches
                    ),
                }
            )
        axis.plot(
            [row["mean_kl"] for row in subset],
            [row["head_disagreement_rate"] for row in subset],
            marker=marker,
            color=color,
            label=variant,
        )
    axis.set_xlabel("KL (species-mapped role || direct role)")
    axis.set_ylabel("Head disagreement rate")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240)
    plt.close(figure)


def _write_markdown(path: Path, aggregate: dict[str, object], paired: dict[str, object]) -> None:
    full = aggregate["full"]
    no_consistency = aggregate["no_consistency"]
    lines = [
        "# 分层风险收缩与双头一致性分析",
        "",
        "## 主要结果",
        "",
        f"- 完整模型的矿物种类准确率为 {full['species_accuracy']['mean']:.2%}，将预测种类按固定映射转换为选矿角色后的硬角色准确率为 {full['hard_species_role_accuracy']['mean']:.2%}。",
        f"- 在矿物种类预测错误的样本中，仍有 {full['species_wrong_role_correct_rate_among_species_errors']['mean']:.2%} 保留了正确选矿角色；对应整体硬决策风险收缩 {full['empirical_hard_role_risk_contraction']['mean']:.2%}。",
        f"- 加入一致性约束后，平均 KL 为 {full['mean_kl_mapped_to_direct']['mean']:.4f}，无一致性约束时为 {no_consistency['mean_kl_mapped_to_direct']['mean']:.4f}。",
        f"- 加入一致性约束后，双头分歧率为 {full['head_disagreement_rate']['mean']:.2%}，无一致性约束时为 {no_consistency['head_disagreement_rate']['mean']:.2%}。",
        "",
        "## 理论与实验口径",
        "",
        "固定映射下，若细粒度种类预测错误但预测种类与真实种类属于同一选矿角色，则粗粒度角色决策仍正确。因此硬角色风险不高于种类风险，二者差值等于这类被角色映射挽救的样本比例。KL 与总变差用于评价直接角色头和种类聚合角色头的分布一致性；Pinsker 不等式只约束分布差异，不保证分类准确率必然提高。",
        "",
        "本结果来自固定公开标本图像测试集的三个随机种子，不等同于工业分选回收率或精矿品位。三种子结果仅报告均值、样本标准差和配对差值，不宣称统计显著性。",
        "",
        "## 配对差值（完整模型减去无一致性约束）",
        "",
        f"- KL：{paired['mean_kl_mapped_to_direct']['mean']:.4f}",
        f"- 总变差：{paired['mean_total_variation']['mean']:.4f}",
        f"- 双头分歧率：{paired['head_disagreement_rate']['mean']:.2%}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hierarchical risk contraction and consistency.")
    parser.add_argument("--probability-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = {
        "full": "formal_hierarchical_efficientnet_b0_seed",
        "no_consistency": "formal_hierarchical_no_consistency_seed",
    }
    run_metrics: dict[str, dict[str, dict[str, object]]] = {variant: {} for variant in variants}
    flat_rows = []
    decile_rows = []
    for variant, prefix in variants.items():
        for seed in (20260727, 20260728, 20260729):
            run_name = f"{prefix}{seed}"
            input_path = args.probability_root / run_name / "test_hierarchical_probabilities.csv"
            metrics = calculate_hierarchy_metrics(_load_rows(input_path))
            run_metrics[variant][str(seed)] = metrics
            flat_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **{key: value for key, value in metrics.items() if not isinstance(value, list)},
                }
            )
            for row in metrics["kl_deciles"]:
                decile_rows.append({"variant": variant, "seed": seed, **row})
    aggregate = {variant: aggregate_runs(values) for variant, values in run_metrics.items()}
    paired_differences = {}
    for metric in aggregate["full"]:
        differences = [
            float(run_metrics["full"][str(seed)][metric])
            - float(run_metrics["no_consistency"][str(seed)][metric])
            for seed in (20260727, 20260728, 20260729)
        ]
        paired_differences[metric] = {"values": differences, **_mean_std(differences)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "hierarchy_consistency_per_seed.csv", flat_rows)
    _write_rows(args.output_dir / "hierarchy_consistency_kl_deciles.csv", decile_rows)
    _plot_deciles(decile_rows, args.figure)
    payload = {
        "analysis": "empirical hard-label role-risk contraction and dual-head consistency",
        "theoretical_scope": (
            "Hard species predictions induce role predictions through the fixed species-role map; "
            "KL/TV results quantify distribution consistency and do not guarantee accuracy gains."
        ),
        "run_metrics": run_metrics,
        "aggregate": aggregate,
        "paired_full_minus_no_consistency": paired_differences,
        "figure": str(args.figure.resolve()),
    }
    with (args.output_dir / "hierarchy_consistency_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _write_markdown(
        args.output_dir / "hierarchy_consistency_summary.md", aggregate, paired_differences
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
