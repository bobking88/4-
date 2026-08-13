from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train_mineral_classifier import CLASS_LABELS


def calculate_prediction_metrics(rows):
    if not rows:
        raise ValueError("Prediction rows must not be empty.")
    true = [row["true_label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    species_rows = defaultdict(list)
    for row in rows:
        species_rows[row["mineral_label"]].append(row)
    species_recall = {
        species: sum(row["predicted_label"] == row["true_label"] for row in values) / len(values)
        for species, values in sorted(species_rows.items())
    }
    target = [row for row in rows if row["true_label"] == "target_mineral"]
    return {
        "row_count": len(rows),
        "accuracy": accuracy_score(true, predicted),
        "macro_precision": precision_score(true, predicted, labels=CLASS_LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(true, predicted, labels=CLASS_LABELS, average="macro", zero_division=0),
        "macro_f1": f1_score(true, predicted, labels=CLASS_LABELS, average="macro", zero_division=0),
        "target_recall": sum(row["predicted_label"] == "target_mineral" for row in target) / len(target) if target else None,
        "role_recall": {
            role: sum(row["predicted_label"] == role for row in rows if row["true_label"] == role)
            / sum(row["true_label"] == role for row in rows)
            for role in CLASS_LABELS
            if any(row["true_label"] == role for row in rows)
        },
        "species_counts": dict(sorted(Counter(row["mineral_label"] for row in rows).items())),
        "species_role_recall": species_recall,
    }


def common_subset_rows(full_rows, ablation_rows):
    common = {row["image_id"] for row in full_rows} & {row["image_id"] for row in ablation_rows}
    full_by_id = {row["image_id"]: row for row in full_rows}
    ablation_by_id = {row["image_id"]: row for row in ablation_rows}
    ids = sorted(common)
    return [full_by_id[image_id] for image_id in ids], [ablation_by_id[image_id] for image_id in ids]


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mean_std(values):
    values = [float(value) for value in values]
    return {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else 0.0}


def _aggregate(metrics_by_seed):
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "target_recall")
    result = {name: _mean_std([metrics[name] for metrics in metrics_by_seed.values()]) for name in names}
    result["role_recall"] = {
        role: _mean_std([metrics["role_recall"][role] for metrics in metrics_by_seed.values()])
        for role in CLASS_LABELS
    }
    return result


def _source_metrics(training_root):
    result = {"baseline": {}, "hierarchical": {}}
    for seed in (20260727, 20260728, 20260729):
        for model in result:
            run = training_root / f"source_holdout_{model}_seed{seed}" / "test_metrics.json"
            metrics = _read_json(run)
            result[model][str(seed)] = {
                "accuracy": metrics["accuracy"], "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"], "macro_f1": metrics["macro_f1"],
                "target_recall": metrics["class_recall"]["target_mineral"],
                "role_recall": metrics["class_recall"],
            }
    aggregate = {model: _aggregate(values) for model, values in result.items()}
    paired = {}
    for metric in ("accuracy", "macro_f1", "target_recall"):
        values = [
            result["hierarchical"][str(seed)][metric] - result["baseline"][str(seed)][metric]
            for seed in (20260727, 20260728, 20260729)
        ]
        paired[metric] = {"values": values, **_mean_std(values)}
    paired["role_recall"] = {}
    for role in CLASS_LABELS:
        values = [
            result["hierarchical"][str(seed)]["role_recall"][role]
            - result["baseline"][str(seed)]["role_recall"][role]
            for seed in (20260727, 20260728, 20260729)
        ]
        paired["role_recall"][role] = {"values": values, **_mean_std(values)}
    return {"per_seed": result, "aggregate": aggregate, "paired_hierarchical_minus_baseline": paired}


def _proxy_metrics(training_root, fixed_training_root):
    result = {"full_common_subset": {}, "no_magnetite_proxy": {}}
    for seed in (20260727, 20260728, 20260729):
        full = _read_csv(fixed_training_root / f"formal_hierarchical_efficientnet_b0_seed{seed}" / "test_predictions.csv")
        ablation = _read_csv(training_root / f"proxy_ablation_hierarchical_seed{seed}" / "test_predictions.csv")
        full_common, ablation_common = common_subset_rows(full, ablation)
        if [row["image_id"] for row in full_common] != [row["image_id"] for row in ablation_common]:
            raise ValueError("Proxy comparison subsets do not align.")
        result["full_common_subset"][str(seed)] = calculate_prediction_metrics(full_common)
        result["no_magnetite_proxy"][str(seed)] = calculate_prediction_metrics(ablation_common)
    aggregate = {model: _aggregate(values) for model, values in result.items()}
    species = {}
    for mineral in ("ilmenite", "titanomagnetite"):
        species[mineral] = {
            model: {
                "count": next(iter(result[model].values()))["species_counts"].get(mineral, 0),
                **_mean_std([metrics["species_role_recall"].get(mineral, 0) for metrics in result[model].values()]),
            }
            for model in result
        }
    return {"per_seed": result, "aggregate": aggregate, "target_species": species}


def _plot_source(source, path):
    metrics = ("macro_f1", "target_recall")
    labels = ("Macro F1", "Target recall")
    x = range(len(metrics)); width = 0.34
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for index, (model, color) in enumerate((("baseline", "#D1495B"), ("hierarchical", "#1768AC"))):
        means = [source["aggregate"][model][metric]["mean"] for metric in metrics]
        stds = [source["aggregate"][model][metric]["sample_std"] for metric in metrics]
        positions = [value + (index - 0.5) * width for value in x]
        axis.bar(positions, means, width=width, yerr=stds, capsize=4, label=model, color=color)
    axis.set_xticks(list(x), labels); axis.set_ylabel("Score"); axis.set_ylim(0, 0.85)
    axis.grid(axis="y", alpha=0.25); axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=240); plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize paper supplementary experiments.")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--fixed-training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_metrics(args.training_root)
    proxy = _proxy_metrics(args.training_root, args.fixed_training_root)
    payload = {"source_holdout": source, "proxy_ablation_common_subset": proxy,
               "limitations": ["Three seeds do not establish statistical significance.",
                               "Titanomagnetite has four common test images; its recall is descriptive only."]}
    with (args.output_dir / "paper_experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
    rows = []
    for experiment, aggregate in (("source_baseline", source["aggregate"]["baseline"]),
                                  ("source_hierarchical", source["aggregate"]["hierarchical"]),
                                  ("proxy_full_common", proxy["aggregate"]["full_common_subset"]),
                                  ("proxy_no_magnetite", proxy["aggregate"]["no_magnetite_proxy"])):
        rows.append({"experiment": experiment, **{f"{m}_{s}": aggregate[m][s]
                    for m in ("accuracy", "macro_f1", "target_recall") for s in ("mean", "sample_std")}})
    with (args.output_dir / "paper_experiment_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _plot_source(source, args.figure)
    print(json.dumps({"source": source["aggregate"], "proxy_target_species": proxy["target_species"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
