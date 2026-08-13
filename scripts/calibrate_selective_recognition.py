from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import beta

from train_mineral_classifier import CLASS_LABELS


THRESHOLDS = tuple(round(index * 0.05, 2) for index in range(20))
DELTAS = (0.10, 0.15, 0.20)
ALPHA = 0.05


def split_validation_groups(rows, seed: int = 20260813):
    if not rows:
        raise ValueError("Validation rows must not be empty.")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row.get("split_group_id", "").strip()
        if not group:
            raise ValueError("Every validation row must have split_group_id.")
        groups[group].append(row)
    by_role: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group, group_rows in groups.items():
        roles = {row["four_class_label"] for row in group_rows}
        if len(roles) != 1:
            raise ValueError(f"Validation group {group} spans multiple roles.")
        by_role[roles.pop()].append((group, group_rows))
    fit, certify = [], []
    rng = random.Random(seed)
    for role in sorted(by_role):
        role_groups = list(by_role[role])
        rng.shuffle(role_groups)
        fit_count = 0
        certify_count = 0
        for _, group_rows in role_groups:
            if fit_count <= certify_count:
                fit.extend(group_rows)
                fit_count += len(group_rows)
            else:
                certify.extend(group_rows)
                certify_count += len(group_rows)
    if not fit or not certify:
        raise ValueError("Grouped validation split produced an empty subset.")
    return fit, certify


def multiclass_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("Logits and labels have incompatible shapes.")
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    baseline = multiclass_nll(logits, labels)
    result = minimize_scalar(
        lambda log_temperature: multiclass_nll(logits / math.exp(log_temperature), labels),
        bounds=(-5.0, 5.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    temperature = math.exp(float(result.x))
    if not result.success or multiclass_nll(logits / temperature, labels) > baseline + 1e-10:
        return 1.0
    return temperature


def clopper_pearson_upper(errors: int, total: int, confidence: float) -> float:
    if total < 0 or errors < 0 or errors > total:
        raise ValueError("Error counts must satisfy 0 <= errors <= total.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between zero and one.")
    if total == 0 or errors == total:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, total - errors))


def select_certified_threshold(rows, thresholds=THRESHOLDS, delta=0.15, alpha=ALPHA):
    if not rows or not thresholds:
        raise ValueError("Certification rows and thresholds must not be empty.")
    simultaneous_confidence = 1.0 - alpha / len(thresholds)
    evaluated = []
    for threshold in thresholds:
        retained = [row for row in rows if float(row["calibrated_confidence"]) >= threshold]
        errors = sum(not bool(row["correct"]) for row in retained)
        upper = clopper_pearson_upper(errors, len(retained), simultaneous_confidence)
        evaluated.append(
            {
                "threshold": float(threshold),
                "retained_count": len(retained),
                "coverage": len(retained) / len(rows),
                "errors": errors,
                "empirical_risk": errors / len(retained) if retained else None,
                "simultaneous_upper_bound": upper,
                "certified": bool(retained) and upper <= delta,
            }
        )
    candidates = [item for item in evaluated if item["certified"]]
    if not candidates:
        return {"status": "no_certified_threshold", "delta": delta, "evaluated": evaluated}
    selected = max(candidates, key=lambda item: (item["coverage"], -item["threshold"]))
    return {"status": "certified", "delta": delta, "selected": selected, "evaluated": evaluated}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _logits_labels(rows):
    logits = np.asarray(
        [[float(row[f"role_logit_{label}"]) for label in CLASS_LABELS] for row in rows]
    )
    labels = np.asarray([CLASS_LABELS.index(row["four_class_label"]) for row in rows])
    return logits, labels


def prepare_rows(rows, temperature: float):
    logits, labels = _logits_labels(rows)
    probabilities = softmax(logits / temperature)
    predictions = probabilities.argmax(axis=1)
    prepared = []
    for row, confidence, predicted, true_label in zip(
        rows, probabilities.max(axis=1), predictions, labels
    ):
        prepared.append(
            {
                **row,
                "calibrated_confidence": float(confidence),
                "predicted_class_id": int(predicted),
                "correct": bool(predicted == true_label),
            }
        )
    return prepared, probabilities, labels


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    one_hot = np.eye(probabilities.shape[1])[labels]
    ece = 0.0
    reliability = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (
            confidences <= upper if index == bins - 1 else confidences < upper
        )
        count = int(mask.sum())
        if count:
            accuracy = float(correct[mask].mean())
            confidence = float(confidences[mask].mean())
            ece += count / len(labels) * abs(accuracy - confidence)
        else:
            accuracy = None
            confidence = None
        reliability.append(
            {"bin": index + 1, "lower": lower, "upper": upper, "count": count,
             "accuracy": accuracy, "mean_confidence": confidence}
        )
    return {
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_15_bin": ece,
        "accuracy": float(correct.mean()),
        "reliability": reliability,
    }


def selective_metrics(rows, threshold: float):
    retained = [row for row in rows if float(row["calibrated_confidence"]) >= threshold]
    target = [row for row in retained if row["four_class_label"] == "target_mineral"]
    titanium = [row for row in retained if row["four_class_label"] == "ti_bearing_negative"]
    metallic = [row for row in retained if row["four_class_label"] == "metallic_hard_negative"]
    predicted_target = lambda row: int(row["predicted_class_id"]) == 0
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    errors = sum(not bool(row["correct"]) for row in retained)
    return {
        "threshold": threshold,
        "retained_count": len(retained),
        "coverage": len(retained) / len(rows),
        "selective_risk": ratio(errors, len(retained)),
        "target_miss_rate": ratio(sum(not predicted_target(row) for row in target), len(target)),
        "titanium_intrusion_rate": ratio(sum(predicted_target(row) for row in titanium), len(titanium)),
        "metallic_intrusion_rate": ratio(sum(predicted_target(row) for row in metallic), len(metallic)),
    }


def risk_coverage_curve(rows):
    ordered = sorted(rows, key=lambda row: float(row["calibrated_confidence"]), reverse=True)
    errors = 0
    curve = []
    for index, row in enumerate(ordered, start=1):
        errors += not bool(row["correct"])
        curve.append({"retained_count": index, "coverage": index / len(rows), "risk": errors / index})
    aurc = float(np.mean([row["risk"] for row in curve]))
    return curve, aurc


def _mean_std(values):
    values = [float(value) for value in values]
    return {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else 0.0}


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _plot_reliability(rows, path: Path):
    figure, axis = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    for state, color, marker in (("before", "#D1495B", "s"), ("after", "#1768AC", "o")):
        subset = [row for row in rows if row["state"] == state and int(row["count"]) > 0]
        axis.plot([row["mean_confidence"] for row in subset], [row["accuracy"] for row in subset],
                  marker=marker, color=color, label=state)
    axis.set_xlabel("Mean confidence"); axis.set_ylabel("Observed accuracy")
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.grid(alpha=0.25); axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=240); plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate and certify selective recognition.")
    parser.add_argument("--probability-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260813)
    return parser.parse_args()


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = (20260727, 20260728, 20260729)
    results = {}; reliability_rows = []; curve_rows = []
    for seed in seeds:
        run = f"formal_hierarchical_efficientnet_b0_seed{seed}"
        run_dir = args.probability_root / run
        validation = _read_rows(run_dir / "val_hierarchical_probabilities.csv")
        test = _read_rows(run_dir / "test_hierarchical_probabilities.csv")
        fit_rows, certification_rows = split_validation_groups(validation, args.split_seed)
        fit_logits, fit_labels = _logits_labels(fit_rows)
        temperature = fit_temperature(fit_logits, fit_labels)
        _, before_prob, test_labels = prepare_rows(test, 1.0)
        test_prepared, after_prob, _ = prepare_rows(test, temperature)
        cert_prepared, _, _ = prepare_rows(certification_rows, temperature)
        before = calibration_metrics(before_prob, test_labels)
        after = calibration_metrics(after_prob, test_labels)
        curve, aurc = risk_coverage_curve(test_prepared)
        for row in curve: curve_rows.append({"seed": seed, **row})
        certificates = {}
        for delta in DELTAS:
            certificate = select_certified_threshold(cert_prepared, THRESHOLDS, delta, ALPHA)
            if certificate["status"] == "certified":
                threshold = float(certificate["selected"]["threshold"])
                certificate["test_evaluation"] = selective_metrics(test_prepared, threshold)
            certificates[str(delta)] = certificate
        results[str(seed)] = {
            "temperature": temperature,
            "validation_fit_count": len(fit_rows),
            "risk_certification_count": len(certification_rows),
            "validation_group_overlap": len(
                {row['split_group_id'] for row in fit_rows} &
                {row['split_group_id'] for row in certification_rows}
            ),
            "test_before_calibration": before,
            "test_after_calibration": after,
            "test_aurc": aurc,
            "certificates": certificates,
        }
        for state, metrics in (("before", before), ("after", after)):
            for row in metrics["reliability"]: reliability_rows.append({"seed": seed, "state": state, **row})
    aggregate = {
        "temperature": _mean_std([results[str(seed)]["temperature"] for seed in seeds]),
        "test_ece_before": _mean_std([results[str(seed)]["test_before_calibration"]["ece_15_bin"] for seed in seeds]),
        "test_ece_after": _mean_std([results[str(seed)]["test_after_calibration"]["ece_15_bin"] for seed in seeds]),
        "test_nll_before": _mean_std([results[str(seed)]["test_before_calibration"]["nll"] for seed in seeds]),
        "test_nll_after": _mean_std([results[str(seed)]["test_after_calibration"]["nll"] for seed in seeds]),
        "test_aurc": _mean_std([results[str(seed)]["test_aurc"] for seed in seeds]),
    }
    averaged_reliability = []
    for state in ("before", "after"):
        for bin_id in range(1, 16):
            subset = [row for row in reliability_rows if row["state"] == state and row["bin"] == bin_id]
            defined = [row for row in subset if row["count"]]
            averaged_reliability.append({
                "state": state, "bin": bin_id, "count": sum(int(row["count"]) for row in subset),
                "accuracy": mean(float(row["accuracy"]) for row in defined) if defined else None,
                "mean_confidence": mean(float(row["mean_confidence"]) for row in defined) if defined else None,
            })
    _write_csv(args.output_dir / "calibration_reliability_source.csv", reliability_rows)
    _write_csv(args.output_dir / "risk_coverage_source.csv", curve_rows)
    _plot_reliability(averaged_reliability, args.figure)
    payload = {
        "protocol": "validation group split: temperature fit / disjoint risk certification / one-shot test",
        "thresholds": THRESHOLDS, "deltas": DELTAS, "alpha": ALPHA,
        "bonferroni_simultaneous": True, "results": results, "aggregate": aggregate,
    }
    with (args.output_dir / "calibrated_selective_recognition_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
