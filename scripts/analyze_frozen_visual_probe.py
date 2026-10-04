"""Post-hoc descriptive capacity audit; no feature extraction or model fitting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from frozen_visual_probe import transition_counts
from run_frozen_visual_probe import DEFAULT_OUTPUT, ROOT, snapshot
from run_tc_oos_rsg_experiments import _file_sha256, _write_json


def capacity_counts(q: torch.Tensor, p: torch.Tensor, labels: torch.Tensor) -> dict:
    promoted = (q.argmax(1) == 0) & (p.argmax(1) != 0)
    below = p[:, 0] < .25
    true = promoted & (labels == 0)
    false = promoted & (labels != 0)
    increased = q[:, 0] > p[:, 0]+1e-12
    if bool((promoted & below & ~increased).any()):
        raise ValueError("Target victory below 1/4 without mass increase contradicts simplex capacity.")
    return {"new_correct_targets": int(true.sum()), "new_correct_targets_base_below_quarter": int((true & below).sum()),
            "new_false_targets_base_below_quarter": int((false & below).sum()),
            "promotions_without_target_mass_increase": int((promoted & ~increased).sum()),
            "target_mass_increased_rows": int(increased.sum())}


def analyze(root: Path, output: Path) -> dict:
    summary = json.loads((output / "development_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((output / "registered_protocol.json").read_text(encoding="utf-8"))
    data, source = snapshot(root, protocol)
    if source != summary["source_snapshot"]:
        raise ValueError("Registered sources changed.")
    runs = []
    for r in summary["runs"]:
        if r["arm"] != "EH":
            continue
        name = f"{r['architecture']}_EH_seed{r['seed']}/stop_raw_predictions.csv"
        path = output / name
        if _file_sha256(path) != summary["artifact_sha256"][name]:
            raise ValueError("Prediction hash changed.")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        d = data[1]
        if len(rows) != 340 or [row["image_id"] for row in rows] != [row["image_id"] for row in d["records"]]:
            raise ValueError("Prediction identities changed.")
        q = torch.tensor([[float(row[f"prob_{k}"]) for k in range(4)] for row in rows], dtype=torch.float64)
        e_name = f"{r['architecture']}_E_seed{r['seed']}/stop_raw_predictions.csv"
        e_path = output / e_name
        if _file_sha256(e_path) != summary["artifact_sha256"][e_name]:
            raise ValueError("Paired probability-only prediction hash changed.")
        with e_path.open(encoding="utf-8", newline="") as handle:
            e_rows = list(csv.DictReader(handle))
        if [row["image_id"] for row in e_rows] != [row["image_id"] for row in rows]:
            raise ValueError("Paired probability-only identities changed.")
        e_q = torch.tensor([[float(row[f"prob_{k}"]) for k in range(4)] for row in e_rows], dtype=torch.float64)
        gained = (q.argmax(1) == 0) & (d["p"].argmax(1) != 0) & (d["labels"] == 0)
        details = [{"image_id": d["records"][i]["image_id"], "base_target_probability": float(d["p"][i, 0]),
                    "probe_target_probability": float(q[i, 0]), "base_below_quarter": bool(d["p"][i, 0] < .25)}
                   for i in gained.nonzero().flatten().tolist()]
        runs.append({"architecture": r["architecture"], "seed": r["seed"], **capacity_counts(q, d["p"], d["labels"]),
                     "relative_to_same_seed_E": transition_counts(q, e_q, d["labels"]), "new_correct_target_details": details})
    result = {"status": "POST_HOC_DESCRIPTIVE_CAPACITY_AUDIT", "summary_sha256": _file_sha256(output / "development_summary.json"),
              "runs": runs, "new_training": False, "new_images_loaded": False,
              "claim_boundary": "Observed unconstrained-head capacity changes do not establish a safe residual architecture, calibrated posterior, independent efficacy or a new general theorem."}
    _write_json(output / "capacity_analysis.json", result)
    fields = ["nll", "accuracy", "macro_f1", "target_recall", "ti_intrusion_to_target", "metallic_intrusion_to_target"]
    with (output / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["architecture", "arm", "version", "seed_count", *[f"{field}_{stat}" for field in fields for stat in ("mean", "sample_sd")]])
        for architecture, family in summary["aggregate"].items():
            for arm, versions in family.items():
                for version, metrics in versions.items():
                    writer.writerow([architecture, arm, version, 3, *[metrics[field][stat] for field in fields for stat in ("mean", "sample_sd")]])
        for arm, control in summary["retained_global_controls"].items():
            writer.writerow(["retained_control", arm, "frozen", 1, *[value for field in fields for value in (control["metrics"]["stop"][field], "")]])
        for arm, control in summary["retained_learned_controls"].items():
            writer.writerow(["retained_control", arm, "frozen", 3, *[control[field][stat] for field in fields for stat in ("mean", "sample_sd")]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
