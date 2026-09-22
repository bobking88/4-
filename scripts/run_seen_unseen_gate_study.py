"""Compare gate supervision from expert-seen and expert-unseen images.

This is a validation-only mechanism diagnostic.  It freezes one completed
expert checkpoint and never loads the project's original test split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from hrgv_network import (
    HierarchicalRiskGatedVerificationNet,
    apply_residual_target_verifiers,
    regret_gate_targets,
    weighted_soft_gate_loss,
)
from mineral_hierarchy import validate_species_role_mapping
from train_hierarchical_mineral_classifier import HierarchicalMineralImageDataset
from train_hrgv_mineral_classifier import build_role_matrix
from train_mineral_classifier import create_transforms, load_manifest_records, require_training_dependencies


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_supervision_records(expert_records, seen_records, unseen_records):
    """Enforce the paired-count and group-isolation contract before training."""
    if not seen_records or not unseen_records:
        raise ValueError("seen and unseen supervision must both be non-empty")
    seen_strata = Counter((r.mineral_label, r.four_class_label) for r in seen_records)
    unseen_strata = Counter((r.mineral_label, r.four_class_label) for r in unseen_records)
    if seen_strata != unseen_strata:
        raise ValueError("seen and unseen species-role counts do not match exactly")

    expert_ids = {r.image_id for r in expert_records}
    seen_ids = [r.image_id for r in seen_records]
    unseen_ids = [r.image_id for r in unseen_records]
    if len(set(seen_ids)) != len(seen_ids) or len(set(unseen_ids)) != len(unseen_ids):
        raise ValueError("gate supervision contains duplicate image ids")
    if set(seen_ids) & set(unseen_ids):
        raise ValueError("seen and unseen supervision share image ids")
    if not set(seen_ids).issubset(expert_ids):
        raise ValueError("seen supervision must be a subset of expert images")
    if set(unseen_ids) & expert_ids:
        raise ValueError("unseen images are present in the expert manifest")

    expert_groups = {r.split_group_id for r in expert_records}
    seen_groups = {r.split_group_id for r in seen_records}
    unseen_groups = {r.split_group_id for r in unseen_records}
    if unseen_groups & expert_groups:
        raise ValueError("unseen groups are present in the expert manifest")
    if seen_groups & unseen_groups:
        raise ValueError("seen and unseen supervision share groups")

    return {
        "seen_count": len(seen_records),
        "unseen_count": len(unseen_records),
        "exact_species_role_count_match": True,
        "seen_subset_of_expert": True,
        "unseen_absent_from_expert": True,
        "unseen_group_overlap_with_expert": 0,
        "seen_unseen_group_overlap": 0,
        "strata": {
            f"{mineral}|{role}": count
            for (mineral, role), count in sorted(seen_strata.items())
        },
    }


def select_earliest_best(history, key):
    if not history:
        raise ValueError("history must be non-empty")
    return min(history, key=lambda row: (float(row[key]), int(row["epoch"])))


def final_probabilities(gate, cached):
    mixture = gate * cached["direct"] + (1.0 - gate) * cached["mapped"]
    return apply_residual_target_verifiers(mixture, cached["ti"], cached["metal"])


def evaluate_gate(gate, cached):
    labels = cached["labels"]
    mixture = gate * cached["direct"] + (1.0 - gate) * cached["mapped"]
    final = apply_residual_target_verifiers(mixture, cached["ti"], cached["metal"])
    epsilon = torch.finfo(final.dtype).eps
    predictions = final.argmax(1)
    confusion = torch.bincount(labels * 4 + predictions, minlength=16).reshape(4, 4).double()
    f1_denominator = confusion.sum(0) + confusion.sum(1)
    direct_true = cached["direct"].gather(1, labels[:, None]).clamp_min(epsilon)
    mapped_true = cached["mapped"].gather(1, labels[:, None]).clamp_min(epsilon)
    fused_true = mixture.gather(1, labels[:, None]).clamp_min(epsilon)
    oracle_true = torch.maximum(direct_true, mapped_true)
    return {
        "pre_verifier_nll_nats": float(F.nll_loss(mixture.clamp_min(epsilon).log(), labels)),
        "final_nll_nats": float(F.nll_loss(final.clamp_min(epsilon).log(), labels)),
        "routing_regret_nats": float((-fused_true.log() + oracle_true.log()).mean()),
        "final_accuracy": float((predictions == labels).double().mean()),
        "final_macro_f1": float(torch.where(f1_denominator > 0, 2 * confusion.diag() / f1_denominator, 0).mean()),
        "final_target_recall": float(confusion[0, 0] / confusion[0].sum()),
        "final_ti_intrusion": float(confusion[1, 0] / confusion[1].sum()),
        "final_metal_intrusion": float(confusion[3, 0] / confusion[3].sum()),
        "mean_gate": float(gate.mean()),
        "confusion_matrix": confusion.int().tolist(),
    }


def extract_cache(model, records, mapping, transform, dependencies, device, label):
    dataset = HierarchicalMineralImageDataset(records, mapping, transform)
    loader = dependencies["DataLoader"](dataset, batch_size=32, shuffle=False, num_workers=0)
    chunks = {key: [] for key in ("features", "direct", "mapped", "labels", "ti", "metal", "original_gate")}
    captured = []
    hook = model.gate_network.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            for batch_index, (images, labels, _) in enumerate(loader):
                captured.clear()
                outputs = model(images.to(device))
                chunks["features"].append(captured[0])
                chunks["direct"].append(outputs["direct_role_probabilities"])
                chunks["mapped"].append(outputs["mapped_role_probabilities"])
                chunks["labels"].append(labels.to(device))
                chunks["ti"].append(outputs["ti_target_probability"])
                chunks["metal"].append(outputs["metallic_target_probability"])
                chunks["original_gate"].append(outputs["gate"])
                if batch_index % 20 == 0:
                    print(f"extract {label} batch {batch_index}/{len(loader)}", flush=True)
    finally:
        hook.remove()
    return {key: torch.cat(values) for key, values in chunks.items()}


def save_predictions(path, records, gate, cached):
    final = final_probabilities(gate, cached)
    predictions = final.argmax(1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image_id", "split_group_id", "mineral_label", "true_class_id",
            "predicted_class_id", "gate", "prob_0", "prob_1", "prob_2", "prob_3",
        ])
        for record, predicted, gate_value, probabilities in zip(records, predictions, gate, final, strict=True):
            writer.writerow([
                record.image_id,
                record.split_group_id,
                record.mineral_label,
                record.class_id,
                int(predicted),
                float(gate_value),
                *[float(value) for value in probabilities],
            ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-manifest", type=Path, default=ROOT / "outputs/training/gate_supervision_manifests_v1/expert.csv")
    parser.add_argument("--seen-manifest", type=Path, default=ROOT / "outputs/training/gate_supervision_manifests_v1/gate_seen.csv")
    parser.add_argument("--unseen-manifest", type=Path, default=ROOT / "outputs/training/gate_supervision_manifests_v1/gate_unseen.csv")
    parser.add_argument("--expert-dir", type=Path, default=ROOT / "outputs/training/gate_supervision_expert_seed20260920_retry1")
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\成信工科研\人工智能选矿\数据集\mindat_manual_positive_v1"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/training/seen_unseen_gate_study_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    dependencies = require_training_dependencies()
    expert_records = load_manifest_records(args.expert_manifest, args.dataset_root)
    seen_records = load_manifest_records(args.seen_manifest, args.dataset_root)
    unseen_records = load_manifest_records(args.unseen_manifest, args.dataset_root)
    audit = audit_supervision_records(expert_records, seen_records, unseen_records)
    stop_records = [record for record in expert_records if record.split == "val"]
    if not stop_records:
        raise ValueError("expert manifest contains no validation/stop records")
    if args.audit_only:
        print(json.dumps({**audit, "stop_count": len(stop_records)}, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists():
        raise RuntimeError("Output already exists; inspect it before any rerun.")
    args.output_dir.mkdir(parents=True)

    environment_path = args.expert_dir / "environment.json"
    checkpoint_path = args.expert_dir / "best_model.pt"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    mapping = validate_species_role_mapping(expert_records)
    torch_device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if torch_device_name == "auto":
        torch_device_name = "cpu"
    if torch_device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(torch_device_name)
    torch.set_num_threads(4)

    model = HierarchicalRiskGatedVerificationNet(
        dependencies["models"],
        build_role_matrix(mapping, torch),
        pretrained=False,
        backbone_name=environment.get("backbone", "efficientnet_b0"),
        embedding_dim=int(environment.get("embedding_dim", 128)),
        gate_hidden_dim=int(environment.get("gate_hidden_dim", 128)),
        verifier_mode=environment.get("verifier_mode", "residual"),
        ti_verifier_threshold=float(environment.get("ti_verifier_threshold", 0.5)),
        metallic_verifier_threshold=float(environment.get("metallic_verifier_threshold", 0.5)),
        ti_verifier_strength=float(environment.get("ti_verifier_strength", 1.0)),
        metallic_verifier_strength=float(environment.get("metallic_verifier_strength", 1.0)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    frozen_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    _, transform = create_transforms(224, dependencies["transforms"])
    caches = {
        "seen": extract_cache(model, seen_records, mapping, transform, dependencies, device, "seen"),
        "unseen": extract_cache(model, unseen_records, mapping, transform, dependencies, device, "unseen"),
        "stop": extract_cache(model, stop_records, mapping, transform, dependencies, device, "stop"),
    }
    epochs = 2 if args.smoke_run else args.epochs
    seeds = (20260912,) if args.smoke_run else (20260912, 20260913, 20260914)
    objectives = ("final_nll", "final_nll_plus_regret")
    source_records = {"seen": seen_records, "unseen": unseen_records}
    protocol = {
        "scope": "validation-only paired expert-seen versus expert-unseen gate supervision",
        "supervision_audit": audit,
        "expert_stop_count": len(stop_records),
        "gate_seeds": list(seeds),
        "objectives": list(objectives),
        "epochs": epochs,
        "batch_size": 256,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "selection": "lowest final validation NLL; earliest tie",
        "auxiliary_regret_weight": 0.1,
        "target_temperature": 0.2,
        "gap_temperature": 0.5,
        "test_access": False,
        "smoke_run": args.smoke_run,
        "hashes": {
            "expert_manifest": sha256(args.expert_manifest),
            "seen_manifest": sha256(args.seen_manifest),
            "unseen_manifest": sha256(args.unseen_manifest),
            "expert_environment": sha256(environment_path),
            "expert_checkpoint": sha256(checkpoint_path),
            "script": sha256(Path(__file__)),
            "hrgv_network": sha256(ROOT / "scripts/hrgv_network.py"),
        },
        "limitations": "one expert checkpoint; stop set reused for gate selection; mechanism diagnostic, not independent confirmation",
    }
    write_json(args.output_dir / "registered_protocol.json", protocol)

    stop = caches["stop"]
    equal_gate = torch.full((len(stop["labels"]), 1), 0.5, device=device)
    summary = {
        "references": {
            "equal": evaluate_gate(equal_gate, stop),
            "original_joint_gate": evaluate_gate(stop["original_gate"], stop),
        },
        "runs": {},
    }
    for seed in seeds:
        torch.manual_seed(seed)
        template = copy.deepcopy(model.gate_network).requires_grad_(True)
        for layer in template.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for source in ("seen", "unseen"):
            train = caches[source]
            targets = regret_gate_targets(train["direct"], train["mapped"], train["labels"], 0.2, 0.5, torch)
            for objective in objectives:
                torch.manual_seed(seed)
                gate_network = copy.deepcopy(template)
                optimizer = torch.optim.AdamW(gate_network.parameters(), lr=0.001, weight_decay=0.0001)
                history = []
                states = []
                for epoch in range(1, epochs + 1):
                    gate_network.train()
                    batch_losses = []
                    for indices in torch.randperm(len(train["labels"]), device=device).split(256):
                        gate = torch.sigmoid(gate_network(train["features"][indices]))
                        final = final_probabilities(gate, {key: values[indices] for key, values in train.items()})
                        loss = F.nll_loss(final.clamp_min(torch.finfo(final.dtype).eps).log(), train["labels"][indices])
                        if objective == "final_nll_plus_regret":
                            loss = loss + 0.1 * weighted_soft_gate_loss(
                                gate,
                                targets["soft_oracle_gate"][indices],
                                targets["gate_gap_weight"][indices],
                                torch,
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        batch_losses.append(float(loss.detach()))
                    gate_network.eval()
                    with torch.no_grad():
                        stop_gate = torch.sigmoid(gate_network(stop["features"]))
                        row = {
                            "epoch": epoch,
                            "mean_training_batch_loss": sum(batch_losses) / len(batch_losses),
                            **evaluate_gate(stop_gate, stop),
                        }
                    history.append(row)
                    states.append(copy.deepcopy(gate_network.state_dict()))
                    if epoch == 1 or epoch % 10 == 0:
                        print(source, seed, objective, row, flush=True)
                selected = select_earliest_best(history, "final_nll_nats")
                selected_state = states[int(selected["epoch"]) - 1]
                gate_network.load_state_dict(selected_state)
                gate_network.eval()
                with torch.no_grad():
                    selected_gate = torch.sigmoid(gate_network(stop["features"]))
                run_name = f"{source}_{objective}_seed{seed}"
                run_dir = args.output_dir / run_name
                run_dir.mkdir()
                torch.save(selected_state, run_dir / "gate_state.pt")
                write_json(run_dir / "history.json", history)
                write_json(run_dir / "selected_metrics.json", selected)
                write_json(run_dir / "epoch30_metrics.json", history[-1])
                save_predictions(run_dir / "stop_predictions.csv", stop_records, selected_gate, stop)
                summary["runs"][run_name] = {
                    "source": source,
                    "objective": objective,
                    "seed": seed,
                    "selected": selected,
                    "last_epoch": history[-1],
                    "training_count": len(source_records[source]),
                }

    if not all(torch.equal(value.cpu(), frozen_state[key]) for key, value in model.state_dict().items()):
        raise AssertionError("frozen expert parameters or buffers changed")
    if not all(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("frozen expert accumulated gradients")
    summary["frozen_expert_parameters_and_buffers_unchanged"] = True
    summary["frozen_expert_gradients_absent"] = True
    summary["test_access"] = False
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
