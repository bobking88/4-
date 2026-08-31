from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn

from hrgv_network import HRGVLossWeights, HierarchicalRiskGatedVerificationNet
from train_hierarchical_mineral_classifier import (
    compute_inverse_frequency_weights,
    create_hierarchical_dataloaders,
)
from train_hrgv_mineral_classifier import build_role_matrix, run_epoch
from train_mineral_classifier import (
    CLASS_LABELS,
    compute_class_weights,
    configure_torch_home,
    load_manifest_records,
    require_training_dependencies,
    resolve_device_name,
    set_seed,
    split_records,
)
from mineral_hierarchy import validate_species_role_mapping
from train_role_aware_mineral_classifier import select_prediction_records


HIGH_PRECISION_FIELDS = (
    "image_id",
    "split_group_id",
    "true_label",
    "gate",
    "direct_true_probability",
    "mapped_true_probability",
    "fused_true_probability",
    "hard_oracle_gate",
    "soft_oracle_gate",
    "routing_regret_nll",
)


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _record_true_label(record: Any) -> Any:
    if isinstance(record, dict):
        return record["true_label"]
    return record.four_class_label


def _format_probability(value: float) -> str:
    return f"{float(value):.15f}".rstrip("0").rstrip(".")


def build_high_precision_rows(
    records: Sequence[Any],
    *,
    gates: Sequence[float],
    direct_true_probabilities: Sequence[float],
    mapped_true_probabilities: Sequence[float],
    fused_true_probabilities: Sequence[float],
    hard_oracle_gates: Sequence[float],
    soft_oracle_gates: Sequence[float],
    routing_regrets_nll: Sequence[float],
) -> list[dict[str, str]]:
    sequences = (
        records,
        gates,
        direct_true_probabilities,
        mapped_true_probabilities,
        fused_true_probabilities,
        hard_oracle_gates,
        soft_oracle_gates,
        routing_regrets_nll,
    )
    if len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("All replay routing fields must have matching lengths.")
    rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        rows.append(
            {
                "image_id": str(_record_value(record, "image_id")),
                "split_group_id": str(_record_value(record, "split_group_id")),
                "true_label": str(_record_true_label(record)),
                "gate": _format_probability(gates[index]),
                "direct_true_probability": _format_probability(
                    direct_true_probabilities[index]
                ),
                "mapped_true_probability": _format_probability(
                    mapped_true_probabilities[index]
                ),
                "fused_true_probability": _format_probability(
                    fused_true_probabilities[index]
                ),
                "hard_oracle_gate": _format_probability(hard_oracle_gates[index]),
                "soft_oracle_gate": _format_probability(soft_oracle_gates[index]),
                "routing_regret_nll": _format_probability(routing_regrets_nll[index]),
            }
        )
    return rows


def _environment_bool(environment: dict[str, Any], name: str) -> bool:
    return bool(environment.get(name, False))


def replay_routing_diagnostics(
    environment_path: Path,
    checkpoint_path: Path,
    output_csv: Path,
    *,
    manifest: Path | None = None,
    dataset_root: Path | None = None,
    device_name: str = "auto",
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 224,
    torch_home: Path | None = None,
) -> dict[str, object]:
    """Replay a saved HRGV checkpoint and export full-precision routing diagnostics."""
    environment = json.loads(Path(environment_path).read_text(encoding="utf-8"))
    manifest_path = Path(manifest or environment["manifest"])
    dataset_path = Path(dataset_root or environment["dataset_root"])
    if not Path(checkpoint_path).is_file():
        raise ValueError(f"Checkpoint is missing: {checkpoint_path}")
    if torch_home is not None:
        configure_torch_home(torch_home)
    dependencies = require_training_dependencies()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")
    device = torch.device(resolve_device_name(device_name, torch.cuda.is_available()))
    set_seed(int(environment["seed"]), torch)

    records = load_manifest_records(manifest_path, dataset_path)
    mapping = validate_species_role_mapping(records)
    records_by_split = split_records(records)
    loader_args = SimpleNamespace(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    loaders = create_hierarchical_dataloaders(
        loader_args, records_by_split, mapping, dependencies, device
    )
    role_matrix = build_role_matrix(mapping, torch)
    model = HierarchicalRiskGatedVerificationNet(
        dependencies["models"],
        role_matrix=role_matrix,
        pretrained=False,
        backbone_name=str(environment.get("backbone", "efficientnet_b0")),
        embedding_dim=int(environment.get("embedding_dim", 128)),
        gate_hidden_dim=int(environment.get("gate_hidden_dim", 128)),
        fixed_gate=environment.get("fixed_gate"),
        disable_verifiers=_environment_bool(environment, "disable_verifiers"),
        verifier_mode=str(environment.get("verifier_mode", "residual")),
        ti_verifier_threshold=float(environment.get("ti_verifier_threshold", 0.5)),
        metallic_verifier_threshold=float(
            environment.get("metallic_verifier_threshold", 0.5)
        ),
        ti_verifier_strength=float(environment.get("ti_verifier_strength", 1.0)),
        metallic_verifier_strength=float(
            environment.get("metallic_verifier_strength", 1.0)
        ),
        detach_verifier_features=_environment_bool(
            environment, "detach_verifier_features"
        ),
        detach_gate_features=_environment_bool(environment, "detach_gate_features"),
        enable_cgdc=_environment_bool(environment, "enable_cgdc"),
        cgdc_shared_features=_environment_bool(environment, "cgdc_shared_features"),
        cgdc_unconditional=_environment_bool(environment, "cgdc_unconditional"),
        adapter_bottleneck_dim=int(environment.get("adapter_bottleneck_dim", 128)),
        calibration_hidden_dim=int(environment.get("calibration_hidden_dim", 256)),
        enable_rpg=_environment_bool(environment, "enable_rpg"),
        rpg_entropy_mode=str(environment.get("rpg_entropy_mode", "partitioned")),
        enable_mrpg=_environment_bool(environment, "enable_mrpg"),
        mrpg_between_mode=str(environment.get("mrpg_between_mode", "monotone")),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    role_counts = [
        sum(record.class_id == class_id for record in records_by_split["train"])
        for class_id in range(len(CLASS_LABELS))
    ]
    species_counts = [
        sum(record.mineral_label == label for record in records_by_split["train"])
        for label in mapping.species_labels
    ]
    role_weights = torch.tensor(
        compute_class_weights(role_counts), dtype=torch.float32, device=device
    )
    loss_weights = HRGVLossWeights(
        direct=float(environment.get("lambda_direct", 0.25)),
        species=float(environment.get("lambda_species", 0.5)),
        consistency=float(environment.get("lambda_consistency", 0.1)),
        verifier=float(environment.get("lambda_verifier", 0.5)),
        contrast=float(environment.get("lambda_contrast", 0.1)),
        gate_regret=float(environment.get("lambda_gate_regret", 0.0)),
        decomposition=float(environment.get("lambda_decomposition", 0.0)),
        calibration=float(environment.get("lambda_calibration", 0.0)),
    )
    result = run_epoch(
        model,
        loaders["test"],
        mapping,
        nn.NLLLoss(weight=role_weights),
        nn.CrossEntropyLoss(weight=role_weights),
        nn.CrossEntropyLoss(
            weight=torch.tensor(
                compute_inverse_frequency_weights(species_counts),
                dtype=torch.float32,
                device=device,
            )
        ),
        nn.CrossEntropyLoss(),
        None,
        device,
        loss_weights,
        float(environment.get("contrast_temperature", 0.1)),
        float(environment.get("gate_regret_temperature", 0.2)),
        float(environment.get("gate_gap_temperature", 0.5)),
        _environment_bool(environment, "hard_gate_target"),
        _environment_bool(environment, "unweighted_gate_regret"),
        None,
    )
    records_for_predictions = select_prediction_records(
        records_by_split["test"], len(result["gates"])
    )
    rows = build_high_precision_rows(
        records_for_predictions,
        gates=result["gates"],
        direct_true_probabilities=result["direct_true_probabilities"],
        mapped_true_probabilities=result["mapped_true_probabilities"],
        fused_true_probabilities=result["fused_true_probabilities"],
        hard_oracle_gates=result["hard_oracle_gates"],
        soft_oracle_gates=result["soft_oracle_gates"],
        routing_regrets_nll=result["routing_regrets_nll"],
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HIGH_PRECISION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "environment": str(environment_path),
        "checkpoint": str(checkpoint_path),
        "output_csv": str(output_csv),
        "device": str(device),
        "count": len(rows),
        "mean_routing_regret_nll": result["mean_routing_regret_nll"],
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay an HRGV checkpoint for high-precision routing diagnostics."
    )
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--torch-home", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = replay_routing_diagnostics(
        args.environment,
        args.checkpoint,
        args.output_csv,
        manifest=args.manifest,
        dataset_root=args.dataset_root,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        torch_home=args.torch_home,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
