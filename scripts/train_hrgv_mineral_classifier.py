from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from analyze_misclassifications import PREDICTION_FIELDS, build_prediction_rows, write_csv
from hrgv_network import (
    HRGVLossWeights,
    HierarchicalRiskGatedVerificationNet,
    compute_hrgv_losses,
    gate_routing_diagnostics,
    regret_gate_targets,
)
from mineral_hierarchy import SpeciesRoleMapping, validate_species_role_mapping
from train_hierarchical_mineral_classifier import (
    compute_inverse_frequency_weights,
    create_hierarchical_dataloaders,
    write_history,
)
from train_mineral_classifier import (
    CLASS_LABELS,
    calculate_metrics,
    compute_class_weights,
    configure_torch_home,
    load_manifest_records,
    require_training_dependencies,
    resolve_device_name,
    set_seed,
    split_records,
    write_json,
)
from train_role_aware_mineral_classifier import select_prediction_records


HRGV_PREDICTION_FIELDS = [
    *PREDICTION_FIELDS,
    "direct_predicted_label",
    "mapped_predicted_label",
    "gate",
    "ti_target_probability",
    "metallic_target_probability",
    "expert_js_divergence",
    "direct_true_probability",
    "mapped_true_probability",
    "fused_true_probability",
    "hard_oracle_gate",
    "soft_oracle_gate",
    "gate_gap_weight",
    "gate_selection_correct",
    "routing_regret_nll",
    "weighted_gate_error",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Hierarchical Risk-Gated Verification Network (HRGV-Net)."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--lambda-direct", type=float, default=0.25)
    parser.add_argument("--lambda-species", type=float, default=0.50)
    parser.add_argument("--lambda-consistency", type=float, default=0.10)
    parser.add_argument("--lambda-verifier", type=float, default=0.50)
    parser.add_argument("--lambda-contrast", type=float, default=0.10)
    parser.add_argument("--lambda-gate-regret", type=float, default=0.0)
    parser.add_argument("--gate-regret-temperature", type=float, default=0.20)
    parser.add_argument("--gate-gap-temperature", type=float, default=0.50)
    parser.add_argument("--disable-gate-regret", action="store_true")
    parser.add_argument("--hard-gate-target", action="store_true")
    parser.add_argument("--unweighted-gate-regret", action="store_true")
    parser.add_argument("--detach-gate-features", action="store_true")
    parser.add_argument("--couple-gate-features", action="store_true")
    parser.add_argument("--contrast-temperature", type=float, default=0.10)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--gate-hidden-dim", type=int, default=128)
    parser.add_argument("--fixed-gate", type=float)
    parser.add_argument("--disable-verifiers", action="store_true")
    parser.add_argument(
        "--verifier-mode",
        choices=("residual", "multiplicative", "disabled"),
        default="residual",
    )
    parser.add_argument("--ti-verifier-threshold", type=float, default=0.5)
    parser.add_argument("--metallic-verifier-threshold", type=float, default=0.5)
    parser.add_argument("--ti-verifier-strength", type=float, default=1.0)
    parser.add_argument("--metallic-verifier-strength", type=float, default=1.0)
    parser.add_argument(
        "--couple-verifier-features",
        action="store_true",
        help="Allow verifier losses to update the shared backbone for the coupling ablation.",
    )
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".torch-cache",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    loss_weights = HRGVLossWeights(
        direct=args.lambda_direct,
        species=args.lambda_species,
        consistency=args.lambda_consistency,
        verifier=args.lambda_verifier,
        contrast=args.lambda_contrast,
        gate_regret=args.lambda_gate_regret,
    )
    loss_weights.validate()
    if args.fixed_gate is not None and not 0.0 <= args.fixed_gate <= 1.0:
        raise ValueError("The fixed gate must lie in [0, 1].")
    thresholds = (args.ti_verifier_threshold, args.metallic_verifier_threshold)
    if any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("Verifier thresholds must lie in (0, 1].")
    strengths = (args.ti_verifier_strength, args.metallic_verifier_strength)
    if any(value < 0 for value in strengths):
        raise ValueError("Verifier strengths must be non-negative.")
    if min(args.epochs, args.batch_size, args.image_size, args.patience) <= 0:
        raise ValueError("Epochs, batch size, image size, and patience must be positive.")
    if min(args.embedding_dim, args.gate_hidden_dim) <= 0:
        raise ValueError("Embedding and gate hidden dimensions must be positive.")
    if args.contrast_temperature <= 0:
        raise ValueError("Contrast temperature must be positive.")
    if args.gate_regret_temperature <= 0 or args.gate_gap_temperature <= 0:
        raise ValueError("Gate temperatures must be positive.")
    if args.detach_gate_features and args.couple_gate_features:
        raise ValueError("Gate features cannot be both detached and coupled.")


def build_role_matrix(mapping: SpeciesRoleMapping, torch) -> torch.Tensor:
    matrix = torch.zeros(
        (len(CLASS_LABELS), len(mapping.species_labels)), dtype=torch.float32
    )
    matrix[
        torch.tensor(mapping.species_role_ids, dtype=torch.long),
        torch.arange(len(mapping.species_labels)),
    ] = 1.0
    return matrix


def calculate_hrgv_risk_metrics(
    targets: Sequence[int], predictions: Sequence[int]
) -> dict[str, float]:
    if len(targets) != len(predictions):
        raise ValueError("Target and prediction lengths must match.")

    def conditional_rate(actual_role: int, predicted_role: int) -> float:
        eligible = [index for index, actual in enumerate(targets) if actual == actual_role]
        if not eligible:
            return float("nan")
        return sum(predictions[index] == predicted_role for index in eligible) / len(eligible)

    target_recall = conditional_rate(0, 0)
    return {
        "target_recall": target_recall,
        "target_miss_rate": 1.0 - target_recall,
        "ti_to_target_intrusion_rate": conditional_rate(1, 0),
        "metallic_to_target_intrusion_rate": conditional_rate(3, 0),
    }


def verifier_subset_accuracy(
    role_targets: Sequence[int],
    target_probabilities: Sequence[float],
    negative_role_id: int,
) -> float:
    if len(role_targets) != len(target_probabilities):
        raise ValueError("Role and verifier probability lengths must match.")
    eligible = [
        index for index, role in enumerate(role_targets) if role in (0, negative_role_id)
    ]
    if not eligible:
        return float("nan")
    correct = 0
    for index in eligible:
        predicted_target = target_probabilities[index] >= 0.5
        correct += predicted_target == (role_targets[index] == 0)
    return correct / len(eligible)


def summarize_gate_routing(
    gate_selection_correct: Sequence[bool],
    routing_regrets_nll: Sequence[float],
    weighted_gate_errors: Sequence[float],
    one_right_one_wrong: Sequence[bool],
) -> dict[str, float]:
    sequences = (
        gate_selection_correct,
        routing_regrets_nll,
        weighted_gate_errors,
        one_right_one_wrong,
    )
    if len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("All gate routing diagnostic fields must have matching lengths.")
    count = len(gate_selection_correct)
    if count == 0:
        raise ValueError("Gate routing diagnostics cannot be empty.")
    disagreement_indices = [
        index for index, value in enumerate(one_right_one_wrong) if value
    ]
    disagreement_accuracy = float("nan")
    if disagreement_indices:
        disagreement_accuracy = sum(
            bool(gate_selection_correct[index]) for index in disagreement_indices
        ) / len(disagreement_indices)
    return {
        "gate_selection_accuracy": sum(bool(value) for value in gate_selection_correct) / count,
        "one_right_gate_selection_accuracy": disagreement_accuracy,
        "mean_routing_regret_nll": sum(routing_regrets_nll) / count,
        "mean_weighted_gate_error": sum(weighted_gate_errors) / count,
    }


def build_hrgv_prediction_rows(
    records,
    final_prediction_ids: Sequence[int],
    confidences: Sequence[float],
    direct_prediction_ids: Sequence[int],
    mapped_prediction_ids: Sequence[int],
    gates: Sequence[float],
    ti_target_probabilities: Sequence[float],
    metallic_target_probabilities: Sequence[float],
    expert_js_divergences: Sequence[float],
    direct_true_probabilities: Sequence[float],
    mapped_true_probabilities: Sequence[float],
    fused_true_probabilities: Sequence[float],
    hard_oracle_gates: Sequence[float],
    soft_oracle_gates: Sequence[float],
    gate_gap_weights: Sequence[float],
    gate_selection_correct: Sequence[bool],
    routing_regrets_nll: Sequence[float],
    weighted_gate_errors: Sequence[float],
) -> list[dict[str, str]]:
    sequences = (
        records,
        final_prediction_ids,
        confidences,
        direct_prediction_ids,
        mapped_prediction_ids,
        gates,
        ti_target_probabilities,
        metallic_target_probabilities,
        expert_js_divergences,
        direct_true_probabilities,
        mapped_true_probabilities,
        fused_true_probabilities,
        hard_oracle_gates,
        soft_oracle_gates,
        gate_gap_weights,
        gate_selection_correct,
        routing_regrets_nll,
        weighted_gate_errors,
    )
    if len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("All HRGV prediction fields must have matching lengths.")
    rows = build_prediction_rows(records, final_prediction_ids, confidences)
    for index, row in enumerate(rows):
        direct_id = direct_prediction_ids[index]
        mapped_id = mapped_prediction_ids[index]
        if direct_id not in range(len(CLASS_LABELS)) or mapped_id not in range(len(CLASS_LABELS)):
            raise ValueError("Invalid direct or mapped prediction id.")
        row.update(
            {
                "direct_predicted_label": CLASS_LABELS[direct_id],
                "mapped_predicted_label": CLASS_LABELS[mapped_id],
                "gate": f"{gates[index]:.6f}",
                "ti_target_probability": f"{ti_target_probabilities[index]:.6f}",
                "metallic_target_probability": f"{metallic_target_probabilities[index]:.6f}",
                "expert_js_divergence": f"{expert_js_divergences[index]:.6f}",
                "direct_true_probability": f"{direct_true_probabilities[index]:.6f}",
                "mapped_true_probability": f"{mapped_true_probabilities[index]:.6f}",
                "fused_true_probability": f"{fused_true_probabilities[index]:.6f}",
                "hard_oracle_gate": f"{hard_oracle_gates[index]:.6f}",
                "soft_oracle_gate": f"{soft_oracle_gates[index]:.6f}",
                "gate_gap_weight": f"{gate_gap_weights[index]:.6f}",
                "gate_selection_correct": "1" if gate_selection_correct[index] else "0",
                "routing_regret_nll": f"{routing_regrets_nll[index]:.6f}",
                "weighted_gate_error": f"{weighted_gate_errors[index]:.6f}",
            }
        )
    return rows


def run_epoch(
    model,
    loader,
    mapping,
    final_role_criterion,
    direct_role_criterion,
    species_criterion,
    verifier_criterion,
    optimizer,
    device,
    weights,
    contrast_temperature,
    gate_target_temperature,
    gate_gap_temperature,
    hard_gate_target,
    unweighted_gate_regret,
    max_batches: int | None,
):
    training = optimizer is not None
    model.train(training)
    loss_names = (
        "loss",
        "final_role_loss",
        "direct_role_loss",
        "species_loss",
        "consistency_loss",
        "ti_verifier_loss",
        "metallic_verifier_loss",
        "verifier_loss",
        "contrast_loss",
        "gate_regret_loss",
    )
    totals = {name: 0.0 for name in loss_names}
    result = {
        "role_targets": [],
        "final_predictions": [],
        "direct_predictions": [],
        "mapped_predictions": [],
        "species_targets": [],
        "species_predictions": [],
        "confidences": [],
        "gates": [],
        "ti_target_probabilities": [],
        "metallic_target_probabilities": [],
        "expert_js_divergences": [],
        "direct_true_probabilities": [],
        "mapped_true_probabilities": [],
        "fused_true_probabilities": [],
        "hard_oracle_gates": [],
        "soft_oracle_gates": [],
        "gate_gap_weights": [],
        "gate_selection_correct": [],
        "routing_regrets_nll": [],
        "weighted_gate_errors": [],
        "one_right_one_wrong": [],
    }
    total_count = 0
    with torch.set_grad_enabled(training):
        for batch_index, (images, roles, species) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, roles, species = images.to(device), roles.to(device), species.to(device)
            outputs = model(images)
            loss, terms = compute_hrgv_losses(
                outputs=outputs,
                role_labels=roles,
                species_labels=species,
                mapping=mapping,
                final_role_criterion=final_role_criterion,
                direct_role_criterion=direct_role_criterion,
                species_criterion=species_criterion,
                verifier_criterion=verifier_criterion,
                weights=weights,
                temperature=contrast_temperature,
                torch=torch,
                gate_target_temperature=gate_target_temperature,
                gate_gap_temperature=gate_gap_temperature,
                hard_gate_target=hard_gate_target,
                unweighted_gate_regret=unweighted_gate_regret,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            final_probabilities = outputs["final_role_probabilities"]
            confidence, final_prediction = final_probabilities.max(dim=1)
            batch_size = roles.size(0)
            total_count += batch_size
            totals["loss"] += float(loss.item()) * batch_size
            for name, term in terms.items():
                totals[name] += float(term.item()) * batch_size
            result["role_targets"].extend(roles.cpu().tolist())
            result["final_predictions"].extend(final_prediction.cpu().tolist())
            result["direct_predictions"].extend(
                outputs["direct_role_probabilities"].argmax(dim=1).cpu().tolist()
            )
            result["mapped_predictions"].extend(
                outputs["mapped_role_probabilities"].argmax(dim=1).cpu().tolist()
            )
            result["species_targets"].extend(species.cpu().tolist())
            result["species_predictions"].extend(
                outputs["species_logits"].argmax(dim=1).cpu().tolist()
            )
            result["confidences"].extend(confidence.cpu().tolist())
            result["gates"].extend(outputs["gate"].squeeze(1).cpu().tolist())
            result["ti_target_probabilities"].extend(
                outputs["ti_target_probability"].squeeze(1).cpu().tolist()
            )
            result["metallic_target_probabilities"].extend(
                outputs["metallic_target_probability"].squeeze(1).cpu().tolist()
            )
            result["expert_js_divergences"].extend(
                outputs["expert_js_divergence"].squeeze(1).cpu().tolist()
            )
            gate_targets = regret_gate_targets(
                outputs["direct_role_probabilities"],
                outputs["mapped_role_probabilities"],
                roles,
                target_temperature=gate_target_temperature,
                gap_temperature=gate_gap_temperature,
                torch=torch,
                hard_target=hard_gate_target,
                unweighted=unweighted_gate_regret,
            )
            routing = gate_routing_diagnostics(
                outputs["gate"],
                outputs["direct_role_probabilities"],
                outputs["mapped_role_probabilities"],
                roles,
                torch,
            )
            result["direct_true_probabilities"].extend(
                routing["direct_true_probability"].squeeze(1).detach().cpu().tolist()
            )
            result["mapped_true_probabilities"].extend(
                routing["mapped_true_probability"].squeeze(1).detach().cpu().tolist()
            )
            result["fused_true_probabilities"].extend(
                routing["fused_true_probability"].squeeze(1).detach().cpu().tolist()
            )
            result["hard_oracle_gates"].extend(
                routing["hard_oracle_gate"].squeeze(1).detach().cpu().tolist()
            )
            result["soft_oracle_gates"].extend(
                gate_targets["soft_oracle_gate"].squeeze(1).cpu().tolist()
            )
            result["gate_gap_weights"].extend(
                gate_targets["gate_gap_weight"].squeeze(1).cpu().tolist()
            )
            result["gate_selection_correct"].extend(
                routing["hard_gate_selection_correct"].squeeze(1).detach().cpu().tolist()
            )
            result["routing_regrets_nll"].extend(
                routing["routing_regret_nll"].squeeze(1).detach().cpu().tolist()
            )
            result["weighted_gate_errors"].extend(
                routing["weighted_gate_error"].squeeze(1).detach().cpu().tolist()
            )
            result["one_right_one_wrong"].extend(
                routing["one_right_one_wrong"].squeeze(1).detach().cpu().tolist()
            )
    if total_count == 0:
        raise RuntimeError("No batches were evaluated.")
    result["losses"] = {name: value / total_count for name, value in totals.items()}
    result["mean_gate"] = sum(result["gates"]) / total_count
    result["mean_expert_js_divergence"] = sum(result["expert_js_divergences"]) / total_count
    result.update(
        summarize_gate_routing(
            result["gate_selection_correct"],
            result["routing_regrets_nll"],
            result["weighted_gate_errors"],
            result["one_right_one_wrong"],
        )
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    configure_torch_home(args.torch_home)
    dependencies = require_training_dependencies()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")
    device = torch.device(resolve_device_name(args.device, torch.cuda.is_available()))
    if args.smoke_run:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.num_workers = 0
        args.patience = 1
        args.no_pretrained = True
        max_batches = 2
    else:
        max_batches = None
    set_seed(args.seed, torch)
    records = load_manifest_records(args.manifest, args.dataset_root)
    mapping = validate_species_role_mapping(records)
    records_by_split = split_records(records)
    role_counts = [
        sum(record.class_id == class_id for record in records_by_split["train"])
        for class_id in range(len(CLASS_LABELS))
    ]
    species_counts = [
        sum(record.mineral_label == label for record in records_by_split["train"])
        for label in mapping.species_labels
    ]
    loaders = create_hierarchical_dataloaders(
        args, records_by_split, mapping, dependencies, device
    )
    role_matrix = build_role_matrix(mapping, torch)
    model = HierarchicalRiskGatedVerificationNet(
        dependencies["models"],
        role_matrix=role_matrix,
        pretrained=not args.no_pretrained,
        embedding_dim=args.embedding_dim,
        gate_hidden_dim=args.gate_hidden_dim,
        fixed_gate=args.fixed_gate,
        disable_verifiers=args.disable_verifiers,
        verifier_mode=args.verifier_mode,
        ti_verifier_threshold=args.ti_verifier_threshold,
        metallic_verifier_threshold=args.metallic_verifier_threshold,
        ti_verifier_strength=args.ti_verifier_strength,
        metallic_verifier_strength=args.metallic_verifier_strength,
        detach_verifier_features=not args.couple_verifier_features,
        detach_gate_features=args.detach_gate_features and not args.couple_gate_features,
    ).to(device)
    role_weight_tensor = torch.tensor(
        compute_class_weights(role_counts), dtype=torch.float32, device=device
    )
    final_role_criterion = nn.NLLLoss(weight=role_weight_tensor)
    direct_role_criterion = nn.CrossEntropyLoss(weight=role_weight_tensor)
    species_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            compute_inverse_frequency_weights(species_counts),
            dtype=torch.float32,
            device=device,
        )
    )
    verifier_criterion = nn.CrossEntropyLoss()
    effective_gate_regret = 0.0 if args.disable_gate_regret else args.lambda_gate_regret
    loss_weights = HRGVLossWeights(
        direct=args.lambda_direct,
        species=args.lambda_species,
        consistency=args.lambda_consistency,
        verifier=args.lambda_verifier,
        contrast=args.lambda_contrast,
        gate_regret=effective_gate_regret,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_verifier_mode = "disabled" if args.disable_verifiers else args.verifier_mode
    model_name = f"hrgv_efficientnet_b0_{effective_verifier_mode}"
    if args.disable_verifiers:
        model_name += "_no_verifiers"
    if args.fixed_gate is not None:
        model_name += f"_fixed_gate_{args.fixed_gate:.2f}"
    if args.couple_verifier_features:
        model_name += "_coupled_verifier_features"
    if effective_gate_regret > 0:
        model_name += "_rsg"
    if args.hard_gate_target:
        model_name += "_hard_gate_target"
    if args.unweighted_gate_regret:
        model_name += "_unweighted_gate_regret"
    if args.couple_gate_features:
        model_name += "_coupled_gate_features"
    write_json(
        args.output_dir / "environment.json",
        {
            "timestamp": datetime.now().isoformat(),
            "torch_version": torch.__version__,
            "device": str(device),
            "model": model_name,
            "manifest": str(args.manifest),
            "dataset_root": str(args.dataset_root),
            "seed": args.seed,
            "class_labels": list(CLASS_LABELS),
            "species_labels": list(mapping.species_labels),
            "species_role_ids": list(mapping.species_role_ids),
            "role_matrix": role_matrix.tolist(),
            "role_counts_train": dict(zip(CLASS_LABELS, role_counts)),
            "species_counts_train": dict(zip(mapping.species_labels, species_counts)),
            "lambda_direct": loss_weights.direct,
            "lambda_species": loss_weights.species,
            "lambda_consistency": loss_weights.consistency,
            "lambda_verifier": loss_weights.verifier,
            "lambda_contrast": loss_weights.contrast,
            "lambda_gate_regret": loss_weights.gate_regret,
            "gate_regret_temperature": args.gate_regret_temperature,
            "gate_gap_temperature": args.gate_gap_temperature,
            "hard_gate_target": args.hard_gate_target,
            "unweighted_gate_regret": args.unweighted_gate_regret,
            "contrast_temperature": args.contrast_temperature,
            "embedding_dim": args.embedding_dim,
            "gate_hidden_dim": args.gate_hidden_dim,
            "fixed_gate": args.fixed_gate,
            "disable_verifiers": args.disable_verifiers,
            "verifier_mode": effective_verifier_mode,
            "ti_verifier_threshold": args.ti_verifier_threshold,
            "metallic_verifier_threshold": args.metallic_verifier_threshold,
            "ti_verifier_strength": args.ti_verifier_strength,
            "metallic_verifier_strength": args.metallic_verifier_strength,
            "detach_verifier_features": not args.couple_verifier_features,
            "detach_gate_features": args.detach_gate_features
            and not args.couple_gate_features,
            "smoke_run": args.smoke_run,
        },
    )
    history: list[dict[str, object]] = []
    best_f1, without_improvement = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model,
            loaders["train"],
            mapping,
            final_role_criterion,
            direct_role_criterion,
            species_criterion,
            verifier_criterion,
            optimizer,
            device,
            loss_weights,
            args.contrast_temperature,
            args.gate_regret_temperature,
            args.gate_gap_temperature,
            args.hard_gate_target,
            args.unweighted_gate_regret,
            max_batches,
        )
        val_result = run_epoch(
            model,
            loaders["val"],
            mapping,
            final_role_criterion,
            direct_role_criterion,
            species_criterion,
            verifier_criterion,
            None,
            device,
            loss_weights,
            args.contrast_temperature,
            args.gate_regret_temperature,
            args.gate_gap_temperature,
            args.hard_gate_target,
            args.unweighted_gate_regret,
            max_batches,
        )
        val_metrics = calculate_metrics(
            val_result["role_targets"], val_result["final_predictions"], dependencies
        )
        scheduler.step(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                **{
                    f"train_{key}": value
                    for key, value in train_result["losses"].items()
                },
                "train_mean_gate": train_result["mean_gate"],
                "train_mean_expert_js_divergence": train_result[
                    "mean_expert_js_divergence"
                ],
                "train_gate_selection_accuracy": train_result[
                    "gate_selection_accuracy"
                ],
                "train_one_right_gate_selection_accuracy": train_result[
                    "one_right_gate_selection_accuracy"
                ],
                "train_mean_routing_regret_nll": train_result[
                    "mean_routing_regret_nll"
                ],
                "train_mean_weighted_gate_error": train_result[
                    "mean_weighted_gate_error"
                ],
                **{f"val_{key}": value for key, value in val_result["losses"].items()},
                "val_mean_gate": val_result["mean_gate"],
                "val_mean_expert_js_divergence": val_result[
                    "mean_expert_js_divergence"
                ],
                "val_gate_selection_accuracy": val_result[
                    "gate_selection_accuracy"
                ],
                "val_one_right_gate_selection_accuracy": val_result[
                    "one_right_gate_selection_accuracy"
                ],
                "val_mean_routing_regret_nll": val_result[
                    "mean_routing_regret_nll"
                ],
                "val_mean_weighted_gate_error": val_result[
                    "mean_weighted_gate_error"
                ],
                **{
                    key: value
                    for key, value in val_metrics.items()
                    if key not in {"class_recall", "confusion_matrix"}
                },
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['losses']['loss']:.4f} "
            f"val_loss={val_result['losses']['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_gate={val_result['mean_gate']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1, without_improvement = val_metrics["macro_f1"], 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_labels": CLASS_LABELS,
                    "species_labels": mapping.species_labels,
                    "species_role_ids": mapping.species_role_ids,
                    "role_matrix": role_matrix,
                    "best_val_macro_f1": best_f1,
                },
                args.output_dir / "best_model.pt",
            )
        else:
            without_improvement += 1
            if without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break
    write_history(args.output_dir / "metrics_history.csv", history)
    checkpoint = torch.load(
        args.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = run_epoch(
        model,
        loaders["test"],
        mapping,
        final_role_criterion,
        direct_role_criterion,
        species_criterion,
        verifier_criterion,
        None,
        device,
        loss_weights,
        args.contrast_temperature,
        args.gate_regret_temperature,
        args.gate_gap_temperature,
        args.hard_gate_target,
        args.unweighted_gate_regret,
        max_batches,
    )
    test_metrics = calculate_metrics(
        test_result["role_targets"], test_result["final_predictions"], dependencies
    )
    test_metrics.update(
        calculate_hrgv_risk_metrics(
            test_result["role_targets"], test_result["final_predictions"]
        )
    )
    test_metrics.update(
        {
            **{f"test_{key}": value for key, value in test_result["losses"].items()},
            "species_accuracy": sum(
                actual == predicted
                for actual, predicted in zip(
                    test_result["species_targets"], test_result["species_predictions"]
                )
            )
            / len(test_result["species_targets"]),
            "ti_verifier_subset_accuracy": verifier_subset_accuracy(
                test_result["role_targets"],
                test_result["ti_target_probabilities"],
                negative_role_id=1,
            ),
            "metallic_verifier_subset_accuracy": verifier_subset_accuracy(
                test_result["role_targets"],
                test_result["metallic_target_probabilities"],
                negative_role_id=3,
            ),
            "mean_gate": test_result["mean_gate"],
            "mean_expert_js_divergence": test_result["mean_expert_js_divergence"],
            "gate_selection_accuracy": test_result["gate_selection_accuracy"],
            "one_right_gate_selection_accuracy": test_result[
                "one_right_gate_selection_accuracy"
            ],
            "mean_routing_regret_nll": test_result["mean_routing_regret_nll"],
            "mean_weighted_gate_error": test_result["mean_weighted_gate_error"],
            "best_val_macro_f1": best_f1,
        }
    )
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    with (args.output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_LABELS])
        for label, values in zip(CLASS_LABELS, test_metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    prediction_records = select_prediction_records(
        records_by_split["test"], len(test_result["final_predictions"])
    )
    prediction_rows = build_hrgv_prediction_rows(
        records=prediction_records,
        final_prediction_ids=test_result["final_predictions"],
        confidences=test_result["confidences"],
        direct_prediction_ids=test_result["direct_predictions"],
        mapped_prediction_ids=test_result["mapped_predictions"],
        gates=test_result["gates"],
        ti_target_probabilities=test_result["ti_target_probabilities"],
        metallic_target_probabilities=test_result["metallic_target_probabilities"],
        expert_js_divergences=test_result["expert_js_divergences"],
        direct_true_probabilities=test_result["direct_true_probabilities"],
        mapped_true_probabilities=test_result["mapped_true_probabilities"],
        fused_true_probabilities=test_result["fused_true_probabilities"],
        hard_oracle_gates=test_result["hard_oracle_gates"],
        soft_oracle_gates=test_result["soft_oracle_gates"],
        gate_gap_weights=test_result["gate_gap_weights"],
        gate_selection_correct=test_result["gate_selection_correct"],
        routing_regrets_nll=test_result["routing_regrets_nll"],
        weighted_gate_errors=test_result["weighted_gate_errors"],
    )
    write_csv(
        args.output_dir / "test_predictions.csv", prediction_rows, HRGV_PREDICTION_FIELDS
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), **test_metrics},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
