from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from analyze_misclassifications import PREDICTION_FIELDS, build_prediction_rows, write_csv
from mineral_hierarchy import SpeciesRoleMapping, aggregate_role_probabilities, validate_species_role_mapping
from train_mineral_classifier import (
    CLASS_LABELS,
    calculate_metrics,
    compute_class_weights,
    configure_torch_home,
    create_transforms,
    load_manifest_records,
    require_training_dependencies,
    resolve_device_name,
    set_seed,
    split_records,
    write_json,
)
from train_role_aware_mineral_classifier import (
    compute_binary_class_weights,
    compute_role_aware_contrastive_loss,
    select_prediction_records,
    target_binary_labels,
)


def species_target_tensor(records, mapping: SpeciesRoleMapping, torch) -> torch.Tensor:
    """Return immutable species indices aligned to the frozen mapping."""
    return torch.tensor(
        [mapping.species_to_index[record.mineral_label] for record in records], dtype=torch.long
    )


def compute_hierarchy_consistency_loss(
    role_logits: torch.Tensor,
    species_logits: torch.Tensor,
    mapping: SpeciesRoleMapping,
    torch,
) -> torch.Tensor:
    """Align direct role predictions with role probabilities aggregated from species logits."""
    species_probabilities = torch.softmax(species_logits, dim=1)
    mapped_role_probabilities = aggregate_role_probabilities(species_probabilities, mapping, torch)
    mapped_role_probabilities = mapped_role_probabilities.clamp_min(torch.finfo(species_probabilities.dtype).eps)
    mapped_role_probabilities = mapped_role_probabilities / mapped_role_probabilities.sum(dim=1, keepdim=True)
    return functional.kl_div(
        functional.log_softmax(role_logits, dim=1),
        mapped_role_probabilities,
        reduction="batchmean",
    )


def compute_inverse_frequency_weights(class_counts: list[int]) -> list[float]:
    if any(count <= 0 for count in class_counts):
        raise ValueError("Every training species must have at least one image.")
    inverse = [1.0 / count for count in class_counts]
    normalizer = sum(inverse) / len(inverse)
    return [weight / normalizer for weight in inverse]


class HierarchicalMineralImageDataset:
    def __init__(self, records, mapping: SpeciesRoleMapping, transform) -> None:
        self.records = records
        self.mapping = mapping
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        return (
            self.transform(image),
            record.class_id,
            self.mapping.species_to_index[record.mineral_label],
        )


def create_hierarchical_dataloaders(args, records_by_split, mapping, dependencies, device):
    DataLoader = dependencies["DataLoader"]
    train_transform, evaluation_transform = create_transforms(args.image_size, dependencies["transforms"])
    datasets = {
        "train": HierarchicalMineralImageDataset(records_by_split["train"], mapping, train_transform),
        "val": HierarchicalMineralImageDataset(records_by_split["val"], mapping, evaluation_transform),
        "test": HierarchicalMineralImageDataset(records_by_split["test"], mapping, evaluation_transform),
    }
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **loader_options),
        "val": DataLoader(datasets["val"], shuffle=False, **loader_options),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_options),
    }


class HierarchicalRoleAwareEfficientNet(nn.Module):
    def __init__(self, models, num_roles: int, num_species: int, pretrained: bool, embedding_dim: int) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(p=base_model.classifier[0].p, inplace=False)
        feature_dim = base_model.classifier[1].in_features
        self.role_head = nn.Linear(feature_dim, num_roles)
        self.species_head = nn.Linear(feature_dim, num_species)
        self.binary_head = nn.Linear(feature_dim, 2)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.avgpool(self.features(images)), 1)
        dropped_features = self.dropout(features)
        return (
            self.role_head(dropped_features),
            self.species_head(dropped_features),
            self.binary_head(features),
            functional.normalize(self.projection_head(features), p=2, dim=1),
        )


def compute_losses(
    outputs,
    role_labels: torch.Tensor,
    species_labels: torch.Tensor,
    mapping: SpeciesRoleMapping,
    role_criterion,
    species_criterion,
    binary_criterion,
    lambda_species: float,
    lambda_consistency: float,
    lambda_binary: float,
    lambda_contrast: float,
    temperature: float,
    torch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    role_logits, species_logits, binary_logits, embeddings = outputs
    role_loss = role_criterion(role_logits, role_labels)
    species_loss = species_criterion(species_logits, species_labels)
    binary_loss = binary_criterion(binary_logits, target_binary_labels(role_labels))
    consistency_loss = compute_hierarchy_consistency_loss(role_logits, species_logits, mapping, torch)
    contrast_loss = compute_role_aware_contrastive_loss(embeddings, role_labels, temperature, torch)
    total_loss = (
        role_loss
        + lambda_species * species_loss
        + lambda_consistency * consistency_loss
        + lambda_binary * binary_loss
        + lambda_contrast * contrast_loss
    )
    return total_loss, {
        "role_loss": role_loss,
        "species_loss": species_loss,
        "binary_loss": binary_loss,
        "consistency_loss": consistency_loss,
        "contrast_loss": contrast_loss,
    }


def run_epoch(
    model,
    loader,
    mapping,
    role_criterion,
    species_criterion,
    binary_criterion,
    optimizer,
    device,
    args,
    max_batches: int | None,
):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "role_loss": 0.0, "species_loss": 0.0, "binary_loss": 0.0, "consistency_loss": 0.0, "contrast_loss": 0.0}
    role_targets: list[int] = []
    role_predictions: list[int] = []
    species_targets: list[int] = []
    species_predictions: list[int] = []
    confidences: list[float] = []
    total_count = 0
    with torch.set_grad_enabled(training):
        for batch_index, (images, roles, species) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, roles, species = images.to(device), roles.to(device), species.to(device)
            outputs = model(images)
            loss, terms = compute_losses(
                outputs, roles, species, mapping, role_criterion, species_criterion, binary_criterion,
                args.lambda_species, args.lambda_consistency, args.lambda_binary, args.lambda_contrast,
                args.contrast_temperature, torch,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            role_probabilities = torch.softmax(outputs[0], dim=1)
            confidence, role_prediction = role_probabilities.max(dim=1)
            species_prediction = outputs[1].argmax(dim=1)
            batch_size = roles.size(0)
            total_count += batch_size
            totals["loss"] += float(loss.item()) * batch_size
            for key, term in terms.items():
                totals[key] += float(term.item()) * batch_size
            role_targets.extend(roles.cpu().tolist())
            role_predictions.extend(role_prediction.cpu().tolist())
            species_targets.extend(species.cpu().tolist())
            species_predictions.extend(species_prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    if total_count == 0:
        raise RuntimeError("No batches were evaluated.")
    return (
        {key: value / total_count for key, value in totals.items()},
        role_targets,
        role_predictions,
        species_targets,
        species_predictions,
        confidences,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hierarchical role-aware mineral classifier.")
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
    parser.add_argument("--lambda-species", type=float, default=0.50)
    parser.add_argument("--lambda-consistency", type=float, default=0.10)
    parser.add_argument("--lambda-binary", type=float, default=0.25)
    parser.add_argument("--lambda-contrast", type=float, default=0.10)
    parser.add_argument("--contrast-temperature", type=float, default=0.10)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--torch-home", type=Path, default=Path(__file__).resolve().parents[1] / ".torch-cache")
    return parser.parse_args()


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    if min(args.lambda_species, args.lambda_consistency, args.lambda_binary, args.lambda_contrast) < 0:
        raise ValueError("Loss weights must be non-negative.")
    configure_torch_home(args.torch_home)
    dependencies = require_training_dependencies()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")
    device = torch.device(resolve_device_name(args.device, torch.cuda.is_available()))
    if args.smoke_run:
        args.epochs, args.batch_size, args.num_workers, args.patience, args.no_pretrained = 1, min(args.batch_size, 4), 0, 1, True
        max_batches = 2
    else:
        max_batches = None
    set_seed(args.seed, torch)
    records = load_manifest_records(args.manifest, args.dataset_root)
    mapping = validate_species_role_mapping(records)
    records_by_split = split_records(records)
    role_counts = [sum(record.class_id == class_id for record in records_by_split["train"]) for class_id in range(len(CLASS_LABELS))]
    species_counts = [sum(record.mineral_label == label for record in records_by_split["train"]) for label in mapping.species_labels]
    loaders = create_hierarchical_dataloaders(args, records_by_split, mapping, dependencies, device)
    model = HierarchicalRoleAwareEfficientNet(
        dependencies["models"], len(CLASS_LABELS), len(mapping.species_labels), not args.no_pretrained, args.embedding_dim
    ).to(device)
    role_criterion = nn.CrossEntropyLoss(weight=torch.tensor(compute_class_weights(role_counts), dtype=torch.float32, device=device))
    species_criterion = nn.CrossEntropyLoss(weight=torch.tensor(compute_inverse_frequency_weights(species_counts), dtype=torch.float32, device=device))
    binary_criterion = nn.CrossEntropyLoss(weight=torch.tensor(compute_binary_class_weights(records_by_split), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "environment.json", {
        "timestamp": datetime.now().isoformat(), "torch_version": torch.__version__, "device": str(device),
        "model": "hierarchical_role_aware_efficientnet_b0", "manifest": str(args.manifest),
        "dataset_root": str(args.dataset_root), "seed": args.seed, "species_labels": list(mapping.species_labels),
        "species_role_ids": list(mapping.species_role_ids), "role_counts_train": dict(zip(CLASS_LABELS, role_counts)),
        "species_counts_train": dict(zip(mapping.species_labels, species_counts)),
        "lambda_species": args.lambda_species, "lambda_consistency": args.lambda_consistency,
        "lambda_binary": args.lambda_binary, "lambda_contrast": args.lambda_contrast,
        "contrast_temperature": args.contrast_temperature, "embedding_dim": args.embedding_dim, "smoke_run": args.smoke_run,
    })
    history: list[dict[str, object]] = []
    best_f1, without_improvement = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        train_values = run_epoch(model, loaders["train"], mapping, role_criterion, species_criterion, binary_criterion, optimizer, device, args, max_batches)
        val_values = run_epoch(model, loaders["val"], mapping, role_criterion, species_criterion, binary_criterion, None, device, args, max_batches)
        val_losses, val_targets, val_predictions, _, _, _ = val_values
        val_metrics = calculate_metrics(val_targets, val_predictions, dependencies)
        scheduler.step(val_metrics["macro_f1"])
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train_values[0].items()}, **{f"val_{key}": value for key, value in val_losses.items()}, **{key: value for key, value in val_metrics.items() if key not in {"class_recall", "confusion_matrix"}}, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch} train_loss={train_values[0]['loss']:.4f} val_loss={val_losses['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1, without_improvement = val_metrics["macro_f1"], 0
            torch.save({"model_state_dict": model.state_dict(), "class_labels": CLASS_LABELS, "species_labels": mapping.species_labels, "best_val_macro_f1": best_f1}, args.output_dir / "best_model.pt")
        else:
            without_improvement += 1
            if without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break
    write_history(args.output_dir / "metrics_history.csv", history)
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_values = run_epoch(model, loaders["test"], mapping, role_criterion, species_criterion, binary_criterion, None, device, args, max_batches)
    test_losses, test_targets, test_predictions, species_targets, species_predictions, confidences = test_values
    test_metrics = calculate_metrics(test_targets, test_predictions, dependencies)
    test_metrics.update({"test_loss": test_losses["loss"], "test_role_loss": test_losses["role_loss"], "test_species_loss": test_losses["species_loss"], "test_binary_loss": test_losses["binary_loss"], "test_consistency_loss": test_losses["consistency_loss"], "test_contrast_loss": test_losses["contrast_loss"], "species_accuracy": sum(actual == predicted for actual, predicted in zip(species_targets, species_predictions)) / len(species_targets), "best_val_macro_f1": best_f1})
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_LABELS])
        for label, values in zip(CLASS_LABELS, test_metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    prediction_records = select_prediction_records(records_by_split["test"], len(test_predictions))
    write_csv(args.output_dir / "test_predictions.csv", build_prediction_rows(prediction_records, test_predictions, confidences), PREDICTION_FIELDS)
    print(json.dumps({"output_dir": str(args.output_dir), **test_metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
