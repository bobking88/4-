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
from train_mineral_classifier import (
    CLASS_LABELS,
    MineralImageDataset,
    calculate_metrics,
    compute_class_weights,
    configure_torch_home,
    create_dataloaders,
    create_transforms,
    load_manifest_records,
    require_training_dependencies,
    resolve_device_name,
    set_seed,
    split_records,
    write_json,
)


TARGET_CLASS_ID = CLASS_LABELS.index("target_mineral")
ROLE_AWARE_NEGATIVE_IDS = frozenset(
    {
        CLASS_LABELS.index("ti_bearing_negative"),
        CLASS_LABELS.index("metallic_hard_negative"),
    }
)


def target_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map the target proxy class to one and all other classes to zero."""
    return (labels == TARGET_CLASS_ID).long()


def compute_binary_inverse_frequency_weights(class_counts: list[int]) -> list[float]:
    """Return mean-normalized inverse-frequency weights for the binary head."""
    if len(class_counts) != 2 or any(count <= 0 for count in class_counts):
        raise ValueError("Binary class weights require two positive class counts.")
    total = sum(class_counts)
    weights = [total / (2 * count) for count in class_counts]
    normalizer = sum(weights) / len(weights)
    return [weight / normalizer for weight in weights]


def compute_role_aware_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    torch,
) -> torch.Tensor:
    """Pull same-class samples together and separate target from two hard-negative roles.

    Only target-vs-Ti-bearing and target-vs-metallic pairs form negative terms. This
    leaves gangue and Ti-bearing-vs-metallic relations to the primary classifier.
    """
    if temperature <= 0:
        raise ValueError("Contrast temperature must be positive.")
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.size(0) != labels.size(0):
        raise ValueError("Embeddings must be [batch, features] and labels must match the batch size.")
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0

    normalized = functional.normalize(embeddings, p=2, dim=1)
    similarities = normalized @ normalized.T / temperature
    batch_size = labels.size(0)
    identity = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    losses: list[torch.Tensor] = []
    for index in range(batch_size):
        anchor_label = int(labels[index].item())
        positive_mask = (labels == anchor_label) & ~identity[index]
        if not bool(positive_mask.any()):
            continue
        if anchor_label == TARGET_CLASS_ID:
            risk_negative_mask = torch.zeros_like(labels, dtype=torch.bool)
            for class_id in ROLE_AWARE_NEGATIVE_IDS:
                risk_negative_mask |= labels == class_id
        elif anchor_label in ROLE_AWARE_NEGATIVE_IDS:
            risk_negative_mask = labels == TARGET_CLASS_ID
        else:
            continue
        comparison_mask = positive_mask | risk_negative_mask
        if not bool(risk_negative_mask.any()):
            continue
        positive_logsumexp = torch.logsumexp(similarities[index][positive_mask], dim=0)
        comparison_logsumexp = torch.logsumexp(similarities[index][comparison_mask], dim=0)
        losses.append(-(positive_logsumexp - comparison_logsumexp))
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


class RoleAwareEfficientNet(nn.Module):
    def __init__(self, models, num_classes: int, pretrained: bool, embedding_dim: int = 128) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.dropout = base_model.classifier[0]
        feature_dim = base_model.classifier[1].in_features
        self.four_class_head = nn.Linear(feature_dim, num_classes)
        self.binary_head = nn.Linear(feature_dim, 2)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(images)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        four_class_logits = self.four_class_head(self.dropout(features))
        binary_logits = self.binary_head(features)
        embedding = functional.normalize(self.projection_head(features), p=2, dim=1)
        return four_class_logits, binary_logits, embedding


def compute_binary_class_weights(records_by_split: dict[str, list]) -> list[float]:
    counts = [0, 0]
    for record in records_by_split["train"]:
        counts[1 if record.class_id == TARGET_CLASS_ID else 0] += 1
    return compute_binary_inverse_frequency_weights(counts)


def select_prediction_records(records: list, prediction_count: int) -> list:
    """Align prediction records with partial evaluation used by a smoke run."""
    if prediction_count < 0 or prediction_count > len(records):
        raise ValueError("Prediction count must be within the available record range.")
    return records[:prediction_count]


def compute_losses(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    four_class_criterion,
    binary_criterion,
    lambda_binary: float,
    lambda_contrast: float,
    temperature: float,
    torch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    four_class_logits, binary_logits, embeddings = outputs
    four_class_loss = four_class_criterion(four_class_logits, labels)
    binary_loss = binary_criterion(binary_logits, target_binary_labels(labels))
    contrast_loss = compute_role_aware_contrastive_loss(embeddings, labels, temperature, torch)
    total_loss = four_class_loss + lambda_binary * binary_loss + lambda_contrast * contrast_loss
    return total_loss, {
        "four_class_loss": four_class_loss,
        "binary_loss": binary_loss,
        "contrast_loss": contrast_loss,
    }


def run_epoch(
    model,
    loader,
    four_class_criterion,
    binary_criterion,
    optimizer,
    device,
    lambda_binary: float,
    lambda_contrast: float,
    temperature: float,
    max_batches: int | None,
) -> tuple[dict[str, float], list[int], list[int], list[float]]:
    is_training = optimizer is not None
    model.train(is_training)
    aggregate_losses = {"loss": 0.0, "four_class_loss": 0.0, "binary_loss": 0.0, "contrast_loss": 0.0}
    total_count = 0
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    with torch.set_grad_enabled(is_training):
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss, loss_terms = compute_losses(
                outputs,
                labels,
                four_class_criterion,
                binary_criterion,
                lambda_binary,
                lambda_contrast,
                temperature,
                torch,
            )
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            four_class_logits = outputs[0]
            probabilities = torch.softmax(four_class_logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            batch_size = labels.size(0)
            total_count += batch_size
            aggregate_losses["loss"] += float(loss.item()) * batch_size
            for key, term in loss_terms.items():
                aggregate_losses[key] += float(term.item()) * batch_size
            targets.extend(labels.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    if total_count == 0:
        raise RuntimeError("No batches were evaluated.")
    return ({key: value / total_count for key, value in aggregate_losses.items()}, targets, predictions, confidences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a role-aware hard-negative mineral classifier.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--lambda-binary", type=float, default=0.25)
    parser.add_argument("--lambda-contrast", type=float, default=0.10)
    parser.add_argument("--contrast-temperature", type=float, default=0.10)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".torch-cache",
    )
    return parser.parse_args()


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    if args.lambda_binary < 0 or args.lambda_contrast < 0:
        raise ValueError("Loss weights must be non-negative.")
    torch_home = configure_torch_home(args.torch_home)
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
    records_by_split = split_records(records)
    class_counts = [
        sum(record.class_id == class_id for record in records_by_split["train"])
        for class_id in range(len(CLASS_LABELS))
    ]
    four_class_weights = compute_class_weights(class_counts)
    binary_weights = compute_binary_class_weights(records_by_split)
    loaders = create_dataloaders(args, records_by_split, dependencies, device)
    model = RoleAwareEfficientNet(
        dependencies["models"], len(CLASS_LABELS), not args.no_pretrained, args.embedding_dim
    ).to(device)
    four_class_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(four_class_weights, dtype=torch.float32, device=device)
    )
    binary_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(binary_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "class_labels": list(CLASS_LABELS),
        "class_counts_train": dict(zip(CLASS_LABELS, class_counts)),
        "four_class_weights": dict(zip(CLASS_LABELS, four_class_weights)),
        "binary_class_weights": {"non_target": binary_weights[0], "target_proxy": binary_weights[1]},
        "manifest": str(args.manifest),
        "dataset_root": str(args.dataset_root),
        "torch_home": str(torch_home),
        "seed": args.seed,
        "model": "role_aware_efficientnet_b0",
        "lambda_binary": args.lambda_binary,
        "lambda_contrast": args.lambda_contrast,
        "contrast_temperature": args.contrast_temperature,
        "embedding_dim": args.embedding_dim,
        "smoke_run": args.smoke_run,
    }
    write_json(args.output_dir / "environment.json", environment)
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_losses, _, _, _ = run_epoch(
            model, loaders["train"], four_class_criterion, binary_criterion, optimizer, device,
            args.lambda_binary, args.lambda_contrast, args.contrast_temperature, max_batches,
        )
        val_losses, val_targets, val_predictions, _ = run_epoch(
            model, loaders["val"], four_class_criterion, binary_criterion, None, device,
            args.lambda_binary, args.lambda_contrast, args.contrast_temperature, max_batches,
        )
        val_metrics = calculate_metrics(val_targets, val_predictions, dependencies)
        scheduler.step(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_losses["loss"],
                "train_four_class_loss": train_losses["four_class_loss"],
                "train_binary_loss": train_losses["binary_loss"],
                "train_contrast_loss": train_losses["contrast_loss"],
                "val_loss": val_losses["loss"],
                "val_four_class_loss": val_losses["four_class_loss"],
                "val_binary_loss": val_losses["binary_loss"],
                "val_contrast_loss": val_losses["contrast_loss"],
                **{key: value for key, value in val_metrics.items() if key not in {"class_recall", "confusion_matrix"}},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_losses['loss']:.4f} val_loss={val_losses['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "role_aware_efficientnet_b0",
                    "class_labels": CLASS_LABELS,
                    "best_val_macro_f1": best_f1,
                },
                args.output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break
    write_history(args.output_dir / "metrics_history.csv", history)
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_losses, test_targets, test_predictions, test_confidences = run_epoch(
        model, loaders["test"], four_class_criterion, binary_criterion, None, device,
        args.lambda_binary, args.lambda_contrast, args.contrast_temperature, max_batches,
    )
    test_metrics = calculate_metrics(test_targets, test_predictions, dependencies)
    test_metrics.update(
        {
            "test_loss": test_losses["loss"],
            "test_four_class_loss": test_losses["four_class_loss"],
            "test_binary_loss": test_losses["binary_loss"],
            "test_contrast_loss": test_losses["contrast_loss"],
            "best_val_macro_f1": best_f1,
        }
    )
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_LABELS])
        for label, values in zip(CLASS_LABELS, test_metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    prediction_records = select_prediction_records(records_by_split["test"], len(test_predictions))
    prediction_rows = build_prediction_rows(prediction_records, test_predictions, test_confidences)
    write_csv(args.output_dir / "test_predictions.csv", prediction_rows, PREDICTION_FIELDS)
    print(json.dumps({"output_dir": str(args.output_dir), **test_metrics}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
