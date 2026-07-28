from __future__ import annotations

import csv
import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


CLASS_LABELS = (
    "target_mineral",
    "ti_bearing_negative",
    "gangue_negative",
    "metallic_hard_negative",
)
CLASS_TO_ID = {label: index for index, label in enumerate(CLASS_LABELS)}
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    image_path: Path
    mineral_label: str
    four_class_label: str
    class_id: int
    mindat_photo_id: str
    split_group_id: str
    split: str


def load_manifest_records(
    manifest_path: Path,
    dataset_root: Path,
) -> list[ManifestRecord]:
    required_fields = {
        "image_id",
        "relative_path",
        "mineral_label",
        "four_class_label",
        "four_class_id",
        "mindat_photo_id",
        "split_group_id",
        "split",
    }
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError("Manifest is missing required columns.")
        records: list[ManifestRecord] = []
        for row in reader:
            label = row["four_class_label"]
            try:
                class_id = int(row["four_class_id"])
            except ValueError as exc:
                raise ValueError(f"Invalid class id for {row['image_id']}") from exc
            if label not in CLASS_TO_ID or CLASS_TO_ID[label] != class_id:
                raise ValueError(
                    f"Invalid four-class mapping for {row['image_id']}: {label} -> {class_id}"
                )
            if row["split"] not in VALID_SPLITS:
                raise ValueError(f"Invalid split for {row['image_id']}: {row['split']}")
            image_path = dataset_root.joinpath(*Path(row["relative_path"]).parts)
            if not image_path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")
            records.append(
                ManifestRecord(
                    image_id=row["image_id"],
                    image_path=image_path,
                    mineral_label=row["mineral_label"],
                    four_class_label=label,
                    class_id=class_id,
                    mindat_photo_id=row["mindat_photo_id"],
                    split_group_id=row["split_group_id"],
                    split=row["split"],
                )
            )
    return records


def compute_class_weights(class_counts: Iterable[int]) -> list[float]:
    counts = list(class_counts)
    if len(counts) != len(CLASS_LABELS):
        raise ValueError("Class count length does not match the four-class mapping.")
    if any(count <= 0 for count in counts):
        raise ValueError("Every class must have at least one training image.")
    inverse = [1.0 / count for count in counts]
    mean_inverse = sum(inverse) / len(inverse)
    return [value / mean_inverse for value in inverse]


def resolve_device_name(requested_device: str, cuda_available: bool) -> str:
    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested_device


def configure_torch_home(cache_path: Path) -> Path:
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_path)
    return cache_path


class MineralImageDataset:
    def __init__(self, records: list[ManifestRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), record.class_id


def require_training_dependencies():
    try:
        import numpy as np
        import torch
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements-training.txt "
            "into the project .venv-training environment first."
        ) from exc
    return {
        "np": np,
        "torch": torch,
        "nn": nn,
        "DataLoader": DataLoader,
        "models": models,
        "transforms": transforms,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
    }


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_transforms(image_size: int, transforms):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def build_model(model_name: str, num_classes: int, pretrained: bool, models, nn):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def split_records(records: list[ManifestRecord]) -> dict[str, list[ManifestRecord]]:
    result = {split: [] for split in sorted(VALID_SPLITS)}
    for record in records:
        result[record.split].append(record)
    for split, split_records_list in result.items():
        if not split_records_list:
            raise ValueError(f"No records found for required split: {split}")
        split_records_list.sort(key=lambda record: record.image_id)
    return result


def create_dataloaders(args, records_by_split, dependencies, device):
    torch = dependencies["torch"]
    DataLoader = dependencies["DataLoader"]
    train_transform, evaluation_transform = create_transforms(
        args.image_size, dependencies["transforms"]
    )
    datasets = {
        "train": MineralImageDataset(records_by_split["train"], train_transform),
        "val": MineralImageDataset(records_by_split["val"], evaluation_transform),
        "test": MineralImageDataset(records_by_split["test"], evaluation_transform),
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
        "datasets": datasets,
    }


def run_epoch(model, loader, criterion, optimizer, device, torch, max_batches: int | None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    predictions: list[int] = []
    targets: list[int] = []
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
    if total_count == 0:
        raise RuntimeError("No batches were evaluated.")
    return total_loss / total_count, targets, predictions


def calculate_metrics(targets, predictions, dependencies) -> dict[str, object]:
    labels = list(range(len(CLASS_LABELS)))
    return {
        "accuracy": float(dependencies["accuracy_score"](targets, predictions)),
        "macro_precision": float(
            dependencies["precision_score"](targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            dependencies["recall_score"](targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            dependencies["f1_score"](targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "class_recall": {
            CLASS_LABELS[index]: float(value)
            for index, value in enumerate(
                dependencies["recall_score"](
                    targets, predictions, labels=labels, average=None, zero_division=0
                )
            )
        },
        "confusion_matrix": dependencies["confusion_matrix"](targets, predictions, labels=labels).tolist(),
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a four-class mineral image baseline.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"D:\成信工科研\人工智能选矿\数据集\dataset_audit\dataset_split_manifest.csv"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\成信工科研\人工智能选矿\数据集\mindat_manual_positive_v1"),
    )
    parser.add_argument("--model", choices=("resnet50", "efficientnet_b0"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".torch-cache",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_home = configure_torch_home(args.torch_home)
    dependencies = require_training_dependencies()
    torch = dependencies["torch"]
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
    class_weights = compute_class_weights(class_counts)
    loaders = create_dataloaders(args, records_by_split, dependencies, device)
    model = build_model(
        args.model,
        num_classes=len(CLASS_LABELS),
        pretrained=not args.no_pretrained,
        models=dependencies["models"],
        nn=dependencies["nn"],
    ).to(device)
    criterion = dependencies["nn"].CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("outputs") / "training" / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "class_labels": list(CLASS_LABELS),
        "class_counts_train": dict(zip(CLASS_LABELS, class_counts)),
        "class_weights": dict(zip(CLASS_LABELS, class_weights)),
        "manifest": str(args.manifest),
        "dataset_root": str(args.dataset_root),
        "torch_home": str(torch_home),
        "seed": args.seed,
        "model": args.model,
        "smoke_run": args.smoke_run,
    }
    write_json(output_dir / "environment.json", environment)

    history: list[dict[str, object]] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, loaders["train"], criterion, optimizer, device, torch, max_batches
        )
        val_loss, val_targets, val_predictions = run_epoch(
            model, loaders["val"], criterion, None, device, torch, max_batches
        )
        val_metrics = calculate_metrics(val_targets, val_predictions, dependencies)
        scheduler.step(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{key: value for key, value in val_metrics.items() if key not in {"class_recall", "confusion_matrix"}},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "class_labels": CLASS_LABELS,
                    "best_val_macro_f1": best_f1,
                },
                output_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    with (output_dir / "metrics_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_targets, test_predictions = run_epoch(
        model, loaders["test"], criterion, None, device, torch, max_batches
    )
    test_metrics = calculate_metrics(test_targets, test_predictions, dependencies)
    test_metrics["test_loss"] = test_loss
    test_metrics["best_val_macro_f1"] = best_f1
    write_json(output_dir / "test_metrics.json", test_metrics)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *CLASS_LABELS])
        for label, values in zip(CLASS_LABELS, test_metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    print(json.dumps({"output_dir": str(output_dir), **test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
