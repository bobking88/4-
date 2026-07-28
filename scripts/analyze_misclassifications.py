from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from train_mineral_classifier import (
    CLASS_LABELS,
    MineralImageDataset,
    build_model,
    configure_torch_home,
    create_transforms,
    load_manifest_records,
    require_training_dependencies,
    resolve_device_name,
    split_records,
)


PREDICTION_FIELDS = [
    "image_id",
    "mindat_photo_id",
    "mineral_label",
    "true_label",
    "predicted_label",
    "confidence",
    "is_correct",
    "error_pair",
    "split_group_id",
    "image_path",
]


def build_prediction_rows(records, prediction_ids, confidences) -> list[dict[str, str]]:
    if not (len(records) == len(prediction_ids) == len(confidences)):
        raise ValueError("Record, prediction, and confidence lengths must match.")
    rows: list[dict[str, str]] = []
    for record, prediction_id, confidence in zip(records, prediction_ids, confidences):
        if prediction_id < 0 or prediction_id >= len(CLASS_LABELS):
            raise ValueError(f"Invalid prediction id: {prediction_id}")
        predicted_label = CLASS_LABELS[prediction_id]
        is_correct = prediction_id == record.class_id
        rows.append(
            {
                "image_id": record.image_id,
                "mindat_photo_id": record.mindat_photo_id,
                "mineral_label": record.mineral_label,
                "true_label": record.four_class_label,
                "predicted_label": predicted_label,
                "confidence": f"{confidence:.6f}",
                "is_correct": str(is_correct).lower(),
                "error_pair": "" if is_correct else f"{record.four_class_label}__as__{predicted_label}",
                "split_group_id": record.split_group_id,
                "image_path": str(record.image_path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_error_pairs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    errors = [row for row in rows if row["is_correct"] == "false"]
    counts = Counter(row["error_pair"] for row in errors)
    total = len(rows)
    return [
        {
            "error_pair": pair,
            "count": str(count),
            "share_of_test": f"{count / total:.6f}",
        }
        for pair, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_error_contact_sheets(
    rows: list[dict[str, str]],
    output_dir: Path,
    max_per_pair: int,
) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["is_correct"] == "false":
            grouped[row["error_pair"]].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair, pair_rows in grouped.items():
        selected = sorted(pair_rows, key=lambda row: float(row["confidence"]), reverse=True)[:max_per_pair]
        columns = 4
        cell_width, cell_height = 260, 290
        rows_needed = (len(selected) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * cell_width, rows_needed * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(selected):
            image = Image.open(row["image_path"]).convert("RGB")
            image.thumbnail((240, 230))
            x = (index % columns) * cell_width + 10
            y = (index // columns) * cell_height + 10
            canvas.paste(image, (x, y))
            draw.text(
                (x, y + 235),
                f"{row['image_id']} | {row['mineral_label']}\n{row['true_label']} -> {row['predicted_label']}\nconf={row['confidence']}",
                fill="black",
            )
        canvas.save(output_dir / f"{pair}.jpg", quality=90)


def run_inference(model, loader, device, torch) -> tuple[list[int], list[float]]:
    model.eval()
    prediction_ids: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            prediction_ids.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return prediction_ids, confidences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export test-set error analysis for a trained mineral classifier.")
    parser.add_argument("--model", choices=("resnet50", "efficientnet_b0"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-per-pair", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".torch-cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_home(args.torch_home)
    dependencies = require_training_dependencies()
    torch = dependencies["torch"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")
    device = torch.device(resolve_device_name(args.device, torch.cuda.is_available()))
    records_by_split = split_records(load_manifest_records(args.manifest, args.dataset_root))
    test_records = records_by_split["test"]
    _, evaluation_transform = create_transforms(args.image_size, dependencies["transforms"])
    loader = dependencies["DataLoader"](
        MineralImageDataset(test_records, evaluation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(args.model, len(CLASS_LABELS), False, dependencies["models"], dependencies["nn"])
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    prediction_ids, confidences = run_inference(model, loader, device, torch)
    prediction_rows = build_prediction_rows(test_records, prediction_ids, confidences)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "test_predictions.csv", prediction_rows, PREDICTION_FIELDS)
    error_rows = [row for row in prediction_rows if row["is_correct"] == "false"]
    write_csv(args.output_dir / "misclassified_samples.csv", error_rows, PREDICTION_FIELDS)
    summary_rows = summarize_error_pairs(prediction_rows)
    write_csv(
        args.output_dir / "error_pair_summary.csv",
        summary_rows,
        ["error_pair", "count", "share_of_test"],
    )
    build_error_contact_sheets(
        prediction_rows,
        args.output_dir / "error_contact_sheets",
        args.max_per_pair,
    )
    print(
        f"Exported {len(prediction_rows)} test predictions and {len(error_rows)} misclassified samples "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
