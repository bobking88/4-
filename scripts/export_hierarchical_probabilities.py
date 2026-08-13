from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from mineral_hierarchy import aggregate_role_probabilities, validate_species_role_mapping
from train_hierarchical_mineral_classifier import (
    HierarchicalMineralImageDataset,
    HierarchicalRoleAwareEfficientNet,
)
from train_mineral_classifier import (
    CLASS_LABELS,
    configure_torch_home,
    create_transforms,
    load_manifest_records,
    require_training_dependencies,
    resolve_device_name,
    split_records,
)


def load_audit_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise ValueError("Audit manifest must contain image_id.")
        return {row["image_id"]: dict(row) for row in reader}


def build_probability_rows(records, role_logits, species_logits, mapping, metadata, torch):
    if len(records) != role_logits.size(0) or len(records) != species_logits.size(0):
        raise ValueError("Record and logit lengths must match.")
    if role_logits.ndim != 2 or role_logits.size(1) != len(CLASS_LABELS):
        raise ValueError("Role logits must match the four-role label mapping.")
    if species_logits.ndim != 2 or species_logits.size(1) != len(mapping.species_labels):
        raise ValueError("Species logits must match the frozen species mapping.")

    role_probabilities = torch.softmax(role_logits, dim=1)
    species_probabilities = torch.softmax(species_logits, dim=1)
    mapped_role_probabilities = aggregate_role_probabilities(species_probabilities, mapping, torch)
    role_predictions = role_probabilities.argmax(dim=1)
    species_predictions = species_probabilities.argmax(dim=1)

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        source = metadata.get(record.image_id, {})
        row: dict[str, object] = {
            "image_id": record.image_id,
            "mindat_photo_id": record.mindat_photo_id,
            "mineral_label": record.mineral_label,
            "four_class_label": record.four_class_label,
            "four_class_id": record.class_id,
            "species_id": mapping.species_to_index[record.mineral_label],
            "split_group_id": record.split_group_id,
            "split": getattr(record, "split", ""),
            "image_path": str(record.image_path),
            "locality": source.get("locality", ""),
            "photographer_or_credit": source.get("photographer_or_credit", ""),
            "source_site": source.get("source_site", ""),
            "detail_page_url": source.get("detail_page_url", ""),
            "predicted_role": CLASS_LABELS[int(role_predictions[index])],
            "predicted_species": mapping.species_labels[int(species_predictions[index])],
            "role_confidence": float(role_probabilities[index].max()),
            "species_confidence": float(species_probabilities[index].max()),
        }
        for class_id, label in enumerate(CLASS_LABELS):
            row[f"role_logit_{label}"] = float(role_logits[index, class_id])
            row[f"role_probability_{label}"] = float(role_probabilities[index, class_id])
            row[f"mapped_role_probability_{label}"] = float(
                mapped_role_probabilities[index, class_id]
            )
        for species_id, label in enumerate(mapping.species_labels):
            row[f"species_logit_{label}"] = float(species_logits[index, species_id])
            row[f"species_probability_{label}"] = float(
                species_probabilities[index, species_id]
            )
        rows.append(row)
    return rows


def infer_logits(model, loader, device, torch):
    role_batches = []
    species_batches = []
    model.eval()
    with torch.no_grad():
        for images, _, _ in loader:
            role_logits, species_logits, _, _ = model(images.to(device))
            role_batches.append(role_logits.cpu())
            species_batches.append(species_logits.cpu())
    return torch.cat(role_batches), torch.cat(species_batches)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty probability export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hierarchical model posterior probabilities.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument(
        "--torch-home", type=Path, default=Path(__file__).resolve().parents[1] / ".torch-cache"
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

    records = load_manifest_records(args.manifest, args.dataset_root)
    records_by_split = split_records(records)
    mapping = validate_species_role_mapping(records)
    metadata = load_audit_metadata(args.audit_manifest)
    _, evaluation_transform = create_transforms(args.image_size, dependencies["transforms"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if tuple(checkpoint.get("class_labels", ())) != tuple(CLASS_LABELS):
        raise ValueError("Checkpoint role labels do not match the current mapping.")
    if tuple(checkpoint.get("species_labels", ())) != tuple(mapping.species_labels):
        raise ValueError("Checkpoint species labels do not match the current manifest.")

    model = HierarchicalRoleAwareEfficientNet(
        dependencies["models"],
        len(CLASS_LABELS),
        len(mapping.species_labels),
        pretrained=False,
        embedding_dim=args.embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "audit_manifest": str(args.audit_manifest.resolve()) if args.audit_manifest else None,
        "device": str(device),
        "class_labels": list(CLASS_LABELS),
        "species_labels": list(mapping.species_labels),
        "splits": {},
    }
    for split in ("val", "test"):
        split_records_list = records_by_split[split]
        dataset = HierarchicalMineralImageDataset(split_records_list, mapping, evaluation_transform)
        loader = dependencies["DataLoader"](
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        role_logits, species_logits = infer_logits(model, loader, device, torch)
        rows = build_probability_rows(
            split_records_list, role_logits, species_logits, mapping, metadata, torch
        )
        output_path = args.output_dir / f"{split}_hierarchical_probabilities.csv"
        write_csv(output_path, rows)
        summary["splits"][split] = {"rows": len(rows), "output": str(output_path.resolve())}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "probability_export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
