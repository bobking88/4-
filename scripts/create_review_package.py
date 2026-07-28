from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


def resolve_image_path(dataset_root: Path, relative_path: str) -> Path:
    return dataset_root.joinpath(*Path(relative_path).parts)


def stratified_sample(
    rows: Iterable[dict[str, str]],
    per_mineral: int,
    seed: int,
) -> list[dict[str, str]]:
    by_mineral: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_mineral[row["mineral_label"]].append(dict(row))

    selected: list[dict[str, str]] = []
    for mineral in sorted(by_mineral):
        candidates = by_mineral[mineral]
        if len(candidates) <= per_mineral:
            mineral_sample = sorted(candidates, key=lambda row: row["image_id"])
        else:
            rng = random.Random(f"{seed}:{mineral}")
            by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in candidates:
                by_split[row["split"]].append(row)
            mineral_sample = []
            for split in ("train", "val", "test"):
                split_rows = by_split.get(split, [])
                if split_rows and len(mineral_sample) < per_mineral:
                    mineral_sample.append(rng.choice(split_rows))
            selected_ids = {row["image_id"] for row in mineral_sample}
            remaining = [
                row for row in candidates if row["image_id"] not in selected_ids
            ]
            rng.shuffle(remaining)
            mineral_sample.extend(remaining[: per_mineral - len(mineral_sample)])
            mineral_sample.sort(key=lambda row: row["image_id"])
        selected.extend(mineral_sample)

    return selected


def _render_contact_sheet(
    rows: list[dict[str, str]],
    dataset_root: Path,
    output_path: Path,
    columns: int,
    tile_size: int,
) -> int:
    label_height = 48
    canvas_rows = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new(
        "RGB",
        (columns * tile_size, canvas_rows * (tile_size + label_height)),
        color="#FFFFFF",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    missing = 0

    for index, row in enumerate(rows):
        x = (index % columns) * tile_size
        y = (index // columns) * (tile_size + label_height)
        image_path = resolve_image_path(dataset_root, row["relative_path"])
        if image_path.exists():
            with Image.open(image_path) as source:
                thumbnail = ImageOps.contain(
                    source.convert("RGB"),
                    (tile_size - 8, tile_size - 8),
                )
            tile = Image.new("RGB", (tile_size, tile_size), color="#F0F3F4")
            tile.paste(
                thumbnail,
                ((tile_size - thumbnail.width) // 2, (tile_size - thumbnail.height) // 2),
            )
        else:
            tile = Image.new("RGB", (tile_size, tile_size), color="#FDECEC")
            missing += 1
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.text((8, 8), "MISSING IMAGE", fill="#9B1C1C", font=font)
        canvas.paste(tile, (x, y))
        draw.rectangle(
            (x, y, x + tile_size - 1, y + tile_size + label_height - 1),
            outline="#AEBBC1",
        )
        draw.text(
            (x + 5, y + tile_size + 4),
            f"{int(row['review_order']):03d}  {row['image_id']}  {row['split']}",
            fill="#173F4F",
            font=font,
        )
        filename = Path(row["relative_path"]).name
        if len(filename) > 31:
            filename = f"{filename[:28]}..."
        draw.text(
            (x + 5, y + tile_size + 22),
            filename,
            fill="#465A63",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return missing


def create_review_package(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    per_mineral: int = 50,
    seed: int = 20260727,
    columns: int = 5,
    tile_size: int = 220,
) -> dict[str, object]:
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    sample = stratified_sample(manifest_rows, per_mineral=per_mineral, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "contact_sheets"
    by_mineral: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample:
        by_mineral[row["mineral_label"]].append(row)

    review_rows: list[dict[str, str]] = []
    missing_images = 0
    page_size = columns * 5
    for mineral in sorted(by_mineral):
        mineral_rows = sorted(by_mineral[mineral], key=lambda row: row["image_id"])
        for review_order, row in enumerate(mineral_rows, start=1):
            row["review_order"] = str(review_order)
        for page_index, start in enumerate(range(0, len(mineral_rows), page_size), start=1):
            page_rows = mineral_rows[start : start + page_size]
            contact_name = f"{mineral}_{page_index:03d}.jpg"
            missing_images += _render_contact_sheet(
                rows=page_rows,
                dataset_root=dataset_root,
                output_path=contact_dir / contact_name,
                columns=columns,
                tile_size=tile_size,
            )
            for tile_index, row in enumerate(page_rows, start=1):
                review_rows.append(
                    {
                        "review_order": row["review_order"],
                        "contact_sheet": contact_name,
                        "tile_index": str(tile_index),
                        **{key: value for key, value in row.items() if key != "review_order"},
                        "review_decision": "",
                        "review_reason": "",
                        "expert_note": "",
                        "reviewer": "",
                        "review_date": "",
                    }
                )

    review_fields = [
        "review_order",
        "contact_sheet",
        "tile_index",
        "image_id",
        "relative_path",
        "mineral_label",
        "four_class_label",
        "four_class_id",
        "mindat_photo_id",
        "split_group_id",
        "split",
        "review_decision",
        "review_reason",
        "expert_note",
        "reviewer",
        "review_date",
    ]
    review_path = output_dir / "review_queue.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_rows)

    counts = {
        mineral: len(rows) for mineral, rows in sorted(by_mineral.items())
    }
    summary: dict[str, object] = {
        "manifest_path": str(manifest_path),
        "dataset_root": str(dataset_root),
        "sample_count": len(review_rows),
        "per_mineral_limit": per_mineral,
        "seed": seed,
        "missing_images": missing_images,
        "sample_by_mineral": counts,
        "contact_sheet_count": len(list(contact_dir.glob("*.jpg"))),
    }
    (output_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "复核说明.md").write_text(
        "# 数据集人工抽检说明\n\n"
        "1. 打开 `contact_sheets` 中的图片，按 `review_order` 和 `image_id` 查看样本。\n"
        "2. 在 `review_queue.csv` 的 `review_decision` 列填写："
        "`keep`、`exclude` 或 `needs_expert`。\n"
        "3. 若填写 `exclude`，在 `review_reason` 中填写原因，例如："
        "`mixed_minerals`、`subject_too_small`、`blurred`、`too_dark`、"
        "`hand_or_scale`、`wrong_label`。\n"
        "4. `needs_expert` 用于矿物主体或标签无法确认的样本，并在 "
        "`expert_note` 中说明疑点。\n"
        "5. 不要修改 `image_id`、路径、类别和 split 列。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mineral image review package.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-mineral", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=220)
    args = parser.parse_args()
    summary = create_review_package(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        per_mineral=args.per_mineral,
        seed=args.seed,
        columns=args.columns,
        tile_size=args.tile_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
