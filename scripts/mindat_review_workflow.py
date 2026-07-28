#!/usr/bin/env python3
"""Build and apply manual review queues for Mindat mineral image datasets."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


REVIEW_FIELDS = [
    "review_id",
    "review_status",
    "review_decision",
    "review_notes",
    "dataset_id",
    "dataset_class",
    "mineral_label",
    "local_path",
    "page_title",
    "locality",
    "mindat_photo_id",
    "detail_page_url",
    "original_width",
    "original_height",
    "sha256",
    "screening_decision",
    "notes",
]

DECISION_TO_FOLDER = {
    "keep_as_magnetite": "selected_positive/magnetite_proxy",
    "exclude_mixed": "rejected_or_reference",
    "needs_expert": "mixed_uncertain",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fit_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return canvas


def make_contact_sheets(root: Path, rows: List[Dict[str, str]], out_dir: Path, per_sheet: int = 30) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tile = 180
    label_h = 56
    cols = 5
    rows_per_sheet = (per_sheet + cols - 1) // cols
    font = safe_font(13)
    small_font = safe_font(11)

    for sheet_index in range(0, len(rows), per_sheet):
        chunk = rows[sheet_index : sheet_index + per_sheet]
        sheet_no = sheet_index // per_sheet + 1
        sheet = Image.new("RGB", (cols * tile, rows_per_sheet * (tile + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(chunk):
            image_path = root / row["local_path"]
            col = idx % cols
            row_no = idx // cols
            x = col * tile
            y = row_no * (tile + label_h)
            try:
                thumb = fit_image(image_path, tile)
            except (UnidentifiedImageError, OSError):
                thumb = Image.new("RGB", (tile, tile), "#eeeeee")
            sheet.paste(thumb, (x, y))
            title = (row.get("page_title") or "")[:24]
            photo_id = row.get("mindat_photo_id") or ""
            draw.text((x + 6, y + tile + 4), row["review_id"], fill="black", font=font)
            draw.text((x + 6, y + tile + 22), f"id:{photo_id} {title}", fill="#333333", font=small_font)
        sheet.save(out_dir / f"mixed_uncertain_contact_sheet_{sheet_no:03d}.jpg", quality=92)


def build_review_queue(root: Path, manifest_name: str, output_name: str) -> Path:
    rows = read_rows(root / "metadata" / manifest_name)
    queue = []
    for row in rows:
        if row.get("screening_decision") != "mixed_uncertain_review":
            continue
        item = {field: row.get(field, "") for field in REVIEW_FIELDS}
        item["review_id"] = f"MR{len(queue) + 1:04d}"
        item["review_status"] = "pending"
        item["review_decision"] = ""
        item["review_notes"] = ""
        queue.append(item)

    review_dir = root / "metadata" / "review_mixed_uncertain"
    out = review_dir / output_name
    write_csv(out, REVIEW_FIELDS, queue)
    make_contact_sheets(root, queue, review_dir / "contact_sheets")
    return out


def apply_review_decisions(root: Path, review_csv: Path, output_name: str) -> Path:
    rows = read_rows(review_csv)
    applied = []
    for row in rows:
        decision = (row.get("review_decision") or "").strip()
        if decision not in DECISION_TO_FOLDER:
            continue
        src = root / row["local_path"]
        if not src.exists():
            row["review_notes"] = (row.get("review_notes", "") + "; source file missing").strip("; ")
            applied.append(row)
            continue
        target_dir = root / DECISION_TO_FOLDER[decision]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / src.name
        if not target.exists():
            shutil.copy2(src, target)
        row["review_status"] = "applied"
        row["review_notes"] = (row.get("review_notes", "") + f"; copied_to={target.relative_to(root).as_posix()}").strip("; ")
        applied.append(row)

    out = root / "metadata" / "review_mixed_uncertain" / output_name
    write_csv(out, REVIEW_FIELDS, applied)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mindat review workflow")
    parser.add_argument("--root", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-mixed-queue")
    build.add_argument("--manifest-name", default="mindat_manual_manifest.csv")
    build.add_argument("--output-name", default="mixed_uncertain_review_queue.csv")

    apply = sub.add_parser("apply-decisions")
    apply.add_argument("--review-csv", required=True)
    apply.add_argument("--output-name", default="mixed_uncertain_review_applied.csv")

    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "build-mixed-queue":
        out = build_review_queue(root, args.manifest_name, args.output_name)
        print(f"Review queue ready: {out}")
    elif args.command == "apply-decisions":
        out = apply_review_decisions(root, Path(args.review_csv).resolve(), args.output_name)
        print(f"Applied review decisions: {out}")


if __name__ == "__main__":
    main()
