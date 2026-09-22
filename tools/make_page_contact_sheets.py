"""Build four-page contact sheets for whole-document visual QA."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _page_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def build_sheets(page_dir: Path, output_dir: Path, pages_per_sheet: int = 4) -> list[Path]:
    pages = sorted(page_dir.glob("*.png"), key=_page_number)
    if not pages:
        raise FileNotFoundError(f"No PNG pages in {page_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    exports: list[Path] = []
    cell_width, cell_height, label_height, gap = 760, 1075, 28, 14
    for start in range(0, len(pages), pages_per_sheet):
        batch = pages[start:start + pages_per_sheet]
        sheet = Image.new("RGB", (cell_width * 2 + gap * 3, (cell_height + label_height) * 2 + gap * 3), "#D7DDE1")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(batch):
            row, column = divmod(index, 2)
            x = gap + column * (cell_width + gap)
            y = gap + row * (cell_height + label_height + gap)
            with Image.open(path) as source:
                page = ImageOps.contain(source.convert("RGB"), (cell_width, cell_height))
            paste_x = x + (cell_width - page.width) // 2
            paste_y = y + (cell_height - page.height) // 2
            sheet.paste(page, (paste_x, paste_y))
            label = f"Page {_page_number(path)}"
            draw.rectangle((x, y + cell_height, x + cell_width, y + cell_height + label_height), fill="white")
            draw.text((x + 8, y + cell_height + 7), label, font=font, fill="#1F303A")
        end = start + len(batch)
        destination = output_dir / f"pages_{start + 1:03d}_{end:03d}.jpg"
        sheet.save(destination, quality=88, optimize=True)
        exports.append(destination)
    return exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in build_sheets(args.page_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
