import argparse
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name-contains", default="")
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--chars", type=int, default=900)
    args = parser.parse_args()

    root = Path("后下论文")
    for pdf_path in sorted(root.glob("*.pdf")):
        if args.name_contains.lower() not in pdf_path.name.lower():
            continue
        try:
            reader = PdfReader(str(pdf_path))
            first_pages = reader.pages[: args.pages]
            text = " ".join(
                " ".join((page.extract_text() or "").split())
                for page in first_pages
            )
            print(f"FILE: {pdf_path.name}")
            print(f"PAGES: {len(reader.pages)}")
            print(f"TEXT: {text[:args.chars]}")
        except Exception as exc:
            print(f"FILE: {pdf_path.name}")
            print(f"ERROR: {type(exc).__name__}: {exc}")
        print("---")


if __name__ == "__main__":
    main()
