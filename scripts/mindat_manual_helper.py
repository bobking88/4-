#!/usr/bin/env python3
"""
Mindat manual-download helper.

This helper does not bypass website protections. It organizes images that a
researcher downloads manually, normalizes filenames, computes image metadata,
and writes a traceable manifest for reports and papers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_ROOT = Path("数据集/mindat_manual_positive_v1")

CLASS_INFO = {
    "magnetite": {
        "dataset_class": "magnetite_proxy",
        "folder": "raw_positive/magnetite_proxy",
        "prefix": "magnetite",
        "label_cn": "磁铁矿",
    },
    "magnetite_proxy": {
        "dataset_class": "magnetite_proxy",
        "folder": "raw_positive/magnetite_proxy",
        "prefix": "magnetite",
        "label_cn": "磁铁矿",
    },
    "ilmenite": {
        "dataset_class": "ilmenite_ti_mineral",
        "folder": "raw_positive/ilmenite_ti_mineral",
        "prefix": "ilmenite",
        "label_cn": "钛铁矿",
    },
    "ilmenite_ti_mineral": {
        "dataset_class": "ilmenite_ti_mineral",
        "folder": "raw_positive/ilmenite_ti_mineral",
        "prefix": "ilmenite",
        "label_cn": "钛铁矿",
    },
    "titanomagnetite": {
        "dataset_class": "titanomagnetite_core",
        "folder": "raw_positive/titanomagnetite_core",
        "prefix": "titanomagnetite",
        "label_cn": "钛磁铁矿",
    },
    "titanomagnetite_core": {
        "dataset_class": "titanomagnetite_core",
        "folder": "raw_positive/titanomagnetite_core",
        "prefix": "titanomagnetite",
        "label_cn": "钛磁铁矿",
    },
    "perovskite": {
        "dataset_class": "perovskite_ti_bearing_negative",
        "folder": "raw_negative/ti_bearing_negative/perovskite",
        "prefix": "perovskite",
        "label_cn": "Perovskite",
    },
    "rutile": {
        "dataset_class": "rutile_ti_bearing_negative",
        "folder": "raw_negative/ti_bearing_negative/rutile",
        "prefix": "rutile",
        "label_cn": "Rutile",
    },
    "anatase": {
        "dataset_class": "anatase_ti_bearing_negative",
        "folder": "raw_negative/ti_bearing_negative/anatase",
        "prefix": "anatase",
        "label_cn": "Anatase",
    },
    "mixed_uncertain": {
        "dataset_class": "mixed_uncertain",
        "folder": "mixed_uncertain",
        "prefix": "mixed_uncertain",
        "label_cn": "混合不确定",
    },
    "rejected_or_reference": {
        "dataset_class": "rejected_or_reference",
        "folder": "rejected_or_reference",
        "prefix": "reference",
        "label_cn": "剔除或参考",
    },
}

LOG_FIELDS = [
    "source_filename",
    "mineral_label",
    "mindat_photo_id",
    "detail_page_url",
    "download_source_url",
    "windows_referrer_url",
    "windows_host_url",
    "page_title",
    "locality",
    "photographer_or_credit",
    "license_or_rights",
    "screening_decision",
    "notes",
]

MANIFEST_FIELDS = [
    "dataset_id",
    "class_group",
    "dataset_class",
    "mineral_label",
    "mineral_label_cn",
    "local_path",
    "source_site",
    "source_type",
    "detail_page_url",
    "download_source_url",
    "windows_referrer_url",
    "windows_host_url",
    "mindat_photo_id",
    "page_title",
    "locality",
    "photographer_or_credit",
    "license_or_rights",
    "original_width",
    "original_height",
    "file_size_bytes",
    "sha256",
    "resolution_pass",
    "screening_decision",
    "manual_review_required",
    "download_or_register_time",
    "filename_rule",
    "notes",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
PHOTO_ID_RE = re.compile(r"photo-(\d+)\.html", re.I)
FILENAME_PHOTO_ID_RE = re.compile(r"_p(\d{3,})_", re.I)
MINDAT_PHOTO_URL_RE = re.compile(r"https?://(?:www\.)?mindat\.org/photo-(\d+)\.html", re.I)


def norm_label(value: str) -> str:
    key = (value or "").strip().lower().replace(" ", "_")
    if key not in CLASS_INFO:
        raise ValueError(f"Unknown mineral_label: {value!r}. Use magnetite, ilmenite, titanomagnetite, mixed_uncertain, or rejected_or_reference.")
    return key


def extract_photo_id(url: str, explicit_id: str = "") -> str:
    explicit_id = re.sub(r"\D", "", explicit_id or "")
    if explicit_id:
        return explicit_id
    match = PHOTO_ID_RE.search(url or "")
    return match.group(1) if match else ""


def extract_mindat_photo_url(text: str) -> str:
    match = MINDAT_PHOTO_URL_RE.search(text or "")
    return match.group(0) if match else ""


def photo_id_from_filename(name: str) -> str:
    match = FILENAME_PHOTO_ID_RE.search(name)
    return match.group(1) if match else ""


def read_zone_identifier(path: Path) -> Dict[str, str]:
    """Read Windows Mark-of-the-Web alternate data stream when available."""
    stream_path = f"{path}:Zone.Identifier"
    try:
        text = Path(stream_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    data: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def infer_label_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    joined = " ".join(parts + [name])

    if "titanomagnetite_core" in parts or re.search(r"titanomagnetite|titanomagnetit|titanium[-_ ]bearing[-_ ]magnetite", joined):
        return "titanomagnetite"
    if "ilmenite_ti_mineral" in parts or re.search(r"ilmenite|ilmenit|ilmenita|ilménite", joined):
        return "ilmenite"
    if "magnetite_proxy" in parts or re.search(r"magnetite|magnetit|lodestone", joined):
        return "magnetite"
    if "rutile" in joined or "rutil" in joined:
        return "rutile"
    if "anatase" in joined or "anatas" in joined:
        return "anatase"
    if "mixed_uncertain" in parts:
        return "mixed_uncertain"
    if "rejected_or_reference" in parts:
        return "rejected_or_reference"
    return ""


def default_page_title(label: str, photo_id: str) -> str:
    titles = {
        "magnetite": "Magnetite photo from Mindat",
        "ilmenite": "Ilmenite photo from Mindat",
        "titanomagnetite": "Titanomagnetite photo from Mindat",
        "rutile": "Rutile photo from Mindat",
        "anatase": "Anatase photo from Mindat",
        "mixed_uncertain": "Mixed or uncertain Mindat photo",
        "rejected_or_reference": "Rejected or reference Mindat photo",
    }
    title = titles.get(label, "Mindat photo")
    return f"{title} ({padded_photo_id(photo_id)})" if photo_id else title


def padded_photo_id(photo_id: str) -> str:
    photo_id = re.sub(r"\D", "", photo_id or "")
    return f"p{int(photo_id):06d}" if photo_id else "manual"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_info(path: Path) -> Dict[str, object]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        width, height = img.size
    return {
        "original_width": width,
        "original_height": height,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "resolution_pass": "yes" if width >= 300 and height >= 300 else "no",
    }


def ensure_dirs(root: Path) -> None:
    for info in CLASS_INFO.values():
        (root / info["folder"]).mkdir(parents=True, exist_ok=True)
    (root / "incoming_downloads").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "selected_positive/magnetite_proxy").mkdir(parents=True, exist_ok=True)
    (root / "selected_positive/ilmenite_ti_mineral").mkdir(parents=True, exist_ok=True)
    (root / "selected_positive/titanomagnetite_core").mkdir(parents=True, exist_ok=True)


def write_template(root: Path, force: bool = False) -> Path:
    ensure_dirs(root)
    path = root / "metadata" / "mindat_manual_download_log.csv"
    if path.exists() and not force:
        return path
    rows = [
        {
            "source_filename": "example_downloaded_image.jpg",
            "mineral_label": "magnetite",
            "mindat_photo_id": "58476",
            "detail_page_url": "https://www.mindat.org/photo-58476.html",
            "download_source_url": "",
            "windows_referrer_url": "",
            "windows_host_url": "",
            "page_title": "Magnetite example title",
            "locality": "copy locality from Mindat page",
            "photographer_or_credit": "copy photographer/uploader/copyright line",
            "license_or_rights": "copy rights/license note from Mindat page",
            "screening_decision": "keep_prelim",
            "notes": "delete this example row before real use",
        },
        {
            "source_filename": "example_uncertain.jpg",
            "mineral_label": "mixed_uncertain",
            "mindat_photo_id": "",
            "detail_page_url": "",
            "download_source_url": "",
            "windows_referrer_url": "",
            "windows_host_url": "",
            "page_title": "Perovskite, Magnetite",
            "locality": "",
            "photographer_or_credit": "",
            "license_or_rights": "",
            "screening_decision": "mixed_uncertain",
            "notes": "title contains multiple minerals; do not use as positive until reviewed",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def auto_log(root: Path, output_name: str = "mindat_manual_download_log_auto.csv") -> Path:
    ensure_dirs(root)
    scan_roots = [
        root / "incoming_downloads",
        root / "incoming_downloads" / "magnetite",
        root / "incoming_downloads" / "ilmenite",
        root / "incoming_downloads" / "titanomagnetite",
        root / "incoming_downloads" / "perovskite",
        root / "incoming_downloads" / "rutile",
        root / "incoming_downloads" / "anatase",
        root / "raw_positive" / "magnetite_proxy",
        root / "raw_positive" / "ilmenite_ti_mineral",
        root / "raw_positive" / "titanomagnetite_core",
        root / "raw_negative" / "ti_bearing_negative" / "perovskite",
        root / "raw_negative" / "ti_bearing_negative" / "rutile",
        root / "raw_negative" / "ti_bearing_negative" / "anatase",
        root / "mixed_uncertain",
        root / "rejected_or_reference",
    ]

    files: List[Path] = []
    seen_paths = set()
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            files.append(path)

    rows: List[Dict[str, str]] = []
    for path in files:
        zone = read_zone_identifier(path)
        zone_text = "\n".join(f"{k}={v}" for k, v in zone.items())
        detail_url = extract_mindat_photo_url(zone.get("ReferrerUrl", ""))
        if not detail_url:
            detail_url = extract_mindat_photo_url(zone.get("HostUrl", ""))
        if not detail_url:
            detail_url = extract_mindat_photo_url(zone_text)
        host_url = zone.get("HostUrl", "")
        referrer_url = zone.get("ReferrerUrl", "")
        download_source_url = host_url or referrer_url

        photo_id = extract_photo_id(detail_url)
        if not photo_id:
            photo_id = photo_id_from_filename(path.name)
        if not detail_url and photo_id:
            detail_url = f"https://www.mindat.org/photo-{int(photo_id)}.html"

        label = infer_label_from_path(path)
        screening_decision = "keep_prelim" if label in {"magnetite", "ilmenite", "titanomagnetite", "perovskite", "rutile", "anatase"} else (label or "needs_review")
        notes = []
        if not label:
            label = "magnetite"
            screening_decision = "needs_review"
            notes.append("mineral_label not inferred; change this row manually")
        if not photo_id:
            notes.append("mindat_photo_id not found; fill from Mindat photo page URL if available")
        if not detail_url:
            notes.append("detail_page_url not found; browser did not expose source URL")

        try:
            rel_source = path.relative_to(root).as_posix()
        except ValueError:
            rel_source = str(path)

        rows.append(
            {
                "source_filename": rel_source,
                "mineral_label": label,
                "mindat_photo_id": photo_id,
                "detail_page_url": detail_url,
                "download_source_url": download_source_url,
                "windows_referrer_url": referrer_url,
                "windows_host_url": host_url,
                "page_title": default_page_title(label, photo_id),
                "locality": "",
                "photographer_or_credit": "",
                "license_or_rights": "Mindat photo page rights; verify per photo before republication",
                "screening_decision": screening_decision,
                "notes": "; ".join(notes),
            }
        )

    out = root / "metadata" / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        out = unique_path(out)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return out


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 10000):
        candidate = path.with_name(f"{stem}_dup{idx:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many duplicate filenames for {path}")


def organize_from_log(root: Path, log_path: Path, move: bool = False) -> List[Dict[str, str]]:
    ensure_dirs(root)
    incoming = root / "incoming_downloads"
    rows = list(csv.DictReader(log_path.open(encoding="utf-8-sig")))
    manifest_rows: List[Dict[str, str]] = []
    class_counters: Dict[str, int] = {}

    for row in rows:
        source_filename = (row.get("source_filename") or "").strip()
        if not source_filename or source_filename.lower().startswith("example_"):
            continue
        label_key = norm_label(row.get("mineral_label", ""))
        info = CLASS_INFO[label_key]
        source_path = Path(source_filename)
        if not source_path.is_absolute():
            root_relative = root / source_filename
            source_path = root_relative if root_relative.exists() else incoming / source_filename
        if not source_path.exists():
            print(f"Missing file, skipped: {source_path}")
            continue
        if source_path.suffix.lower() not in IMAGE_EXTS:
            print(f"Not an image, skipped: {source_path}")
            continue

        photo_id = extract_photo_id(row.get("detail_page_url", ""), row.get("mindat_photo_id", ""))
        class_counters[info["dataset_class"]] = class_counters.get(info["dataset_class"], 0) + 1
        seq = class_counters[info["dataset_class"]]
        suffix = ".jpg" if source_path.suffix.lower() == ".jpeg" else source_path.suffix.lower()
        filename = f"{info['prefix']}_mindat_{padded_photo_id(photo_id)}_{seq:03d}_raw{suffix}"
        dest = root / info["folder"] / filename

        if not dest.exists():
            if move:
                shutil.move(str(source_path), str(dest))
            else:
                shutil.copy2(source_path, dest)

        try:
            metrics = image_info(dest)
        except (UnidentifiedImageError, OSError) as exc:
            print(f"Bad image, skipped metadata: {dest} ({exc})")
            continue

        manifest_rows.append(row_to_manifest(root, dest, row, info, photo_id, metrics))

    return manifest_rows


def row_to_manifest(
    root: Path,
    image_path: Path,
    log_row: Dict[str, str],
    info: Dict[str, str],
    photo_id: str,
    metrics: Dict[str, object],
) -> Dict[str, str]:
    class_group = "positive" if info["dataset_class"] not in {"mixed_uncertain", "rejected_or_reference"} else info["dataset_class"]
    dataset_id = f"{info['dataset_class']}_mindat_{padded_photo_id(photo_id)}_{metrics['sha256'][:10]}"
    return {
        "dataset_id": dataset_id,
        "class_group": class_group,
        "dataset_class": info["dataset_class"],
        "mineral_label": info["prefix"],
        "mineral_label_cn": info["label_cn"],
        "local_path": image_path.relative_to(root).as_posix(),
        "source_site": "Mindat",
        "source_type": "manual_download",
        "detail_page_url": log_row.get("detail_page_url", ""),
        "download_source_url": log_row.get("download_source_url", ""),
        "windows_referrer_url": log_row.get("windows_referrer_url", ""),
        "windows_host_url": log_row.get("windows_host_url", ""),
        "mindat_photo_id": photo_id,
        "page_title": log_row.get("page_title", ""),
        "locality": log_row.get("locality", ""),
        "photographer_or_credit": log_row.get("photographer_or_credit", ""),
        "license_or_rights": log_row.get("license_or_rights", ""),
        "original_width": str(metrics["original_width"]),
        "original_height": str(metrics["original_height"]),
        "file_size_bytes": str(metrics["file_size_bytes"]),
        "sha256": str(metrics["sha256"]),
        "resolution_pass": str(metrics["resolution_pass"]),
        "screening_decision": log_row.get("screening_decision", "keep_prelim"),
        "manual_review_required": "yes",
        "download_or_register_time": datetime.now().isoformat(timespec="seconds"),
        "filename_rule": "mineral_mindat_pPHOTOID_SEQ_raw.ext",
        "notes": log_row.get("notes", ""),
    }


def scan_existing(root: Path) -> List[Dict[str, str]]:
    ensure_dirs(root)
    rows: List[Dict[str, str]] = []
    for label_key, info in CLASS_INFO.items():
        # Avoid scanning duplicate aliases.
        if label_key != info["dataset_class"] and label_key not in {"magnetite", "ilmenite", "titanomagnetite", "perovskite", "rutile", "anatase", "mixed_uncertain", "rejected_or_reference"}:
            continue
        folder = root / info["folder"]
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                metrics = image_info(path)
            except (UnidentifiedImageError, OSError):
                continue
            photo_id = photo_id_from_filename(path.name)
            log_stub = {
                "detail_page_url": f"https://www.mindat.org/photo-{int(photo_id)}.html" if photo_id else "",
                "download_source_url": "",
                "windows_referrer_url": "",
                "windows_host_url": "",
                "page_title": "",
                "locality": "",
                "photographer_or_credit": "",
                "license_or_rights": "",
                "screening_decision": "keep_prelim" if info["dataset_class"] not in {"mixed_uncertain", "rejected_or_reference"} else info["dataset_class"],
                "notes": "scanned from existing file; fill missing Mindat fields manually if needed",
            }
            rows.append(row_to_manifest(root, path, log_stub, info, photo_id, metrics))
    return rows


def write_manifest(root: Path, rows: List[Dict[str, str]], filename: str) -> Path:
    out = root / "metadata" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Mindat manual download helper")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Mindat manual dataset root")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init", help="Create folders and CSV log template")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing log template")

    auto_p = sub.add_parser("auto-log", help="Scan downloaded files and auto-fill a manual download log")
    auto_p.add_argument("--output-name", default="mindat_manual_download_log_auto.csv", help="Output CSV filename under metadata")

    org_p = sub.add_parser("organize", help="Copy/move incoming images according to manual log")
    org_p.add_argument("--log", default="", help="CSV log path; defaults to metadata/mindat_manual_download_log.csv")
    org_p.add_argument("--move", action="store_true", help="Move files from incoming_downloads instead of copying")

    sub.add_parser("scan", help="Scan existing organized images and write manifest")

    args = parser.parse_args()
    root = Path(args.root).resolve()

    if args.command == "init":
        path = write_template(root, force=args.force)
        print(f"Template ready: {path}")
        print(f"Put downloaded images here: {root / 'incoming_downloads'}")
        return

    if args.command == "auto-log":
        path = auto_log(root, output_name=args.output_name)
        print(f"Auto log ready: {path}")
        return

    if args.command == "organize":
        log_path = Path(args.log).resolve() if args.log else root / "metadata" / "mindat_manual_download_log.csv"
        if not log_path.exists():
            raise SystemExit(f"Log file not found: {log_path}. Run init first.")
        rows = organize_from_log(root, log_path, move=args.move)
        out = write_manifest(root, rows, "mindat_manual_manifest.csv")
        print(f"Organized rows: {len(rows)}")
        print(f"Manifest: {out}")
        return

    if args.command == "scan":
        rows = scan_existing(root)
        out = write_manifest(root, rows, "mindat_manual_manifest_scanned.csv")
        print(f"Scanned rows: {len(rows)}")
        print(f"Manifest: {out}")
        return


if __name__ == "__main__":
    main()
