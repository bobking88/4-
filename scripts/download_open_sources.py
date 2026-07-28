"""Download supplementary mineral images with provenance metadata.

Sources:
- Wikimedia Commons category API: original image URL + file-page metadata.
- RRUFF sample pages: optional sample-page records; image extraction is conservative.

This script deliberately labels all downloaded images as keep_prelim. Final visual
screening remains a human step because category membership is not a guarantee that
the specimen occupies most of the frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ROOT = Path(r"D:\成信工科研\人工智能选矿\数据集\open_source_supplement_v1")
UA = "VTM-research-dataset/1.0 (research provenance downloader)"


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, path: Path) -> bytes:
    req = Request(url, headers={"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    path.write_bytes(data)
    return data


def ext(meta: dict, key: str) -> str:
    value = meta.get(key, {})
    if isinstance(value, dict):
        return str(value.get("value", ""))
    return str(value or "")


def safe_name(value: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    return re.sub(r"\s+", " ", value).strip(" .")[:180] or "unnamed"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def thumbnail_url(original_url: str, width: int = 500) -> str:
    """Derive a Wikimedia thumbnail URL without requesting the original file."""
    marker = "/commons/"
    if marker not in original_url:
        return original_url
    prefix, rest = original_url.split(marker, 1)
    filename = rest.rsplit("/", 1)[-1]
    return f"{prefix}/commons/thumb/{rest}/{width}px-{filename}"


def commons_category(category: str, label: str, limit: int, root: Path) -> list[dict]:
    out_dir = root / "images" / "commons" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cont = ""
    while len(rows) < limit:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "1600",
            "format": "json",
            "origin": "*",
        }
        if cont:
            params["gcmcontinue"] = cont
        api_url = "https://commons.wikimedia.org/w/api.php?" + urlencode(params)
        payload = fetch_json(api_url)
        pages = list((payload.get("query", {}).get("pages", {}) or {}).values())
        if not pages:
            break
        for page in pages:
            if len(rows) >= limit:
                break
            info = (page.get("imageinfo") or [{}])[0]
            title = page.get("title", "")
            name = title.removeprefix("File:")
            mime = info.get("mime", "")
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            # Exclude diagrams, charts, maps and microscopy from the visual set;
            # keep them in the log as rejected_reference for auditability.
            text = name.lower()
            reject_hint = any(k in text for k in (
                "diagram", "structure", "process", "production", "patent",
                "microsection", "thin section", "polarized", "ultraviolet",
                "mining equipment", "railroad", "wagon", "tailings",
            ))
            original_url = info.get("url", "")
            # Always use the bounded thumbnail for acquisition. The original
            # file URL is retained separately for provenance and citation.
            download_url = thumbnail_url(original_url)
            file_page = "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_"), safe="/:()")
            decision = "rejected_reference" if reject_hint or width < 300 or height < 300 else "keep_prelim"
            local_name = f"commons_{label}_{page.get('pageid','unknown')}_{safe_name(name)}"
            suffix = Path(name).suffix.lower() or ".jpg"
            local_path = out_dir / (local_name + suffix)
            digest = ""
            if decision == "keep_prelim" and download_url:
                try:
                    data = download(download_url, local_path)
                    digest = sha256(data)
                except Exception as exc:
                    decision = "download_failed"
                    local_path = Path("")
                    digest = str(exc)
            rows.append({
                "source": "Wikimedia Commons",
                "source_filename": name,
                "mineral_label": label,
                "source_record_id": str(page.get("pageid", "")),
                "detail_page_url": file_page,
                "download_source_url": download_url,
                "original_file_url": original_url,
                "page_title": title,
                "author_or_credit": ext(info.get("extmetadata", {}), "Artist"),
                "license_or_rights": ext(info.get("extmetadata", {}), "LicenseShortName"),
                "license_url": ext(info.get("extmetadata", {}), "LicenseUrl"),
                "width": width,
                "height": height,
                "mime": mime,
                "screening_decision": decision,
                "local_path": str(local_path) if local_path else "",
                "sha256": digest,
                "notes": "Category membership is source evidence; visual suitability requires manual review.",
            })
        cont = (payload.get("continue") or {}).get("gcmcontinue", "")
        if not cont:
            break
        time.sleep(0.3)
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["source", "mineral_label"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--limit-per-category", type=int, default=100)
    args = ap.parse_args()
    root = args.root
    rows: list[dict] = []
    rows += commons_category("Magnetite", "magnetite_proxy", args.limit_per_category, root)
    rows += commons_category("Ilmenite", "ilmenite_ti_mineral", args.limit_per_category, root)
    write_rows(rows, root / "metadata" / "open_source_images_manifest.csv")
    print(json.dumps({
        "root": str(root),
        "rows": len(rows),
        "downloaded": sum(bool(r.get("sha256")) and len(r["sha256"]) == 64 for r in rows),
        "keep_prelim": sum(r.get("screening_decision") == "keep_prelim" for r in rows),
        "rejected_reference": sum(r.get("screening_decision") == "rejected_reference" for r in rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
