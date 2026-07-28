#!/usr/bin/env python3
"""
Build a traceable V-Ti related mineral image dataset.

The script downloads a conservative seed dataset from source pages with
metadata, applies objective first-pass filters, and writes a CSV manifest that
can be opened directly in Excel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

MINDAT_PHOTO_RE = re.compile(r"""href=["']([^"']*photo-\d+\.html[^"']*)["']""", re.I)
META_RE_TEMPLATE = r"""<meta[^>]+(?:property|name)=["']{key}["'][^>]+content=["']([^"']+)["']"""
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
IMG_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["'][^>]*>""", re.I)
COPYRIGHT_RE = re.compile(r"(?:copyright|©|&copy;|photographer|photo by|uploaded by)[^<]{0,160}", re.I)
PHOTO_ID_RE = re.compile(r"photo-(\d+)\.html", re.I)

KNOWN_MINERAL_TERMS = [
    "magnetite",
    "ilmenite",
    "titanomagnetite",
    "titanium-bearing magnetite",
    "perovskite",
    "titanite",
    "sphene",
    "apatite",
    "fluorapatite",
    "vesuvianite",
    "hematite",
    "rutile",
    "pyroxene",
    "olivine",
    "quartz",
    "calcite",
    "feldspar",
    "biotite",
    "hornblende",
]

MICROGRAPH_TERMS = [
    "sem",
    "bse",
    "tem",
    "thin section",
    "micrograph",
    "microscope",
    "polarized",
    "xrd",
    "eds",
    "diagram",
    "map",
]

SOURCES = [
    {
        "source_type": "mindat",
        "source_site": "Mindat",
        "mineral_label": "Magnetite",
        "dataset_class": "magnetite_proxy",
        "class_group": "positive",
        "source_url": "https://www.mindat.org/gm/2538",
        "max_keep": 20,
    },
    {
        "source_type": "mindat",
        "source_site": "Mindat",
        "mineral_label": "Ilmenite",
        "dataset_class": "ilmenite_ti_mineral",
        "class_group": "positive",
        "source_url": "https://www.mindat.org/gm/2013",
        "max_keep": 20,
    },
    {
        "source_type": "mindat",
        "source_site": "Mindat",
        "mineral_label": "Titanomagnetite",
        "dataset_class": "titanomagnetite_core",
        "class_group": "positive",
        "source_url": "https://www.mindat.org/min-10309.html",
        "max_keep": 20,
    },
    {
        "source_type": "commons_category",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Magnetite",
        "dataset_class": "magnetite_proxy",
        "class_group": "positive",
        "category": "Category:Magnetite",
        "source_url": "https://commons.wikimedia.org/wiki/Category:Magnetite",
        "max_keep": 80,
    },
    {
        "source_type": "commons_category",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Ilmenite",
        "dataset_class": "ilmenite_ti_mineral",
        "class_group": "positive",
        "category": "Category:Ilmenite",
        "source_url": "https://commons.wikimedia.org/wiki/Category:Ilmenite",
        "max_keep": 80,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Titanomagnetite",
        "dataset_class": "titanomagnetite_core",
        "class_group": "positive",
        "search": "Titanomagnetite",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=Titanomagnetite",
        "max_keep": 40,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Magnetite",
        "dataset_class": "magnetite_proxy",
        "class_group": "positive",
        "search": "Magnetite mineral specimen",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=Magnetite+mineral+specimen",
        "max_keep": 50,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Ilmenite",
        "dataset_class": "ilmenite_ti_mineral",
        "class_group": "positive",
        "search": "Ilmenite mineral specimen",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=Ilmenite+mineral+specimen",
        "max_keep": 50,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Ilmenite",
        "dataset_class": "ilmenite_ti_mineral",
        "class_group": "positive",
        "search": "Ilmenit mineral",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=Ilmenit+mineral",
        "max_keep": 50,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Titanomagnetite",
        "dataset_class": "titanomagnetite_core",
        "class_group": "positive",
        "search": "Titanomagnetit",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=Titanomagnetit",
        "max_keep": 40,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Titanomagnetite",
        "dataset_class": "titanomagnetite_core",
        "class_group": "positive",
        "search": "titaniferous magnetite",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=titaniferous+magnetite",
        "max_keep": 40,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Magnetite",
        "dataset_class": "magnetite_proxy",
        "class_group": "positive",
        "search": "Magnetit mineral",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=Magnetit+mineral",
        "max_keep": 50,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Magnetite",
        "dataset_class": "magnetite_proxy",
        "class_group": "positive",
        "search": "lodestone mineral",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=lodestone+mineral",
        "max_keep": 30,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Ilmenite",
        "dataset_class": "ilmenite_ti_mineral",
        "class_group": "positive",
        "search": "ilmenita mineral",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=ilmenita+mineral",
        "max_keep": 40,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Ilmenite",
        "dataset_class": "ilmenite_ti_mineral",
        "class_group": "positive",
        "search": "ilménite mineral",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=ilm%C3%A9nite+mineral",
        "max_keep": 40,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Titanomagnetite",
        "dataset_class": "titanomagnetite_core",
        "class_group": "positive",
        "search": "titanium bearing magnetite",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=titanium+bearing+magnetite",
        "max_keep": 40,
    },
    {
        "source_type": "commons_search",
        "source_site": "Wikimedia Commons",
        "mineral_label": "Titanomagnetite",
        "dataset_class": "titanomagnetite_core",
        "class_group": "positive",
        "search": "\"titanomagnetite\" mineral",
        "source_url": "https://commons.wikimedia.org/w/index.php?search=%22titanomagnetite%22+mineral",
        "max_keep": 40,
    },
]

CSV_FIELDS = [
    "dataset_id",
    "class_group",
    "dataset_class",
    "mineral_label",
    "label_role",
    "local_path",
    "source_site",
    "source_type",
    "source_url",
    "detail_page_url",
    "image_url",
    "source_original_image_url",
    "source_photo_id",
    "title",
    "description",
    "locality",
    "author_or_credit",
    "license_or_rights",
    "license_url",
    "original_width",
    "original_height",
    "file_size_bytes",
    "sha256",
    "download_time",
    "resolution_pass",
    "label_clear",
    "multiple_minerals_in_title",
    "micrograph_like",
    "filter_decision",
    "filter_reasons",
    "manual_review_required",
    "screening_rule_version",
    "notes",
]


def fetch(url: str, binary: bool = False, retries: int = 2, sleep_s: float = 0.4, timeout_s: int = 15):
    last_error = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            if sleep_s:
                time.sleep(sleep_s)
            return data if binary else data.decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.5 + attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = html.unescape(re.sub(r"<[^>]+>", " ", str(value)))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def meta_content(page: str, key: str) -> str:
    pattern = re.compile(META_RE_TEMPLATE.format(key=re.escape(key)), re.I | re.S)
    match = pattern.search(page)
    if match:
        return clean_text(match.group(1))
    return ""


def title_content(page: str) -> str:
    match = TITLE_RE.search(page)
    return clean_text(match.group(1)) if match else ""


def absolutize(url: str, base: str) -> str:
    return urllib.parse.urljoin(base, html.unescape(url))


def slug(value: str, max_len: int = 80) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", value).strip("_")
    return (value[:max_len].strip("_") or "item")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_photo_id_from_url(url: str) -> str:
    match = PHOTO_ID_RE.search(url)
    return match.group(1) if match else ""


def extract_mindat_photo_links(page: str, base_url: str) -> List[str]:
    links = []
    seen = set()
    for match in MINDAT_PHOTO_RE.finditer(page):
        link = absolutize(match.group(1), base_url)
        link = link.split("#", 1)[0]
        if link not in seen:
            seen.add(link)
            links.append(link)
    return links


def candidate_mindat_pages(source_url: str, max_pages: int) -> Iterable[str]:
    yield source_url
    if "/gm/" in source_url:
        for page_no in range(2, max_pages + 1):
            sep = "&" if "?" in source_url else "?"
            yield f"{source_url}{sep}page={page_no}"


def extract_mindat_image_url(photo_page: str, detail_url: str) -> str:
    for key in ("og:image", "twitter:image"):
        value = meta_content(photo_page, key)
        if value:
            return absolutize(value, detail_url)

    candidates = []
    for match in IMG_RE.finditer(photo_page):
        src = absolutize(match.group(1), detail_url)
        lower = src.lower()
        if any(token in lower for token in ["imagecache", "/photos/", "mindat"]):
            candidates.append(src)
    # Prefer imagecache/photos assets and avoid tiny icons/logos.
    for src in candidates:
        lower = src.lower()
        if not any(token in lower for token in ["logo", "icon", "avatar", "sprite"]):
            return src
    return candidates[0] if candidates else ""


def extract_rights_hint(page: str) -> str:
    fragments = [clean_text(m.group(0)) for m in COPYRIGHT_RE.finditer(page)]
    unique = []
    for item in fragments:
        if item and item not in unique:
            unique.append(item)
    return " | ".join(unique[:3])


def mineral_term_hits(text: str) -> List[str]:
    lower = text.lower()
    hits = []
    for term in KNOWN_MINERAL_TERMS:
        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lower):
            hits.append(term)
    return hits


def is_micrograph_like(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in MICROGRAPH_TERMS)


def is_title_mixed(title: str, target: str) -> bool:
    lower_title = title.lower()
    target_lower = target.lower()
    hits = mineral_term_hits(title)
    if len(set(hits)) >= 2:
        return True
    if "," in lower_title and target_lower in lower_title:
        # Mindat titles such as "Perovskite, Magnetite" are intentionally
        # kept outside the positive class unless a human reviewer confirms the
        # subject.
        return True
    return False


def extension_from_url(url: str, content_type: str = "") -> str:
    path = urllib.parse.urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"]:
        return ".jpg" if ext == ".jpeg" else ext
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    return ".jpg"


def save_image(
    image_data: bytes,
    out_dir: Path,
    filename_stem: str,
    source_url: str,
) -> Tuple[Path, int, int, int, str]:
    digest = sha256_bytes(image_data)
    ext = extension_from_url(source_url)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{filename_stem}_{digest[:10]}{ext}"
    path.write_bytes(image_data)
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
    except UnidentifiedImageError:
        path.unlink(missing_ok=True)
        raise
    return path, width, height, path.stat().st_size, digest


def evaluate_record(
    mineral_label: str,
    title: str,
    description: str,
    width: int,
    height: int,
    source_type: str,
) -> Tuple[bool, bool, bool, bool, str, str, str]:
    text = f"{title} {description}"
    resolution_pass = width >= 300 and height >= 300
    label_clear = mineral_label.lower() in text.lower() or source_type.startswith("commons")
    mixed = is_title_mixed(title, mineral_label)
    micro = is_micrograph_like(text)
    reasons = []
    manual_review = "yes"

    if not resolution_pass:
        decision = "reject_low_resolution"
        reasons.append("image resolution below 300x300")
        manual_review = "no"
    elif mixed:
        decision = "mixed_uncertain"
        reasons.append("title contains multiple mineral names or comma-separated mineral title")
    elif micro:
        decision = "review_micrograph_like"
        reasons.append("title/description suggests microscope or analytical image")
    elif not label_clear:
        decision = "review_label_unclear"
        reasons.append("mineral name not explicit in title/description")
    else:
        decision = "keep_prelim"
        reasons.append("resolution and source label passed; subject-area/blur check still needs visual review")

    return (
        resolution_pass,
        label_clear,
        mixed,
        micro,
        decision,
        "; ".join(reasons),
        manual_review,
    )


def commons_api(params: Dict[str, str]) -> Dict:
    base = "https://commons.wikimedia.org/w/api.php"
    params = {"format": "json", **params}
    url = f"{base}?{urllib.parse.urlencode(params)}"
    return json.loads(fetch(url, binary=False, sleep_s=0.4))


def iter_commons_category_files(category: str, limit: int) -> Iterable[Dict]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": "50",
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "1200",
    }
    yielded = 0
    cont: Dict[str, str] = {}
    while yielded < limit * 4:
        data = commons_api({**params, **cont})
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            yielded += 1
            yield page
            if yielded >= limit * 4:
                break
        if "continue" not in data:
            break
        cont = data["continue"]


def iter_commons_search_files(search: str, limit: int) -> Iterable[Dict]:
    fetched_titles = 0
    offset = 0
    while fetched_titles < limit * 4:
        data = commons_api(
            {
                "action": "query",
                "list": "search",
                "srnamespace": "6",
                "srlimit": "50",
                "sroffset": str(offset),
                "srsearch": search,
            }
        )
        batch = [item["title"] for item in data.get("query", {}).get("search", [])]
        if not batch:
            return
        fetched_titles += len(batch)
        for start in range(0, len(batch), 50):
            titles = batch[start : start + 50]
            data2 = commons_api(
                {
                    "action": "query",
                    "titles": "|".join(titles),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata",
                    "iiurlwidth": "1200",
                }
            )
            for page in data2.get("query", {}).get("pages", {}).values():
                yield page
        if "continue" not in data:
            break
        offset = int(data["continue"].get("sroffset", offset + 50))


def build_commons_record(page: Dict, source: Dict) -> Optional[Dict]:
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    info = infos[0]
    title = clean_text(page.get("title", ""))
    mime = info.get("mime", "")
    if not mime.startswith("image/"):
        return None
    lower_title = title.lower()
    if any(term in lower_title for term in MICROGRAPH_TERMS):
        # Keep possible micrographs in metadata only if downloaded by Mindat;
        # for Commons seed data we skip them before downloading.
        return None

    ext = (info.get("url") or "").lower()
    if ext.endswith(".svg") or ext.endswith(".gif"):
        return None

    meta = info.get("extmetadata") or {}

    def m(key: str) -> str:
        return clean_text((meta.get(key) or {}).get("value", ""))

    detail_url = f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'), safe=':/_')}"
    return {
        "source_url": source["source_url"],
        "detail_page_url": detail_url,
        "image_url": info.get("thumburl") or info.get("url", ""),
        "source_original_image_url": info.get("url", ""),
        "source_photo_id": str(page.get("pageid", "")),
        "title": title,
        "description": m("ImageDescription"),
        "locality": "",
        "author_or_credit": m("Artist") or m("Credit"),
        "license_or_rights": m("LicenseShortName") or m("UsageTerms"),
        "license_url": m("LicenseUrl"),
    }


def iter_mindat_photo_records(source: Dict, max_pages: int) -> Iterable[Dict]:
    photo_links: List[str] = []
    seen = set()
    for page_url in candidate_mindat_pages(source["source_url"], max_pages):
        try:
            page = fetch(page_url, sleep_s=0.5)
        except RuntimeError:
            continue
        for link in extract_mindat_photo_links(page, page_url):
            if link not in seen:
                seen.add(link)
                photo_links.append(link)

    if not photo_links and "/min-" in source["source_url"]:
        # Some mineral pages have related image links only in the text payload.
        try:
            page = fetch(source["source_url"], sleep_s=0.5)
            photo_links = extract_mindat_photo_links(page, source["source_url"])
        except RuntimeError:
            photo_links = []

    for detail_url in photo_links:
        try:
            page = fetch(detail_url, sleep_s=0.7)
        except RuntimeError:
            continue
        title = meta_content(page, "og:title") or title_content(page)
        description = meta_content(page, "og:description")
        image_url = extract_mindat_image_url(page, detail_url)
        if not image_url:
            continue
        yield {
            "source_url": source["source_url"],
            "detail_page_url": detail_url,
            "image_url": image_url,
            "source_original_image_url": image_url,
            "source_photo_id": source_photo_id_from_url(detail_url),
            "title": title,
            "description": description,
            "locality": "",
            "author_or_credit": extract_rights_hint(page),
            "license_or_rights": "Mindat photo page rights; verify per photo before republication",
            "license_url": detail_url,
        }


def collect_records(source: Dict, max_pages: int) -> Iterable[Dict]:
    if source["source_type"] == "mindat":
        yield from iter_mindat_photo_records(source, max_pages)
    elif source["source_type"] == "commons_category":
        for page in iter_commons_category_files(source["category"], source["max_keep"]):
            record = build_commons_record(page, source)
            if record:
                yield record
    elif source["source_type"] == "commons_search":
        for page in iter_commons_search_files(source["search"], source["max_keep"]):
            record = build_commons_record(page, source)
            if record:
                yield record


def download_source(source: Dict, output_root: Path, max_pages: int) -> List[Dict]:
    records = []
    kept_count = 0
    source_seen_urls = set()
    class_dir = output_root / source["class_group"] / source["dataset_class"]
    mixed_dir = output_root / "mixed_uncertain"
    reject_dir = output_root / "rejected"

    for raw in collect_records(source, max_pages):
        if raw["image_url"] in source_seen_urls:
            continue
        source_seen_urls.add(raw["image_url"])
        if kept_count >= source["max_keep"]:
            break

        title = raw.get("title", "")
        description = raw.get("description", "")

        try:
            image_data = fetch(raw["image_url"], binary=True, retries=1, sleep_s=0.15, timeout_s=12)
        except RuntimeError:
            continue

        # Save initially to class folder, then move-like save destination is
        # chosen after dimensions and filters are known.
        try:
            tmp_path, width, height, file_size, digest = save_image(
                image_data,
                output_root / "_tmp_downloads",
                f"{source['dataset_class']}_{source['source_site'].lower().replace(' ', '')}_{kept_count + 1:04d}",
                raw["image_url"],
            )
        except (UnidentifiedImageError, OSError):
            continue

        (
            resolution_pass,
            label_clear,
            mixed,
            micro,
            decision,
            filter_reasons,
            manual_review,
        ) = evaluate_record(
            source["mineral_label"],
            title,
            description,
            width,
            height,
            source["source_type"],
        )

        if decision == "mixed_uncertain":
            final_dir = mixed_dir
        elif decision.startswith("reject_"):
            final_dir = reject_dir
        else:
            final_dir = class_dir
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / tmp_path.name
        if final_path.exists():
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(final_path)

        dataset_id = (
            f"{source['dataset_class']}_"
            f"{source['source_site'].lower().replace(' ', '')}_"
            f"{raw.get('source_photo_id') or digest[:10]}"
        )
        rel_path = final_path.relative_to(output_root).as_posix()

        row = {
            "dataset_id": dataset_id,
            "class_group": source["class_group"],
            "dataset_class": source["dataset_class"],
            "mineral_label": source["mineral_label"],
            "label_role": "positive_subclass" if decision != "mixed_uncertain" else "uncertain_not_positive",
            "local_path": rel_path,
            "source_site": source["source_site"],
            "source_type": source["source_type"],
            "source_url": raw.get("source_url", ""),
            "detail_page_url": raw.get("detail_page_url", ""),
            "image_url": raw.get("image_url", ""),
            "source_original_image_url": raw.get("source_original_image_url", raw.get("image_url", "")),
            "source_photo_id": raw.get("source_photo_id", ""),
            "title": title,
            "description": description,
            "locality": raw.get("locality", ""),
            "author_or_credit": raw.get("author_or_credit", ""),
            "license_or_rights": raw.get("license_or_rights", ""),
            "license_url": raw.get("license_url", ""),
            "original_width": width,
            "original_height": height,
            "file_size_bytes": file_size,
            "sha256": digest,
            "download_time": datetime.now().isoformat(timespec="seconds"),
            "resolution_pass": "yes" if resolution_pass else "no",
            "label_clear": "yes" if label_clear else "no",
            "multiple_minerals_in_title": "yes" if mixed else "no",
            "micrograph_like": "yes" if micro else "no",
            "filter_decision": decision,
            "filter_reasons": filter_reasons,
            "manual_review_required": manual_review,
            "screening_rule_version": "vtm_image_screening_v1_2026-07-04",
            "notes": "First-pass automatic screening; verify mineral-subject area >50%, blur, exposure, and objects manually before final paper use.",
        }
        records.append(row)
        if decision in ("keep_prelim", "review_micrograph_like", "review_label_unclear"):
            kept_count += 1

    tmp_dir = output_root / "_tmp_downloads"
    if tmp_dir.exists():
        for child in tmp_dir.iterdir():
            child.unlink(missing_ok=True)
        tmp_dir.rmdir()

    return records


def write_csv(rows: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(output_root: Path, rows: List[Dict]) -> None:
    counts: Dict[str, int] = {}
    decisions: Dict[str, int] = {}
    for row in rows:
        counts[row["dataset_class"]] = counts.get(row["dataset_class"], 0) + 1
        decisions[row["filter_decision"]] = decisions.get(row["filter_decision"], 0) + 1
    lines = [
        "# V-Ti related mineral image seed dataset",
        "",
        "This dataset is a traceable seed set for project conclusion and paper experiments.",
        "It uses source-labelled mineral specimen images and records provenance for every file.",
        "",
        "Important wording: these images are V-Ti related useful mineral specimen images,",
        "not a verified industrial vanadium-titanium magnetite ore belt dataset.",
        "",
        "## Directory labels",
        "",
        "- positive/titanomagnetite_core: core positive subclass.",
        "- positive/magnetite_proxy: proxy positive subclass for Fe oxide target mineral.",
        "- positive/ilmenite_ti_mineral: Ti-bearing useful mineral positive subclass.",
        "- mixed_uncertain: source title suggests multiple minerals or unclear subject; do not use as positive until reviewed.",
        "- rejected: first-pass rejected images, mainly low resolution.",
        "",
        "## First-pass screening rules",
        "",
        "- Keep preliminarily: resolution >= 300x300, clear mineral/source label, no multi-mineral title.",
        "- Put into mixed_uncertain: title contains multiple mineral names, such as 'Perovskite, Magnetite'.",
        "- Review separately: possible microscope/analytical images.",
        "- Manual review still needed before final paper use: mineral subject >50%, exposure, blur, hand/coin/ruler dominance.",
        "",
        "## Counts",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")
    lines.append("")
    lines.append("## Filter decisions")
    lines.append("")
    for key in sorted(decisions):
        lines.append(f"- {key}: {decisions[key]}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("Open `metadata/vtm_mineral_image_manifest.csv` with Excel.")
    lines.append("Each row records source URL, detail page URL, image URL, source label, license/rights hint, dimensions, checksum, and screening decision.")
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="数据集/vtm_mineral_images_v1",
        help="Output dataset directory, relative to current workspace by default.",
    )
    parser.add_argument("--max-pages", type=int, default=3, help="Mindat gallery pages to try per mineral.")
    parser.add_argument("--source", choices=["all", "mindat", "commons"], default="all")
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=0,
        help="Optional cap for each configured source. 0 keeps the built-in source limits.",
    )
    parser.add_argument(
        "--only-class",
        choices=["all", "magnetite_proxy", "ilmenite_ti_mineral", "titanomagnetite_core"],
        default="all",
        help="Limit run to one dataset class.",
    )
    args = parser.parse_args()

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected = []
    for source in SOURCES:
        if args.source == "mindat" and source["source_site"] != "Mindat":
            continue
        if args.source == "commons" and source["source_site"] == "Mindat":
            continue
        if args.only_class != "all" and source["dataset_class"] != args.only_class:
            continue
        source_copy = dict(source)
        if args.per_source_limit > 0:
            source_copy["max_keep"] = min(source_copy["max_keep"], args.per_source_limit)
        selected.append(source_copy)

    all_rows: List[Dict] = []
    output_csv = output_root / "metadata" / "vtm_mineral_image_manifest.csv"
    for source in selected:
        print(
            f"Collecting {source['source_site']} - {source['mineral_label']} -> {source['dataset_class']}",
            flush=True,
        )
        rows = download_source(source, output_root, args.max_pages)
        print(f"  rows: {len(rows)}", flush=True)
        all_rows.extend(rows)
        write_csv(all_rows, output_csv)
        write_readme(output_root, all_rows)

    write_csv(all_rows, output_csv)
    write_readme(output_root, all_rows)
    print(f"Done. Images/metadata saved to: {output_root}")
    print(f"Manifest: {output_csv}")


if __name__ == "__main__":
    main()
