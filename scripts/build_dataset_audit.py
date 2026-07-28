from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(r"D:\成信工科研\人工智能选矿")
DATA_ROOT = PROJECT_ROOT / "数据集"
DATASET_ROOT = DATA_ROOT / "mindat_manual_positive_v1"
METADATA_ROOT = DATASET_ROOT / "metadata"
OUTPUT_ROOT = DATA_ROOT / "dataset_audit"
SEED = 20260725
MIN_DIMENSION = 300
NEAR_DUPLICATE_DISTANCE = 3

CLASS_IDS = {
    "target_mineral": 0,
    "ti_bearing_negative": 1,
    "gangue_negative": 2,
    "metallic_hard_negative": 3,
}

MINERAL_ALIASES = {
    "ilmenite_ti_mineral": "ilmenite",
    "magnetite_proxy": "magnetite",
    "titanomagnetite_core": "titanomagnetite",
}

CORE_METADATA_FIELDS = ("mindat_photo_id", "detail_page_url", "page_title")


def normalize_mineral(value: str) -> str:
    key = str(value or "").strip().lower()
    return MINERAL_ALIASES.get(key, key)


def normalize_path(value: str) -> str:
    return str(value or "").strip().replace("\\", "/")


def metadata_score(row: dict[str, str]) -> int:
    preferred = (
        "mindat_photo_id",
        "detail_page_url",
        "download_source_url",
        "page_title",
        "locality",
        "photographer_or_credit",
        "license_or_rights",
        "screening_decision",
    )
    return sum(bool(str(row.get(field, "")).strip()) for field in preferred)


def load_metadata() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}

    for csv_path in sorted(METADATA_ROOT.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for source_row in csv.DictReader(handle):
                row = {key: str(value or "").strip() for key, value in source_row.items()}
                source_path = (
                    row.get("final_dataset_path")
                    or row.get("local_path")
                    or row.get("source_filename")
                    or ""
                )
                filename = Path(normalize_path(source_path)).name.lower()
                if not filename:
                    continue
                mineral = normalize_mineral(row.get("mineral_label", ""))
                row["_metadata_file"] = csv_path.name
                row["_metadata_score"] = str(metadata_score(row))

                pair_key = (mineral, filename)
                current = by_pair.get(pair_key)
                if current is None or metadata_score(row) > metadata_score(current):
                    by_pair[pair_key] = row

                current_any = by_name.get(filename)
                if current_any is None or metadata_score(row) > metadata_score(current_any):
                    by_name[filename] = row

    return by_pair, by_name


def discover_images() -> list[tuple[Path, str, str]]:
    discovered: list[tuple[Path, str, str]] = []

    positive_root = DATASET_ROOT / "raw_positive"
    for mineral_dir in sorted(path for path in positive_root.iterdir() if path.is_dir()):
        mineral = normalize_mineral(mineral_dir.name)
        for image_path in sorted(path for path in mineral_dir.iterdir() if path.is_file()):
            discovered.append((image_path, mineral, "target_mineral"))

    negative_root = DATASET_ROOT / "raw_negative"
    for class_dir in sorted(path for path in negative_root.iterdir() if path.is_dir()):
        class_label = class_dir.name
        if class_label not in CLASS_IDS:
            continue
        for mineral_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            mineral = normalize_mineral(mineral_dir.name)
            for image_path in sorted(path for path in mineral_dir.iterdir() if path.is_file()):
                discovered.append((image_path, mineral, class_label))

    return discovered


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(image: Image.Image) -> int:
    grayscale = ImageOps.exif_transpose(image).convert("L")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = grayscale.resize((9, 8), resampling)
    pixels = list(resized.getdata())
    result = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            result = (result << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return result


def parse_photo_id(filename: str) -> str:
    match = re.search(r"_p0*(\d+)", filename.lower())
    return match.group(1) if match else ""


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def profile_images(
    images: list[tuple[Path, str, str]],
    by_pair: dict[tuple[str, str], dict[str, str]],
    by_name: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for image_index, (image_path, mineral, class_label) in enumerate(images, start=1):
        relative_path = image_path.relative_to(DATASET_ROOT).as_posix()
        filename = image_path.name
        metadata = by_pair.get((mineral, filename.lower())) or by_name.get(filename.lower()) or {}
        photo_id = str(metadata.get("mindat_photo_id", "")).strip() or parse_photo_id(filename)

        file_hash = ""
        dhash_value: int | None = None
        width = 0
        height = 0
        image_format = ""
        image_mode = ""
        corrupt = False
        error_message = ""

        try:
            file_hash = file_sha256(image_path)
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_format = str(image.format or image_path.suffix.lstrip(".")).upper()
                image_mode = str(image.mode or "")
                dhash_value = image_dhash(image)
        except Exception as exc:
            corrupt = True
            error_message = f"{type(exc).__name__}: {exc}"

        missing_fields = [
            field
            for field in CORE_METADATA_FIELDS
            if not str(metadata.get(field, "")).strip()
            and not (field == "mindat_photo_id" and photo_id)
        ]

        dimension_pass = bool(width >= MIN_DIMENSION and height >= MIN_DIMENSION)
        quality_status = "pass"
        if corrupt:
            quality_status = "exclude_corrupt"
        elif not dimension_pass:
            quality_status = "exclude_low_resolution"

        rows.append(
            {
                "image_id": f"VTM-{image_index:06d}",
                "relative_path": relative_path,
                "file_name": filename,
                "mineral_label": mineral,
                "four_class_label": class_label,
                "four_class_id": CLASS_IDS[class_label],
                "source_site": metadata.get("source_site", "") or "mindat.org",
                "source_type": metadata.get("source_type", "") or "online_mineral_gallery",
                "mindat_photo_id": photo_id,
                "detail_page_url": metadata.get("detail_page_url", ""),
                "download_source_url": metadata.get("download_source_url", ""),
                "page_title": metadata.get("page_title", ""),
                "locality": metadata.get("locality", ""),
                "photographer_or_credit": metadata.get("photographer_or_credit", ""),
                "license_or_rights": metadata.get("license_or_rights", ""),
                "screening_decision": metadata.get("screening_decision", ""),
                "manual_review_required": metadata.get("manual_review_required", ""),
                "metadata_source_file": metadata.get("_metadata_file", ""),
                "metadata_status": "complete_core" if not missing_fields else "missing_core",
                "metadata_missing_fields": "|".join(missing_fields),
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 6) if height else "",
                "megapixels": round(width * height / 1_000_000, 4) if width and height else 0,
                "file_size_bytes": image_path.stat().st_size,
                "image_format": image_format,
                "image_mode": image_mode,
                "sha256": file_hash,
                "dhash64": f"{dhash_value:016x}" if dhash_value is not None else "",
                "_dhash_int": dhash_value,
                "dimension_pass": dimension_pass,
                "quality_status": quality_status,
                "quality_error": error_message,
                "exact_duplicate_group_size": 1,
                "near_duplicate_group_id": "",
                "near_duplicate_group_size": 1,
                "split_group_id": "",
                "split": "",
            }
        )

    return rows


def group_duplicates(rows: list[dict[str, object]]) -> None:
    union_find = UnionFind(len(rows))
    exact_groups: dict[str, list[int]] = defaultdict(list)
    photo_groups: dict[str, list[int]] = defaultdict(list)

    for index, row in enumerate(rows):
        sha256 = str(row["sha256"])
        photo_id = str(row["mindat_photo_id"])
        if sha256:
            exact_groups[sha256].append(index)
        if photo_id:
            photo_groups[photo_id].append(index)

    for group in exact_groups.values():
        for member in group[1:]:
            union_find.union(group[0], member)

    for group in photo_groups.values():
        for member in group[1:]:
            union_find.union(group[0], member)

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        dhash_value = row["_dhash_int"]
        if dhash_value is None:
            continue
        candidates: set[int] = set()
        for chunk_index in range(8):
            shift = chunk_index * 8
            chunk_value = (int(dhash_value) >> shift) & 0xFF
            candidates.update(buckets[(chunk_index, chunk_value)])

        current_ratio = float(row["aspect_ratio"]) if row["aspect_ratio"] != "" else 0.0
        for candidate in candidates:
            candidate_row = rows[candidate]
            candidate_ratio = (
                float(candidate_row["aspect_ratio"])
                if candidate_row["aspect_ratio"] != ""
                else 0.0
            )
            if current_ratio and candidate_ratio and abs(current_ratio - candidate_ratio) > 0.03:
                continue
            candidate_hash = candidate_row["_dhash_int"]
            if candidate_hash is None:
                continue
            if (int(dhash_value) ^ int(candidate_hash)).bit_count() <= NEAR_DUPLICATE_DISTANCE:
                union_find.union(index, candidate)

        for chunk_index in range(8):
            shift = chunk_index * 8
            chunk_value = (int(dhash_value) >> shift) & 0xFF
            buckets[(chunk_index, chunk_value)].append(index)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        clusters[union_find.find(index)].append(index)

    for cluster_members in clusters.values():
        image_ids = sorted(str(rows[index]["image_id"]) for index in cluster_members)
        group_id = "DG-" + hashlib.sha1("|".join(image_ids).encode("utf-8")).hexdigest()[:12]
        minerals = {str(rows[index]["mineral_label"]) for index in cluster_members}
        classes = {str(rows[index]["four_class_label"]) for index in cluster_members}
        cross_label_conflict = len(minerals) > 1 or len(classes) > 1

        for index in cluster_members:
            rows[index]["near_duplicate_group_id"] = group_id
            rows[index]["near_duplicate_group_size"] = len(cluster_members)
            rows[index]["split_group_id"] = group_id

        if cross_label_conflict and len(cluster_members) > 1:
            for index in cluster_members:
                if rows[index]["quality_status"] == "pass":
                    rows[index]["quality_status"] = "exclude_near_label_conflict"

    for exact_members in exact_groups.values():
        if len(exact_members) <= 1:
            continue
        minerals = {str(rows[index]["mineral_label"]) for index in exact_members}
        classes = {str(rows[index]["four_class_label"]) for index in exact_members}
        for index in exact_members:
            rows[index]["exact_duplicate_group_size"] = len(exact_members)
        if len(minerals) > 1 or len(classes) > 1:
            for index in exact_members:
                rows[index]["quality_status"] = "exclude_exact_label_conflict"
        else:
            ordered = sorted(exact_members, key=lambda item: str(rows[item]["relative_path"]))
            for index in ordered[1:]:
                if rows[index]["quality_status"] == "pass":
                    rows[index]["quality_status"] = "exclude_exact_duplicate"


def assign_splits(rows: list[dict[str, object]]) -> None:
    rng = random.Random(SEED)
    eligible_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["quality_status"] == "pass":
            eligible_groups[str(row["split_group_id"])].append(index)
        else:
            row["split"] = "excluded"

    groups_by_mineral: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for group_id, members in eligible_groups.items():
        minerals = {str(rows[index]["mineral_label"]) for index in members}
        if len(minerals) != 1:
            for index in members:
                rows[index]["quality_status"] = "exclude_group_label_conflict"
                rows[index]["split"] = "excluded"
            continue
        mineral = next(iter(minerals))
        groups_by_mineral[mineral].append((group_id, members))

    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    for mineral, groups in groups_by_mineral.items():
        rng.shuffle(groups)
        groups.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(members) for _, members in groups)
        targets = {split: total * ratio for split, ratio in split_ratios.items()}
        assigned = Counter()

        for _, members in groups:
            split = max(
                split_ratios,
                key=lambda name: (targets[name] - assigned[name]) / max(targets[name], 1),
            )
            for index in members:
                rows[index]["split"] = split
            assigned[split] += len(members)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nested_counts(
    rows: list[dict[str, object]],
    outer_field: str,
    inner_field: str,
    predicate=None,
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if predicate and not predicate(row):
            continue
        result[str(row[outer_field])][str(row[inner_field])] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(result.items())}


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    candidate_count = len(rows)
    eligible_count = sum(row["quality_status"] == "pass" for row in rows)
    missing_metadata = sum(row["metadata_status"] != "complete_core" for row in rows)
    near_groups = {
        str(row["near_duplicate_group_id"])
        for row in rows
        if int(row["near_duplicate_group_size"]) > 1
    }

    return {
        "dataset_root": str(DATASET_ROOT),
        "grain": "one row per physical image file in raw_positive/raw_negative",
        "generated_seed": SEED,
        "minimum_dimension": MIN_DIMENSION,
        "near_duplicate_hamming_threshold": NEAR_DUPLICATE_DISTANCE,
        "candidate_images": candidate_count,
        "eligible_images": eligible_count,
        "excluded_images": candidate_count - eligible_count,
        "metadata_missing_core_images": missing_metadata,
        "metadata_missing_core_rate": round(missing_metadata / candidate_count, 6)
        if candidate_count
        else 0,
        "quality_status_counts": dict(sorted(Counter(str(row["quality_status"]) for row in rows).items())),
        "candidate_by_class": dict(
            sorted(Counter(str(row["four_class_label"]) for row in rows).items())
        ),
        "eligible_by_class": dict(
            sorted(
                Counter(
                    str(row["four_class_label"])
                    for row in rows
                    if row["quality_status"] == "pass"
                ).items()
            )
        ),
        "candidate_by_mineral": dict(
            sorted(Counter(str(row["mineral_label"]) for row in rows).items())
        ),
        "eligible_by_mineral": dict(
            sorted(
                Counter(
                    str(row["mineral_label"])
                    for row in rows
                    if row["quality_status"] == "pass"
                ).items()
            )
        ),
        "split_by_class": nested_counts(
            rows,
            "four_class_label",
            "split",
            predicate=lambda row: row["quality_status"] == "pass",
        ),
        "split_by_mineral": nested_counts(
            rows,
            "mineral_label",
            "split",
            predicate=lambda row: row["quality_status"] == "pass",
        ),
        "exact_duplicate_files": sum(int(row["exact_duplicate_group_size"]) > 1 for row in rows),
        "near_duplicate_groups": len(near_groups),
        "near_duplicate_files": sum(int(row["near_duplicate_group_size"]) > 1 for row in rows),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    by_pair, by_name = load_metadata()
    images = discover_images()
    rows = profile_images(images, by_pair, by_name)
    group_duplicates(rows)
    assign_splits(rows)

    public_fields = [
        "image_id",
        "relative_path",
        "file_name",
        "mineral_label",
        "four_class_label",
        "four_class_id",
        "source_site",
        "source_type",
        "mindat_photo_id",
        "detail_page_url",
        "download_source_url",
        "page_title",
        "locality",
        "photographer_or_credit",
        "license_or_rights",
        "screening_decision",
        "manual_review_required",
        "metadata_source_file",
        "metadata_status",
        "metadata_missing_fields",
        "width",
        "height",
        "aspect_ratio",
        "megapixels",
        "file_size_bytes",
        "image_format",
        "image_mode",
        "sha256",
        "dhash64",
        "dimension_pass",
        "quality_status",
        "quality_error",
        "exact_duplicate_group_size",
        "near_duplicate_group_id",
        "near_duplicate_group_size",
        "split_group_id",
        "split",
    ]

    rows.sort(key=lambda row: str(row["image_id"]))
    master_path = OUTPUT_ROOT / "dataset_master_manifest.csv"
    write_csv(master_path, rows, public_fields)

    split_fields = [
        "image_id",
        "relative_path",
        "mineral_label",
        "four_class_label",
        "four_class_id",
        "mindat_photo_id",
        "split_group_id",
        "split",
    ]
    split_rows = [row for row in rows if row["quality_status"] == "pass"]
    write_csv(OUTPUT_ROOT / "dataset_split_manifest.csv", split_rows, split_fields)

    issue_rows = [
        row
        for row in rows
        if row["quality_status"] != "pass"
        or row["metadata_status"] != "complete_core"
        or int(row["near_duplicate_group_size"]) > 1
    ]
    write_csv(OUTPUT_ROOT / "dataset_quality_issues.csv", issue_rows, public_fields)

    class_mapping_rows = [
        {
            "four_class_id": class_id,
            "four_class_label": class_label,
            "included_minerals": "|".join(
                sorted(
                    {
                        str(row["mineral_label"])
                        for row in rows
                        if row["four_class_label"] == class_label
                    }
                )
            ),
        }
        for class_label, class_id in CLASS_IDS.items()
    ]
    write_csv(
        OUTPUT_ROOT / "dataset_class_mapping.csv",
        class_mapping_rows,
        ["four_class_id", "four_class_label", "included_minerals"],
    )

    summary = build_summary(rows)
    (OUTPUT_ROOT / "dataset_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
