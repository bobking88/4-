from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from mineral_hierarchy import SpeciesRoleMapping, validate_species_role_mapping
from train_mineral_classifier import load_manifest_records


DEFAULT_SEED = 20260801
DEFAULT_CANDIDATE_SIZES = (2, 3, 4)
SCENARIOS = ("role_consistent", "role_conflict")


def _validate_mapping(mapping: SpeciesRoleMapping) -> None:
    if len(mapping.species_labels) != len(mapping.species_role_ids):
        raise ValueError("Species labels and role IDs must have the same length.")
    if len(mapping.species_labels) < 2:
        raise ValueError("At least two species are required for candidate sets.")
    if len(set(mapping.species_labels)) != len(mapping.species_labels):
        raise ValueError("Species labels must be unique.")


def _validate_candidate_sizes(candidate_sizes: Iterable[int]) -> tuple[int, ...]:
    sizes = tuple(int(size) for size in candidate_sizes)
    if not sizes:
        raise ValueError("At least one candidate size is required.")
    if any(size < 2 for size in sizes):
        raise ValueError("Candidate sizes must be at least 2.")
    if len(set(sizes)) != len(sizes):
        raise ValueError("Candidate sizes must be unique.")
    return sizes


def _make_row(
    scenario: str,
    candidate_indices: list[int],
    mapping: SpeciesRoleMapping,
) -> dict[str, object]:
    candidate_labels = [mapping.species_labels[index] for index in candidate_indices]
    candidate_roles = [mapping.species_role_ids[index] for index in candidate_indices]
    return {
        "candidate_size": len(candidate_indices),
        "scenario": scenario,
        "true_species_label": candidate_labels[0],
        "true_role_id": candidate_roles[0],
        "candidate_species_labels": candidate_labels,
        "candidate_role_ids": candidate_roles,
        "species_unique": len(set(candidate_labels)) == 1,
        "role_unique": len(set(candidate_roles)) == 1,
    }


def build_candidate_set_rows(
    mapping: SpeciesRoleMapping,
    candidate_sizes: Iterable[int] = DEFAULT_CANDIDATE_SIZES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, object]]:
    """Build reproducible candidate sets using logical species-role assignments only."""
    _validate_mapping(mapping)
    sizes = _validate_candidate_sizes(candidate_sizes)
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for size in sizes:
        for species_index, role_id in enumerate(mapping.species_role_ids):
            same_role = [
                index
                for index, candidate_role_id in enumerate(mapping.species_role_ids)
                if candidate_role_id == role_id and index != species_index
            ]
            other_role = [
                index
                for index, candidate_role_id in enumerate(mapping.species_role_ids)
                if candidate_role_id != role_id
            ]
            if len(same_role) >= size - 1:
                rows.append(
                    _make_row(
                        "role_consistent",
                        [species_index, *rng.sample(same_role, size - 1)],
                        mapping,
                    )
                )
            if len(other_role) >= size - 1:
                rows.append(
                    _make_row(
                        "role_conflict",
                        [species_index, *rng.sample(other_role, size - 1)],
                        mapping,
                    )
                )
    if not rows:
        raise ValueError("No candidate sets can be constructed for the requested sizes.")
    return rows


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"row_count": 0, "species_unique_count": 0, "species_unique_rate": 0.0, "role_unique_count": 0, "role_unique_rate": 0.0}
    row_count = len(rows)
    species_unique_count = sum(bool(row["species_unique"]) for row in rows)
    role_unique_count = sum(bool(row["role_unique"]) for row in rows)
    return {
        "row_count": row_count,
        "species_unique_count": species_unique_count,
        "species_unique_rate": species_unique_count / row_count,
        "role_unique_count": role_unique_count,
        "role_unique_rate": role_unique_count / row_count,
    }


def summarize_candidate_sets(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize logical identifiability without using visual or model predictions."""
    by_scenario = {
        scenario: _summarize_rows([row for row in rows if row["scenario"] == scenario])
        for scenario in SCENARIOS
    }
    by_candidate_size = {
        str(size): _summarize_rows([row for row in rows if row["candidate_size"] == size])
        for size in sorted({int(row["candidate_size"]) for row in rows})
    }
    return {
        "row_count": len(rows),
        "scenario_count": dict(Counter(str(row["scenario"]) for row in rows)),
        "species_unique_count": sum(bool(row["species_unique"]) for row in rows),
        "species_unique_rate": (
            sum(bool(row["species_unique"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "role_unique_count": sum(bool(row["role_unique"]) for row in rows),
        "role_unique_rate": sum(bool(row["role_unique"]) for row in rows) / len(rows) if rows else 0.0,
        "role_consistent": by_scenario["role_consistent"],
        "role_conflict": by_scenario["role_conflict"],
        "by_candidate_size": by_candidate_size,
    }


def _markdown_report(
    summary: dict[str, object],
    mapping: SpeciesRoleMapping,
    manifest: Path,
    dataset_root: Path,
    split_counts: dict[str, int],
    seed: int,
    candidate_sizes: tuple[int, ...],
) -> str:
    lines = [
        "# Controlled Candidate-Set Role Identifiability Validation",
        "",
        "This is a **controlled logical validation** of candidate-set identifiability. It uses the frozen species-to-role mapping and does not infer visual labels from raw images or model predictions.",
        "",
        "## Configuration",
        "",
        f"- Manifest: `{manifest}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Fixed seed: `{seed}`",
        f"- Candidate sizes: `{', '.join(map(str, candidate_sizes))}`",
        f"- Species count: `{len(mapping.species_labels)}`",
        f"- Fixed split consumed unchanged: `{split_counts}`",
        "",
        "## Results",
        "",
        "| Scenario | Rows | Species unique rate | Role unique rate |",
        "|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        result = summary[scenario]
        lines.append(
            f"| {scenario} | {result['row_count']} | {result['species_unique_rate']:.2%} | {result['role_unique_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Role-consistent candidate sets are expected to preserve role identifiability even when species identity is ambiguous. Role-conflict candidate sets intentionally include another role and therefore do not guarantee a unique role.",
            "",
            "This result validates the stated logical condition only. It is not a visual-label audit, a claim about image truth, or an industrial separation or recovery estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled logical validation of candidate-set role identifiability."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-sizes", type=int, nargs="+", default=DEFAULT_CANDIDATE_SIZES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest_records(args.manifest, args.dataset_root)
    mapping = validate_species_role_mapping(records)
    split_counts = dict(sorted(Counter(record.split for record in records).items()))
    candidate_sizes = _validate_candidate_sizes(args.candidate_sizes)
    rows = build_candidate_set_rows(mapping, candidate_sizes, args.seed)
    summary = summarize_candidate_sets(rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "validation_type": "controlled logical validation",
        "uses_visual_labels": False,
        "uses_model_predictions": False,
        "manifest": str(args.manifest),
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "candidate_sizes": list(candidate_sizes),
        "fixed_split_consumed_unchanged": True,
        "split_counts": split_counts,
        "species_labels": list(mapping.species_labels),
        "species_role_ids": list(mapping.species_role_ids),
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "role_identifiability_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "role_identifiability_summary.md").write_text(
        _markdown_report(
            summary, mapping, args.manifest, args.dataset_root, split_counts, args.seed, candidate_sizes
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
