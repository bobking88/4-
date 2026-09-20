"""Partition only original training groups for an expert-unseen gate diagnostic."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def partition_training_rows(rows, seed=20260919):
    if not rows or any(r['split'] != 'train' for r in rows):
        raise ValueError('Only nonempty original training rows are allowed.')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs.')
    groups = defaultdict(list)
    for row in rows:
        if not row['split_group_id']:
            raise ValueError('Missing duplicate group.')
        groups[row['split_group_id']].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        labels = {(r['mineral_label'], r['four_class_id']) for r in members}
        if len(labels) != 1:
            raise ValueError('Mixed-label group requires explicit adjudication.')
        strata[next(iter(labels))].append(group)
    assignment = {}
    for label, ids in sorted(strata.items()):
        if len(ids) < 3:
            raise ValueError(f'Insufficient groups for {label}.')
        ordered = sorted(ids, key=lambda g: (hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest(), g))
        held = max(1, round(.15*len(ids)))
        for i, group in enumerate(ordered):
            assignment[group] = 'expert_stop' if i < held else 'gate_fit' if i < 2*held else 'expert_fit'
    return [{**r, 'experiment_subset': assignment[r['split_group_id']]} for r in sorted(rows, key=lambda r:r['image_id'])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Refusing to overwrite an existing partition.')
    original_bytes = args.manifest.read_bytes()
    with args.manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    seen = defaultdict(set)
    for row in rows:
        seen[row['split_group_id']].add(row['split'])
    if any(len(s)>1 for s in seen.values()):
        raise ValueError('Original duplicate group crosses splits.')
    partition = partition_training_rows([r for r in rows if r['split']=='train'])
    counts = Counter(r['experiment_subset'] for r in partition)
    by_species = defaultdict(Counter)
    for row in partition:
        by_species[row['mineral_label']][row['experiment_subset']] += 1
    assert args.manifest.read_bytes() == original_bytes
    args.output_dir.mkdir(parents=True)
    with (args.output_dir/'training_partition.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, 'experiment_subset'])
        writer.writeheader()
        writer.writerows(partition)
    result = {
        'seed': 20260919, 'source_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'training metadata only; no image access or new model training',
        'allocation': 'species-role strata; duplicate groups ranked by seeded SHA256; 15% groups each stop/gate, remainder fit; rounded with minimum one',
        'counts': dict(counts), 'by_species': dict(by_species),
        'original_split_column_preserved': True, 'original_manifest_unchanged': True,
        'warning': 'assignment metadata, not directly a trainer manifest; gate_fit must be absent from expert fitting and checkpoint selection',
    }
    (args.output_dir/'audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
