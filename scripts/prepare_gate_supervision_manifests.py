"""Export isolated expert manifests and count-matched gate supervision controls."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'outputs/training/gate_supervision_partition_v1/training_partition.csv'
    out = ROOT/'outputs/training/gate_supervision_manifests_v1'
    if out.exists():
        raise ValueError('Refusing to overwrite manifests.')
    with source.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    fit = [r for r in rows if r['experiment_subset']=='expert_fit']
    stop = [r for r in rows if r['experiment_subset']=='expert_stop']
    unseen = [r for r in rows if r['experiment_subset']=='gate_fit']
    key = lambda r: (r['mineral_label'], r['four_class_id'])
    target = Counter(key(r) for r in unseen)
    groups = defaultdict(list)
    for r in fit:
        groups[r['split_group_id']].append(r)
    seen = []
    remaining = target.copy()
    for group in sorted(groups, key=lambda g: hashlib.sha256(f'20260920:{g}'.encode()).hexdigest()):
        members = groups[group]
        labels = {key(r) for r in members}
        if len(labels) != 1:
            raise ValueError('Mixed-label duplicate group.')
        label = next(iter(labels))
        if len(members) <= remaining[label]:
            seen.extend(members)
            remaining[label] -= len(members)
    if any(remaining.values()):
        raise ValueError('Exact count match unavailable without splitting a group.')
    assert Counter(key(r) for r in seen) == target
    group_sets = [{r['split_group_id'] for r in part} for part in (fit, stop, unseen)]
    assert all(not group_sets[i]&group_sets[j] for i in range(3) for j in range(i))
    expert = [{**r, 'split': 'train'} for r in fit]+[{**r, 'split': 'val'} for r in stop]
    assert not {r['image_id'] for r in expert}&{r['image_id'] for r in unseen}
    out.mkdir(parents=True)
    hashes = {}
    for name, records in [('expert', expert), ('gate_seen', seen), ('gate_unseen', unseen)]:
        path = out/f'{name}.csv'
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(records, key=lambda r:r['image_id']))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {'expert_fit':len(fit), 'expert_stop':len(stop), 'gate_seen':len(seen), 'gate_unseen':len(unseen), 'exact_species_role_count_match':True, 'unseen_absent_from_expert_manifest':True, 'duplicate_group_overlap':False, 'original_val_test_used':False, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'output_sha256':hashes}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
