"""Within-checkpoint routing diagnostics; no training or configuration selection."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def audit(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError(f'Empty or duplicate samples: {path}')
    values = []
    for row in rows:
        a, b, g = (float(row[k]) for k in ('direct_true_probability', 'mapped_true_probability', 'gate'))
        if not (0 < a <= 1 and 0 < b <= 1 and 0 <= g <= 1):
            raise ValueError(f'Invalid positive posterior: {row["image_id"]}')
        oracle = -math.log(max(a, b))
        fused = -math.log(g*a+(1-g)*b)
        equal = -math.log((a+b)/2)
        values.append((oracle, fused, equal))
    return {
        'samples': len(rows), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'oracle_nll_nats': mean(v[0] for v in values),
        'learned_fusion_nll_nats': mean(v[1] for v in values),
        'equal_fusion_nll_nats': mean(v[2] for v in values),
        'learned_routing_regret_nats': mean(v[1]-v[0] for v in values),
        'equal_routing_regret_nats': mean(v[2]-v[0] for v in values),
        'learned_minus_equal_nll_nats': mean(v[1]-v[2] for v in values),
        'max_decomposition_error': max(abs((v[1]-v[2])-((v[1]-v[0])-(v[2]-v[0]))) for v in values),
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    source = root / 'outputs/paper_experiments_v3/rsg_theory_replay'
    paths = sorted(source.glob('*/*.csv'))
    if len(paths) != 9:
        raise ValueError(f'Expected nine registered replays, found {len(paths)}')
    result = {
        'evidence_type': 'posthoc_frozen_checkpoint_descriptive_diagnostic',
        'unit': 'nats, not percentage points',
        'scope': 'pre-verifier fusion; identical experts within each comparison; no causal training inference',
        'runs': {str(p.relative_to(source)): audit(p) for p in paths},
    }
    output = root / 'outputs/theory/frozen_expert_routing_audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, run in result['runs'].items():
        print(name, f"learned-equal={run['learned_minus_equal_nll_nats']:+.6f} nats")
