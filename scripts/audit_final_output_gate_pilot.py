"""Read-only validation replay of already-selected frozen-expert pilot gates."""
import copy
import csv
import json
from pathlib import Path

import torch

from run_frozen_expert_gate_pilot import (
    BASE, OUT as PILOT, ROOT, digest, save_json,
    HierarchicalRiskGatedVerificationNet, build_role_matrix,
    create_transforms, HierarchicalMineralImageDataset,
    load_manifest_records, require_training_dependencies,
    split_records, validate_species_role_mapping,
)
from hrgv_network import apply_residual_target_verifiers, regret_gate_targets


def classification(p, y):
    pred = p.argmax(1)
    cm = torch.bincount(y*4+pred, minlength=16).reshape(4, 4).double()
    denom = cm.sum(0)+cm.sum(1)
    f1 = torch.where(denom > 0, 2*cm.diag()/denom, 0)
    return {
        'nll_nats': float(-p.gather(1, y[:, None]).clamp_min(1e-15).log().mean()),
        'accuracy': float((pred == y).double().mean()),
        'macro_f1': float(f1.mean()),
        'target_recall': float(cm[0, 0]/cm[0].sum()),
        'ti_intrusion': float(cm[1, 0]/cm[1].sum()),
        'metal_intrusion': float(cm[3, 0]/cm[3].sum()),
    }


def main():
    outdir = ROOT/'outputs/theory/final_output_gate_pilot'
    if outdir.exists():
        raise RuntimeError('Refusing to overwrite a previous audit.')
    env = json.loads((BASE/'environment.json').read_text(encoding='utf-8'))
    prior = json.loads((PILOT/'summary.json').read_text())
    deps = require_training_dependencies()
    records = load_manifest_records(Path(env['manifest']), Path(env['dataset_root']))
    mapping = validate_species_role_mapping(records)
    rows = split_records(records)['val']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    model = HierarchicalRiskGatedVerificationNet(
        deps['models'], build_role_matrix(mapping, torch), pretrained=False,
        embedding_dim=int(env.get('embedding_dim', 128)),
        gate_hidden_dim=int(env.get('gate_hidden_dim', 128)), verifier_mode='residual',
    ).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(BASE/'best_model.pt', map_location=device, weights_only=False)['model_state_dict'])
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    _, transform = create_transforms(224, deps['transforms'])
    loader = deps['DataLoader'](HierarchicalMineralImageDataset(rows, mapping, transform), batch_size=32, shuffle=False, num_workers=0)
    captured = []
    hook = model.gate_network.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    chunks = {k: [] for k in ('features', 'a', 'b', 'ti', 'metal', 'y')}
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(loader):
            captured.clear()
            result = model(images.to(device))
            values = (captured[0], result['direct_role_probabilities'], result['mapped_role_probabilities'], result['ti_target_probability'], result['metallic_target_probability'], labels.to(device))
            for key, value in zip(chunks, values):
                chunks[key].append(value)
            if i % 10 == 0:
                print(f'validation replay {i}/{len(loader)}', flush=True)
    hook.remove()
    cache = {k: torch.cat(v) for k, v in chunks.items()}
    a, b = cache['a'].double(), cache['b'].double()
    ti, metal, y = cache['ti'].double(), cache['metal'].double(), cache['y']
    c = torch.ones_like(a)
    c[:, :1] = torch.exp(-torch.relu(1-2*ti)-torch.relu(1-2*metal))
    za, zb = (c*a).sum(1, keepdim=True), (c*b).sum(1, keepdim=True)
    targets = regret_gate_targets(a, b, y, .2, .5, torch)
    gates = {'equal': torch.full_like(ti, .5)}
    hashes = {'expert': digest(BASE/'best_model.pt'), 'script': digest(Path(__file__)), 'network': digest(ROOT/'scripts/hrgv_network.py')}
    with torch.no_grad():
        for name in prior['runs']:
            path = PILOT/name/'gate_state.pt'
            gate = copy.deepcopy(model.gate_network)
            gate.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            gate.eval()
            gates[name] = torch.sigmoid(gate(cache['features'])).double()
            hashes[name] = digest(path)
    outdir.mkdir(parents=True)
    summary = {'scope': 'posthoc validation diagnostic; no retraining, test access, or configuration selection', 'samples': len(rows), 'expert_seed': 20260728, 'hashes': hashes, 'runs': {}}
    for name, g in gates.items():
        m = g*a+(1-g)*b
        q = apply_residual_target_verifiers(m, ti, metal)
        effective = g*za/(g*za+(1-g)*zb)
        rho = g*a.gather(1, y[:, None])/m.gather(1, y[:, None])
        pre, final = g-rho, effective-rho
        auxiliary = targets['gate_gap_weight']*(g-targets['soft_oracle_gate'])
        summary['runs'][name] = {
            'pre_verifier': classification(m, y), 'final': classification(q, y),
            'pre_final_gradient_sign_conflicts': int((pre*final < -1e-12).sum()),
            'aux_final_gradient_sign_conflicts': int((auxiliary*final < -1e-12).sum()),
            'max_effective_gate_shift': float((effective-g).abs().max()),
        }
        reference = prior['equal'] if name == 'equal' else prior['runs'][name]
        assert abs(summary['runs'][name]['pre_verifier']['nll_nats']-reference['fusion_nll_nats']) < 2e-5
        with (outdir/f'{name}.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image_id', 'split_group_id', 'label', 'gate', 'effective_gate', 'pre_gradient', 'final_gradient', 'raw_aux_gradient'])
            matrix = torch.cat((g, effective, pre, final, auxiliary), 1).cpu().tolist()
            writer.writerows([row.image_id, row.split_group_id, row.class_id, *values] for row, values in zip(rows, matrix))
    assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items())
    summary['frozen_state_unchanged'] = True
    summary['gradient_scope'] = 'per-sample partial derivative wrt gate logit, unweighted NLL; auxiliary before batch normalization; not shared-parameter gradient conflict'
    save_json(outdir/'summary.json', summary)
    print(json.dumps(summary['runs'], indent=2), flush=True)


if __name__ == '__main__':
    main()
