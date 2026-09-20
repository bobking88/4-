"""Validation-only gate training with a single frozen HRGV expert checkpoint."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hrgv_network import HierarchicalRiskGatedVerificationNet, regret_gate_targets, weighted_soft_gate_loss, apply_residual_target_verifiers
from mineral_hierarchy import validate_species_role_mapping
from train_hierarchical_mineral_classifier import HierarchicalMineralImageDataset
from train_hrgv_mineral_classifier import build_role_matrix
from train_mineral_classifier import load_manifest_records, require_training_dependencies, split_records, create_transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/training/frozen_expert_gate_pilot'
BASE = ROOT / 'outputs/training/formal_hrgv_residual_complete_seed20260728'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def final_gate_nll(g, direct, mapped, labels, ti, metal):
    p = apply_residual_target_verifiers(g*direct+(1-g)*mapped, ti, metal)
    return F.nll_loss(p.clamp_min(torch.finfo(p.dtype).eps).log(), labels)


def metrics(gate, cached):
    probabilities = gate * cached['direct'] + (1-gate) * cached['mapped']
    y = cached['labels']
    true = probabilities.gather(1, y[:, None]).clamp_min(torch.finfo(probabilities.dtype).eps)
    oracle = torch.maximum(cached['direct'], cached['mapped']).gather(1, y[:, None]).clamp_min(torch.finfo(probabilities.dtype).eps)
    result = {
        'fusion_nll_nats': float(-true.log().mean()),
        'routing_regret_nats': float((-true.log()+oracle.log()).mean()),
        'accuracy': float((probabilities.argmax(1) == y).float().mean()),
    }
    if 'ti' in cached:
        final = apply_residual_target_verifiers(probabilities, cached['ti'], cached['metal'])
        pred = final.argmax(1)
        cm = torch.bincount(y*4+pred, minlength=16).reshape(4, 4).double()
        denominator = cm.sum(0)+cm.sum(1)
        result.update({
            'final_nll_nats': float(F.nll_loss(final.clamp_min(torch.finfo(final.dtype).eps).log(), y)),
            'final_accuracy': float((pred == y).double().mean()),
            'final_macro_f1': float(torch.where(denominator > 0, 2*cm.diag()/denominator, 0).mean()),
            'final_target_recall': float(cm[0, 0]/cm[0].sum()),
            'final_ti_intrusion': float(cm[1, 0]/cm[1].sum()),
            'final_metal_intrusion': float(cm[3, 0]/cm[3].sum()),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-output-study', action='store_true')
    args = parser.parse_args()
    output_dir = ROOT/'outputs/training/final_output_gate_study' if args.final_output_study else OUT
    objectives = ('fusion_nll', 'final_nll', 'final_nll_plus_regret') if args.final_output_study else ('fusion_nll', 'regret_bce')
    selection_key = 'final_nll_nats' if args.final_output_study else 'fusion_nll_nats'
    if output_dir.exists():
        raise RuntimeError('Output already exists; inspect it before any rerun.')
    environment_path = BASE / 'environment.json'
    checkpoint_path = BASE / 'best_model.pt'
    environment = json.loads(environment_path.read_text(encoding='utf-8'))
    manifest = Path(environment['manifest'])
    dependencies = require_training_dependencies()
    records = load_manifest_records(manifest, Path(environment['dataset_root']))
    mapping = validate_species_role_mapping(records)
    by_split = split_records(records)
    output_dir.mkdir(parents=True)
    save_json(output_dir / 'registered_protocol.json', {
        'scope': 'exploratory validation-only; one fixed expert checkpoint; final output' if args.final_output_study else 'exploratory validation-only; pre-verifier fusion; one fixed expert checkpoint',
        'expert_seed': 20260728, 'gate_seeds': [20260912, 20260913, 20260914],
        'objectives': ['equal', *objectives],
        'epochs': 30, 'batch_size': 256, 'optimizer': 'AdamW', 'lr': .001,
        'weight_decay': .0001, 'selection': f'lowest validation {selection_key}; earliest tie',
        'auxiliary_weight': .1 if args.final_output_study else None,
        'final_output_study': args.final_output_study,
        'transforms': 'existing deterministic evaluation transform for both train and val',
        'target_temperature': .2, 'gap_temperature': .5,
        'test_access': False, 'fresh_gate_initialization': True,
        'limitations': 'experts trained on training samples and selected using this validation set; no independent confirmation',
        'hashes': {str(p): digest(p) for p in (manifest, environment_path, checkpoint_path, Path(__file__), ROOT/'scripts/hrgv_network.py')},
    })
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    model = HierarchicalRiskGatedVerificationNet(
        dependencies['models'], build_role_matrix(mapping, torch), pretrained=False,
        embedding_dim=int(environment.get('embedding_dim', 128)),
        gate_hidden_dim=int(environment.get('gate_hidden_dim', 128)),
        verifier_mode='residual',
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    frozen_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    _, transform = create_transforms(224, dependencies['transforms'])
    cache = {}
    captured = []
    hook = model.gate_network.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    for split in ('train', 'val'):
        dataset = HierarchicalMineralImageDataset(by_split[split], mapping, transform)
        loader = dependencies['DataLoader'](dataset, batch_size=32, shuffle=False, num_workers=0)
        chunks = {k: [] for k in ('features', 'direct', 'mapped', 'labels')}
        if args.final_output_study:
            chunks.update({'ti': [], 'metal': []})
        with torch.no_grad():
            for batch_index, (images, labels, _) in enumerate(loader):
                captured.clear()
                output = model(images.to(device))
                chunks['features'].append(captured[0])
                chunks['direct'].append(output['direct_role_probabilities'])
                chunks['mapped'].append(output['mapped_role_probabilities'])
                chunks['labels'].append(labels.to(device))
                if args.final_output_study:
                    chunks['ti'].append(output['ti_target_probability'])
                    chunks['metal'].append(output['metallic_target_probability'])
                if batch_index % 40 == 0:
                    print(f'extract {split} batch {batch_index}/{len(loader)}', flush=True)
        cache[split] = {k: torch.cat(v) for k, v in chunks.items()}
    hook.remove()
    train, val = cache['train'], cache['val']
    targets = regret_gate_targets(train['direct'], train['mapped'], train['labels'], .2, .5, torch)
    summary = {'equal': metrics(torch.full((len(val['labels']), 1), .5, device=device), val), 'runs': {}}
    for seed in (20260912, 20260913, 20260914):
        torch.manual_seed(seed)
        template = copy.deepcopy(model.gate_network).requires_grad_(True)
        for layer in template.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for objective in objectives:
            torch.manual_seed(seed)
            gate_net = copy.deepcopy(template)
            optimizer = torch.optim.AdamW(gate_net.parameters(), lr=.001, weight_decay=.0001)
            history, best_state, best_metric = [], None, float('inf')
            for epoch in range(30):
                gate_net.train()
                for indices in torch.randperm(len(train['labels']), device=device).split(256):
                    g = torch.sigmoid(gate_net(train['features'][indices]))
                    if objective == 'fusion_nll':
                        p = g*train['direct'][indices]+(1-g)*train['mapped'][indices]
                        loss = F.nll_loss(p.clamp_min(torch.finfo(p.dtype).eps).log(), train['labels'][indices])
                    elif objective.startswith('final_nll'):
                        loss = final_gate_nll(g, train['direct'][indices], train['mapped'][indices], train['labels'][indices], train['ti'][indices], train['metal'][indices])
                        if objective == 'final_nll_plus_regret':
                            loss = loss + .1*weighted_soft_gate_loss(g, targets['soft_oracle_gate'][indices], targets['gate_gap_weight'][indices], torch)
                    else:
                        loss = weighted_soft_gate_loss(g, targets['soft_oracle_gate'][indices], targets['gate_gap_weight'][indices], torch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                gate_net.eval()
                with torch.no_grad():
                    record = {'epoch': epoch+1, **metrics(torch.sigmoid(gate_net(val['features'])), val)}
                history.append(record)
                if record[selection_key] < best_metric:
                    best_metric, best_state = record[selection_key], copy.deepcopy(gate_net.state_dict())
                if (epoch+1) % 10 == 0:
                    print(seed, objective, record, flush=True)
            gate_net.load_state_dict(best_state)
            run_dir = output_dir / f'{objective}_seed{seed}'
            run_dir.mkdir()
            torch.save(best_state, run_dir/'gate_state.pt')
            save_json(run_dir/'history.json', history)
            with torch.no_grad():
                gates = torch.sigmoid(gate_net(val['features']))
                selected = min(history, key=lambda r: r[selection_key])
            summary['runs'][run_dir.name] = selected
            with (run_dir/'val_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['image_id', 'split_group_id', 'class_id', 'gate'])
                writer.writerows((r.image_id, r.split_group_id, r.class_id, float(g)) for r, g in zip(by_split['val'], gates))
    assert all(torch.equal(v.cpu(), frozen_state[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    summary['frozen_expert_parameters_and_buffers_unchanged'] = True
    summary['test_access'] = False
    save_json(output_dir/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
