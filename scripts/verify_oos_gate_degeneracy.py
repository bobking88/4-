"""Numerically verify the sufficient gate-degeneracy condition."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hrgv_network import apply_residual_target_verifiers


ROOT = Path(__file__).resolve().parents[1]


def evaluate_pair(direct, mapped):
    logit = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
    gate = logit.sigmoid()
    mixture = gate * direct + (1.0 - gate) * mapped
    ti = torch.tensor([[0.3]], dtype=torch.float64)
    metal = torch.tensor([[0.7]], dtype=torch.float64)
    final = apply_residual_target_verifiers(mixture, ti, metal)
    loss = -final[0, 0].log()
    gradient = torch.autograd.grad(loss, logit)[0]
    direct_loss = -direct[0, 0].log()
    mapped_loss = -mapped[0, 0].log()
    gap_weight = torch.tanh((mapped_loss - direct_loss).abs() / 0.5)
    return {
        "gate": float(gate.detach()),
        "final_nll": float(loss.detach()),
        "final_nll_gradient": float(gradient.detach()),
        "expert_true_class_loss_gap": float((mapped_loss - direct_loss).detach()),
        "gap_weight": float(gap_weight.detach()),
    }


def compute_examples():
    equal = torch.tensor([[0.7, 0.1, 0.1, 0.1]], dtype=torch.float64)
    return {
        "equal_experts": evaluate_pair(equal, equal.clone()),
        "distinct_experts": evaluate_pair(
            equal,
            torch.tensor([[0.2, 0.3, 0.3, 0.2]], dtype=torch.float64),
        ),
    }


def main():
    result = compute_examples()
    result["interpretation"] = (
        "Equal expert posteriors make the fused and verified posterior independent "
        "of the gate; distinct posteriors restore a non-zero risk and regret signal."
    )
    output = ROOT / "outputs/theory/oos_gate_degeneracy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
