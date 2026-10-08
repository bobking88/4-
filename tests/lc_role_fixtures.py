"""Synthetic-only fixtures shared by the LC-RFA acceptance tests."""
import hashlib

import torch


def raw(n=120, seed=7):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n, dtype=torch.long) % 4
    return {
        "p": torch.softmax(torch.randn(n, 4, generator=generator, dtype=torch.float64), 1),
        "evidence": torch.randn(n, 18, generator=generator, dtype=torch.float64),
        "contradiction": torch.rand(n, 1, generator=generator, dtype=torch.float64),
        "H": torch.randn(n, 1280, generator=generator, dtype=torch.float64),
        "labels": labels,
        "records": [dict(image_id=f"synthetic-{i}", split_group_id=f"group-{i}",
                         relative_path=f"synthetic/{i}.jpg", four_class_id=str(int(labels[i])))
                    for i in range(n)],
        "hashes": {},
    }


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
