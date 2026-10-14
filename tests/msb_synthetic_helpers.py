"""Synthetic MSB fixtures; never open project data."""
import copy
import torch


def make_prepared_fixture(n=340, seed=7):
    generator = torch.Generator().manual_seed(seed)
    counts = [66, 130, 97, 47] if n == 340 else [n // 4] * 4
    if sum(counts) != n or min(counts) < 3:
        raise ValueError("Fixture requires four populated classes.")
    labels = torch.cat([torch.full((count,), k, dtype=torch.long)
                        for k, count in enumerate(counts)])
    assignment = torch.cat([torch.arange(count) % 3 for count in counts])
    batch = dict(h=torch.randn(n, 64, generator=generator, dtype=torch.float64),
                 e=torch.randn(n, 23, generator=generator, dtype=torch.float64),
                 log_anchor=torch.log_softmax(torch.randn(n, 4, generator=generator,
                                                        dtype=torch.float64), 1),
                 labels=labels,
                 records=[dict(image_id=f"synthetic-{i}", split_group_id=f"group-{i}",
                               four_class_id=str(int(labels[i])), tc_subset="projector_fit")
                          for i in range(n)])
    folds = []
    for fold in range(3):
        train = torch.where(assignment != fold)[0]
        validation = torch.where(assignment == fold)[0]
        def subset(indices):
            return {k: copy.deepcopy([batch[k][i] for i in indices.tolist()])
                    if k == "records" else batch[k][indices].clone()
                    for k in batch}
        folds.append(dict(inner_fold=fold, train_indices=train,
                          validation_indices=validation, train=subset(train),
                          validation=subset(validation),
                          state=dict(components=64,
                                     train_ids=[batch["records"][i]["image_id"]
                                                for i in train.tolist()])))
    return dict(batch=batch, folds=folds, assignment=assignment, counts=counts)
