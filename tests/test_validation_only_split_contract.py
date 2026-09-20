import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_mineral_classifier import split_records
import train_hierarchical_mineral_classifier as hierarchy


class ValidationOnlyTests(unittest.TestCase):
    def test_explicit_validation_only_accepts_missing_test(self):
        rows = [SimpleNamespace(split=s, image_id=s) for s in ('train', 'val')]
        result = split_records(rows, required_splits=('train', 'val'))
        self.assertEqual(result['test'], [])
        with self.assertRaises(ValueError):
            split_records(rows)

    def test_loader_does_not_construct_test_dataset(self):
        args = SimpleNamespace(validation_only=True, image_size=224, batch_size=2, num_workers=0)
        records = {'train': ['fit'], 'val': ['stop']}
        deps = {'transforms': None, 'DataLoader': lambda dataset, **kwargs: dataset}
        with patch.object(hierarchy, 'create_transforms', return_value=('train_transform', 'val_transform')), patch.object(hierarchy, 'HierarchicalMineralImageDataset', side_effect=lambda rows, mapping, transform: rows):
            loaders = hierarchy.create_hierarchical_dataloaders(args, records, None, deps, SimpleNamespace(type='cpu'))
        self.assertEqual(set(loaders), {'train', 'val'})


if __name__ == '__main__':
    unittest.main()
