import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_gate_supervision_partition import partition_training_rows


class PartitionTests(unittest.TestCase):
    def rows(self):
        return [dict(image_id=f'i{i}', split_group_id=f'g{i}', mineral_label='m', four_class_id='0', split='train') for i in range(20)]

    def test_deterministic_order_and_group_integrity(self):
        rows = self.rows()
        rows.append({**rows[0], 'image_id': 'duplicate-view'})
        a = partition_training_rows(rows)
        b = partition_training_rows(list(reversed(rows)))
        self.assertEqual({r['image_id']:r['experiment_subset'] for r in a}, {r['image_id']:r['experiment_subset'] for r in b})
        self.assertEqual(len({r['experiment_subset'] for r in a if r['split_group_id']=='g0'}), 1)
        self.assertEqual({r['experiment_subset'] for r in a}, {'expert_fit', 'expert_stop', 'gate_fit'})
        self.assertTrue(all(r['split']=='train' for r in a))
        self.assertTrue(all('experiment_subset' not in r for r in rows))

    def test_reject_nontraining(self):
        rows = self.rows()
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            partition_training_rows(rows)

    def test_reject_ambiguous_group(self):
        rows = self.rows()
        rows.append({**rows[0], 'image_id': 'conflict', 'mineral_label': 'other'})
        with self.assertRaises(ValueError):
            partition_training_rows(rows)

    def test_reject_insufficient_groups(self):
        with self.assertRaises(ValueError):
            partition_training_rows(self.rows()[:2])


if __name__ == '__main__':
    unittest.main()
