import json
from pathlib import Path
import tempfile
import unittest

from peva_sim.evaluation_inventory import load_evaluation_cases


class EvaluationInventoryTest(unittest.TestCase):
    def write(self, directory, name, rows):
        path = Path(directory) / name
        path.write_text(json.dumps({
            'config': {'x': 1}, 'field_sha256': 'field', 'scenarios': rows}),
            encoding='utf-8')
        return path

    @staticmethod
    def row(identifier, split, time, lat, lon):
        return {'id': identifier, 'split': split, 'support': {
            'time': time, 'latitude': lat, 'longitude': lon}}

    def test_external_ood_inventory_preserves_training_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            training = self.write(directory, 'train.json', [
                self.row('train', 'train', [0, 1], [0, 1], [0, 1])])
            evaluation = self.write(directory, 'eval.json', [
                self.row('ood', 'ood_test', [3, 4], [3, 4], [3, 4])])
            cases, provenance = load_evaluation_cases(
                training, evaluation, 'ood_test', {'x': 1}, 'field')
            self.assertEqual([row['id'] for row in cases], ['ood'])
            self.assertNotEqual(provenance['training_inventory_sha256'],
                                provenance['scenario_inventory_sha256'])

    def test_overlap_and_field_change_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            training = self.write(directory, 'train.json', [
                self.row('train', 'train', [0, 1], [0, 1], [0, 1])])
            overlap = self.write(directory, 'overlap.json', [
                self.row('ood', 'ood_test', [.5, 2], [3, 4], [3, 4])])
            with self.assertRaises(ValueError):
                load_evaluation_cases(training, overlap, 'ood_test',
                                      {'x': 1}, 'field')
            changed = json.loads(overlap.read_text())
            changed['field_sha256'] = 'other'
            overlap.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                load_evaluation_cases(training, overlap, 'ood_test',
                                      {'x': 1}, 'field')


if __name__ == '__main__':
    unittest.main()
