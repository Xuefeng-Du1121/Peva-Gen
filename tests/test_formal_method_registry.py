import unittest
from argparse import Namespace
from pathlib import Path
import tempfile
from unittest.mock import patch
from peva_sim.run_formal_study import METHOD_FLAGS
from peva_sim import run_formal_study


class FormalMethodRegistryTest(unittest.TestCase):
    def test_communication_methods_are_not_duplicate_labels(self):
        self.assertIn('--no-communication', METHOD_FLAGS['mappo'])
        self.assertNotIn('--no-communication', METHOD_FLAGS['commnet-mappo'])
        self.assertEqual(set(METHOD_FLAGS['pvf-mappo']),
                         {'--pvf', '--no-communication'})
        self.assertEqual(METHOD_FLAGS['pvf-commnet-mappo'], ('--pvf',))

    def test_legacy_alias_is_explicitly_identical(self):
        self.assertEqual(METHOD_FLAGS['no-communication'], METHOD_FLAGS['mappo'])

    def test_plan_uses_active_python_not_untracked_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / 'inventory.json'
            inventory.write_text('{"status":"frozen formal protocol"}')
            args = Namespace(
                out='test-plan', inventory=str(inventory), transport_aux=None,
                scope='formal-paper', steps=256, rollout=256, seeds=[0],
                methods=['ippo'], eval_split='validation', unlock_test=False,
                episodes_per_scenario=1, gae_lambda=.95, device='cpu')
            fake_output = Path(directory) / 'output'
            with patch.object(run_formal_study, 'validate_args',
                              return_value=fake_output):
                plan, _ = run_formal_study.build_plan(args)
            for stage in ('train', 'evaluate'):
                command = plan['jobs'][0][stage]
                self.assertEqual(command[0], run_formal_study.sys.executable)
                self.assertNotIn('run.sh', command)


if __name__ == '__main__':
    unittest.main()
