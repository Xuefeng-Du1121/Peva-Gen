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
            self.assertEqual(plan['runtime'], run_formal_study.runtime_fingerprint())

    def test_frozen_plan_rejects_runtime_change(self):
        plan = {'source_sha256': {}, 'input_sha256': {},
                'runtime': {'python': 'different'}}
        with self.assertRaisesRegex(RuntimeError, 'runtime changed'):
            run_formal_study._assert_frozen(plan)

    def test_resume_command_preserves_frozen_command_and_adds_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'train' / 'ippo-seed0'
            output.mkdir(parents=True)
            (output / 'resume.pt').write_bytes(b'checkpoint')
            command = ['python', '-m', 'trainer', '--out', str(output)]
            resumed = run_formal_study._resume_train_command(command)
            self.assertEqual(command, ['python', '-m', 'trainer',
                                       '--out', str(output)])
            self.assertEqual(resumed[-2:], ['--resume', str(output / 'resume.pt')])

    def test_execute_runs_independent_jobs_concurrently(self):
        jobs = [
            {'id': f'ippo-seed{seed}', 'method': 'ippo', 'seed': seed,
             'train': ['python', '--out', f'train-{seed}'],
             'evaluate': ['python', '--out', f'eval-{seed}']}
            for seed in range(3)
        ]
        plan = {'jobs': jobs, 'source_sha256': {}, 'input_sha256': {}}
        active = 0
        peak = 0
        lock = run_formal_study.threading.Lock()

        class Process:
            pid = 123

            def wait(self):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                run_formal_study.time.sleep(.02)
                with lock:
                    active -= 1
                return 0

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_formal_study.subprocess, 'Popen',
                             side_effect=lambda *args, **kwargs: Process()), \
                patch.object(run_formal_study, '_validate_stage_artifact'), \
                patch.object(run_formal_study, '_assert_frozen'):
            run_formal_study.execute(plan, Path(directory), max_parallel=3)
            self.assertGreaterEqual(peak, 2)

    def test_rejects_nonpositive_parallelism(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'positive'):
                run_formal_study.execute(
                    {'jobs': [], 'source_sha256': {}, 'input_sha256': {}},
                    Path(directory), max_parallel=0)


if __name__ == '__main__':
    unittest.main()
