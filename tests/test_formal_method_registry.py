import unittest
from peva_sim.run_formal_study import METHOD_FLAGS


class FormalMethodRegistryTest(unittest.TestCase):
    def test_communication_methods_are_not_duplicate_labels(self):
        self.assertIn('--no-communication', METHOD_FLAGS['mappo'])
        self.assertNotIn('--no-communication', METHOD_FLAGS['commnet-mappo'])
        self.assertEqual(set(METHOD_FLAGS['pvf-mappo']),
                         {'--pvf', '--no-communication'})
        self.assertEqual(METHOD_FLAGS['pvf-commnet-mappo'], ('--pvf',))

    def test_legacy_alias_is_explicitly_identical(self):
        self.assertEqual(METHOD_FLAGS['no-communication'], METHOD_FLAGS['mappo'])


if __name__ == '__main__':
    unittest.main()
