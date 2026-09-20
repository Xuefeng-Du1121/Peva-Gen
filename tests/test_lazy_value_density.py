import unittest
import numpy as np

from peva_sim.env import Config, MaritimeSAR


class LazyValueDensityTest(unittest.TestCase):
    def test_disabled_density_skips_kde_and_preserves_shape(self):
        env = MaritimeSAR(Config(particles=10, grid_side=4))
        env.value_density_enabled = False
        env.pvf = lambda: (_ for _ in ()).throw(AssertionError('KDE evaluated'))
        observation, _ = env.reset(1)
        np.testing.assert_array_equal(observation['value_density'], np.zeros(16))
        action = np.zeros((env.cfg.n_uavs, 2))
        observation, *_ = env.step(action)
        np.testing.assert_array_equal(observation['value_density'], np.zeros(16))

    def test_enabled_density_retains_existing_behavior(self):
        env = MaritimeSAR(Config(particles=10, grid_side=4))
        calls = []
        env.pvf = lambda: calls.append(True) or np.ones(16)
        observation, _ = env.reset(1)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(observation['value_density'], np.ones(16))


if __name__ == '__main__':
    unittest.main()
