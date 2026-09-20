import unittest
import numpy as np
from peva_sim.belief_diagnostics import particle_diagnostics, summarize_particles


class ParticleDiagnosticsTest(unittest.TestCase):
    def test_scale_invariance_and_missing_targets(self):
        particles = np.array([[[0., 0.], [4., 0.]]] * 3)
        weights = np.array([[1., 3.], [0., 0.], [1., 3.]])
        truth = np.array([[3., 0.]] * 3)
        found = [False, False, True]
        a = particle_diagnostics(particles, weights, truth, found)
        b = particle_diagnostics(particles, weights * 1e-300, truth, found)
        for index in (0, 1, 2, 4):
            np.testing.assert_allclose(a[index], b[index], equal_nan=True)
        self.assertEqual(a[1][0], 0.)
        self.assertAlmostEqual(a[2][0], 1.6)
        self.assertTrue(np.isnan(a[2][1:]).all())
        report = summarize_particles([a[1]], [a[2]], [a[3]], 2)
        self.assertEqual(report['valid_target_steps'], 1)
        self.assertAlmostEqual(report['effective_sample_size_fraction'], .8)

    def test_empty_posterior_is_not_a_perfect_prediction(self):
        a = particle_diagnostics(np.zeros((1, 2, 2)), np.zeros((1, 2)),
                                 np.zeros((1, 2)), [False])
        report = summarize_particles([a[1]], [a[2]], [a[3]], 2)
        self.assertIsNone(report['posterior_mean_error_m'])
        self.assertEqual(report['valid_target_steps'], 0)


if __name__ == '__main__':
    unittest.main()
