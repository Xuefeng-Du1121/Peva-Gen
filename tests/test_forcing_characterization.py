import unittest

from peva_sim.characterize_forcing import compare_to_training


class ForcingCharacterizationTest(unittest.TestCase):
    def test_training_match_has_zero_standardized_mean_distance(self):
        reference = {
            "u_mean": 1.0, "v_mean": -2.0, "speed_mean": 3.0,
            "u_sd": 0.5, "v_sd": 0.25, "speed_sd": 0.75,
        }
        candidate = dict(reference)
        result = compare_to_training(candidate, reference)
        self.assertAlmostEqual(result["standardized_mean_distance_l2"], 0.0)
        self.assertAlmostEqual(result["speed_sd_ratio"], 1.0)

    def test_shifted_candidate_has_positive_distance(self):
        reference = {
            "u_mean": 0.0, "v_mean": 0.0, "speed_mean": 1.0,
            "u_sd": 1.0, "v_sd": 1.0, "speed_sd": 1.0,
        }
        candidate = {**reference, "u_mean": 2.0, "speed_mean": 3.0}
        result = compare_to_training(candidate, reference)
        self.assertAlmostEqual(result["standardized_mean_distance_l2"], 2.8284271247)
        self.assertEqual(result["standardized_mean_z"], [2.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
