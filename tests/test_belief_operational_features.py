from types import SimpleNamespace

import numpy as np

from peva_sim.multitarget_features import belief_operational_features


def test_operational_features_use_decoded_belief_relative_to_each_uav():
    config = SimpleNamespace(
        n_uavs=2, n_targets=2, region_m=100.0,
        speed_mps=10.0, horizon_s=100.0)
    decoded = np.asarray([
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        [[0.0, 0.0, -1.0, -1.0], [0.0, 0.0, -1.0, -1.0]],
    ], dtype=np.float32)
    positions = np.asarray([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    features = belief_operational_features(decoded, positions, config)
    expected = np.asarray([
        [0.5, 0.5, 1.0, 1.0, np.sqrt(50**2 + 50**2) / 1000,
         np.sqrt(100**2 + 100**2) / 1000],
        [-0.5, -0.5, -1.0, -1.0, np.sqrt(50**2 + 50**2) / 1000,
         np.sqrt(100**2 + 100**2) / 1000],
    ], dtype=np.float32)
    np.testing.assert_allclose(features, expected, rtol=1e-6, atol=1e-6)


def test_operational_features_reject_nonfinite_belief():
    config = SimpleNamespace(
        n_uavs=1, n_targets=1, region_m=100.0,
        speed_mps=10.0, horizon_s=100.0)
    decoded = np.asarray([[[np.nan, 0.0]]], dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "finite"):
        belief_operational_features(decoded, np.zeros((1, 2)), config)
