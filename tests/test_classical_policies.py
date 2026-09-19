import numpy as np

from peva_sim.classical_policies import CBBAPVF, make_policy


class _Cfg:
    sensing_m = 500.0
    region_m = 40000.0
    grid_side = 20
    speed_mps = 20.0


def _observation():
    return {
        "prior_grid_m": np.asarray(
            [[1000.0, 1000.0], [39000.0, 1000.0],
             [1000.0, 39000.0], [39000.0, 39000.0]]),
        "positions_m": np.asarray(
            [[2000.0, 2000.0], [38000.0, 2000.0],
             [2000.0, 38000.0], [38000.0, 38000.0]]),
        "value_density": np.asarray([1.0, .9, .8, .7]),
    }


def test_cbba_is_registered_and_returns_bounded_actions():
    policy = make_policy("cbba")
    action = policy.act(_observation(), _Cfg())
    assert isinstance(policy, CBBAPVF)
    assert action.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(action, axis=1), _Cfg.speed_mps)
