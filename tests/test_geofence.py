import numpy as np

from peva_sim.env import geofence_velocity


def test_geofence_reflects_only_outward_components():
    positions = np.asarray([[2.0, 50.0], [95.0, 80.0], [40.0, 40.0]])
    velocities = np.asarray([[-5.0, 3.0], [10.0, 30.0], [2.0, -4.0]])
    safe, endpoint, intervened = geofence_velocity(
        positions, velocities, dt_s=1.0, region_m=100.0)
    np.testing.assert_array_equal(
        safe, [[5.0, 3.0], [-10.0, -30.0], [2.0, -4.0]])
    np.testing.assert_array_equal(
        endpoint, [[7.0, 53.0], [85.0, 50.0], [42.0, 36.0]])
    np.testing.assert_array_equal(intervened, [True, True, False])


def test_geofence_rejects_displacement_larger_than_domain():
    with np.testing.assert_raises_regex(ValueError, "exceeds geofence"):
        geofence_velocity(
            np.asarray([[50.0, 50.0]]), np.asarray([[250.0, 0.0]]),
            dt_s=1.0, region_m=100.0)
