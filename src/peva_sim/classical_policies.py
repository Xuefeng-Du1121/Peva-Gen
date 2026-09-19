"""Deterministic non-learning policies for the common SAR evaluator."""
import numpy as np


def _unit_velocity(delta, speed):
    delta = np.asarray(delta, dtype=np.float64)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return (delta / np.maximum(norm, 1e-12) * speed).astype(np.float64)


def _assign_grid(observation, cfg, utility, travel_scale_m=5000.0):
    points = np.asarray(observation["prior_grid_m"], dtype=np.float64)
    positions = np.asarray(observation["positions_m"], dtype=np.float64)
    available = np.asarray(utility, dtype=np.float64).copy()
    if available.shape != (len(points),) or not np.isfinite(available).all():
        raise ValueError("Invalid grid utility")
    actions = []
    exclusion = max(1.5 * cfg.sensing_m, cfg.region_m / cfg.grid_side)
    for position in positions:
        distance = np.linalg.norm(points - position, axis=1)
        score = available / (1 + distance / travel_scale_m)
        selected = int(np.argmax(score))
        actions.append(points[selected] - position)
        radius2 = np.sum((points - points[selected]) ** 2, axis=1)
        available *= 1 - np.exp(-radius2 / (2 * exclusion ** 2))
    return _unit_velocity(np.asarray(actions), cfg.speed_mps)


class GreedyProbability:
    name = "greedy-probability"

    def reset(self, observation):
        return None

    def act(self, observation, cfg, state=None):
        factor = cfg.detection_p * np.exp(
            -observation["time_s"] / cfg.survival_tau_s)
        probability_density = (
            np.asarray(observation["value_density"], dtype=np.float64)
            / max(float(factor), np.finfo(np.float64).tiny))
        return _assign_grid(observation, cfg, probability_density)


class PVFGreedy:
    name = "pvf-greedy"

    def reset(self, observation):
        return None

    def act(self, observation, cfg, state=None):
        return _assign_grid(
            observation, cfg,
            np.asarray(observation["value_density"], dtype=np.float64))


class BayesianInformationGain:
    name = "bayesian-information-gain"

    def reset(self, observation):
        return None

    def act(self, observation, cfg, state=None):
        factor = cfg.detection_p * np.exp(
            -observation["time_s"] / cfg.survival_tau_s)
        density = (
            np.asarray(observation["value_density"], dtype=np.float64)
            / max(float(factor), np.finfo(np.float64).tiny))
        cell_area = (cfg.region_m / cfg.grid_side) ** 2
        occupancy = np.clip(density * cell_area, 1e-12, 1 - 1e-12)
        entropy = -(occupancy * np.log(occupancy)
                    + (1 - occupancy) * np.log1p(-occupancy))
        return _assign_grid(observation, cfg, entropy)


class LawnmowerCoverage:
    name = "lawnmower-coverage"

    def __init__(self):
        self.waypoints = None
        self.indices = None

    def reset(self, observation):
        axis = np.unique(np.asarray(observation["prior_grid_m"])[:, 0])
        spacing = axis[1] - axis[0]
        cfg_region = float(axis[-1] + spacing / 2)
        n_agents = len(observation["positions_m"])
        margin = cfg_region / 40
        lane_spacing = cfg_region / 50
        self.waypoints = []
        for agent in range(n_agents):
            low = agent * cfg_region / n_agents
            high = (agent + 1) * cfg_region / n_agents
            ys = np.arange(low + lane_spacing / 2, high, lane_spacing)
            points = []
            for index, y in enumerate(ys):
                endpoints = (margin, cfg_region - margin)
                if index % 2:
                    endpoints = endpoints[::-1]
                points.extend([(endpoints[0], y), (endpoints[1], y)])
            self.waypoints.append(np.asarray(points, dtype=np.float64))
        self.indices = np.zeros(n_agents, dtype=np.int64)

    def act(self, observation, cfg, state=None):
        if self.waypoints is None:
            self.reset(observation)
        positions = np.asarray(observation["positions_m"], dtype=np.float64)
        targets = []
        threshold = cfg.speed_mps * cfg.dt_s * 1.5
        for agent, position in enumerate(positions):
            path = self.waypoints[agent]
            index = int(self.indices[agent])
            if np.linalg.norm(path[index] - position) <= threshold:
                index = (index + 1) % len(path)
                self.indices[agent] = index
            targets.append(path[index])
        return _unit_velocity(np.asarray(targets) - positions, cfg.speed_mps)


class CBBAPVF:
    """Deterministic CBBA-style auction over value-grid task cells.

    This is a classical coordination baseline: agents greedily bid for
    distinct high-value cells using travel-adjusted PVF utility. It has no
    learned policy or privileged target state.
    """

    name = "cbba"

    def reset(self, observation):
        return None

    def act(self, observation, cfg, state=None):
        points = np.asarray(observation["prior_grid_m"], dtype=np.float64)
        positions = np.asarray(observation["positions_m"], dtype=np.float64)
        utility = np.asarray(observation["value_density"], dtype=np.float64)
        if utility.shape != (len(points),) or not np.isfinite(utility).all():
            raise ValueError("Invalid grid utility")
        distance = np.linalg.norm(
            points[None, :, :] - positions[:, None, :], axis=-1)
        bids = utility[None, :] / (1.0 + distance / 5000.0)
        available = np.ones(len(points), dtype=bool)
        assignments = []
        exclusion = max(1.5 * cfg.sensing_m,
                        cfg.region_m / cfg.grid_side)
        for agent in np.argsort(-np.max(bids, axis=1)):
            scores = bids[agent].copy()
            scores[~available] = -np.inf
            selected = int(np.argmax(scores))
            assignments.append((int(agent), selected))
            radius2 = np.sum((points - points[selected]) ** 2, axis=1)
            available &= radius2 > exclusion ** 2
            if not available.any():
                available[:] = True
        targets = np.empty_like(positions)
        for agent, selected in assignments:
            targets[agent] = points[selected]
        return _unit_velocity(targets - positions, cfg.speed_mps)


class OracleTarget:
    name = "oracle-target"

    def reset(self, observation):
        return None

    def act(self, observation, cfg, state=None):
        if state is None:
            raise ValueError("Oracle requires privileged state")
        positions = np.asarray(observation["positions_m"], dtype=np.float64)
        targets = np.asarray(state["targets_m"], dtype=np.float64)
        found = np.asarray(state["found"], dtype=bool)
        active = targets[~found]
        if len(active) == 0:
            return np.zeros_like(positions)
        assigned = active[np.arange(len(positions)) % len(active)]
        return _unit_velocity(assigned - positions, cfg.speed_mps)


POLICIES = {
    policy.name: policy for policy in (
        GreedyProbability, PVFGreedy, BayesianInformationGain,
        LawnmowerCoverage, CBBAPVF, OracleTarget)
}


def make_policy(name):
    try:
        return POLICIES[name]()
    except KeyError as error:
        raise ValueError("Unknown classical policy: " + name) from error
