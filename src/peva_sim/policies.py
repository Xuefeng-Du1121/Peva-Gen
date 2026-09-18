import numpy as np

def pvf_greedy(obs, cfg):
    """Assignment heuristic from public prior; no access to target truth."""
    points = obs["prior_grid_m"]
    utility = obs["value_density"].copy()
    actions = []
    for pos in obs["positions_m"]:
        distances = np.linalg.norm(points-pos, axis=1)
        score = utility / (1 + distances/5000)
        selected = int(np.argmax(score))
        delta = points[selected]-pos
        actions.append(delta/max(np.linalg.norm(delta), 1e-12)*cfg.speed_mps)
        # Soft exclusion avoids identical assignments, not an optimal planner.
        utility *= 1-np.exp(-np.sum((points-points[selected])**2, axis=1)/(2*1500**2))
    return np.array(actions)

def random_policy(obs, cfg, rng):
    angles = rng.uniform(0, 2*np.pi, cfg.n_uavs)
    return np.column_stack((np.cos(angles), np.sin(angles)))*cfg.speed_mps
