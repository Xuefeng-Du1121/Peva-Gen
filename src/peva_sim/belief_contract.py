"""Shared train/deployment contract. No privileged state enters context."""
import numpy as np

CONTEXT_SCHEMA = "sar-local-pvf-grid-v1"
TARGET_SCHEMA = "single-current-target-centered-region-v1"


def local_context(local, config):
    """Own xy, time, four nearest contact slots, relative PVF and PVF mass.

    Coordinates use the fixed mission frame; no longitude, dataset row index,
    true target position, or undelivered teammate observation is accepted.
    Grid ordering is the simulator's ordering, identical at train and inference.
    """
    pos = np.asarray(local["position_m"], dtype=np.float64)
    contacts = np.asarray(local["contacts_m"], dtype=np.float64).reshape(-1, 2)
    grid = np.asarray(local["prior_grid_m"], dtype=np.float64)
    value = np.asarray(local["value_density"], dtype=np.float64)
    if pos.shape != (2,) or grid.shape != (config.grid_side**2, 2) or value.shape != (len(grid),):
        raise ValueError("context geometry does not match configuration")
    if not all(np.isfinite(x).all() for x in (pos, contacts, grid, value)) or (value < 0).any():
        raise ValueError("invalid local context")
    time = float(local["time_s"])
    if not np.isfinite(time) or not 0 <= time <= config.horizon_s:
        raise ValueError("invalid mission time")
    if len(contacts):
        # Deterministic tie-breaking; input ordering cannot change the encoding.
        d2 = ((contacts-pos)**2).sum(-1)
        contacts = contacts[np.lexsort((contacts[:,1], contacts[:,0], d2))[:4]]
    slots = np.zeros((4,3), dtype=np.float64)
    for j, contact in enumerate(contacts):
        slots[j,:2] = (contact-pos)/config.region_m
        slots[j,2] = 1
    maximum = float(value.max())
    normalized = value/maximum if maximum > 0 else np.zeros_like(value)
    # Dimensionless density integral proxy, retaining survival/detection amplitude.
    cell_area = (config.region_m/config.grid_side)**2
    mass = np.log1p(value.sum()*cell_area)
    return np.concatenate((pos/config.region_m, [time/config.horizon_s],
                           slots.ravel(), normalized, [mass])).astype(np.float32)


def swarm_context(observation, config):
    """Batch wrapper uses exactly the same local builder as deployment."""
    return np.stack([local_context({
        "position_m": observation["positions_m"][i],
        "contacts_m": observation["contacts_m"][i],
        "time_s": observation["time_s"],
        "prior_grid_m": observation["prior_grid_m"],
        "value_density": observation["value_density"]}, config)
        for i in range(config.n_uavs)])


def training_target(privileged_state, config):
    """Current target only; caller must enforce training split provenance."""
    target = np.asarray(privileged_state["targets_m"], dtype=np.float64)
    if target.shape != (1,2):
        raise ValueError("v1 target contract supports exactly one target")
    if not np.isfinite(target).all() or ((target < 0) | (target > config.region_m)).any():
        raise ValueError("target outside mission frame")
    return (2*target[0]/config.region_m-1).astype(np.float32)


def decode_target(latent, config):
    return (np.asarray(latent)+1)*config.region_m/2
