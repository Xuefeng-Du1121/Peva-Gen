"""Synthetic maritime SAR; SI units throughout. No real-data claims."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class Config:
    region_m: float = 40000.
    n_uavs: int = 6
    n_targets: int = 3
    dt_s: float = 30.
    horizon_s: float = 10800.
    speed_mps: float = 20.
    sensing_m: float = 500.
    survival_tau_s: float = 9000.
    detection_p: float = .85
    packet_drop: float = .1
    radio_range_m: float = 10000.
    payload_budget_bytes: int = 64
    header_bytes: int = 16
    particles: int = 2000
    prior_sigma_m: float = 3000.
    diffusion_m2s: float = 2.
    windage: float = .02
    forecast_bias_mps: float = 0.
    clutter_mean: float = .2
    position_noise_m: float = 50.
    step_cost: float = 0.
    grid_side: int = 20
    coverage_mode: str = "conditional"  # legacy default for old checkpoint configs

    def __post_init__(self):
        if self.coverage_mode not in ("conditional", "intensity"):
            raise ValueError("coverage_mode must be conditional or intensity")
        for k in ("region_m", "n_uavs", "n_targets", "dt_s", "horizon_s",
                  "speed_mps", "sensing_m", "survival_tau_s", "particles",
                  "prior_sigma_m", "grid_side"):
            if getattr(self, k) <= 0:
                raise ValueError(k + " must be positive")
        for k in ("detection_p", "packet_drop"):
            if not 0 <= getattr(self, k) <= 1:
                raise ValueError(k + " must be in [0,1]")
        for k in ("radio_range_m", "payload_budget_bytes", "header_bytes",
                  "diffusion_m2s", "clutter_mean", "position_noise_m", "step_cost"):
            if getattr(self, k) < 0:
                raise ValueError(k + " must be nonnegative")
        if not np.isclose(self.horizon_s / self.dt_s, round(self.horizon_s / self.dt_s)):
            raise ValueError("horizon must contain an integer number of steps")

def reflect(x, size):
    y = np.mod(x, 2 * size)
    return np.where(y <= size, y, 2 * size - y)


def geofence_velocity(position, velocity, dt_s, region_m):
    """Apply a deterministic rectangular geofence safety shield.

    An outward normal component is reflected before execution.  This prevents
    clipping from creating absorbing boundary states while retaining the
    policy's tangential component and speed.  All policies, learned or
    classical, pass through the same kinematic constraint.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if position.shape != velocity.shape or position.shape[-1] != 2:
        raise ValueError("position and velocity must be matched (...,2) arrays")
    proposed = position + velocity * dt_s
    crossed = (proposed < 0) | (proposed > region_m)
    safe_velocity = np.where(crossed, -velocity, velocity)
    endpoint = position + safe_velocity * dt_s
    if np.any(endpoint < -1e-9) or np.any(endpoint > region_m + 1e-9):
        raise ValueError("One-step displacement exceeds geofence assumptions")
    return safe_velocity, np.clip(endpoint, 0, region_m), crossed.any(axis=-1)

def segment_distance(points, starts, ends):
    """Distance from P points to N swept flight segments, shape (P,N)."""
    delta = ends - starts
    offset = points[:, None, :] - starts[None, :, :]
    frac = np.sum(offset * delta[None], axis=-1) / np.maximum(np.sum(delta**2, axis=-1), 1e-12)
    closest = starts[None] + np.clip(frac, 0, 1)[..., None] * delta[None]
    return np.linalg.norm(points[:, None] - closest, axis=-1)

class MaritimeSAR:
    """Gym-style reset/step, joint continuous velocities in m/s.

    Shared coverage posterior is a synchronous service, accounted separately.
    Detection confirmation includes target identity (an explicit simplifying assumption).
    """
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.ready = False
        self.coverage_update_enabled = True

    def flow(self, xy, t):
        c = self.cfg
        x, y = xy[..., 0] / c.region_m, xy[..., 1] / c.region_m
        return np.stack((.35 + .15*np.sin(2*np.pi*y + t/7200),
                         .12*np.cos(2*np.pi*x - t/7200)), axis=-1) + c.windage*np.array([5., 1.])

    def reset(self, seed=0):
        c = self.cfg
        streams = np.random.SeedSequence(seed).spawn(4)
        self.world_rng, self.sensor_rng, self.belief_rng, self.radio_rng = [
            np.random.default_rng(s) for s in streams]
        self.t = 0.
        self.done = False
        self.ready = True
        self.datum = np.array([.5, .5])*c.region_m
        self.targets = reflect(self.datum + self.world_rng.normal(0, c.prior_sigma_m, (c.n_targets, 2)), c.region_m)
        self.particles = reflect(self.datum + self.belief_rng.normal(0, c.prior_sigma_m, (c.n_targets, c.particles, 2)), c.region_m)
        self.weights = np.full((c.n_targets, c.particles), 1/c.particles)
        self.found = np.zeros(c.n_targets, dtype=bool)
        self.detected_at = np.full(c.n_targets, np.nan)
        self.uavs = np.column_stack((np.linspace(.3, .7, c.n_uavs)*c.region_m,
                                     np.full(c.n_uavs, .3*c.region_m)))
        axis = (np.arange(c.grid_side)+.5)*c.region_m/c.grid_side
        self.grid = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        self.contacts = [np.empty((0, 2)) for _ in range(c.n_uavs)]
        self.inbox = [[] for _ in range(c.n_uavs)]
        self._exchange_stamp = None
        self.bytes = dict(peer_payload=0, peer_header=0, shared_uplink=0, shared_downlink=0)
        self.geofence_interventions = 0
        return self.observation(), self.info()

    def pvf(self):
        c = self.cfg
        # Gaussian KDE in square metres; integrated expected detection reward density.
        h = max(c.sensing_m, c.prior_sigma_m*c.particles**(-1/6))
        value = np.zeros(len(self.grid))
        for g in range(c.n_targets):
            if self.found[g]:
                continue
            for j in range(0, len(self.grid), 32):
                d = (self.grid[j:j+32, None] - self.particles[g, None])/h
                value[j:j+32] += np.exp(-.5*np.sum(d*d, -1)) @ self.weights[g] / (2*np.pi*h*h)
        return value * c.detection_p * np.exp(-self.t/c.survival_tau_s)

    def observation(self):
        # No target coordinates or hidden world state in policy observations.
        return {"positions_m": self.uavs.copy(), "time_s": self.t,
                "prior_grid_m": self.grid.copy(), "value_density": self.pvf(),
                "contacts_m": [x.copy() for x in self.contacts],
                "inbox": [list(x) for x in self.inbox]}

    def local_observations(self):
        """Per-agent inputs: own position/contacts, received messages, shared prior.

        Teammate positions and hidden target truth are not exposed.
        """
        grid_value = self.pvf()
        return {i: {"position_m": self.uavs[i].copy(), "time_s": self.t,
                    "prior_grid_m": self.grid.copy(), "value_density": grid_value.copy(),
                    "contacts_m": self.contacts[i].copy(), "inbox": list(self.inbox[i])}
                for i in range(self.cfg.n_uavs)}

    def info(self):
        c = self.cfg
        times = np.where(self.found, self.detected_at, c.horizon_s)
        return {"success_rate": float(self.found.mean()),
                "ttd_all": float(np.mean(times/c.horizon_s)),
                "survival_score": float(np.sum(np.exp(-self.detected_at[self.found]/c.survival_tau_s))/c.n_targets),
                "time_s": self.t, "bytes": dict(self.bytes),
                "geofence_interventions": int(self.geofence_interventions)}

    def state(self):
        """Privileged state for CTDE/debug only; never feed this to decentralized actors."""
        return {"targets_m": self.targets.copy(), "found": self.found.copy(),
                "uavs_m": self.uavs.copy(), "time_s": self.t}

    def advance_targets(self, next_time):
        c = self.cfg
        return reflect(self.targets + (self.flow(self.targets, self.t) + c.forecast_bias_mps)*c.dt_s
                       + self.world_rng.normal(0, np.sqrt(2*c.diffusion_m2s*c.dt_s), self.targets.shape), c.region_m)

    def exchange_messages(self, messages):
        """One pre-action radio phase per mission step, at current positions.

        No world time or sensing advances. This models a synchronous exchange
        completed within the control interval, not measured radio latency.
        Calling step(messages=...) in the same interval is prohibited.
        """
        if not self.ready or self.done:
            raise RuntimeError("reset required before communicating")
        if getattr(self, "_exchange_stamp", None) == (id(self.world_rng), self.t):
            raise RuntimeError("only one pre-action exchange per control step")
        c = self.cfg
        if len(messages) != c.n_uavs or any(not isinstance(m, bytes) or len(m)>c.payload_budget_bytes for m in messages):
            raise ValueError("messages must be bytes within per-UAV payload budget")
        self.inbox = [[] for _ in range(c.n_uavs)]
        distance = np.linalg.norm(self.uavs[:, None]-self.uavs[None, :], axis=-1)
        for sender, payload in enumerate(messages):
            if not payload:
                continue
            self.bytes["peer_payload"] += len(payload)
            self.bytes["peer_header"] += c.header_bytes
            for receiver in range(c.n_uavs):
                if sender != receiver and distance[sender,receiver] <= c.radio_range_m and self.radio_rng.random() >= c.packet_drop:
                    self.inbox[receiver].append((sender,payload))
        self._exchange_stamp = (id(self.world_rng), self.t)
        return [list(inbox) for inbox in self.inbox]

    def step(self, actions, messages=None):
        if not self.ready or self.done:
            raise RuntimeError("reset required before stepping")
        if messages is not None and getattr(self, "_exchange_stamp", None) == (id(self.world_rng), self.t):
            raise ValueError("pre-action exchange already sent this step's packets")
        c = self.cfg
        a = np.asarray(actions, dtype=float)
        if a.shape != (c.n_uavs, 2) or not np.isfinite(a).all():
            raise ValueError("actions must be finite (n_uavs,2) velocities in m/s")
        if messages is None:
            messages = [b""]*c.n_uavs
        if len(messages) != c.n_uavs or any(not isinstance(m, bytes) or len(m)>c.payload_budget_bytes for m in messages):
            raise ValueError("messages must be bytes within per-UAV payload budget")
        a = a * np.minimum(1., c.speed_mps/np.maximum(np.linalg.norm(a, axis=1), 1e-12))[:, None]
        start = self.uavs.copy()
        a, self.uavs, intervened = geofence_velocity(
            start, a, c.dt_s, c.region_m)
        self.geofence_interventions += int(intervened.sum())
        scale = np.sqrt(2*c.diffusion_m2s*c.dt_s)
        self.targets = self.advance_targets(self.t + c.dt_s)
        self.particles = reflect(self.particles + self.flow(self.particles, self.t)*c.dt_s
                                 + self.belief_rng.normal(0, scale, self.particles.shape), c.region_m)
        self.t += c.dt_s
        dist = segment_distance(self.targets, start, self.uavs)
        prob = c.detection_p*np.maximum(0, 1-(dist/c.sensing_m)**2)
        hits = self.sensor_rng.random(prob.shape) < prob
        hits[self.found] = False
        new = hits.any(axis=1)
        self.detected_at[new] = self.t
        self.found |= new
        self.contacts = []
        for i in range(c.n_uavs):
            real = self.targets[hits[:, i]] + self.sensor_rng.normal(0, c.position_noise_m, (hits[:, i].sum(), 2))
            count = self.sensor_rng.poisson(c.clutter_mean)
            theta = self.sensor_rng.uniform(0, 2*np.pi, count)
            radius = c.sensing_m*np.sqrt(self.sensor_rng.random(count))
            false = self.uavs[i] + np.column_stack((np.cos(theta), np.sin(theta)))*radius[:, None]
            self.contacts.append(np.concatenate((real, false)))
        for g in range(c.n_targets):
            if self.found[g]:
                self.weights[g].fill(0)
            elif self.coverage_update_enabled:
                d = segment_distance(self.particles[g], start, self.uavs)
                miss = np.prod(1-c.detection_p*np.maximum(0, 1-(d/c.sensing_m)**2), axis=1)
                w = self.weights[g]*miss
                total = w.sum()
                if c.coverage_mode == "intensity":
                    # Manuscript: expected undetected intensity loses mass under
                    # negative evidence. Never resurrect zero or underflowed mass.
                    self.weights[g] = w
                else:
                    # Legacy conditional posterior given a known remaining target.
                    self.weights[g] = w/total if total > 1e-300 else np.full(c.particles, 1/c.particles)
        self.inbox = [[] for _ in range(c.n_uavs)]
        radio_dist = np.linalg.norm(self.uavs[:, None]-self.uavs[None, :], axis=-1)
        for i, payload in enumerate(messages):
            if not payload:
                continue
            self.bytes["peer_payload"] += len(payload)
            self.bytes["peer_header"] += c.header_bytes
            for j in range(c.n_uavs):
                if i != j and radio_dist[i,j] <= c.radio_range_m and self.radio_rng.random() >= c.packet_drop:
                    self.inbox[j].append((i, payload))
        # Reference protocol: upload swept segments (4 float32) and confirmed IDs (uint32);
        # broadcast grid float32 plus timestamp float64. Both include packet headers.
        self.bytes["shared_uplink"] += c.n_uavs*(16+c.header_bytes) + int(new.sum())*4
        self.bytes["shared_downlink"] += len(self.grid)*4+8+c.header_bytes
        reward = float(new.sum()*np.exp(-self.t/c.survival_tau_s)
                       - c.step_cost*np.linalg.norm(a, axis=1).sum())
        terminated = bool(self.found.all())
        truncated = bool(self.t >= c.horizon_s and not terminated)
        self.done = terminated or truncated
        return self.observation(), reward, terminated, truncated, self.info()

    def config_dict(self):
        return asdict(self.cfg)
