"""Publication MAPPO primitives: recurrent decentralized actor and ValueNorm."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn

from .communication import MESSAGE_DIM, encode_message
from .joint_transport import quantize_message


def features(obs, cfg, use_pvf):
    """Matched decentralized inputs with one-hot roles for symmetry breaking."""
    rows = []
    for agent in range(cfg.n_uavs):
        pos = np.asarray(obs["positions_m"][agent], dtype=np.float32)
        contacts = np.asarray(obs["contacts_m"][agent], dtype=np.float32)
        if len(contacts):
            order = np.argsort(np.linalg.norm(contacts - pos, axis=1))[:4]
            contacts = contacts[order]
        slots = np.zeros((4, 3), dtype=np.float32)
        for slot, contact in enumerate(contacts):
            slots[slot, :2] = (contact - pos) / cfg.region_m
            slots[slot, 2] = 1.0
        field = np.asarray(obs["value_density"], dtype=np.float32)
        if use_pvf:
            field = field / max(float(field.max()), 1e-12)
        else:
            field = np.zeros_like(field)
        role = np.eye(cfg.n_uavs, dtype=np.float32)[agent]
        rows.append(np.concatenate((
            pos / cfg.region_m,
            np.asarray([obs["time_s"] / cfg.horizon_s], dtype=np.float32),
            slots.ravel(), field, role)))
    return np.asarray(rows, dtype=np.float32)


def critic_state(env):
    state, cfg = env.state(), env.cfg
    return np.concatenate((
        state["uavs_m"].ravel() / cfg.region_m,
        state["targets_m"].ravel() / cfg.region_m,
        state["found"].astype(np.float32),
        np.asarray([state["time_s"] / cfg.horizon_s], dtype=np.float32)
    )).astype(np.float32)


class ValueNorm(nn.Module):
    """Streaming scalar normalization with checkpointable float64 statistics."""
    def __init__(self, epsilon=1e-5, clip=10.0):
        super().__init__()
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("var", torch.ones((), dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values):
        values = torch.as_tensor(values, dtype=torch.float64, device=self.mean.device).reshape(-1)
        if values.numel() == 0:
            return
        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_count = torch.as_tensor(values.numel(), dtype=torch.float64, device=values.device)
        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
            self.count.copy_(batch_count)
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        merged_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count
        m2 = m2 + delta.square() * self.count * batch_count / total
        self.mean.copy_(merged_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def normalize(self, values):
        values = torch.as_tensor(values)
        mean = self.mean.to(dtype=values.dtype, device=values.device)
        scale = torch.sqrt(self.var.to(dtype=values.dtype, device=values.device) + self.epsilon)
        return ((values - mean) / scale).clamp(-self.clip, self.clip)

    def denormalize(self, values):
        values = torch.as_tensor(values)
        mean = self.mean.to(dtype=values.dtype, device=values.device)
        scale = torch.sqrt(self.var.to(dtype=values.dtype, device=values.device) + self.epsilon)
        return values * scale + mean


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128, learned_comm=False, role_dim=0):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.role_dim = int(role_dim)
        if self.role_dim < 0 or self.role_dim > self.obs_dim:
            raise ValueError("role_dim must be between zero and obs_dim")
        self.message_dim = MESSAGE_DIM if learned_comm else 0
        if learned_comm:
            self.message_head = nn.Linear(obs_dim, MESSAGE_DIM)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + self.message_dim, hidden_dim), nn.Tanh())
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)
        if self.role_dim:
            self.role_bias = nn.Parameter(torch.empty(self.role_dim, 2))
        else:
            self.register_parameter("role_bias", None)
        self.logstd = nn.Parameter(torch.full((2,), -0.5))

    def initial_state(self, n_agents, device=None):
        return torch.zeros(n_agents, self.hidden_dim, device=device)

    def encoded_messages(self, obs):
        if not self.message_dim:
            raise RuntimeError("This actor has no learned communication head")
        return quantize_message(torch.tanh(self.message_head(obs)))

    def _fuse_messages(self, obs, communication_mask):
        if not self.message_dim:
            if communication_mask is not None:
                raise ValueError("Communication mask supplied to non-communicating actor")
            return obs
        if communication_mask is None:
            raise ValueError("Learned communication requires a delivery mask")
        squeeze = obs.ndim == 2
        batched_obs = obs[None] if squeeze else obs
        batched_mask = (torch.as_tensor(
            communication_mask, dtype=obs.dtype, device=obs.device)[None]
            if squeeze else torch.as_tensor(
                communication_mask, dtype=obs.dtype, device=obs.device))
        if batched_obs.ndim != 3 or batched_mask.shape != (
                batched_obs.shape[0], batched_obs.shape[1], batched_obs.shape[1]):
            raise ValueError("Communication mask must be [batch,receiver,sender]")
        messages = self.encoded_messages(batched_obs)
        denominator = batched_mask.sum(-1, keepdim=True).clamp_min(1.)
        aggregate = torch.einsum("brs,bsd->brd", batched_mask, messages)
        fused = torch.cat((batched_obs, aggregate / denominator), dim=-1)
        return fused[0] if squeeze else fused

    def _action_mean(self, hidden, obs):
        """Use one shared actor with a minimal role-specific output bias.

        The one-hot role suffix is observable onboard and fixed for a mission.
        The two-parameter bias breaks symmetry directly while every transition
        still trains the same encoder, recurrent trunk, and action head.
        """
        mean = self.head(hidden)
        if not self.role_dim:
            return mean
        roles = obs[..., -self.role_dim:]
        if not torch.isfinite(roles).all():
            raise ValueError("Non-finite role features")
        if not torch.allclose(roles.sum(-1), torch.ones_like(roles[..., 0]),
                              atol=1e-5, rtol=0.):
            raise ValueError("Role suffix must be one-hot")
        if bool(((roles < -1e-6) | (roles > 1. + 1e-6)).any()):
            raise ValueError("Role suffix must be one-hot")
        return mean + roles @ self.role_bias

    def step(self, obs, hidden, mask, communication_mask=None):
        mask = torch.as_tensor(mask, dtype=obs.dtype, device=obs.device)
        if obs.ndim == 2 and mask.ndim == 1:
            mask = mask[:, None]
        fused = self._fuse_messages(obs, communication_mask)
        shape = fused.shape[:-1]
        encoded = self.encoder(fused.reshape(-1, fused.shape[-1]))
        flat_hidden = hidden.reshape(-1, self.hidden_dim)
        flat_mask = mask.reshape(-1, 1)
        flat_hidden = self.gru(encoded, flat_hidden * flat_mask)
        hidden = flat_hidden.reshape(*shape, self.hidden_dim)
        return self._action_mean(hidden, obs), self.logstd.clamp(-4, 1), hidden

    def sequence(self, obs, initial_hidden, masks, communication_masks=None):
        if obs.ndim != 4:
            raise ValueError("Expected [time,batch,agent,feature] observations")
        if self.message_dim and communication_masks is None:
            raise ValueError("Communicating sequence requires delivery masks")
        hidden = initial_hidden
        outputs = []
        for step in range(obs.shape[0]):
            delivery = (None if communication_masks is None
                        else communication_masks[step])
            mu, _, hidden = self.step(
                obs[step], hidden, masks[step], delivery)
            outputs.append(mu)
        return torch.stack(outputs), self.logstd.clamp(-4, 1), hidden


class FormalMAPPO(nn.Module):
    def __init__(self, obs_dim, state_dim, hidden_dim=128, learned_comm=False,
                 role_dim=0):
        super().__init__()
        self.actor = RecurrentActor(
            obs_dim, hidden_dim, learned_comm, role_dim=role_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.head.weight, 0.01)
        if self.actor.role_bias is not None:
            nn.init.normal_(self.actor.role_bias, mean=0., std=0.01)
        if learned_comm:
            nn.init.orthogonal_(self.actor.message_head.weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)


@torch.no_grad()
def exchange_actor_messages(actor, observations, env):
    """Transmit actual 42-byte packets and return receiver-by-sender delivery."""
    messages = actor.encoded_messages(observations).detach().cpu().numpy()
    packets = [encode_message(message, relevance=1.) for message in messages]
    inboxes = env.exchange_messages(packets)
    agents = len(packets)
    delivery = np.eye(agents, dtype=np.float32)
    for receiver, inbox in enumerate(inboxes):
        for sender, payload in inbox:
            if payload != packets[sender]:
                raise ValueError("Delivered payload differs from transmitted packet")
            delivery[receiver, sender] = 1.
    return torch.as_tensor(
        delivery, dtype=observations.dtype, device=observations.device), packets


def capture_environment(env):
    """Capture every mutable simulator component needed for exact CPU resume."""
    names = ("t", "done", "ready", "datum", "targets", "particles", "weights",
             "found", "detected_at", "uavs", "grid", "contacts", "inbox", "bytes",
             "geofence_interventions",
             "truth_diffusion_m2s", "truth_drift_error_mps", "active_scenario_id")
    state = {name: deepcopy(getattr(env, name)) for name in names if hasattr(env, name)}
    state["rng"] = {name: deepcopy(getattr(env, name).bit_generator.state)
                    for name in ("world_rng", "sensor_rng", "belief_rng", "radio_rng")}
    state["exchange_at_current"] = (
        getattr(env, "_exchange_stamp", None) == (id(env.world_rng), env.t))
    return state


def restore_environment(env, state):
    """Restore an OceanTrainingPool without serializing its open NetCDF handle."""
    env.reset_for_training_scenario(0, state["active_scenario_id"])
    for name, value in state.items():
        if name not in ("rng", "exchange_at_current"):
            setattr(env, name, deepcopy(value))
    for name, rng_state in state["rng"].items():
        getattr(env, name).bit_generator.state = deepcopy(rng_state)
    env._exchange_stamp = ((id(env.world_rng), env.t)
                           if state.get("exchange_at_current") else None)
    return env.observation()


def stack_rollout(buffer):
    if not buffer:
        raise ValueError("Empty rollout")
    keys = buffer[0].keys()
    if any(row.keys() != keys for row in buffer):
        raise ValueError("Inconsistent rollout records")
    return {key: np.asarray([row[key] for row in buffer]) for key in keys}


def sequence_batches(length, chunk_length, minibatch, generator, device):
    if length % chunk_length or minibatch % chunk_length:
        raise ValueError("rollout and minibatch must be divisible by recurrent chunk length")
    starts = torch.arange(0, length, chunk_length, device=device)
    chunks_per_batch = minibatch // chunk_length
    order = starts[torch.randperm(len(starts), generator=generator, device=device)]
    return list(order.split(chunks_per_batch))
