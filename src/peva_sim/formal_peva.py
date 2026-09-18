"""Formal PEVA-Gen update primitives for recurrent, prioritized MAPPO."""
from copy import deepcopy
import numpy as np
import torch

from .formal_mappo import capture_environment, restore_environment
from .value_replay import ValueReplaySampler


def critic_latent_gradient(critic, state, n_uavs, codec, value_norm,
                           create_graph=False):
    """Differentiate the raw centralized value at neural-decoded targets."""
    expected = 2 * n_uavs + 3 * codec.n_targets + 1
    if state.ndim != 2 or state.shape[1] != expected:
        raise ValueError("Critic state contract mismatch")
    target_end = 2 * n_uavs + codec.state_dim
    target = state[:, 2 * n_uavs:target_end] * 2 - 1
    latent = codec.encode_target(target).detach().requires_grad_(True)
    decoded = (codec.decode(latent) + 1) / 2
    surrogate_state = torch.cat((
        state[:, :2 * n_uavs].detach(),
        decoded,
        state[:, target_end:].detach(),
    ), dim=-1)
    surrogate = value_norm.denormalize(critic(surrogate_state).squeeze(-1))
    gradient = torch.autograd.grad(
        surrogate.sum(), latent, create_graph=create_graph)[0]
    fitted = value_norm.denormalize(critic(state.detach()).squeeze(-1))
    return fitted, gradient


def stable_cosine(first, second, epsilon=1e-6):
    denominator = (first.norm(dim=-1) * second.norm(dim=-1)).clamp_min(epsilon)
    return (first * second).sum(dim=-1) / denominator


def fused_priority(critic_gradient, physics_gradient, beta, alpha=1.0):
    if critic_gradient.shape != physics_gradient.shape:
        raise ValueError("Gradient shapes differ")
    if not np.isfinite([beta, alpha]).all() or not 0 <= beta <= 1 or alpha <= 0:
        raise ValueError("Invalid fusion inputs")
    magnitude = ((1 - beta) * critic_gradient.norm(dim=-1)
                 + beta * physics_gradient.norm(dim=-1))
    return magnitude.clamp(0.1, 10).pow(alpha)


def priority_chunks(priority, chunk_length):
    """Convert transition priorities to recurrent-chunk draw probabilities."""
    priority = np.asarray(priority, dtype=np.float64)
    if priority.ndim != 1 or len(priority) == 0:
        raise ValueError("Expected nonempty transition priorities")
    if chunk_length < 1 or len(priority) % chunk_length:
        raise ValueError("Priority length must be divisible by chunk length")
    if not np.isfinite(priority).all() or (priority <= 0).any():
        raise ValueError("Priorities must be finite and positive")
    return priority.reshape(-1, chunk_length).mean(axis=1)


def sample_chunk_starts(priority, chunk_length, number, rng):
    """Sample whole recurrent chunks and return exact change-of-measure weights."""
    chunk_priority = priority_chunks(priority, chunk_length)
    chunk_index, correction = ValueReplaySampler(chunk_priority).sample(rng, number)
    return chunk_index * chunk_length, correction


def risk_gate(spread, coefficient):
    if coefficient < 0 or not np.isfinite(coefficient):
        raise ValueError("Risk coefficient must be finite and nonnegative")
    if not torch.isfinite(spread).all() or (spread < 0).any():
        raise ValueError("Posterior spread must be finite and nonnegative")
    return 2 * torch.sigmoid(-2 * coefficient * spread)


def weighted_clipped_value_loss(prediction, old_prediction, target, weight,
                                clip=0.2):
    clipped = old_prediction + (prediction - old_prediction).clamp(-clip, clip)
    elementwise = 0.5 * torch.maximum(
        (prediction - target).square(), (clipped - target).square())
    weight = torch.as_tensor(weight, dtype=elementwise.dtype,
                             device=elementwise.device)
    if weight.shape != elementwise.shape:
        raise ValueError("Value correction shape mismatch")
    return (weight * elementwise).mean()


def capture_confirmed_environment(env):
    if not hasattr(env, "ledger") or not hasattr(env, "env"):
        raise ValueError("ConfirmedEnvironment required")
    return {
        "environment": capture_environment(env.env),
        "confirmed": env.ledger._confirmed.copy(),
        "received_payload_bytes": int(env.ledger.received_payload_bytes),
    }


def restore_confirmed_environment(env, state):
    observation = restore_environment(env.env, state["environment"])
    confirmed = np.asarray(state["confirmed"], dtype=bool)
    if confirmed.shape != (env.cfg.n_targets,):
        raise ValueError("Confirmation state shape mismatch")
    env.ledger._confirmed = confirmed.copy()
    env.ledger.received_payload_bytes = int(state["received_payload_bytes"])
    return env._observation(observation)
