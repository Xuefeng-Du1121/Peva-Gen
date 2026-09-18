"""Formal recurrent PEVA-Gen trainer with exact resumability."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from .belief_contract import swarm_context
from .collect_simulated_belief import target_slots
from .confirmed_environment import ConfirmedEnvironment
from .formal_mappo import FormalMAPPO, ValueNorm, critic_state, stack_rollout
from .formal_peva import (
    capture_confirmed_environment, critic_latent_gradient, fused_priority,
    restore_confirmed_environment, risk_gate, sample_chunk_starts, stable_cosine,
    weighted_clipped_value_loss,
)
from .multitarget_features import MultiTargetFeatureAdapter
from .ocean_training_pool import OceanTrainingPool
from .particle_snapshot import snapshot_environment, decoded_physics_gradient
from .ppo_math import disk_action, gae, latent_log_prob
from .protocol import file_sha
from .reproducibility import snapshot_source
from .separate_optimization import step_separate

ROOT = Path(__file__).resolve().parents[2]
VERSION = "formal_recurrent_operational_belief_geofenced_peva_gen_v6"


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_linear_lr(optimizer, base_lr, completed_updates, total_updates):
    factor = max(0.0, 1.0 - completed_updates / total_updates)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor
    return factor


def tensor(value, device):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport-aux", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--recurrent-chunk", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--aux-lr", type=float, default=3e-4)
    parser.add_argument("--rate-weight", type=float, default=1e-3)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--risk-lambda", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--fixed-beta", type=float, default=None)
    parser.add_argument("--priority-alpha", type=float, default=1.0)
    parser.add_argument("--message-weighting", choices=("uniform", "physics_grid"),
                        default="physics_grid")
    parser.add_argument("--belief-samples", type=int, default=2)
    parser.add_argument("--deterministic-belief", action="store_true")
    parser.add_argument("--disable-rsr", action="store_true")
    parser.add_argument("--disable-coverage-update", action="store_true")
    parser.add_argument("--disable-physics-context", action="store_true")
    parser.add_argument("--no-lr-decay", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    positive = ("updates", "rollout", "epochs", "minibatch", "recurrent_chunk",
                "hidden_dim", "snapshot_every", "belief_samples")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("Training counts must be positive")
    if args.deterministic_belief and args.belief_samples != 1:
        parser.error("Deterministic belief requires exactly one sample")
    if args.rollout % args.recurrent_chunk or args.minibatch % args.recurrent_chunk:
        parser.error("Rollout and minibatch must be divisible by recurrent chunk")
    if args.minibatch > args.rollout:
        parser.error("Minibatch cannot exceed rollout")
    if not 0 <= args.gae_lambda <= 1 or not 0 < args.gamma <= 1:
        parser.error("Invalid GAE parameters")
    for key in ("entropy_coef", "alignment_weight", "risk_lambda", "rate_weight"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(key + " must be finite and nonnegative")
    for key in ("actor_lr", "critic_lr", "aux_lr", "priority_alpha"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(key + " must be finite and positive")
    if not 0 <= args.beta_min <= 1:
        parser.error("beta-min must be within [0,1]")
    if args.fixed_beta is not None and not 0 <= args.fixed_beta <= 1:
        parser.error("fixed-beta must be within [0,1]")
    if args.checkpoint_every_steps < 0 or args.stop_after_steps < 0:
        parser.error("Checkpoint steps must be nonnegative")
    return args


def checkpoint_contract(args):
    ignored = {"resume", "stop_after_steps", "checkpoint_every_steps"}
    return {key: value for key, value in vars(args).items() if key not in ignored}


def inventory_training_contract(record):
    """Return fields that determine training semantics, excluding freeze wrappers."""
    keys = (
        "field_sha256", "spec_sha256", "source_sha256",
        "config", "scenarios", "audit",
    )
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(
            "Inventory lacks training-contract fields: " + ", ".join(missing))
    return {key: record[key] for key in keys}


def main(argv=None):
    args = parse_args(argv)
    out = (ROOT / args.out).resolve()
    inventory_path = (ROOT / args.inventory).resolve()
    auxiliary_path = (ROOT / args.transport_aux).resolve()
    if any(not path.is_relative_to(ROOT)
           for path in (out, inventory_path, auxiliary_path)):
        raise SystemExit("Paths must stay under project root")
    resume_path = (ROOT / args.resume).resolve() if args.resume else None
    if resume_path is not None and not resume_path.is_relative_to(ROOT):
        raise SystemExit("Resume checkpoint must stay under project root")
    if resume_path is None:
        if out.exists():
            raise SystemExit("Refusing to overwrite output directory")
        out.mkdir(parents=True)
    elif not out.is_dir() or not resume_path.is_file():
        raise SystemExit("Resume requires existing output and checkpoint")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    base_env = OceanTrainingPool(inventory_path)
    base_env.coverage_update_enabled = not args.disable_coverage_update
    env = ConfirmedEnvironment(base_env)
    cfg = env.cfg
    adapter = MultiTargetFeatureAdapter(
        auxiliary_path, cfg, seed=args.seed + 123000,
        samples=args.belief_samples, message_weighting=args.message_weighting,
        device=device,
        deterministic=args.deterministic_belief,
        physics_context=not args.disable_physics_context)
    source_forcing = adapter.metadata["source_metadata"]["forcing"]
    source_inventory_path = Path(source_forcing["inventory_path"])
    if (not source_inventory_path.is_file()
            or file_sha(source_inventory_path) != source_forcing["inventory_sha256"]):
        raise SystemExit("Auxiliary source inventory is unavailable or changed")
    source_inventory = json.loads(source_inventory_path.read_text())
    current_inventory = json.loads(inventory_path.read_text())
    if (inventory_training_contract(source_inventory)
            != inventory_training_contract(current_inventory)):
        raise SystemExit(
            "Auxiliary and policy inventory training semantics differ")
    adapter.model.requires_grad_(True)
    adapter.model.codec.freeze()
    episode_rng = np.random.default_rng(args.seed + 90000)
    priority_rng = np.random.default_rng(args.seed + 3345000)

    def reset_episode():
        seed = int(episode_rng.integers(0, 2**31 - 1))
        observation, _ = env.reset(seed)
        return observation


    obs = reset_episode()
    state_dim = len(critic_state(env))
    model = FormalMAPPO(
        adapter.feature_dim, state_dim, args.hidden_dim,
        role_dim=cfg.n_uavs).to(device)
    value_norm = ValueNorm().to(device)
    actor_parameters = list(model.actor.parameters())
    critic_parameters = list(model.critic.parameters())
    auxiliary_parameters = [
        parameter for parameter in adapter.model.parameters()
        if parameter.requires_grad
    ]
    actor_optimizer = torch.optim.Adam(
        actor_parameters, lr=args.actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.Adam(
        critic_parameters, lr=args.critic_lr, eps=1e-5)
    auxiliary_optimizer = torch.optim.Adam(
        auxiliary_parameters, lr=args.aux_lr, eps=1e-5)
    sampling_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71003)

    metadata = {
        "version": VERSION,
        "status": "formal candidate; publication requires competence/protocol gates",
        "args": vars(args),
        "contract": checkpoint_contract(args),
        "config": env.config_dict(),
        "device": str(device),
        "policy_parameters": sum(p.numel() for p in model.parameters()),
        "auxiliary_parameters": sum(p.numel() for p in auxiliary_parameters),
        "actor_inputs": (f"observable {adapter.feature_dim}-D cached transport-belief features; "
                         "includes belief-derived relative candidate locations, "
                         "normalized reach times, and one-hot UAV role suffix"),
        "operational_feature_contract": (
            "decoded belief samples only; candidate mean relative to own UAV "
            "plus straight-line reach time normalized by mission horizon; no "
            "privileged target coordinates"),
        "actor_role_conditioning": (
            "fully shared encoder/GRU/action head plus one learned 2-D bias per "
            "fixed mission role; role suffix is observable at deployment"),
        "critic_inputs": "privileged global state; training only",
        "deployment_fusion": ("actor-only deployment; no critic gradient or beta; "
                               "physics-conditioned belief remains observable"),
        "action_feasibility": (
            "shared rectangular-geofence safety shield reflects outward normal "
            "velocity components before execution"),
        "training_fusion": ("training-only adaptive reliability governs priority from "
                            "critic-physics alignment and TD EMA"),
        "recurrent_actor": {"type": "GRUCell", "hidden_dim": args.hidden_dim,
                            "reset_mask": "zero at mission boundary"},
        "prioritized_replay": (
            "fresh on-policy recurrent chunks; frozen draw distribution per update; "
            "exact chunk-level change-of-measure correction"),
        "communication": {
            "peer_payload_bytes": 42,
            "peer_header_bytes": 16,
            "total_peer_packet_bytes": 58,
            "confirmation_protocol": env.protocol_version,
        },
        "value_normalization": "streaming returns; raw metrics unchanged",
        "lr_schedule": "constant" if args.no_lr_decay else "linear by PPO update",
        "action_distribution": "Normal latent with bijective disk transform",
        "field_sha256": file_sha(Path(env.field.path)),
        "forcing": env.forcing_metadata(),
        "inventory_sha256": file_sha(inventory_path),
        "auxiliary_sha256": file_sha(auxiliary_path),
        "auxiliary_metadata": adapter.metadata,
        "source_sha256": file_sha(__file__),
        "exact_resume_scope": (
            "policy,auxiliary,optimizers,ValueNorm,all RNGs,confirmed environment,"
            "episode,RNN,unfinished rollout,reliability EMA"),
    }

    update = 0
    total_steps = 0
    rollout_buffer = []
    completed = []
    episode_return = 0.0
    ema_td = None
    actor_hidden = model.actor.initial_state(cfg.n_uavs, device)
    actor_mask = torch.zeros(cfg.n_uavs, 1, device=device)
    started = time.perf_counter()

    if resume_path is None:
        metadata["source_archive_sha256"] = snapshot_source(out)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    else:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get("version") != VERSION:
            raise ValueError("Unsupported exact-resume checkpoint")
        if saved["contract"] != checkpoint_contract(args):
            raise ValueError("Resume arguments differ from checkpoint contract")
        if saved["metadata"]["inventory_sha256"] != file_sha(inventory_path):
            raise ValueError("Inventory changed since checkpoint")
        if saved["metadata"]["auxiliary_sha256"] != file_sha(auxiliary_path):
            raise ValueError("Auxiliary checkpoint changed since checkpoint")
        model.load_state_dict(saved["model"])
        value_norm.load_state_dict(saved["value_norm"])
        adapter.model.load_state_dict(saved["auxiliary_model"])
        actor_optimizer.load_state_dict(saved["actor_optimizer"])
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        auxiliary_optimizer.load_state_dict(saved["auxiliary_optimizer"])
        update = int(saved["update"])
        total_steps = int(saved["environment_steps"])
        rollout_buffer = deepcopy(saved["rollout_buffer"])
        completed = deepcopy(saved["completed_episodes"])
        episode_return = float(saved["episode_return"])
        ema_td = saved["ema_td"]
        episode_rng.bit_generator.state = deepcopy(saved["episode_rng"])
        priority_rng.bit_generator.state = deepcopy(saved["priority_rng"])
        obs = restore_confirmed_environment(env, saved["environment"])
        actor_hidden = saved["actor_hidden"].to(device)
        actor_mask = saved["actor_mask"].to(device)
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_cpu_rng"].cpu())
        if device.type == "cuda" and saved["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in saved["torch_cuda_rng"]])
        sampling_generator.set_state(saved["sampling_generator"].cpu())
        adapter.rng.set_state(saved["auxiliary_generator"].cpu())
        metadata = saved["metadata"]
        with (out / "resume-events.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "checkpoint": str(resume_path.relative_to(ROOT)),
                "environment_steps": total_steps,
                "update": update,
                "rollout_length": len(rollout_buffer),
            }) + "\n")


    def exact_payload():
        return {
            "version": VERSION,
            "contract": checkpoint_contract(args),
            "metadata": metadata,
            "model": model.state_dict(),
            "value_norm": value_norm.state_dict(),
            "auxiliary_model": adapter.model.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "auxiliary_optimizer": auxiliary_optimizer.state_dict(),
            "update": update,
            "environment_steps": total_steps,
            "rollout_buffer": deepcopy(rollout_buffer),
            "completed_episodes": deepcopy(completed),
            "episode_return": episode_return,
            "ema_td": ema_td,
            "episode_rng": deepcopy(episode_rng.bit_generator.state),
            "priority_rng": deepcopy(priority_rng.bit_generator.state),
            "environment": capture_confirmed_environment(env),
            "actor_hidden": actor_hidden.detach().cpu(),
            "actor_mask": actor_mask.detach().cpu(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
            "sampling_generator": sampling_generator.get_state(),
            "auxiliary_generator": adapter.rng.get_state(),
        }

    def save_exact(path=None):
        atomic_torch_save(exact_payload(), path or out / "resume.pt")

    def inference_payload():
        return {
            "model": model.state_dict(),
            "value_norm": value_norm.state_dict(),
            "auxiliary_model": adapter.model.state_dict(),
            "metadata": metadata,
            "update": update,
            "environment_steps": total_steps,
            "ema_td": ema_td,
        }

    while update < args.updates:
        rollout_reward = sum(float(row["r"]) for row in rollout_buffer)
        adapter.model.eval()
        while len(rollout_buffer) < args.rollout:
            context = swarm_context(obs, cfg)
            target = target_slots(env)
            with torch.no_grad():
                target_z = adapter.model.codec.encode_target(
                    tensor(target, device)[None])
            snapshot = snapshot_environment(
                cfg, env.particles, env.weights, env.found, env.t)
            physics_gradient = decoded_physics_gradient(
                adapter.model.codec, target_z, snapshot)[0].cpu().numpy()
            x = adapter.step(env, obs)
            received = np.zeros((cfg.n_uavs, cfg.n_uavs), dtype=bool)
            for receiver, entries in enumerate(adapter.last["received"]):
                for sender, _ in entries:
                    received[receiver, sender] = True
            state = critic_state(env)
            with torch.no_grad():
                hidden_before = actor_hidden.clone()
                mask_before = actor_mask.clone()
                mu, logstd, actor_hidden = model.actor.step(
                    tensor(x, device), actor_hidden, actor_mask)
                raw = mu + logstd.exp() * torch.randn(
                    mu.shape, dtype=mu.dtype, device=device,
                    generator=sampling_generator)
                logprob = latent_log_prob(mu, logstd, raw)
                action = disk_action(raw, cfg.speed_mps).cpu().numpy()
                value_raw = value_norm.denormalize(
                    model.critic(tensor(state, device)).squeeze(-1)).item()
            scenario_id = env.active_scenario_id
            next_obs, reward, terminated, truncated, info = env.step(action)
            ended = terminated or truncated
            with torch.no_grad():
                next_value_raw = (
                    0.0 if ended else value_norm.denormalize(
                        model.critic(tensor(critic_state(env), device)).squeeze(-1)
                    ).item())
            rollout_buffer.append({
                "x": x,
                "s": state,
                "raw": raw.cpu().numpy(),
                "lp": logprob.cpu().numpy(),
                "v": value_raw,
                "nv": next_value_raw,
                "r": reward,
                "boot": float(not ended),
                "cont": float(not ended),
                "hidden": hidden_before.cpu().numpy(),
                "mask": mask_before.cpu().numpy(),
                "physics_gradient": physics_gradient,
                "context": context,
                "target": target,
                "score": adapter.last["message_scores"].copy(),
                "received": received,
                "spread": adapter.last["spread"].copy(),
            })
            rollout_reward += reward
            episode_return += reward
            total_steps += 1
            obs = next_obs
            actor_mask = torch.ones(cfg.n_uavs, 1, device=device)
            if ended:
                completed.append(dict(
                    info, return_value=episode_return,
                    environment_step=total_steps, scenario_id=scenario_id))
                episode_return = 0.0
                obs = reset_episode()
                actor_hidden.zero_()
                actor_mask.zero_()
            checkpoint_due = (
                args.checkpoint_every_steps
                and total_steps % args.checkpoint_every_steps == 0)
            stopping = (
                args.stop_after_steps and total_steps >= args.stop_after_steps)
            if checkpoint_due or stopping:
                save_exact()
            if stopping:
                (out / "episodes.json").write_text(
                    json.dumps(completed, indent=2))
                env.close()
                return


        arrays = stack_rollout(rollout_buffer)
        advantage_np, returns_np = gae(
            arrays["r"], arrays["v"], arrays["nv"],
            arrays["boot"], arrays["cont"],
            gamma=args.gamma, lam=args.gae_lambda)
        advantage_np = (
            (advantage_np - advantage_np.mean()) / (advantage_np.std() + 1e-8))
        value_norm.update(tensor(returns_np, device))

        x = tensor(arrays["x"], device)
        states = tensor(arrays["s"], device)
        raw = tensor(arrays["raw"], device)
        old_logprob = tensor(arrays["lp"], device)
        advantages = tensor(advantage_np, device)
        targets_norm = value_norm.normalize(tensor(returns_np, device))
        old_values_norm = value_norm.normalize(tensor(arrays["v"], device))
        hidden_before = tensor(arrays["hidden"], device)
        masks = tensor(arrays["mask"], device)
        physics_gradient = tensor(arrays["physics_gradient"], device)
        contexts = tensor(arrays["context"], device)
        targets = tensor(arrays["target"], device)
        scores = tensor(arrays["score"], device)
        received = torch.as_tensor(
            arrays["received"], dtype=torch.bool, device=device)
        spread = tensor(arrays["spread"], device)

        with torch.no_grad():
            check_mu, check_logstd, _ = model.actor.sequence(
                x[:, None], hidden_before[0][None], masks[:, None])
            check_ratio = torch.exp(
                latent_log_prob(check_mu[:, 0], check_logstd, raw)
                - old_logprob)
            ratio_error = float((check_ratio - 1).abs().max().item())
        if ratio_error > 1e-5:
            raise RuntimeError("Stored recurrent behavior probability inconsistent")

        _, critic_gradient = critic_latent_gradient(
            model.critic, states, cfg.n_uavs, adapter.model.codec, value_norm)
        critic_gradient = critic_gradient.detach()
        alignment = float(stable_cosine(
            critic_gradient, physics_gradient).mean().item())
        td_error = (
            tensor(arrays["r"], device)
            + args.gamma * tensor(arrays["boot"], device)
            * tensor(arrays["nv"], device)
            - tensor(arrays["v"], device))
        observed_td = float(td_error.abs().mean().item())
        ema_td = (
            observed_td if ema_td is None
            else 0.95 * ema_td + 0.05 * observed_td)
        normalized_td = ema_td / (1 + ema_td)
        confidence = float(torch.sigmoid(torch.tensor(
            4 * alignment - 4 * normalized_td)).item())
        beta = args.beta_min + (1 - args.beta_min) * (1 - confidence)
        if args.fixed_beta is not None:
            beta = args.fixed_beta
        priority = fused_priority(
            critic_gradient, physics_gradient, beta,
            alpha=args.priority_alpha).detach().cpu().numpy()

        lr_factor = 1.0
        if not args.no_lr_decay:
            lr_factor = set_linear_lr(
                actor_optimizer, args.actor_lr, update, args.updates)
            set_linear_lr(
                critic_optimizer, args.critic_lr, update, args.updates)
            set_linear_lr(
                auxiliary_optimizer, args.aux_lr, update, args.updates)

        chunks_per_epoch = args.rollout // args.recurrent_chunk
        chunks_per_batch = args.minibatch // args.recurrent_chunk
        metric_rows = []
        adapter.model.train()
        for _ in range(args.epochs):
            sampled_starts, sampled_correction = sample_chunk_starts(
                priority, args.recurrent_chunk, chunks_per_epoch, priority_rng)
            for offset in range(0, chunks_per_epoch, chunks_per_batch):
                starts_np = sampled_starts[offset:offset + chunks_per_batch]
                correction_np = sampled_correction[
                    offset:offset + chunks_per_batch]
                starts = torch.as_tensor(
                    starts_np, dtype=torch.long, device=device)
                replay_correction = tensor(correction_np, device)
                rsr_correction = (torch.ones_like(replay_correction)
                                  if args.disable_rsr else replay_correction)
                step_offset = torch.arange(args.recurrent_chunk, device=device)
                indices = starts[:, None] + step_offset[None, :]
                sequence_x = x[indices].permute(1, 0, 2, 3)
                sequence_masks = masks[indices].permute(1, 0, 2, 3)
                initial_hidden = hidden_before[starts]
                mu, logstd, _ = model.actor.sequence(
                    sequence_x, initial_hidden, sequence_masks)
                batch_raw = raw[indices].permute(1, 0, 2, 3)
                batch_old_logprob = old_logprob[indices].permute(1, 0, 2)
                ratio = torch.exp(
                    latent_log_prob(mu, logstd, batch_raw)
                    - batch_old_logprob)
                batch_advantage = advantages[indices].permute(1, 0)[:, :, None]
                surrogate = torch.minimum(
                    ratio * batch_advantage,
                    ratio.clamp(0.8, 1.2) * batch_advantage).mean(dim=-1)
                flat = indices.reshape(-1)
                replay_correction_flat = replay_correction[:, None].expand(
                    -1, args.recurrent_chunk).reshape(-1)
                rsr_correction_flat = rsr_correction[:, None].expand(
                    -1, args.recurrent_chunk).reshape(-1)
                batch_spread = spread[indices].mean(dim=-1).permute(1, 0)
                gate = (torch.ones_like(batch_spread) if args.disable_rsr
                        else risk_gate(batch_spread, args.risk_lambda))
                actor_weight = rsr_correction.clamp(0.2, 5)[None, :] * gate
                actor_loss = -(actor_weight * surrogate).mean()
                entropy = torch.distributions.Normal(
                    mu, logstd.exp()).entropy().sum(-1).mean()
                actor_objective = actor_loss - args.entropy_coef * entropy


                predicted_norm = model.critic(states[flat]).squeeze(-1)
                value_loss = weighted_clipped_value_loss(
                    predicted_norm, old_values_norm[flat],
                    targets_norm[flat], rsr_correction_flat)
                _, current_critic_gradient = critic_latent_gradient(
                    model.critic, states[flat], cfg.n_uavs,
                    adapter.model.codec, value_norm, create_graph=True)
                direction_element = 1 - stable_cosine(
                    current_critic_gradient, physics_gradient[flat])
                direction_loss = direction_element.mean()
                critic_objective = (
                    value_loss + args.alignment_weight * beta * direction_loss)

                auxiliary_losses = adapter.model.loss(
                    contexts[flat], targets[flat], scores[flat],
                    received[flat], replay_correction_flat)
                auxiliary_loss = adapter.model.objective(
                    auxiliary_losses, rate_weight=args.rate_weight)
                if not torch.isfinite(
                        actor_objective + critic_objective + auxiliary_loss):
                    raise FloatingPointError("Nonfinite PEVA objective")
                auxiliary_optimizer.zero_grad()
                auxiliary_loss.backward()
                auxiliary_norm = torch.nn.utils.clip_grad_norm_(
                    auxiliary_parameters, 1.0)
                if not torch.isfinite(auxiliary_norm):
                    raise FloatingPointError("Nonfinite auxiliary gradient")
                auxiliary_optimizer.step()
                actor_norm, critic_norm = step_separate(
                    actor_objective, critic_objective,
                    actor_parameters, critic_parameters,
                    actor_optimizer, critic_optimizer)
                metric_rows.append({
                    "actor_loss": float(actor_loss.detach().item()),
                    "value_loss": float(value_loss.detach().item()),
                    "direction_loss": float(direction_loss.detach().item()),
                    "auxiliary_loss": float(auxiliary_loss.detach().item()),
                    "aux_belief": float(
                        auxiliary_losses["belief"].detach().item()),
                    "aux_relevance": float(
                        auxiliary_losses["relevance"].detach().item()),
                    "aux_rate": float(auxiliary_losses["rate"].detach().item()),
                    "risk_gate": float(gate.detach().mean().item()),
                    "correction": float(
                        rsr_correction.detach().mean().item()),
                    "replay_correction": float(replay_correction.mean().item()),
                    "actor_grad_norm": actor_norm,
                    "critic_grad_norm": critic_norm,
                    "auxiliary_grad_norm": float(auxiliary_norm),
                })

        update += 1
        rollout_buffer.clear()
        mean_metrics = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }
        row = {
            "update": update,
            "environment_steps": total_steps,
            "completed_episodes": len(completed),
            "current_episode_time_s": env.t,
            "rollout_reward": rollout_reward,
            "initial_ratio_max_error": ratio_error,
            "beta": beta,
            "confidence": confidence,
            "alignment": alignment,
            "ema_td": ema_td,
            "priority_min": float(priority.min()),
            "priority_max": float(priority.max()),
            "spread_mean": float(spread.mean().item()),
            "actor_lr": actor_optimizer.param_groups[0]["lr"],
            "critic_lr": critic_optimizer.param_groups[0]["lr"],
            "aux_lr": auxiliary_optimizer.param_groups[0]["lr"],
            "lr_factor": lr_factor,
            "elapsed_s": time.perf_counter() - started,
            **mean_metrics,
        }
        with (out / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        atomic_torch_save(inference_payload(), out / "checkpoint.pt")
        save_exact()
        if update % args.snapshot_every == 0 or update == args.updates:
            atomic_torch_save(
                inference_payload(),
                out / f"checkpoint-step-{total_steps:09d}.pt")
            save_exact(out / f"resume-step-{total_steps:09d}.pt")
            (out / "episodes.json").write_text(
                json.dumps(completed, indent=2))

    (out / "episodes.json").write_text(json.dumps(completed, indent=2))
    env.close()


if __name__ == "__main__":
    main()
