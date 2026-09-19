"""Formal recurrent MAPPO/PVF+MAPPO trainer with exact resumability."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .formal_mappo import (
    FormalMAPPO, ValueNorm, capture_environment, critic_state, features,
    exchange_actor_messages, restore_environment, sequence_batches, stack_rollout,
)
from .ocean_training_pool import OceanTrainingPool
from .ppo_math import disk_action, gae, latent_log_prob
from .protocol import file_sha
from .reproducibility import snapshot_source
from .separate_optimization import clipped_value_loss, step_separate

ROOT = Path(__file__).resolve().parents[2]
VERSION = "formal_recurrent_shared_role_bias_geofenced_mappo_v5"


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
    parser.add_argument("--no-lr-decay", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pvf", action="store_true")
    parser.add_argument("--no-communication", action="store_true",
                        help="train the matched decentralized no-radio actor")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--stop-after-steps", type=int, default=0,
                        help="Testing/maintenance stop after writing an exact-resume checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    positive = ("updates", "rollout", "epochs", "minibatch", "recurrent_chunk",
                "hidden_dim", "snapshot_every")
    if any(getattr(args, key) < 1 for key in positive):
        parser.error("Training counts must be positive")
    if args.rollout % args.recurrent_chunk or args.minibatch % args.recurrent_chunk:
        parser.error("rollout and minibatch must be divisible by recurrent chunk")
    if args.minibatch > args.rollout:
        parser.error("minibatch cannot exceed rollout")
    if not 0 <= args.gae_lambda <= 1 or not 0 < args.gamma <= 1:
        parser.error("Invalid GAE parameters")
    for key in ("entropy_coef", "actor_lr", "critic_lr"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(key + " must be finite and nonnegative")
    if args.actor_lr == 0 or args.critic_lr == 0:
        parser.error("Learning rates must be positive")
    if args.checkpoint_every_steps < 0 or args.stop_after_steps < 0:
        parser.error("Checkpoint/stop steps must be nonnegative")
    return args


def checkpoint_contract(args):
    ignored = {"resume", "stop_after_steps", "checkpoint_every_steps"}
    return {key: value for key, value in vars(args).items() if key not in ignored}


def main(argv=None):
    args = parse_args(argv)
    out = (ROOT / args.out).resolve()
    inventory_path = (ROOT / args.inventory).resolve()
    if not out.is_relative_to(ROOT) or not inventory_path.is_relative_to(ROOT):
        raise SystemExit("Paths must stay under project root")
    resume_path = (ROOT / args.resume).resolve() if args.resume else None
    if resume_path is not None and not resume_path.is_relative_to(ROOT):
        raise SystemExit("Resume checkpoint must stay under project root")
    if resume_path is None:
        if out.exists():
            raise SystemExit("Refusing to overwrite output directory")
        out.mkdir(parents=True)
    elif not out.is_dir() or not resume_path.is_file():
        raise SystemExit("Resume requires an existing output directory and checkpoint")

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

    env = OceanTrainingPool(inventory_path)
    cfg = env.cfg
    episode_rng = np.random.default_rng(args.seed + 90000)

    def reset_episode():
        seed = int(episode_rng.integers(0, 2**31 - 1))
        observation, _ = env.reset(seed)
        return observation

    obs = reset_episode()
    obs_dim = features(obs, cfg, args.pvf).shape[-1]
    state_dim = len(critic_state(env))
    model = FormalMAPPO(
        obs_dim, state_dim, args.hidden_dim,
        learned_comm=not args.no_communication,
        role_dim=cfg.n_uavs).to(device)
    value_norm = ValueNorm().to(device)
    actor_parameters = list(model.actor.parameters())
    critic_parameters = list(model.critic.parameters())
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=args.actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=args.critic_lr, eps=1e-5)
    sampling_generator = torch.Generator(device=device).manual_seed(args.seed + 71003)

    metadata = {
        "version": VERSION,
        "status": "formal candidate; publication status requires competence and protocol gates",
        "args": vars(args),
        "contract": checkpoint_contract(args),
        "config": env.config_dict(),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "actor_inputs": (
            "own position,time,local contacts,one-hot role; matched PVF or zero "
            "channel; delivered learned 32-D peer-message aggregate"),
        "actor_role_conditioning": (
            "fully shared encoder/GRU/action head plus one learned 2-D bias per "
            "fixed mission role; identical for MAPPO and PVF+MAPPO"),
        "critic_inputs": "privileged global state; training only",
        "recurrent_actor": {"type": "GRUCell", "hidden_dim": args.hidden_dim,
                            "reset_mask": "zero on each new mission"},
        "value_normalization": "streaming returns; raw rewards and reported metrics unchanged",
        "lr_schedule": "constant" if args.no_lr_decay else "linear by completed PPO updates",
        "action_distribution": "Normal latent with bijective disk transform",
        "action_feasibility": (
            "shared rectangular-geofence safety shield reflects outward normal "
            "velocity components before execution"),
        "communication": {
            "type": ("none" if args.no_communication else
                     "jointly learned 32-D actor message with uniform aggregation"),
            "payload_codec": "per-packet symmetric int8 with float32 scale/relevance",
            "peer_payload_bytes": 42,
            "peer_header_bytes": cfg.header_bytes,
            "self_message": ("not used" if args.no_communication else
                              "included locally without radio accounting"),
            "delivery": "distance topology and simulator packet loss",
        },
        "mission_timeout": "finite-horizon terminal, zero bootstrap",
        "field_sha256": file_sha(Path(env.field.path)),
        "forcing": env.forcing_metadata(),
        "inventory_sha256": file_sha(inventory_path),
        "source_sha256": file_sha(__file__),
        "exact_resume_scope": "model,optimizers,ValueNorm,all RNGs,environment,episode,RNN,unfinished rollout",
    }

    update = 0
    total_steps = 0
    rollout_buffer = []
    completed = []
    episode_return = 0.0
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
        model.load_state_dict(saved["model"])
        value_norm.load_state_dict(saved["value_norm"])
        actor_optimizer.load_state_dict(saved["actor_optimizer"])
        critic_optimizer.load_state_dict(saved["critic_optimizer"])
        update = int(saved["update"])
        total_steps = int(saved["environment_steps"])
        rollout_buffer = deepcopy(saved["rollout_buffer"])
        completed = deepcopy(saved["completed_episodes"])
        episode_return = float(saved["episode_return"])
        episode_rng.bit_generator.state = deepcopy(saved["episode_rng"])
        obs = restore_environment(env, saved["environment"])
        actor_hidden = saved["actor_hidden"].to(device)
        actor_mask = saved["actor_mask"].to(device)
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_cpu_rng"].cpu())
        if device.type == "cuda" and saved["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["torch_cuda_rng"]])
        sampling_generator.set_state(saved["sampling_generator"].to(device="cpu")
                                     if device.type == "cpu" else saved["sampling_generator"])
        metadata = saved["metadata"]
        with (out / "resume-events.jsonl").open("a") as stream:
            stream.write(json.dumps({"checkpoint": str(resume_path.relative_to(ROOT)),
                                     "environment_steps": total_steps,
                                     "update": update,
                                     "rollout_length": len(rollout_buffer)}) + "\n")

    def exact_payload():
        return {
            "version": VERSION,
            "contract": checkpoint_contract(args),
            "metadata": metadata,
            "model": model.state_dict(),
            "value_norm": value_norm.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "update": update,
            "environment_steps": total_steps,
            "rollout_buffer": deepcopy(rollout_buffer),
            "completed_episodes": deepcopy(completed),
            "episode_return": episode_return,
            "episode_rng": deepcopy(episode_rng.bit_generator.state),
            "environment": capture_environment(env),
            "actor_hidden": actor_hidden.detach().cpu(),
            "actor_mask": actor_mask.detach().cpu(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "sampling_generator": sampling_generator.get_state(),
        }

    def save_exact(path=None):
        atomic_torch_save(exact_payload(), path or out / "resume.pt")

    def inference_payload():
        return {
            "model": model.state_dict(),
            "value_norm": value_norm.state_dict(),
            "metadata": metadata,
            "update": update,
            "environment_steps": total_steps,
        }

    while update < args.updates:
        rollout_reward = sum(float(row["r"]) for row in rollout_buffer)
        while len(rollout_buffer) < args.rollout:
            x = features(obs, cfg, args.pvf)
            state = critic_state(env)
            with torch.no_grad():
                actor_input = tensor(x, device)
                communication_mask = None
                if not args.no_communication:
                    communication_mask, _ = exchange_actor_messages(
                        model.actor, actor_input, env)
                hidden_before = actor_hidden.clone()
                mask_before = actor_mask.clone()
                mu, logstd, actor_hidden = model.actor.step(
                    actor_input, actor_hidden, actor_mask, communication_mask)
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
                next_value_raw = 0.0 if ended else value_norm.denormalize(
                    model.critic(tensor(critic_state(env), device)).squeeze(-1)).item()
            rollout_buffer.append({
                "x": x, "s": state, "raw": raw.cpu().numpy(),
                "lp": logprob.cpu().numpy(), "v": value_raw, "nv": next_value_raw,
                "r": reward, "boot": float(not ended), "cont": float(not ended),
                "hidden": hidden_before.cpu().numpy(), "mask": mask_before.cpu().numpy(),
                "comm": (np.zeros((cfg.n_uavs, cfg.n_uavs), dtype=np.float32)
                         if communication_mask is None else
                         communication_mask.cpu().numpy()),
            })
            rollout_reward += reward
            episode_return += reward
            total_steps += 1
            obs = next_obs
            actor_mask = torch.ones(cfg.n_uavs, 1, device=device)
            if ended:
                completed.append(dict(info, return_value=episode_return,
                                      environment_step=total_steps,
                                      scenario_id=scenario_id))
                episode_return = 0.0
                obs = reset_episode()
                actor_hidden.zero_()
                actor_mask.zero_()
            checkpoint_due = (args.checkpoint_every_steps and
                              total_steps % args.checkpoint_every_steps == 0)
            stopping = args.stop_after_steps and total_steps >= args.stop_after_steps
            if checkpoint_due or stopping:
                save_exact()
            if stopping:
                (out / "episodes.json").write_text(json.dumps(completed, indent=2))
                env.close()
                return

        arrays = stack_rollout(rollout_buffer)
        advantage_np, returns_np = gae(
            arrays["r"], arrays["v"], arrays["nv"], arrays["boot"], arrays["cont"],
            gamma=args.gamma, lam=args.gae_lambda)
        advantage_np = (advantage_np - advantage_np.mean()) / (advantage_np.std() + 1e-8)
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
        communication_masks = (None if args.no_communication else
                               tensor(arrays["comm"], device))

        with torch.no_grad():
            check_mu, check_logstd, _ = model.actor.sequence(
                x[:, None], hidden_before[0][None], masks[:, None],
                None if args.no_communication else communication_masks[:, None])
            check_ratio = torch.exp(
                latent_log_prob(check_mu[:, 0], check_logstd, raw) - old_logprob)
            ratio_error = float((check_ratio - 1).abs().max().item())
        if ratio_error > 1e-5:
            raise RuntimeError("Stored recurrent behavior probability inconsistent")

        lr_factor = 1.0
        if not args.no_lr_decay:
            lr_factor = set_linear_lr(actor_optimizer, args.actor_lr, update, args.updates)
            set_linear_lr(critic_optimizer, args.critic_lr, update, args.updates)
        losses = []
        for _ in range(args.epochs):
            for starts in sequence_batches(
                    args.rollout, args.recurrent_chunk, args.minibatch,
                    sampling_generator, device):
                offsets = torch.arange(args.recurrent_chunk, device=device)
                indices = starts[:, None] + offsets[None, :]
                sequence_x = x[indices].permute(1, 0, 2, 3)
                sequence_masks = masks[indices].permute(1, 0, 2, 3)
                sequence_communication = (None if args.no_communication else
                    communication_masks[indices].permute(1, 0, 2, 3))
                initial_hidden = hidden_before[starts]
                mu, logstd, _ = model.actor.sequence(
                    sequence_x, initial_hidden, sequence_masks,
                    sequence_communication)
                batch_raw = raw[indices].permute(1, 0, 2, 3)
                batch_old_logprob = old_logprob[indices].permute(1, 0, 2)
                ratio = torch.exp(
                    latent_log_prob(mu, logstd, batch_raw) - batch_old_logprob)
                batch_advantage = advantages[indices].permute(1, 0)[:, :, None]
                surrogate = torch.minimum(
                    ratio * batch_advantage,
                    ratio.clamp(0.8, 1.2) * batch_advantage)
                actor_loss = -surrogate.mean()
                entropy = torch.distributions.Normal(
                    mu, logstd.exp()).entropy().sum(-1).mean()
                actor_objective = actor_loss - args.entropy_coef * entropy

                flat = indices.reshape(-1)
                predicted_norm = model.critic(states[flat]).squeeze(-1)
                value_loss = clipped_value_loss(
                    predicted_norm, old_values_norm[flat], targets_norm[flat])
                step_separate(
                    actor_objective, value_loss,
                    actor_parameters, critic_parameters,
                    actor_optimizer, critic_optimizer)
                losses.append(float((actor_objective + value_loss).detach().item()))

        update += 1
        rollout_buffer.clear()
        row = {
            "update": update,
            "environment_steps": total_steps,
            "completed_episodes": len(completed),
            "current_episode_time_s": env.t,
            "rollout_reward": rollout_reward,
            "initial_ratio_max_error": ratio_error,
            "mean_loss": float(np.mean(losses)),
            "value_norm_mean": value_norm.mean.item(),
            "value_norm_var": value_norm.var.item(),
            "actor_lr": actor_optimizer.param_groups[0]["lr"],
            "critic_lr": critic_optimizer.param_groups[0]["lr"],
            "lr_factor": lr_factor,
            "elapsed_s": time.perf_counter() - started,
        }
        with (out / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        atomic_torch_save(inference_payload(), out / "checkpoint.pt")
        save_exact()
        if update % args.snapshot_every == 0 or update == args.updates:
            atomic_torch_save(
                inference_payload(), out / f"checkpoint-step-{total_steps:09d}.pt")
            save_exact(out / f"resume-step-{total_steps:09d}.pt")
            (out / "episodes.json").write_text(json.dumps(completed, indent=2))

    (out / "episodes.json").write_text(json.dumps(completed, indent=2))
    env.close()


if __name__ == "__main__":
    main()
