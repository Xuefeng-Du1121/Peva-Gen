"""Audited deterministic evaluator for formal recurrent MAPPO baselines."""
import argparse
from dataclasses import replace
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

from .audit_simulated_collection import verify_episode_arrays
from .evaluation_trace import audit_trace
from .formal_mappo import (
    FormalMAPPO, ValueNorm, exchange_actor_messages, features)
from .forcing_scenarios import validate_split_support
from .ocean_forced_env import OceanForcedSAR
from .ocean_training_pool import OceanTrainingPool
from .ppo_math import disk_action
from .protocol import file_sha

ROOT = Path(__file__).resolve().parents[2]
VERSION = "formal_recurrent_shared_role_bias_geofenced_mappo_v5"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
    parser.add_argument("--no-communication", action="store_true",
                        help="evaluate a matched decentralized no-radio actor")
    parser.add_argument("--packet-drop", type=float, default=None)
    parser.add_argument("--radio-range-m", type=float, default=None)
    parser.add_argument("--detection-p", type=float, default=None)
    parser.add_argument("--clutter-mean", type=float, default=None)
    parser.add_argument("--belief-diffusion-m2s", type=float, default=None)
    parser.add_argument("--truth-diffusion-min-m2s", type=float, default=None)
    parser.add_argument("--truth-diffusion-max-m2s", type=float, default=None)
    parser.add_argument("--drift-error-bound-mps", type=float, default=None)
    args = parser.parse_args(argv)
    if args.episodes_per_scenario < 1:
        parser.error("episodes-per-scenario must be positive")
    if args.packet_drop is not None and not 0 <= args.packet_drop <= 1:
        parser.error("packet-drop must be in [0,1]")
    if args.radio_range_m is not None and args.radio_range_m <= 0:
        parser.error("radio-range-m must be positive")
    if args.detection_p is not None and not 0 <= args.detection_p <= 1:
        parser.error("detection-p must be in [0,1]")
    for key in ("clutter_mean", "belief_diffusion_m2s",
                "truth_diffusion_min_m2s", "truth_diffusion_max_m2s",
                "drift_error_bound_mps"):
        value = getattr(args, key)
        if value is not None and value < 0:
            parser.error(key + " must be nonnegative")

    truth_diffusion_range = (
        .5 if args.truth_diffusion_min_m2s is None
        else args.truth_diffusion_min_m2s,
        4. if args.truth_diffusion_max_m2s is None
        else args.truth_diffusion_max_m2s)
    if truth_diffusion_range[0] > truth_diffusion_range[1]:
        parser.error("truth diffusion minimum exceeds maximum")
    drift_error_bound_mps = (
        .1 if args.drift_error_bound_mps is None
        else args.drift_error_bound_mps)
    checkpoint, inventory_path, out = [
        (ROOT / value).resolve() for value in
        (args.checkpoint, args.inventory, args.out)]
    if not all(path.is_relative_to(ROOT) for path in (checkpoint, inventory_path, out)):
        parser.error("Paths must stay under project root")
    if out.exists():
        parser.error("Refusing to overwrite evaluation output")

    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    if metadata.get("version") != VERSION:
        parser.error("Unsupported checkpoint architecture")
    pool = OceanTrainingPool(inventory_path)
    if metadata["config"] != pool.config_dict():
        raise ValueError("Evaluation environment differs from training")
    if metadata["inventory_sha256"] != file_sha(inventory_path):
        raise ValueError("Evaluation inventory differs from training")
    overrides = {key: value for key, value in (
        ("packet_drop", args.packet_drop),
        ("radio_range_m", args.radio_range_m),
        ("detection_p", args.detection_p),
        ("clutter_mean", args.clutter_mean),
        ("diffusion_m2s", args.belief_diffusion_m2s))
                 if value is not None}
    cfg = replace(pool.cfg, **overrides)
    robustness_overrides = dict(overrides)
    if (args.truth_diffusion_min_m2s is not None or
            args.truth_diffusion_max_m2s is not None):
        robustness_overrides["truth_diffusion_range_m2s"] = list(
            truth_diffusion_range)
    if args.drift_error_bound_mps is not None:
        robustness_overrides["drift_error_bound_mps"] = drift_error_bound_mps
    inventory = json.loads(inventory_path.read_text())
    cases = [row for row in inventory["scenarios"] if row["split"] == args.split]
    if not cases:
        raise ValueError("No cases for requested split")
    train_cases = [row for row in inventory["scenarios"] if row["split"] == "train"]
    validate_split_support(train_cases + cases)

    learned_comm = "actor.message_head.weight" in saved["model"]
    if learned_comm == args.no_communication:
        raise ValueError("checkpoint communication mode does not match evaluator")
    obs_dim = (saved["model"]["actor.message_head.weight"].shape[1]
               if learned_comm else
               saved["model"]["actor.encoder.0.weight"].shape[1])
    state_dim = saved["model"]["critic.0.weight"].shape[1]
    hidden_dim = saved["model"]["actor.encoder.0.weight"].shape[0]
    model = FormalMAPPO(
        obs_dim, state_dim, hidden_dim, learned_comm=learned_comm,
        role_dim=cfg.n_uavs)
    model.load_state_dict(saved["model"])
    model.eval()
    value_norm = ValueNorm()
    value_norm.load_state_dict(saved["value_norm"])
    value_norm.eval()

    out.mkdir(parents=True)
    source_manifest = {
        path.name: file_sha(path)
        for path in sorted((ROOT / "src/peva_sim").glob("*.py"))}
    (out / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2))
    rows = []
    evaluation_started = time.perf_counter()

    for case_index, case in enumerate(cases):
        env = OceanForcedSAR(
            pool.field, case["origin_lonlat"], case["start_utc"], cfg,
            truth_diffusion_range=truth_diffusion_range,
            drift_error_bound_mps=drift_error_bound_mps)
        for repeat in range(args.episodes_per_scenario):
            seed = 820000 + 1000 * case_index + repeat
            obs, _ = env.reset(seed)
            hidden = model.actor.initial_state(cfg.n_uavs)
            mask = torch.zeros(cfg.n_uavs, 1)
            times, found, actions, rewards = [], [], [], []
            positions, targets, inference_ms = [], [], []
            episode_started = time.perf_counter()
            total_reward = 0.0
            while not env.done:
                times.append(env.t)
                found.append(env.found.copy())
                state = env.state()
                positions.append(state["uavs_m"].copy())
                targets.append(state["targets_m"].copy())
                inference_started = time.perf_counter()
                with torch.no_grad():
                    actor_input = torch.as_tensor(
                        features(obs, cfg, metadata["args"]["pvf"]),
                        dtype=torch.float32)
                    communication_mask = None
                    if learned_comm:
                        communication_mask, _ = exchange_actor_messages(
                            model.actor, actor_input, env)
                    mean, _, hidden = model.actor.step(
                        actor_input, hidden, mask, communication_mask)
                    action = disk_action(mean, cfg.speed_mps).numpy()
                inference_ms.append(
                    (time.perf_counter() - inference_started) * 1000)
                obs, reward, _, _, info = env.step(action)
                mask.fill_(1.0)
                actions.append(action.copy())
                rewards.append(reward)
                total_reward += reward

            trace = out / ("episode-%04d.npz" % len(rows))
            np.savez_compressed(
                trace, time_s=np.asarray(times), found=np.asarray(found),
                terminal_found=env.found.copy(),
                actions_mps=np.asarray(actions), rewards=np.asarray(rewards),
                uavs_m=np.asarray(positions), targets_m=np.asarray(targets))
            record = {
                "scenario_id": case["id"], "seed": seed, "steps": len(times),
                "return_value": total_reward, "info": info,
                "trace_file": trace.name, "trace_sha256": file_sha(trace),
                "runtime": {
                    "episode_wall_s": time.perf_counter() - episode_started,
                    "inference_mean_ms": float(np.mean(inference_ms)),
                    "inference_p95_ms": float(np.percentile(inference_ms, 95)),
                    "inference_total_ms": float(np.sum(inference_ms)),
                },
            }
            verify_episode_arrays(
                np.asarray(times), np.asarray(found), record, vars(cfg),
                peer_payload_bytes=(42 if learned_comm else 0))
            if not np.isclose(
                    total_reward, info["survival_score"] * cfg.n_targets,
                    rtol=1e-9, atol=1e-12):
                raise ValueError("Reward/survival mismatch")
            with np.load(trace) as archive:
                audit_trace(archive, record, vars(cfg))
            trajectory = np.asarray(positions)
            margin = np.minimum(
                np.minimum(trajectory[..., 0], trajectory[..., 1]),
                np.minimum(cfg.region_m - trajectory[..., 0],
                           cfg.region_m - trajectory[..., 1]))
            record["trajectory_diagnostics"] = {
                "boundary_fraction_1km": float(np.mean(margin <= 1000.0)),
                "mean_speed_mps": float(np.linalg.norm(
                    np.asarray(actions), axis=-1).mean()),
                "commanded_mean_speed_mps": float(np.linalg.norm(
                    np.asarray(actions), axis=-1).mean()),
                "executed_mean_speed_mps": float(np.linalg.norm(
                    np.diff(trajectory, axis=0), axis=-1).mean() / cfg.dt_s)
                    if len(trajectory) > 1 else 0.0,
            }
            rows.append(record)
            with (out / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({
                "completed": len(rows),
                "total": len(cases) * args.episodes_per_scenario,
            }), flush=True)
        env.close()

    summary = {
        "status": "formal protocol candidate; performance claim requires frozen multi-seed run",
        "split": args.split,
        "checkpoint_sha256": file_sha(checkpoint),
        "inventory_sha256": file_sha(inventory_path),
        "source_sha256": file_sha(__file__),
        "source_manifest_sha256": file_sha(out / "source-manifest.json"),
        "algorithm": metadata["version"],
        "pvf": metadata["args"]["pvf"],
        "training_steps": saved["environment_steps"],
        "episodes": len(rows),
        "scenarios": [case["id"] for case in cases],
        "records": rows,
        "evaluation_config": vars(cfg),
        "environment_uncertainty": {
            "truth_diffusion_range_m2s": list(truth_diffusion_range),
            "drift_error_bound_mps": drift_error_bound_mps,
        },
        "robustness_overrides": robustness_overrides,
        "policy_communication": {
            "type": ("jointly learned 32-D uniformly aggregated messages"
                     if learned_comm else "none"),
            "peer_payload_bytes_per_transmitting_agent_step": (42 if learned_comm else 0),
            "peer_header_bytes_per_transmitting_agent_step": cfg.header_bytes,
            "actual_delivery": "distance topology and packet loss applied before action",
            "shared_service": "coverage/PVF uplink and downlink counted by environment",
        },
        "runtime": {
            "wall_s": time.perf_counter() - evaluation_started,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "timing_scope": (
                "CPU features, message encoding/radio exchange, aggregation and "
                "recurrent actor inference; excludes env.step and audit IO"),
        },
    }
    for key in ("success_rate", "ttd_all", "survival_score"):
        summary[key] = float(np.mean([row["info"][key] for row in rows]))
    summary["boundary_fraction_1km"] = float(np.mean([
        row["trajectory_diagnostics"]["boundary_fraction_1km"] for row in rows]))
    summary["geofence_interventions_mean"] = float(np.mean([
        row["info"]["geofence_interventions"] for row in rows]))
    summary["communication_bytes_mean"] = {
        key: float(np.mean([row["info"]["bytes"][key] for row in rows]))
        for key in rows[0]["info"]["bytes"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pool.close()


if __name__ == "__main__":
    main()
