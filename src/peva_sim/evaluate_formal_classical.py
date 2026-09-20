"""Audited common evaluator for deterministic non-learning SAR policies."""
import argparse
from dataclasses import replace
import json
import platform
from pathlib import Path
import time
import numpy as np

from .audit_simulated_collection import verify_episode_arrays
from .classical_policies import POLICIES, make_policy
from .evaluation_trace import audit_trace
from .evaluation_inventory import EVALUATION_SPLITS, load_evaluation_cases
from .ocean_forced_env import OceanForcedSAR
from .ocean_training_pool import OceanTrainingPool
from .protocol import file_sha
from .belief_diagnostics import particle_diagnostics, summarize_particles

ROOT = Path(__file__).resolve().parents[2]
VERSION = "formal_classical_evaluator_v1"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--scenario-inventory", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=EVALUATION_SPLITS,
                        default="validation")
    parser.add_argument("--methods", nargs="+", choices=tuple(POLICIES),
                        default=tuple(POLICIES))
    parser.add_argument("--episodes-per-scenario", type=int, default=10)
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
    if args.detection_p is not None and not 0 <= args.detection_p <= 1:
        parser.error("detection-p must be in [0,1]")
    for key in ("radio_range_m", "clutter_mean", "belief_diffusion_m2s",
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
    inventory_path = (ROOT / args.inventory).resolve()
    scenario_inventory_path = ((ROOT / args.scenario_inventory).resolve()
                               if args.scenario_inventory else None)
    out = (ROOT / args.out).resolve()
    if (not inventory_path.is_relative_to(ROOT) or not out.is_relative_to(ROOT)
            or (scenario_inventory_path is not None and
                not scenario_inventory_path.is_relative_to(ROOT))):
        parser.error("Paths must stay under project root")
    if out.exists():
        parser.error("Refusing to overwrite evaluation output")
    pool = OceanTrainingPool(inventory_path)
    overrides = {
        key: value for key, value in (
            ("packet_drop", args.packet_drop),
            ("radio_range_m", args.radio_range_m),
            ("detection_p", args.detection_p),
            ("clutter_mean", args.clutter_mean),
            ("diffusion_m2s", args.belief_diffusion_m2s))
        if value is not None
    }
    cfg = replace(pool.cfg, **overrides)
    robustness_overrides = dict(overrides)
    if (args.truth_diffusion_min_m2s is not None or
            args.truth_diffusion_max_m2s is not None):
        robustness_overrides["truth_diffusion_range_m2s"] = list(
            truth_diffusion_range)
    if args.drift_error_bound_mps is not None:
        robustness_overrides["drift_error_bound_mps"] = drift_error_bound_mps
    cases, evaluation_inventory = load_evaluation_cases(
        inventory_path, scenario_inventory_path, args.split,
        pool.config_dict(), file_sha(Path(pool.field.path)))

    out.mkdir(parents=True)
    source_manifest = {
        path.name: file_sha(path)
        for path in sorted((ROOT / "src/peva_sim").glob("*.py"))
    }
    (out / "source-manifest.json").write_text(
        json.dumps(source_manifest, indent=2))
    all_rows = {}
    evaluation_started = time.perf_counter()


    for method in args.methods:
        policy_rows = []
        for case_index, case in enumerate(cases):
            env = OceanForcedSAR(
                pool.field, case["origin_lonlat"], case["start_utc"], cfg,
                truth_diffusion_range=truth_diffusion_range,
                drift_error_bound_mps=drift_error_bound_mps)
            for repeat in range(args.episodes_per_scenario):
                seed = 820000 + 1000 * case_index + repeat
                observation, _ = env.reset(seed)
                policy = make_policy(method)
                policy.reset(observation)
                times, found, actions, rewards = [], [], [], []
                positions, targets, inference_ms = [], [], []
                belief_errors, belief_ess, belief_mass = [], [], []
                total_reward = 0.0
                episode_started = time.perf_counter()
                while not env.done:
                    times.append(env.t)
                    found.append(env.found.copy())
                    state = env.state()
                    positions.append(state["uavs_m"].copy())
                    targets.append(state["targets_m"].copy())
                    posterior_mean, error, ess, weight_mass, valid = particle_diagnostics(
                        env.particles, env.weights, state["targets_m"], env.found)
                    belief_errors.append(error)
                    belief_ess.append(ess)
                    belief_mass.append(weight_mass)
                    inference_started = time.perf_counter()
                    action = policy.act(
                        observation, cfg,
                        state if method == "oracle-target" else None)
                    inference_ms.append(
                        (time.perf_counter() - inference_started) * 1000)
                    observation, reward, _, _, info = env.step(action)
                    actions.append(action.copy())
                    rewards.append(reward)
                    total_reward += reward

                trace = out / (
                    f"{method}-episode-{len(policy_rows):04d}.npz")
                np.savez_compressed(
                    trace,
                    time_s=np.asarray(times),
                    found=np.asarray(found),
                    terminal_found=env.found.copy(),
                    actions_mps=np.asarray(actions),
                    rewards=np.asarray(rewards),
                    uavs_m=np.asarray(positions),
                    targets_m=np.asarray(targets),
                    belief_error_m=np.asarray(belief_errors),
                    belief_ess=np.asarray(belief_ess),
                    belief_mass=np.asarray(belief_mass))
                record = {
                    "method": method,
                    "oracle": method == "oracle-target",
                    "scenario_id": case["id"],
                    "seed": seed,
                    "steps": len(times),
                    "return_value": total_reward,
                    "info": info,
                    "trace_file": trace.name,
                    "trace_sha256": file_sha(trace),
                    "runtime": {
                        "episode_wall_s": (
                            time.perf_counter() - episode_started),
                        "inference_mean_ms": float(np.mean(inference_ms)),
                        "inference_p95_ms": float(
                            np.percentile(inference_ms, 95)),
                        "inference_total_ms": float(np.sum(inference_ms)),
                    },
                }
                verify_episode_arrays(
                    np.asarray(times), np.asarray(found),
                    record, vars(cfg), peer_payload_bytes=0)
                if not np.isclose(
                        total_reward,
                        info["survival_score"] * cfg.n_targets,
                        rtol=1e-9, atol=1e-12):
                    raise ValueError("Reward/survival mismatch")
                with np.load(trace) as archive:
                    audit_trace(archive, record, vars(cfg))
                trajectory = np.asarray(positions)
                margin = np.minimum(
                    np.minimum(
                        trajectory[..., 0], trajectory[..., 1]),
                    np.minimum(
                        cfg.region_m - trajectory[..., 0],
                        cfg.region_m - trajectory[..., 1]))
                record["trajectory_diagnostics"] = {
                    "boundary_fraction_1km": float(
                        np.mean(margin <= 1000.0)),
                    "mean_speed_mps": float(np.linalg.norm(
                        np.asarray(actions), axis=-1).mean()),
                }
                record["belief_diagnostics"] = summarize_particles(
                    belief_errors, belief_ess, belief_mass, cfg.particles)
                policy_rows.append(record)
                with (out / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps({
                    "method": method,
                    "completed": len(policy_rows),
                    "total": len(cases) * args.episodes_per_scenario,
                }), flush=True)
            env.close()
        all_rows[method] = policy_rows

    probability_pvf_equivalent = None
    if {"greedy-probability", "pvf-greedy"} <= set(all_rows):
        pairs = zip(
            all_rows["greedy-probability"], all_rows["pvf-greedy"])
        equality = []
        for first, second in pairs:
            with np.load(out / first["trace_file"]) as left:
                with np.load(out / second["trace_file"]) as right:
                    equality.append(all(
                        np.array_equal(left[key], right[key], equal_nan=True)
                        for key in left.files))
        probability_pvf_equivalent = bool(all(equality))
        if not probability_pvf_equivalent:
            raise ValueError(
                "Uniform-scalar probability/PVF control unexpectedly differs")

    summaries = {}
    for method, rows in all_rows.items():
        values = {}
        for key in ("success_rate", "ttd_all", "survival_score"):
            sample = np.asarray([row["info"][key] for row in rows])
            values[key] = {
                "mean": float(sample.mean()),
                "std_episode": float(sample.std(ddof=1)) if len(sample) > 1 else 0.0,
            }
        values["boundary_fraction_1km"] = {
            "mean": float(np.mean([
                row["trajectory_diagnostics"]["boundary_fraction_1km"]
                for row in rows])),
        }
        values["communication_bytes_mean"] = {
            key: float(np.mean([
                row["info"]["bytes"][key] for row in rows]))
            for key in rows[0]["info"]["bytes"]
        }
        values["runtime_inference_ms"] = {
            "mean": float(np.mean([
                row["runtime"]["inference_mean_ms"] for row in rows])),
            "p95_episode_mean": float(np.percentile([
                row["runtime"]["inference_mean_ms"] for row in rows], 95)),
        }
        summaries[method] = values

    summary = {
        "version": VERSION,
        "status": "formal protocol candidate; test use requires frozen protocol",
        "split": args.split,
        "inventory_sha256": file_sha(inventory_path),
        **evaluation_inventory,
        "field_sha256": file_sha(Path(pool.field.path)),
        "source_sha256": file_sha(__file__),
        "source_manifest_sha256": file_sha(out / "source-manifest.json"),
        "methods": list(args.methods),
        "oracle_method": "oracle-target",
        "episodes_per_method": len(cases) * args.episodes_per_scenario,
        "scenarios": [case["id"] for case in cases],
        "evaluation_config": vars(cfg),
        "environment_uncertainty": {
            "truth_diffusion_range_m2s": list(truth_diffusion_range),
            "drift_error_bound_mps": drift_error_bound_mps,
        },
        "robustness_overrides": robustness_overrides,
        "probability_pvf_equivalent_under_uniform_scalars": (
            probability_pvf_equivalent),
        "summaries": summaries,
        "records": all_rows,
        "runtime": {
            "wall_s": time.perf_counter() - evaluation_started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "timing_scope": "policy action only; excludes env.step and audit IO",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pool.close()


if __name__ == "__main__":
    main()
