import argparse
import json
import time
from pathlib import Path
import numpy as np
from .env import Config, MaritimeSAR
from .policies import pvf_greedy, random_policy

def save_svg(path, tracks, targets, size):
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
    def point(x):
        return f"{40+720*x[0]/size:.2f},{40+720*(1-x[1]/size):.2f}"
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="820" viewBox="0 0 800 820">',
             '<rect width="800" height="820" fill="white"/>',
             '<rect x="40" y="40" width="720" height="720" fill="#eef6fa" stroke="#555"/>']
    for i in range(tracks.shape[1]):
        coords = " ".join(point(p) for p in tracks[:, i])
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[i%len(colors)]}" stroke-width="2"/>')
    for i in range(targets.shape[1]):
        coords = " ".join(point(p) for p in targets[:, i])
        parts.append(f'<polyline points="{coords}" fill="none" stroke="#222" stroke-dasharray="4 3"/>')
    parts.append('<text x="40" y="795" font-family="sans-serif" font-size="16">Synthetic SAR | colored: UAVs; dashed: hidden target tracks (evaluation only)</text></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--policy", choices=["pvf", "random"], default="pvf")
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("episodes must be positive")
    root = Path(__file__).resolve().parents[2]
    out = (root/args.out).resolve()
    if not out.is_relative_to(root):
        ap.error("output must be inside project")
    out.mkdir(parents=True, exist_ok=False)
    cfg = Config(**json.loads((root/args.config).read_text()))
    results = []
    for episode in range(args.episodes):
        seed = args.seed+episode
        env = MaritimeSAR(cfg)
        obs, _ = env.reset(seed)
        policy_rng = np.random.default_rng(seed+100000)
        tracks, targets = [env.uavs.copy()], [env.targets.copy()]
        started = time.perf_counter()
        total_reward = 0.
        steps = 0
        while not env.done:
            action = pvf_greedy(obs, cfg) if args.policy=="pvf" else random_policy(obs, cfg, policy_rng)
            obs, reward, _, _, info = env.step(action)
            total_reward += reward
            steps += 1
            tracks.append(env.uavs.copy())
            targets.append(env.targets.copy())
        info.update(seed=seed, steps=steps, reward=total_reward, wall_s=time.perf_counter()-started)
        results.append(info)
        np.savez_compressed(out/f"episode-{seed}.npz", uavs=np.array(tracks),
                            targets=np.array(targets), detected_at=env.detected_at)
        if episode == 0:
            save_svg(out/"trajectories.svg", np.array(tracks), np.array(targets), cfg.region_m)
        print(json.dumps(info), flush=True)
    metadata = {"status": "synthetic simulator verification, not PEVA-Gen results",
                "policy": args.policy, "config": env.config_dict(), "episodes": results,
                "mean_success_rate": float(np.mean([r["success_rate"] for r in results])),
                "numpy": np.__version__}
    (out/"results.json").write_text(json.dumps(metadata, indent=2))
    print("Saved", out, flush=True)

if __name__ == "__main__":
    main()
