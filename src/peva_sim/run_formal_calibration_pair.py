"""Frozen paired development calibration for formal MAPPO and PVF+MAPPO."""
import json
import os
from pathlib import Path
import subprocess
import time

from .protocol import file_sha

ROOT = Path(__file__).resolve().parents[2]


def main():
    os.chdir(ROOT)
    output = ROOT / "runs/formal-calibration-pair-seed0-20260913"
    output.mkdir(exist_ok=False)
    inventory = "runs/ocean-forcing-expanded-development-inventory-v2.json"
    sources = {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in sorted((ROOT / "src/peva_sim").glob("*.py"))}
    sources[inventory] = file_sha(ROOT / inventory)
    jobs = []
    for use_pvf in (False, True):
        method = "pvf-mappo" if use_pvf else "mappo"
        train = str((output / method).relative_to(ROOT))
        evaluate = str((output / (method + "-validation")).relative_to(ROOT))
        command = [
            "bash", "run.sh", "-m", "peva_sim.train_formal_mappo",
            "--inventory", inventory, "--out", train,
            "--seed", "0", "--updates", "200", "--rollout", "256",
            "--epochs", "4", "--minibatch", "64", "--recurrent-chunk", "16",
            "--hidden-dim", "128", "--gae-lambda", "0.95",
            "--entropy-coef", "0.01", "--snapshot-every", "50",
            "--checkpoint-every-steps", "12800",
        ]
        if use_pvf:
            command.append("--pvf")
        jobs.append({
            "method": method,
            "train": command,
            "evaluate": [
                "bash", "run.sh", "-m", "peva_sim.evaluate_formal_mappo",
                "--checkpoint", train + "/checkpoint.pt",
                "--inventory", inventory, "--out", evaluate,
                "--episodes-per-scenario", "10",
            ],
        })
    plan = {
        "scope": "development calibration; not formal paper results",
        "purpose": "test recurrent/ValueNorm trainer competence and matched PVF channel",
        "seed": 0,
        "steps_per_method": 51200,
        "selection": "none; final checkpoints evaluated on validation only",
        "test_split": "untouched",
        "jobs": jobs,
        "source_sha256": sources,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2))

    def unchanged():
        return all(file_sha(ROOT / path) == digest
                   for path, digest in sources.items())

    def event(payload):
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), **payload}) + "\n")

    event({"status": "started"})
    running = []
    for job in jobs:
        log = (output / (job["method"] + "-train.log")).open("x")
        process = subprocess.Popen(job["train"], stdout=log, stderr=subprocess.STDOUT)
        running.append((job, process, log))
        event({"status": "training", "method": job["method"], "pid": process.pid})
    failed = False
    for job, process, log in running:
        code = process.wait()
        log.close()
        event({"status": "train-finished", "method": job["method"], "exit": code})
        failed = failed or code != 0
    if failed:
        raise SystemExit("At least one paired training process failed")
    if not unchanged():
        raise SystemExit("Source changed during paired training")

    evaluating = []
    for job in jobs:
        log = (output / (job["method"] + "-evaluate.log")).open("x")
        process = subprocess.Popen(job["evaluate"], stdout=log, stderr=subprocess.STDOUT)
        evaluating.append((job, process, log))
        event({"status": "evaluating", "method": job["method"], "pid": process.pid})
    for job, process, log in evaluating:
        code = process.wait()
        log.close()
        event({"status": "evaluate-finished", "method": job["method"], "exit": code})
        if code:
            raise SystemExit("Paired evaluation failed: " + job["method"])
    if not unchanged():
        raise SystemExit("Source changed during paired evaluation")
    event({"status": "complete"})


if __name__ == "__main__":
    main()
