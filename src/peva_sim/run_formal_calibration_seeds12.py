"""Frozen seeds 1-2 development calibration with predeclared checkpoint grid."""
import json
import os
from pathlib import Path
import subprocess
import time

from .protocol import file_sha

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAME = "formal-calibration-seeds12-20260913-v2"
CHECKPOINT_STEPS = (12800, 25600, 38400, 51200)


def main():
    os.chdir(ROOT)
    output = ROOT / "runs" / OUTPUT_NAME
    output.mkdir(exist_ok=False)
    inventory = "runs/ocean-forcing-expanded-development-inventory-v2.json"
    sources = {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in sorted((ROOT / "src/peva_sim").glob("*.py"))}
    sources[inventory] = file_sha(ROOT / inventory)
    jobs = []
    for seed in (1, 2):
        for use_pvf in (False, True):
            method = "pvf-mappo" if use_pvf else "mappo"
            name = f"seed{seed}-{method}"
            train = str((output / name).relative_to(ROOT))
            command = [
                "bash", "run.sh", "-m", "peva_sim.train_formal_mappo",
                "--inventory", inventory, "--out", train,
                "--seed", str(seed), "--updates", "200", "--rollout", "256",
                "--epochs", "4", "--minibatch", "64", "--recurrent-chunk", "16",
                "--hidden-dim", "128", "--gae-lambda", "0.95",
                "--entropy-coef", "0.01", "--snapshot-every", "50",
                "--checkpoint-every-steps", "12800",
            ]
            if use_pvf:
                command.append("--pvf")
            evaluations = []
            for step in CHECKPOINT_STEPS:
                destination = str(
                    (output / f"{name}-validation-step-{step:09d}").relative_to(ROOT))
                evaluations.append([
                    "bash", "run.sh", "-m", "peva_sim.evaluate_formal_mappo",
                    "--checkpoint", train + f"/checkpoint-step-{step:09d}.pt",
                    "--inventory", inventory, "--out", destination,
                    "--episodes-per-scenario", "10",
                ])
            jobs.append({
                "name": name, "seed": seed, "method": method,
                "train": command, "evaluations": evaluations,
            })
    plan = {
        "scope": "development budget/seed calibration; not formal paper results",
        "seeds": [1, 2],
        "methods": ["mappo", "pvf-mappo"],
        "steps_per_method": 51200,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "selection_rule": "combine with seed0; choose one shared step by mean validation survival score, tie by target success then TTD",
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
    training = []
    for job in jobs:
        log = (output / (job["name"] + "-train.log")).open("x")
        process = subprocess.Popen(job["train"], stdout=log, stderr=subprocess.STDOUT)
        training.append((job, process, log))
        event({"status": "training", "name": job["name"], "pid": process.pid})
    failures = []
    for job, process, log in training:
        code = process.wait()
        log.close()
        event({"status": "train-finished", "name": job["name"], "exit": code})
        if code:
            failures.append(job["name"])
    if failures:
        raise SystemExit("Training failures: " + ",".join(failures))
    if not unchanged():
        raise SystemExit("Source changed during training")

    evaluating = []
    for job in jobs:
        for step, command in zip(CHECKPOINT_STEPS, job["evaluations"]):
            label = f"{job['name']}-step-{step:09d}"
            log = (output / (label + "-evaluate.log")).open("x")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            evaluating.append((label, process, log))
            event({"status": "evaluating", "name": label, "pid": process.pid})
    for label, process, log in evaluating:
        code = process.wait()
        log.close()
        event({"status": "evaluate-finished", "name": label, "exit": code})
        if code:
            failures.append(label)
    if failures:
        raise SystemExit("Evaluation failures: " + ",".join(failures))
    if not unchanged():
        raise SystemExit("Source changed during evaluation")
    event({"status": "complete"})


if __name__ == "__main__":
    main()
