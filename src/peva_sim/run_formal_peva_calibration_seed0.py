"""Frozen seed-0 PEVA-Gen development calibration and validation grid."""
import json
import os
from pathlib import Path
import subprocess
import time

from .protocol import file_sha

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAME = "formal-peva-calibration-seed0-20260914"
CHECKPOINT_STEPS = (12800, 25600, 38400, 51200)


def main():
    os.chdir(ROOT)
    output = ROOT / "runs" / OUTPUT_NAME
    output.mkdir(exist_ok=False)
    inventory = "runs/ocean-forcing-expanded-development-inventory-v2.json"
    auxiliary = "runs/simulated-balanced-36-aux-1000-20260913/checkpoint.pt"
    sources = {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in sorted((ROOT / "src/peva_sim").glob("*.py"))
    }
    sources[inventory] = file_sha(ROOT / inventory)
    sources[auxiliary] = file_sha(ROOT / auxiliary)
    train_output = str((output / "seed0-peva-gen").relative_to(ROOT))
    train = [
        "bash", "run.sh", "-m", "peva_sim.train_formal_peva",
        "--transport-aux", auxiliary,
        "--inventory", inventory,
        "--out", train_output,
        "--seed", "0",
        "--updates", "200",
        "--rollout", "256",
        "--epochs", "4",
        "--minibatch", "64",
        "--recurrent-chunk", "16",
        "--hidden-dim", "128",
        "--gae-lambda", "0.95",
        "--entropy-coef", "0.01",
        "--message-weighting", "physics_grid",
        "--belief-samples", "2",
        "--snapshot-every", "50",
        "--checkpoint-every-steps", "12800",
        "--device", "cuda",
    ]
    evaluations = []
    for step in CHECKPOINT_STEPS:
        destination = str((
            output / f"seed0-peva-gen-validation-step-{step:09d}"
        ).relative_to(ROOT))
        evaluations.append([
            "bash", "run.sh", "-m", "peva_sim.evaluate_formal_peva",
            "--checkpoint",
            train_output + f"/checkpoint-step-{step:09d}.pt",
            "--transport-aux", auxiliary,
            "--inventory", inventory,
            "--out", destination,
            "--split", "validation",
            "--episodes-per-scenario", "10",
            "--device", "cuda",
        ])
    plan = {
        "scope": "development calibration; not formal paper results",
        "seed": 0,
        "method": "peva-gen",
        "steps": 51200,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "selection_rule": (
            "use the one shared budget selected from MAPPO/PVF seeds 0-2; "
            "PEVA checkpoints diagnose competence only and do not alter it"),
        "test_split": "untouched",
        "train": train,
        "evaluations": evaluations,
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
    with (output / "train.log").open("x") as log:
        process = subprocess.Popen(
            train, stdout=log, stderr=subprocess.STDOUT)
        event({"status": "training", "pid": process.pid})
        code = process.wait()
    event({"status": "train-finished", "exit": code})
    if code:
        raise SystemExit("PEVA training failed")
    if not unchanged():
        raise SystemExit("Source changed during training")

    running = []
    for step, command in zip(CHECKPOINT_STEPS, evaluations):
        label = f"validation-step-{step:09d}"
        log = (output / (label + ".log")).open("x")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT)
        running.append((label, process, log))
        event({"status": "evaluating", "name": label, "pid": process.pid})
    failures = []
    for label, process, log in running:
        code = process.wait()
        log.close()
        event({"status": "evaluate-finished", "name": label, "exit": code})
        if code:
            failures.append(label)
    if failures:
        raise SystemExit("PEVA evaluations failed: " + ",".join(failures))
    if not unchanged():
        raise SystemExit("Source changed during evaluation")
    event({"status": "complete"})


if __name__ == "__main__":
    main()
