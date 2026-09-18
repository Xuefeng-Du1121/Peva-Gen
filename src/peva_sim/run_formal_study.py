"""Create and optionally execute a frozen multi-seed formal-study plan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .protocol import file_sha


ROOT = Path(__file__).resolve().parents[2]
METHOD_FLAGS = {
    "mappo": (),
    "pvf-mappo": ("--pvf",),
    "peva-gen": (),
    "peva-no-rsr": ("--disable-rsr",),
    "peva-deterministic-belief": (
        "--deterministic-belief", "--belief-samples", "1"),
    "peva-no-coverage": ("--disable-coverage-update",),
    "peva-no-physics-context": ("--disable-physics-context",),
    "peva-beta0-uniform-no-pvf-context": (
        "--fixed-beta", "0", "--message-weighting", "uniform",
        "--disable-physics-context"),
}
MAPPO_METHODS = {"mappo", "pvf-mappo"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        help="new directory below runs/")
    parser.add_argument("--inventory")
    parser.add_argument("--transport-aux")
    parser.add_argument("--scope", choices=("development-calibration",
                                             "formal-paper"),
                        default="development-calibration")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=range(5))
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_FLAGS),
                        default=("mappo", "pvf-mappo", "peva-gen"))
    parser.add_argument("--eval-split", choices=("validation", "test"),
                        default="validation")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--episodes-per-scenario", type=int, default=100)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"),
                        default="cuda")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--execute-plan",
                      help="execute a previously generated plan.json")
    return parser.parse_args(argv)


def _resolve_input(value: str) -> Path:
    path = (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"input file does not exist: {path}")
    return path


def validate_args(args):
    if args.out is None or args.inventory is None or args.steps is None:
        raise ValueError("--out, --inventory and --steps are required when creating a plan")
    if args.steps <= 0 or args.rollout <= 0 or args.steps % args.rollout:
        raise ValueError("steps must be a positive multiple of rollout")
    if args.episodes_per_scenario <= 0:
        raise ValueError("episodes-per-scenario must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("seeds must be unique non-negative integers")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be unique")
    if args.eval_split == "test" and not args.unlock_test:
        raise ValueError("test evaluation requires --unlock-test")
    if any(m not in MAPPO_METHODS for m in args.methods) and not args.transport_aux:
        raise ValueError("PEVA methods require --transport-aux")
    out = (ROOT / "runs" / args.out).resolve()
    runs = (ROOT / "runs").resolve()
    if out.parent != runs:
        raise ValueError("out must be one new direct child of runs/")
    return out


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_plan(args):
    output = validate_args(args)
    inventory = _resolve_input(args.inventory)
    inventory_record = json.loads(inventory.read_text(encoding="utf-8"))
    if (args.scope == "formal-paper" and
            inventory_record.get("status") != "frozen formal protocol"):
        raise ValueError("formal-paper scope requires a frozen formal inventory")
    auxiliary = (_resolve_input(args.transport_aux)
                 if args.transport_aux else None)
    updates = args.steps // args.rollout
    source_hashes = {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in sorted((ROOT / "src/peva_sim").glob("*.py"))
    }
    input_hashes = {_relative(inventory): file_sha(inventory)}
    if auxiliary:
        input_hashes[_relative(auxiliary)] = file_sha(auxiliary)
    jobs = []
    for method in args.methods:
        for seed in args.seeds:
            label = f"{method}-seed{seed}"
            train_out = output / "train" / label
            eval_out = output / args.eval_split / label
            if method in MAPPO_METHODS:
                train_module = "peva_sim.train_formal_mappo"
                eval_module = "peva_sim.evaluate_formal_mappo"
                extra = list(METHOD_FLAGS[method])
            else:
                train_module = "peva_sim.train_formal_peva"
                eval_module = "peva_sim.evaluate_formal_peva"
                extra = ["--transport-aux", _relative(auxiliary),
                         "--message-weighting", "physics_grid",
                         "--belief-samples", "5"] + list(METHOD_FLAGS[method])
            train = [
                "bash", "run.sh", "-m", train_module,
                "--inventory", _relative(inventory),
                "--out", _relative(train_out), "--seed", str(seed),
                "--updates", str(updates), "--rollout", str(args.rollout),
                "--epochs", "4", "--minibatch", "64",
                "--recurrent-chunk", "16", "--hidden-dim", "128",
                "--gae-lambda", str(args.gae_lambda),
                "--entropy-coef", "0.01", "--snapshot-every", "50",
                "--device", args.device,
            ] + extra
            evaluate = [
                "bash", "run.sh", "-m", eval_module,
                "--checkpoint", _relative(train_out / "checkpoint.pt"),
                "--inventory", _relative(inventory),
                "--out", _relative(eval_out), "--split", args.eval_split,
                "--episodes-per-scenario", str(args.episodes_per_scenario),
            ]
            if method not in MAPPO_METHODS:
                evaluate += ["--transport-aux", _relative(auxiliary),
                             "--device", args.device]
            jobs.append({"id": label, "method": method, "seed": seed,
                         "train": train, "evaluate": evaluate})
    return {
        "schema": "formal-study-plan-v1",
        "created_unix": time.time(),
        "scope": args.scope,
        "selection_policy": (
            "budget and hyperparameters frozen before test evaluation"),
        "test_unlocked": bool(args.unlock_test),
        "steps_per_training_seed": args.steps,
        "rollout": args.rollout,
        "gae_lambda": args.gae_lambda,
        "evaluation_split": args.eval_split,
        "episodes_per_scenario": args.episodes_per_scenario,
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "jobs": jobs,
    }, output


def _assert_frozen(plan):
    for name, digest in {**plan["source_sha256"],
                         **plan["input_sha256"]}.items():
        path = (ROOT / name) if not Path(name).is_absolute() else Path(name)
        if not path.is_file() or file_sha(path) != digest:
            raise RuntimeError(f"frozen file changed or disappeared: {name}")


def _option(command, name):
    index = command.index(name)
    return command[index + 1]


def _validate_stage_artifact(plan, job, stage):
    command = job[stage]
    output = Path(_option(command, "--out"))
    output = (ROOT / output).resolve() if not output.is_absolute() else output
    if stage == "train":
        checkpoint = output / "checkpoint.pt"
        metadata_path = output / "metadata.json"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"missing training checkpoint: {checkpoint}")
        if not metadata_path.is_file():
            raise RuntimeError(f"missing training metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        recorded = metadata.get("args", {})
        if recorded.get("seed") != job["seed"]:
            raise RuntimeError(f"training seed mismatch for {job['id']}")
        steps = recorded.get("updates", 0) * recorded.get("rollout", 0)
        if steps != plan["steps_per_training_seed"]:
            raise RuntimeError(f"training budget mismatch for {job['id']}")
    else:
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"missing evaluation summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        status = summary.get("status")
        if not isinstance(status, str) or not status.strip():
            raise RuntimeError(f"missing evaluation status for {job['id']}")
        if any(word in status.lower() for word in ("projected", "placeholder")):
            raise RuntimeError(f"non-empirical evaluation status for {job['id']}")
        if summary.get("split") != plan["evaluation_split"]:
            raise RuntimeError(f"evaluation split mismatch for {job['id']}")
        records = summary.get("records")
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"empty evaluation records for {job['id']}")
        if summary.get("episodes") != len(records):
            raise RuntimeError(f"evaluation episode count mismatch for {job['id']}")


def execute(plan, output):
    def event(payload):
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), **payload}) + "\n")

    event({"status": "started", "driver_pid": os.getpid()})
    for job in plan["jobs"]:
        for stage in ("train", "evaluate"):
            _assert_frozen(plan)
            log_path = output / "logs" / f"{job['id']}-{stage}.log"
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(job[stage], cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT)
                event({"status": "running", "job": job["id"],
                       "stage": stage, "child_pid": process.pid})
                code = process.wait()
            event({"status": "finished", "job": job["id"],
                   "stage": stage, "returncode": code})
            if code:
                raise SystemExit(f"failed: {job['id']} {stage} ({code})")
            _validate_stage_artifact(plan, job, stage)
    _assert_frozen(plan)
    event({"status": "complete"})


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    if args.execute_plan:
        plan_path = Path(args.execute_plan).resolve()
        if plan_path.name != "plan.json" or not plan_path.is_file():
            raise ValueError("--execute-plan must name an existing plan.json")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("schema") != "formal-study-plan-v1":
            raise ValueError("unsupported formal-study plan schema")
        output = plan_path.parent
        if (output / "events.jsonl").exists():
            raise ValueError("plan has already been started; manual audit is required")
        execute(plan, output)
        return
    plan, output = build_plan(args)
    output.mkdir(parents=False, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2),
                                      encoding="utf-8")
    if args.execute:
        execute(plan, output)


if __name__ == "__main__":
    main()
