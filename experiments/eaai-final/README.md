# EAAI final benchmark

This directory contains the frozen protocol for the expanded PEVA-Gen benchmark. The protocol is written before final training and evaluation. It distinguishes validation stress tests from independent OOD test results and does not authorize a global-optimality claim.

## Current implementation status

Implemented in the repository:

- `greedy-probability`
- `bayesian-information-gain`
- `lawnmower-coverage`
- `pvf-greedy`
- `cbba` (PVF-aware deterministic auction baseline)
- `oracle-target`
- formal `MAPPO`
- formal `IPPO` with a shared local-observation critic and post-transition bootstrap
- formal `CommNet+MAPPO` (the original 32-D uniform learned-message baseline)
- `PVF+MAPPO`
- `PVF+CommNet+MAPPO`
- PEVA-Gen and current ablations

Not yet implemented or audited under this protocol:

- MADDPG;
- MATD3;
- HAPPO;
- MASAC;
- IC3Net;
- TarMAC;
- asynchronous communication and UAV-failure conditions;
- direct belief-quality metrics.

These methods must not be described as completed baselines until their training, evaluation, trace audit, and source-data records exist.

## Execution order

1. Freeze simulator/data inventory and verify split disjointness.
2. Add missing baseline adapters using the common observation/action/channel interface.
3. Run two-seed pilot jobs only for implementation diagnostics.
4. Freeze hyperparameters and run the ten-seed validation matrix.
5. Unlock IID test and OOD test only after validation and source hashes are frozen.
6. Run stress, scaling, mechanism, and hardware-profile evaluations.
7. Aggregate from raw traces, apply the registered statistics, and audit every result.

The paper must not be rewritten from the new experiments until steps 1--7 are complete.

## Compute scheduling

Create an isolated formal-study environment with the locked CUDA 12.8 stack:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-formal-cu128.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
```

Do not use an unconstrained `pip install -e .` for formal runs: the broad
project requirements may resolve a newer PyTorch/CUDA build and make results
incomparable with the frozen experiments.

`peva_sim.run_formal_study` runs one job at a time by default. Independent
method/seed jobs may share a GPU without changing their frozen commands:

```bash
python -m peva_sim.run_formal_study \
  --execute-plan runs/<study>/plan.json --max-parallel 4
```

The event log records the driver PID, child PIDs, execution parallelism, stage
status, and whether a stage was resumed. A previously started plan is never
silently reused. Explicit `--resume-plan` first validates and skips complete
stages; an incomplete training stage may continue only from its nonempty
`resume.pt`. Evaluation output is not automatically overwritten.

On the study server (RTX 4090, nine visible CPU cores), a diagnostic workload
of three independent two-update IPPO jobs took 27.81 s serially, 19.46 s at
two-way parallelism, and 10.69 s at three-way parallelism. Four independent
jobs took 11.24 s at four-way parallelism. Thus the formal queues use at most
four concurrent jobs, subject to a longer pilot and memory check. These timing
runs use seeds 910--943 and are compute diagnostics only, never paper results.
