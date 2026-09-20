# Expanded EAAI Experiment Implementation Plan

This plan is the execution checklist for the frozen benchmark in
`protocol.json`. No manuscript quantitative claim is updated until the
corresponding gate is green and the source artifacts are archived.

## Phase A: infrastructure and auditability

1. Keep all runs under `/home/ubuntu-a/data/peva-gen/eaai-final-experiments`.
2. Use the isolated server `.venv` and the frozen forcing inventory.
3. Every run must write metadata, source manifest, checkpoint hash, episode
   traces, summary JSON, and audit output.
4. Aggregate only from audited traces; retain validation, IID test, OOD test,
   and stress-test labels separately.

Completed: protocol freeze, CBBA classical policy, matched no-communication
MAPPO, and auditable belief diagnostics.

## Phase B: baseline families

Run the following in this order, each first with a one-update smoke test and
then with the frozen 204800-environment-step multi-seed budget (seeds 0--9):

1. Classical: greedy-probability, PVF-greedy, Bayesian information gain,
   lawnmower coverage, CBBA, and privileged oracle-target.
2. Continuous MARL: MAPPO, IPPO (decentralized critics), MADDPG, MATD3,
   HAPPO, and MASAC.
3. Communication MARL: no-communication, CommNet, IC3Net, TarMAC, and an
   asynchronous message schedule.
4. Physics-aware: PVF+MAPPO, PVF+HAPPO, and the complete PEVA-Gen method.

An algorithm is included in the paper only after its implementation, training
entry point, evaluator, trace audit, and source-data manifest exist. A method
that cannot be implemented faithfully under the same observation, action,
training-step, and evaluation budget is reported as a limitation rather than
silently replaced by a proxy.

## Phase C: generalization and mechanism evidence

For every learned method, evaluate IID test first, then:

- OOD ocean forcing and start-time cases;
- packet-loss/radio-range stress;
- target diffusion and forecast-drift stress;
- clutter and detection-probability stress;
- UAV-failure and reduced-team scenarios;
- team-size scaling where the environment supports it.

Record success, survival, normalized TTD, boundary fraction, total/peer/shared
bytes, inference latency, posterior mean error, ESS fraction, remaining belief
mass, coverage gain, and control smoothness. Report only metrics that are
available for every compared method.

## Phase D: statistics and release gates

Use paired episode-level comparisons with exact Wilcoxon tests, Holm correction,
and hierarchical bootstrap confidence intervals over scenarios and seeds.
Freeze CSV/JSON source tables before plotting. Generate PDF and PNG figures,
source-data CSV, a figure manifest, and SHA-256 hashes. The final paper may
claim only “best-performing under the evaluated benchmark/protocol”; it may
not claim global optimality.

## Hardware and runtime

A single RTX 4090 is sufficient for the planned recurrent-policy runs if the
environment is CPU-parallelized and checkpoints are written under the data
directory. Reserve at least 16 CPU cores, 64 GB RAM, 200 GB free disk, and
CUDA/PyTorch matching the isolated environment. Classical and evaluator jobs
are CPU-bound; run them in parallel only when trace-writing bandwidth remains
stable.

## Stop conditions

Stop and report a limitation if a baseline is unstable after three documented
restarts, if its observation/action interface differs materially, or if its
result cannot pass the trace audit. Never fill missing learning curves or
baseline values with projected numbers.
