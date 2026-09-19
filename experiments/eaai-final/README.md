# EAAI final benchmark

This directory contains the frozen protocol for the expanded PEVA-Gen benchmark. The protocol is written before final training and evaluation. It distinguishes validation stress tests from independent OOD test results and does not authorize a global-optimality claim.

## Current implementation status

Implemented in the repository:

- `greedy-probability`
- `bayesian-information-gain`
- `lawnmower-coverage`
- `pvf-greedy`
- `oracle-target`
- formal `MAPPO`
- `PVF+MAPPO`
- PEVA-Gen and current ablations

Not yet implemented or audited under this protocol:

- CBBA;
- IPPO;
- MADDPG;
- MATD3;
- HAPPO;
- MASAC;
- CommNet;
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
