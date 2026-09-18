# PEVA-Gen

Physics- and evidence-value-aligned perception and communication for decentralized multi-UAV maritime search and rescue.

This repository contains the research implementation used for the PEVA-Gen experiments: the maritime simulator, MAPPO and PVF+MAPPO baselines, PEVA-Gen training and evaluation code, ablations, robustness runners, trace auditing, and result aggregation.

The repository does **not** contain the original Copernicus ocean files or trained checkpoints. Those artifacts are distributed separately because of size and upstream data licensing. See [DATA.md](DATA.md) for the data source, download instructions, expected layout, and provenance requirements.

## Reproducibility status

- Code: included in `src/peva_sim/`.
- Tests: included in `tests/`.
- Frozen experiment plans and aggregated results: released separately with the paper supplementary material.
- Hardware used for the reported runs: NVIDIA RTX 4090, CUDA-enabled PyTorch.
- Evaluation uses fixed seeds and separate validation/test splits. Test evaluation must be explicitly unlocked in the runner.

## Installation

Python 3.10 or 3.11 is recommended. Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a GPU installation, install the PyTorch build matching the local CUDA driver before installing the remaining requirements. The default requirements file intentionally does not pin a CUDA-specific wheel.

Run the unit tests:

```bash
pytest -q
```

## Data setup

Download the frozen PEVA-Gen data bundle from the separate [Peva-Gen-data repository](https://github.com/Xuefeng-Du1121/Peva-Gen-data). Do not commit the downloaded NetCDF/array files to this code repository. Set `PEVA_DATA_ROOT` to the extracted data directory, then build the forcing inventory:

```bash
export PEVA_DATA_ROOT=/path/to/peva-gen-data
PYTHONPATH=src python -m peva_sim.build_forcing_inventory \
  --field "$PEVA_DATA_ROOT/ocean_system_data_20260913T024902Z/drift-trajectory-system/data_platform/data/raw/environment/copernicus/multobs_currents_east_china_2018_01.nc" \
  --spec configs/scenario-spec.json \
  --out configs/forcing-inventory.json
```

The inventory is a small manifest containing file names, dimensions, timestamps, checksums, and split membership. It is safe to commit after checking that it contains no private paths.

## Reproduce an experiment

All reported runs should be executed from the repository root with `PYTHONPATH=src` and a frozen inventory. A typical validation plan is:

```bash
PYTHONPATH=src python -m peva_sim.run_formal_study \
  --plan-only \
  --out runs/reproduction-validation \
  --inventory configs/forcing-inventory.json \
  --steps 20000 \
  --seeds 10 11 12 13 14 \
  --methods mappo pvf-mappo peva-gen \
  --eval-split validation
```

Execute the frozen plan with the required PEVA-Gen transport auxiliary checkpoint:

```bash
PYTHONPATH=src python -m peva_sim.run_formal_study \
  --execute-plan runs/reproduction-validation/plan.json \
  --transport-aux /path/to/transport-aux/checkpoint.pt
```

Aggregate and audit the resulting traces:

```bash
PYTHONPATH=src python -m peva_sim.aggregate_formal_results \
  --run-dir runs/reproduction-validation \
  --out runs/reproduction-validation/aggregate

PYTHONPATH=src python -m peva_sim.audit_stage_results \
  --run-dir runs/reproduction-validation
```

The exact step count, rollout length, auxiliary checkpoint checksum, and scenario inventory must match the frozen experiment plan accompanying the paper. Do not infer those values from this abbreviated example.

## Methods

The implementation exposes the following main entry points:

- `peva_sim.train_formal_mappo`: formal MAPPO baseline;
- `peva_sim.train_formal_peva`: PEVA-Gen and ablation training;
- `peva_sim.evaluate_formal_mappo` and `peva_sim.evaluate_formal_peva`: frozen-checkpoint evaluation;
- `peva_sim.run_formal_study`: multi-seed plan generation and execution;
- `peva_sim.aggregate_formal_results`: mean/SD and pairwise result aggregation;
- `peva_sim.audit_stage_results`: trace-level invariant and metric audit.

Deployment code is in `peva_sim.peva_deployment`. The centralized critic and replay buffer are training-time components; deployment uses only decentralized policy, belief, and communication modules.

## Data and code citation

Please cite the PEVA-Gen paper and the upstream Copernicus Marine Service product used by the data bundle. The data repository must preserve upstream attribution and license text. See [DATA.md](DATA.md).

## License

The code is released under the MIT License. This license applies only to the original code in this repository; upstream ocean data remain subject to their own terms.
