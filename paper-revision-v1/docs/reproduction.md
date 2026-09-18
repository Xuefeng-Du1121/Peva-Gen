# PEVA-Gen paper reproduction

The paper tables and figures are generated only from the frozen CSV/JSON artifacts in `results/`. The independent test split contains 20 audited jobs (methods MAPPO, PVF+MAPPO, critic-only EVA-Gen, and PEVA-Gen; seeds 10--14). The robustness files are validation stress-test results, not an independent test split.

## Rebuild tables

```bash
python scripts/build_paper_tables.py --artifacts /path/to/frozen-artifacts --out results
```

For the server archive layout, pass `--test-root`, `--validation-root`, `--ablation-root`, and `--robustness-csv` explicitly. This command also writes the paired test comparison and `results-manifest.json`.

## Recompute communication

```bash
python scripts/recompute_communication.py --test-root /path/to/formal-main-test-seeds10-14-v1-20260918 --out results/communication-recomputed.csv
```

The audit reads `info.bytes` from every `summary.json`, checks each component against the recorded summary, and reports peer payload, peer header, shared uplink, shared downlink, total bytes, and bytes per UAV-step. The protocol is 42-byte quantized payload + 16-byte radio header = 58-byte peer packet (464 bits). A 32-D fp16 reference is 64 bytes (512 bits).

## Rebuild figures

```bash
python scripts/plot_paper_results.py --results results --test-root /path/to/formal-main-test-seeds10-14-v1-20260918 --out figures
```

The script writes PDF, PNG, and source-data CSV files. No learning curve is generated because the frozen archive does not contain a complete auditable checkpoint series.

## Data and artifacts

- Code: https://github.com/Xuefeng-Du1121/Peva-Gen
- Historical forcing data and provenance: https://github.com/Xuefeng-Du1121/Peva-Gen-data
- Large checkpoints are distributed through a GitHub Release or Zenodo artifact with SHA-256 checksums; they are not committed to Git history.

The public repository contains no server absolute paths, passwords, tokens, or private credentials.
