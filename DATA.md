# Data provenance and usage

The simulator is driven by historical ocean forcing obtained from the Copernicus Marine Service. The exact product identifier, geographic bounding box, time range, download date, and preprocessing checksum used for the paper must be recorded in the companion data repository and in the frozen inventory file.

## Separate data repository

Create or use a companion public repository named, for example, `peva-gen-data`. Replace the placeholder below with its final URL before publishing this code repository:

`https://github.com/<OWNER>/peva-gen-data`

The data repository should contain:

1. a `README.md` with the Copernicus product URL and license/attribution;
2. a `MANIFEST.json` listing every downloaded file, SHA-256 checksum, dimensions, timestamps, and split assignment;
3. preprocessing scripts or notebooks sufficient to regenerate the compact forcing files;
4. a small public sample for smoke tests;
5. instructions for obtaining any files that cannot be redistributed directly.

Do not commit credentials, Copernicus access tokens, private server paths, or raw files that the upstream license does not permit redistribution.

## Expected local layout

The code accepts a data root rather than a hard-coded path. A typical extracted bundle is:

```text
<PEVA_DATA_ROOT>/
  forcing/
    train/          # training trajectories
    validation/     # validation trajectories
    test/           # held-out test trajectories
  metadata/
    MANIFEST.json
```

The exact filenames and variables are defined by the inventory. Use the inventory builder instead of editing paths in source code.

## Reproducibility record

For every public result, archive the following alongside the result table:

- Copernicus product identifier and source URL;
- download date and preprocessing version;
- data manifest SHA-256;
- split and scenario identifiers;
- code commit SHA;
- random seeds;
- PyTorch/CUDA versions.

The paper's current evidence is simulation based, even though the forcing is derived from historical ocean data. It should not be described as a real-UAV or field deployment validation.
