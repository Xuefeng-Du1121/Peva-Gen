# Model artifacts

The Git repository contains the complete model implementation, but trained weights are intentionally distributed as release artifacts rather than ordinary Git blobs. This keeps clones small and avoids silently mixing checkpoints from different data manifests.

Before publication, upload the frozen checkpoints to GitHub Releases, Zenodo, or an equivalent archival repository and replace the placeholders below:

| Artifact | Purpose | SHA-256 | Download |
|---|---|---|---|
| `transport-aux-checkpoint.pt` | auxiliary transport model required by PEVA-Gen training | `TO_BE_FILLED` | `TO_BE_FILLED` |
| `peva-gen-validation-seed10-14.tar.zst` | optional validation checkpoints and manifests | `TO_BE_FILLED` | `TO_BE_FILLED` |
| `peva-gen-test-release.tar.zst` | frozen test-evaluation checkpoints | `TO_BE_FILLED` | `TO_BE_FILLED` |

After downloading an artifact, verify its checksum before use. The exact command is:

```bash
sha256sum transport-aux-checkpoint.pt
```

Do not publish a checkpoint without recording its code commit, data-manifest checksum, training configuration, and random seed. A checkpoint alone is not a reproducible model release.
