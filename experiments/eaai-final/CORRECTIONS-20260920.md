# Experiment validity corrections, 2026-09-20

The first IPPO seed-0 run completed 400 updates / 102400 steps, but is
ineligible for benchmark aggregation: the local critic bootstrapped from
the pre-transition observation rather than the post-transition observation.
The collector has been corrected. Retraining must start from initialization
in a new directory; resuming the affected checkpoint is not a valid repair.
Affected output: `eaai-final-experiments/formal/ippo-seed0`.

The two classical validation batches with 20 episodes per method are
development checks, even though their directory names contain `formal`.
They do not satisfy the protocol's 100-episode evaluation requirement and
must not be described as the complete final benchmark. Repeating identical
scenario seeds does not add independent evidence.

Belief diagnostics still require repair before publication: found targets
have zero weights, making their posterior mean and ESS undefined. They must
be excluded using an explicit validity mask, and tiny positive intensity
mass must be normalized without clipping it to 1e-12. ESS and remaining
intensity mass are diagnostics, not monotonically better performance metrics.
Current diagnostics measure the simulator particle posterior, not the learned
PEVA posterior; they cannot substantiate learned-belief superiority.

Protocol v2 now freezes 204800 environment steps (800 updates of 256 steps)
per learned method and seed, using the final checkpoint. The previously
launched 102400-step IPPO run remains excluded because both its bootstrap
observation and its budget differ from the corrected final protocol.

The original trainer's default “MAPPO” actor includes a learned 32-D message
and uniform neighbor aggregation. This is operationally a CommNet-style MAPPO
policy, so it cannot also serve as an independent plain-MAPPO baseline. The
final registry assigns plain MAPPO and PVF+MAPPO no learned peer messages and
uses `commnet-mappo` / `pvf-commnet-mappo` for the uniform learned-message
variants. Historical checkpoints must be relabeled from their recorded
communication metadata; they are not silently treated as plain MAPPO.

The original final-v2 IPPO queue also used the eager value-density path. A
matched-input benchmark showed 18.5 s versus 1.91 s per 256-step update after
skipping the KDE that non-PVF policies immediately discard. Its partial files
are retained but excluded; final-v4 restarts every seed from initialization.

The first clean-clone v3 IPPO plan failed before training because the plan
driver referenced an untracked `run.sh` wrapper. No result was produced. The
driver now records the active isolated Python interpreter in every command;
the failed plan directory is retained as execution evidence.
