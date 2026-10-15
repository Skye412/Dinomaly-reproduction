# V1: Dinomaly + one official AMCM

Approved 2026-10-15: full runs only, Bridge then independently DACL, 10000 steps
each. No smoke, small subset training, preliminary short runs, extra modules,
parameter sweep, or automatic retry. Existing periodic validation/checkpoints
and runtime safety checks remain part of each full run.

V0 is frozen by `baselines/V0_LOCK.json` at commit
`2ca74b144b23f829f63b6778168442762342df9e`. Each dataset has its own V1 run
plan under `plans/`, each published in a dedicated commit and GitHub branch.
The two V1 commits must contain identical model and execution code; only the
DACL publication readiness plan changes in the second commit.

The entire official AMCM source is pinned and attributed in
`models/AMCM_NOTICE.md`. `AMCMTokenAdapter` reads encoder special-token counts
and patch size, reshapes only patches to 28x28, and rejoins the untouched
special prefix. It appends to the existing bottleneck ModuleList, so optimizer,
gradient clipping and full state_dict checkpoints include AMCM parameters.
Original reconstruction initialization occurs first, followed by official AMCM
default initialization. The encoder is explicitly frozen. Teacher feature
targets, loss, decoder, input, dataset filters and metrics stay as in V0.

`launch_v1.py` detaches `run_v1_sequence.py` after Git publication/sync. It makes
two clean detached server Git worktrees under the ignored sequence run folder;
each uses its designated published commit and shares only pretrained weights
and the GPU lock. Logs, optimizer and trained weights are independent. It waits
for >=14500 MiB free on GPU0, starts the full Bridge process, requires successful
completion, then starts a freshly initialized DACL process. Errors halt the
sequence without starting another experiment. No V0 or Bridge-trained weights
are loaded into DACL. Worktrees are retained to keep checkpoints accessible.

Completed configurations, raw metrics and prediction CSVs are copied into
`reproduction/results/<run_id>/` in the main checkout; model weights remain
ignored in the respective run directories. After both complete,
`compare_v1.py` writes Image AUROC, Pixel AUROC/AP/F1-max/AUPRO, validation-fixed
recall and normal false-positive counts, best step, wall time, peak allocated
memory and added parameter count. It reads completed results without inference
or tuning. The first optimizer step checks that every AMCM parameter has a
gradient; the full loop retains shape, finite-loss/gradient and error guards.

This experiment retains the historical 256 evaluator for V0/V1 comparison.
Ground-truth resizing corrections are deferred to one consistent re-evaluation
of both versions. A single seed is screening evidence; no additional runs
are scheduled from its outcome.

Startup-only revision launchfix1 (2026-10-16): the prior queue was verified
waiting with no training child/status and superseded before training. Replacement
sequence ID is `v1-amcm-seed1-20261015-launchfix1`; original run IDs stay fixed.
The tracked `/backbones/weights` ignore rule covers directory symlinks. In the
trainer, a busy shared GPU lock is retried with logged waiting status; after
acquisition memory is checked again under that lock. Insufficient memory releases
the lock and continues waiting before any model initialization. Query/launch
errors retain logged failure status. `console.log`, `log.txt` and other `.log` /
`.txt` training logs are archived alongside raw metrics, with result `.log` files
explicitly allowed by Git ignore rules. No model, loss, budget or evaluator changes.
