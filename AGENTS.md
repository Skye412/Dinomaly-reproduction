# Project instructions

## Fixed bridge anomaly benchmarks

For every subsequent model improvement, evaluate independently on the frozen
`Bridge512_balance` repaired split (`bridge512_balance_repaired`) and
`DACL10k_v3_512_split_seed42` (`dacl_v3_seed42`). Follow
`reproduction/BENCHMARKS.md` for the shared protocol, dataset fingerprints,
reference runs and comparison limits. Publish a run plan before each full run.
Do not change the datasets or select checkpoints/thresholds on test scores.

## Full experiment publication rule

The user requires a GitHub upload before every subsequent full experiment.

- Before launching each full experiment, record its run ID, datasets, seed,
  hyperparameters, intended changes and launch command in a tracked run plan.
- Commit the experiment code and configuration, push to
  `https://github.com/Skye412/Dinomaly-reproduction`, and verify that the remote
  branch points to the intended commit before starting training.
- Synchronize the server checkout to that published commit and verify that its
  tracked code is clean. Retain the commit ID in the run configuration.
- Apply this sequence to reruns and new full experiments even when the model
  code is unchanged: commit an updated run plan before launch.
- Publication is authorized by the user's standing instruction; do not ask for
  repeated confirmation for these experiment commits and pushes.
- A smoke check is not a full experiment. Never treat it as final results.
- Archive completed experiment configurations and raw metric records in
  `reproduction/results/`. Keep datasets, model weights, checkpoints, secrets
  and large runtime files out of Git.

## Source and execution

Preserve upstream attribution and LICENSE. The server checkout is
`/home/skye/data/Skye/Dinomaly`, accessed with SSH host `skye-cong`:
HostName `120.26.32.49`, Port `12139`, User `skye`, IdentityFile `~/.ssh/id_ed25519`.
The user selected this tunnel on 2026-10-16; prefer it over the temporarily
unreachable `skye` private address.
The conda environment is `/home/skye/miniconda3/envs/dinomaly`.
Avoid interrupting or restarting running experiments unless required by the
user's task. Distinguish a completed run from scientific reproduction of all
published results; preserve actual metric values and experiment settings.
