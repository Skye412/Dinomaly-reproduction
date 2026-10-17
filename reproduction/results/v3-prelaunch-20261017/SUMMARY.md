# V3 prelaunch results: V0 reevaluation and 50-step smoke

Both approved stages completed. No full training or native512 appendix launched.

| Dataset | Selected z | Fixed-z=1% test I-AUROC | Actual validation FPR | Smoke loss step1 -> step50 | Peak allocated GiB |
|---|---:|---:|---:|---|---:|
| bridge512 | 1.0% | 80.005274% | 10.032362% | 0.988707 -> 0.075689 | 10.937564 |
| dacl | 1.0% | 58.857708% | 10.679612% | 0.970372 -> 0.092462 | 10.937564 |

Fixed 1% test AUROC and all four pixel metrics match original V0 JSON exactly. Both selected z=1%, so joint-z reevaluation test values equal historical values.
Smoke is diagnostic only; two-image validation subset is not performance evidence. Full-run stability remains unverified.
Memory is peak CUDA allocated, not reserved or nvidia-smi total device usage.
See PROVENANCE.json for actual evaluator and smoke commits. Smoke configs confirm clean commit 6e9c764.
Reference checkpoints are fixed V0 best_val at step10000; no search over old intermediate checkpoints.
No augmentation images were exported or visually reviewed in this run.
