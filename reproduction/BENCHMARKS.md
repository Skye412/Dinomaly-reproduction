# Fixed bridge anomaly detection benchmarks

User decision: 2026-10-15. Every model improvement is evaluated separately on
both datasets below, against the corresponding unchanged Dinomaly baseline.
These are complementary benchmarks, not two directly comparable test sets.

| Dataset / frozen profile | Normal train | Validation normal / anomaly | Test normal / anomaly | Baseline run |
|---|---:|---:|---:|---|
| Bridge512_balance / bridge512_balance_repaired | 2211 | 309 / 309 | 628 / 628 | bridge512-balance-repaired-b16-s1-20261014 |
| DACL10k_v3_512_split_seed42 / dacl_v3_seed42 | 413 | 103 / 1284 | 55 / 920 | dacl-v3-seed42-b16-s1-20261015 |

Roots are under `/home/skye/data/Skye/databases/`. Split CSV SHA256 and counts
are frozen in `dataset_profiles/`; a mismatch stops execution. Dataset content
audits are in `audits/`. Keep original images and annotations unchanged.

## Shared protocol

- Dinomaly-B with pretrained DINOv2-R ViT-B/14, eight linear-attention decoder
  blocks, noisy MLP dropout 0.2, upstream loss and anomaly scoring.
- Only normal training images; seed 1 for initial comparisons, batch 16,
  10000 optimizer steps, StableAdamW AMSGrad, LR 0.002 -> 0.0002 with 100-step
  warmup, weight decay 0.0001. Full frame resized to 392x392, no spatial crop.
- Validate every 2000 steps. Primary checkpoint maximizes validation image
  AUROC, with pixel AP as tie-breaker. Also retain final-step results.
- Image classification threshold maximizes validation F1 for the corresponding
  checkpoint, then remains fixed on test. Never use test to choose a model,
  threshold or training duration. Test F1-max is an explicitly oracle metric.
- Report image AUROC/AP/F1-max and pixel AUROC/AP/F1-max/AUPRO at 256x256.
  Report fixed-threshold precision, recall, F1, confusion counts, FPR,
  specificity and balanced accuracy. Retain raw validation/test score CSVs.
- Bridge mask: native-resolution union of nonzero RGB channels. DACL mask:
  native-resolution union of the 14 damage labels listed in `dacl_dataset.py`;
  exclude five object labels. Shared interpolation/binarization pipeline.

Do not rank improvements by averaging absolute scores across these datasets.
Report paired deltas against each dataset's baseline. If an improvement changes
input size, training budget or preprocessing, label the changed protocol and
add a matched baseline before attributing gains to its architecture.

DACL has only 55 test normals and 94.36% test anomalies; all-positive F1 is
97.10%. AUROC, FPR and pixel localization are essential to interpretation.
DACL content groups do not certify bridge/session independence. Normal means
no annotated damage label, not independently certified healthy structures.

One seed is an initial baseline, not evidence of run-to-run stability. Before
final research claims, run matched baseline/improvement seeds (e.g. 1, 2, 3)
on both datasets and report individual values plus mean and standard deviation.
Additional seeds are future runs, not results already obtained.

## Reference evidence and publication

Bridge512 baseline training commit:
`f3865c383001055917b22e57d3f8be762a6fbab5`; selected step 10000.
Raw results are in `results/bridge512-balance-repaired-b16-s1-20261014/`.
DACL baseline is planned/queued; no DACL full-run metrics are available yet.
Its actual launch commit, parameters, status and metrics are recorded under
`runs/dacl-v3-seed42-b16-s1-20261015/` on the server and archived upon completion.

Every full run requires its own committed plan, GitHub push, remote HEAD
verification and clean matching server checkout before launch. Smoke checks
are execution checks only. Queued jobs refuse to start if that checkout changes.
Do not interrupt another experiment to run these benchmarks.
