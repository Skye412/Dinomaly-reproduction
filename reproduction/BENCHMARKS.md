# Fixed bridge anomaly detection benchmarks

Historical suite decision: 2026-10-15. The 2026-10-18 expanded suite below supersedes this suite for future module decisions. Every model improvement is evaluated separately on
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
DACL baseline completed 10000 steps on 2026-10-15, with step 10000 selected by
validation. Training commit: `a1c8073fbd3a002a806b3876e24f10d2a17456be`.
Raw results are in `results/dacl-v3-seed42-b16-s1-20261015/`.
Test image AUROC is 58.86%, pixel AUROC 67.04%, pixel AP 39.21%, and AUPRO
28.46%. The validation-F1 threshold yields 54 false positives among 55 normals;
its 97.15% image F1 must be interpreted alongside 98.18% FPR.

Every full run requires its own committed plan, GitHub push, remote HEAD
verification and clean matching server checkout before launch. Smoke checks
are execution checks only. Queued jobs refuse to start if that checkout changes.
Do not interrupt another experiment to run these benchmarks.


## Expanded three-dataset V0 suite (2026-10-18; awaiting code review)

| Profile / directory | Normal training | Validation normal / anomaly | Test normal / anomaly |
|---|---:|---:|---:|
| codebrim512 / Bridge512_balance_CODEBRIM | 1944 | 277 / 123 | 555 / 254 |
| s2ds512 / Bridge512_balance_S2DS | 267 | 32 / 186 | 73 / 374 |
| dacl_v3_seed42 / DACL10k_v3_512_split_seed42 | 413 | 103 / 1284 | 55 / 920 |

CODEBRIM totals 4070 images (1294 anomalous); S2DS totals 2226 (1854 anomalous).
The supplied split_protocol.json describes source filtering of the deduplicated
Bridge512_balance, with inherited split assignment and no rebalancing. All
parent IDs are disjoint across splits in each new dataset; supplied audits report
zero exact JPEG duplicates. These are user-derived subsets, not claims of official
benchmark splits. Manifest SHA256 values are frozen in dataset_profiles/*.json.

Both colored datasets use the same RGB palette: black (0,0,0) background,
red (255,0,0) Crack, green (0,255,0) Spalling, blue (0,0,255) Corrosion,
yellow (255,255,0) Efflorescence. All anomalous masks were scanned; no other
colors occurred. The intended structural-damage taxonomy maps exposed rebar to
Corrosion, vegetation and control points to background. The loader unions all
nonzero RGB pixels; it does not relabel or alter existing frozen files.
Source remapping semantics cannot be reconstructed solely from already-converted
masks; this taxonomy must be confirmed against the dataset preparation provenance.
DACL retains its historical 14-label union, including Graffiti/Wetspot and other
labels beyond the four-class taxonomy. It is area-stratified only. Thus the
frozen DACL definition is not strictly identical to the structural-only taxonomy;
changing it would require a new profile and cannot be silently applied to V0.

The tile manifest records crop_x/crop_y in {0,512} x {0,512}, representing
512-square tiles in the inherited 1024-coordinate source tile frame; parent_id
keeps original-parent grouping. The source filtering step performs no new crop.
Earlier original-image extraction, interpolation or padding is not established by
these manifests. Model input is the full 512 tile resized to 392, without cropping.

Native 512 mask union area counts (anomalous images only; bins in order
<0.1%, [0.1%,0.5%), [0.5%,1%), [1%,5%), >=5%):

| Dataset | Train | Validation | Test |
|---|---|---|---|
| CODEBRIM | 1,58,79,199,580 | 0,17,17,37,52 | 0,13,21,52,168 |
| S2DS | 87,240,224,276,467 | 13,32,24,38,79 | 41,93,37,52,151 |

Each stratum uses ALL test normals plus only its anomalous images; other anomaly
strata are excluded. Multiclass images enter every applicable class. Thresholds
remain validation-normal 90th percentile with >= comparison; no per-stratum
threshold fitting. Empty strata yield null metrics and counts. Per-image native
area, pixels, class membership, path and score are included in test_*_v3.json.
This supplemental schema is native512_test_strata_v1; main metrics retain
v3_shared_z_fpr_256_v3 and historical mask path 512 -> 392 -> nearest 256.
Record positive_images_with_empty_resized_mask for the 256 and native512 reports.

Future V0 runs use model_variant=v0, image_size=392, v3_evaluation, seed1,
batch16 and the original 10000-step optimizer/loss/scheduler configuration.
The ten validation z candidates are 0.1,0.25,0.5,1,2,5,10,20,50,100 percent.
Checkpoint/z selection uses validation Image AUROC then Pixel AP. FPR10 is the
primary operating threshold; validation maximum-F1 remains a secondary threshold.
CODEBRIM and DACL validation metrics determine future retention; S2DS is reporting
only. Historical Bridge results remain archived. V3b, CRMC and seeds2/3 are deferred.
No training or checkpoint evaluation is authorized before this code review.
