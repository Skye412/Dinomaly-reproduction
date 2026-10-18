# Three-dataset V0 baseline report

**Run date:** 2026-10-18
**Published code commit:** `bc107fa9aeab22714b58397036b03268c47c9bf2`
**Server:** `skye-cong`, `/home/skye/data/Skye/Dinomaly`
**Sequence status:** completed

This report summarizes the frozen V0 baseline suite. The source JSON/JSONL files are retained in this directory. Model weights, checkpoints, datasets and secrets are excluded.

## Protocol

- Model: original Dinomaly V0 with frozen DINOv2-registers ViT-B/14 encoder, original noisy bottleneck and decoder.
- Input: full 512×512 tile resized to 392×392; no spatial crop.
- Training: normal images only, batch 16, seed 1, 10,000 optimizer steps, validation every 2,000 steps.
- Optimizer/scheduler/loss: the existing V0 configuration (StableAdamW AMSGrad, original reconstruction loss and warm cosine schedule).
- Checkpoint selection: validation Image AUROC, Pixel AP as tie-breaker; test scores never select a checkpoint or threshold.
- Main pixel protocol: anomaly maps and masks evaluated at 256×256. The z/top-fraction value is selected on validation and then fixed for test.
- Primary operating point: the 90th percentile of normal validation image scores (nominal FPR10); empirical validation and test FPR can differ, especially with few normal samples. A threshold maximizing validation F1 is reported as a secondary operating point. Test F1-max is a separate oracle metric, not a deployable threshold.
- Native512 is a supplementary S2DS pixel-only evaluation. It does not select a checkpoint or threshold.

The colored Bridge masks use the union of non-black pixels: red Crack, green Spalling, blue Corrosion, yellow Efflorescence. DACL retains the official historical 14 damage labels. CODEBRIM and S2DS stratification uses original 512 masks; each area/class stratum includes all test normals plus the selected anomalous images.

## Dataset and run summary

| Dataset | Train normal | Val normal/anomaly | Test normal/anomaly | Best step | Training time | Peak allocated/reserved |
|---|---:|---:|---:|---:|---:|---:|
| CODEBRIM-512 | 1,944 | 277/123 | 555/254 | 10000 | 2.43 h | 8.73/10.19 GiB |
| S2DS-512 | 267 | 32/186 | 73/374 | 6000 | 2.43 h | 8.73/10.19 GiB |
| DACL V0 reevaluation | no training | 103/1,284 | 55/920 | 10000 | evaluation only | not recorded as training |

CODEBRIM and S2DS each completed 10,000 steps. DACL reused the existing V0 best-val checkpoint at step 10000.

## Main test results

| Dataset | z/top fraction | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel F1-max | AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CODEBRIM-512 | 10% | 82.04% | 56.74% | 68.23% | 82.88% | 28.06% | 40.09% | 53.33% |
| S2DS-512 | 0.1% | 66.54% | 89.74% | 91.22% | 87.45% | 40.05% | 50.89% | 49.89% |
| DACL | 100% | 72.41% | 97.01% | 97.24% | 67.04% | 39.21% | 45.34% | 28.46% |

The 95% bootstrap intervals for test Image AUROC are CODEBRIM **79.12%–84.91%**, S2DS **59.39%–73.21%**, and DACL **63.75%–80.41%**. These are image-stratified intervals for one seed, not across training seeds or bridge/session groups.

## Validation reference for later module decisions

| Dataset | Selected step | Validation Image AUROC | Validation Pixel AP | Empirical validation FPR at nominal FPR10 |
|---|---:|---:|---:|---:|
| CODEBRIM | 10000 | 70.24% | 18.96% | 10.11% (28/277) |
| DACL | 10000, existing frozen checkpoint | 74.98% | 37.27% | 10.68% (11/103) |
| S2DS (reporting only) | 6000 | 76.44% | 50.47% | 12.50% (4/32) |

DACL is a reevaluation of a previously validation-selected checkpoint, not a new
10000-step training run. It selects z at that fixed checkpoint; intermediate
historical checkpoints are not newly selected by this reevaluation. S2DS final
step 10000 test Image AUROC was 67.45%, but the selected step6000 result remains
66.54%; the higher test number must not replace the validation-selected result.

## Threshold behavior

| Dataset | Threshold rule | Numeric threshold | Recall | Precision | F1 | Normal false positives / normals | Test FPR | TP / FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CODEBRIM | Validation normal 90th percentile | 0.091473171 | 23.23% | 59.60% | 33.43% | 40 / 555 | 7.21% | 59 / 195 |
| CODEBRIM | Validation max-F1 | 0.074636094 | 82.68% | 57.22% | 67.63% | 157 / 555 | 28.29% | 210 / 44 |
| S2DS | Validation normal 90th percentile | 0.209980540 | 31.55% | 88.72% | 46.55% | 15 / 73 | 20.55% | 118 / 256 |
| S2DS | Validation max-F1 | 0.137490004 | 93.85% | 84.58% | 88.97% | 64 / 73 | 87.67% | 351 / 23 |
| DACL | Validation normal 90th percentile | 0.111781707 | 25.11% | 96.65% | 39.86% | 8 / 55 | 14.55% | 231 / 689 |
| DACL | Validation max-F1 | 0.074358411 | 99.57% | 94.82% | 97.14% | 50 / 55 | 90.91% | 916 / 4 |

## Area-stratified Image AUROC and FPR10 recall

### CODEBRIM-512

| Native mask area | Anomaly images | Image AUROC | FPR10 recall |
|---|---:|---:|---:|
| <0.1% | 0 | — | — |
| 0.1–0.5% | 13 | 78.66% | 23.08% |
| 0.5–1% | 21 | 75.52% | 33.33% |
| 1–5% | 52 | 74.07% | 11.54% |
| ≥5% | 168 | 85.58% | 25.60% |

### S2DS-512

| Native mask area | Anomaly images | Image AUROC | FPR10 recall |
|---|---:|---:|---:|
| <0.1% | 41 | 50.05% | 14.63% |
| 0.1–0.5% | 93 | 49.82% | 10.75% |
| 0.5–1% | 37 | 58.83% | 13.51% |
| 1–5% | 52 | 73.10% | 36.54% |
| ≥5% | 151 | 80.94% | 51.66% |

### DACL

| Native mask area | Anomaly images | Image AUROC | FPR10 recall |
|---|---:|---:|---:|
| <0.1% | 6 | 66.06% | 33.33% |
| 0.1–0.5% | 28 | 62.60% | 14.29% |
| 0.5–1% | 24 | 61.52% | 25.00% |
| 1–5% | 141 | 67.20% | 17.73% |
| ≥5% | 721 | 74.22% | 26.91% |

## Class-stratified results

For CODEBRIM and S2DS, multi-class tiles are included in every applicable class. DACL has no colored class breakdown in this protocol.

### CODEBRIM-512

| Class | Anomaly images | Image AUROC | FPR10 recall |
|---|---:|---:|---:|
| Crack | 74 | 77.43% | 16.22% |
| Spalling | 96 | 86.11% | 35.42% |
| Corrosion | 62 | 86.04% | 51.61% |
| Efflorescence | 85 | 81.36% | 9.41% |

### S2DS-512

| Class | Anomaly images | Image AUROC | FPR10 recall |
|---|---:|---:|---:|
| Crack | 139 | 48.82% | 7.91% |
| Spalling | 108 | 81.33% | 54.63% |
| Corrosion | 148 | 80.60% | 54.73% |
| Efflorescence | 70 | 68.63% | 25.71% |

`positive_images_with_empty_resized_mask` is zero in all three main reports.

## S2DS native512 supplement

The best-val checkpoint is step 6000. Using the original 512 mask resolution gives:

| Pixel AUROC | Pixel AP | Pixel F1-max | AUPRO | Pixel anomaly prevalence | Empty resized positive masks |
|---:|---:|---:|---:|---:|---:|
| 87.69% | 39.39% | 50.37% | 50.32% | 11.43% | 0 |

This is an appendix measurement only; it was not used to select the checkpoint or threshold.

The native512-minus-256 pixel changes are +0.24 pp AUROC, -0.66 pp AP,
-0.52 pp F1-max and +0.42 pp AUPRO. Thus the aggregate evidence does not show
a uniform improvement at 512. The mask path, output interpolation and smoothing
in pixel units differ; this is not a 512-input training experiment or a controlled
attribution of fine-crack performance solely to mask resizing. Empty positive
masks being zero does not prove that thin structures retained their full area.

## Interpretation and limits

- CODEBRIM has the strongest image-level test result in this suite (82.04% Image AUROC). Its FPR10 operating point gives 23.23% recall at 7.21% test FPR; the max-F1 point raises recall to 82.68% but increases FPR to 28.29%.
- S2DS has high Image AP because the test set is heavily anomalous (374/447). Its Image AUROC is 66.54%; at FPR10 calibration, recall is 31.55% and empirical test FPR is 20.55% (15/73 normals).
- DACL also has a heavily anomalous test set (920/975). Its max-F1 score is not a low-FPR operating result: the reported test FPR is 90.91% (50/55 normals). Use AUROC, pixel metrics and FPR10 together rather than interpreting F1 alone.
- The three datasets are not directly rankable by one pooled score. Future module retention is defined by CODEBRIM and DACL validation evidence; S2DS is a reporting benchmark.
- These are single-seed baseline results. They do not establish training-seed stability, bridge/session independence, or causal module gains. Future improvements must use the same frozen splits and protocol, and publish a run plan and Git commit before training.

## Source files

- CODEBRIM: `codebrim512-v0-b16-s1-review20261018/`
- S2DS: `s2ds512-v0-b16-s1-review20261018/`
- DACL reevaluation: `dacl-v0-strata-review20261018/`
- S2DS native512: `s2ds512-v0-native-review20261018/`

## Timing clarification

The full sequential execution lasted approximately 5.76 hours from CODEBRIM queue time to final sequence completion. Each training time includes periodic validation and startup before final testing. The earlier chat estimate of about four hours described monitoring time and was not the complete execution duration.
