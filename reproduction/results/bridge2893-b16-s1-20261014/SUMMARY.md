# Bridge2893 normal-only Dinomaly baseline

Completed 2026-10-14 13:48:50 Asia/Shanghai; started 11:18:01; approximately 2h31m.
Training source: `d8253c925e6bb9bba597293c401c03251cb0c691`.
Train: 1014 normal images only; validation: 279; test: 582. Batch 16, seed 1, 10000 steps; full-frame resize 392x392.

Primary checkpoint: step 2000, chosen by validation image AUROC, never by test metrics.

| Metric (%) | Selected by validation (step 2000) | Final step 10000 (reference) |
|---|---:|---:|
| image_auroc | 72.84 | 73.80 |
| image_ap | 68.37 | 69.81 |
| image_f1_max | 71.66 | 72.50 |
| pixel_auroc | 86.56 | 87.83 |
| pixel_ap | 22.64 | 24.84 |
| pixel_f1_max | 35.04 | 36.92 |
| pixel_aupro | 46.07 | 49.43 |

## Fixed validation-derived threshold

Primary checkpoint threshold: 0.0926210805773735.
Image precision/recall/F1: 62.70% / 80.89% / 70.64%.
Confusion matrix: {"tp": 237, "fp": 141, "tn": 148, "fn": 56}. Normal-image false-positive rate: 48.79%.
F1-max in the table uses a test-optimal threshold for paper-style reporting; it is separate from the fixed-threshold F1.

No positive test mask became empty after metric resizing. Original splits and source images were retained.
Checkpoints remain on the server at `/home/skye/data/Skye/Dinomaly/runs/bridge2893-b16-s1-20261014/` (`best_val.pth`, `final.pth`, `latest.pth`).
Included files preserve raw validation/test metrics, prediction scores, configuration, selection metadata and run status.
