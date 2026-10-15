# Bridge512_balance repaired dataset baseline

Started 2026-10-14T21:21:41.017685+08:00; completed 2026-10-15T00:04:00.420957+08:00.
Training source: `f3865c383001055917b22e57d3f8be762a6fbab5`.
Training: 2211 normal images only; validation: 618 (309 normal/309 defect); test: 1256 (628/628).
Source patches: 512x512; network input: full-frame resize 392x392; metrics: 256. Batch 16, seed 1, 10000 steps.

Best checkpoint: step 10000, selected by validation image AUROC; identical to final-step checkpoint.

| Test metric | Percent |
|---|---:|
| image_auroc | 80.01 |
| image_ap | 73.41 |
| image_f1_max | 77.01 |
| pixel_auroc | 82.25 |
| pixel_ap | 29.13 |
| pixel_f1_max | 40.85 |
| pixel_aupro | 51.62 |

## Validation-calibrated image threshold

Threshold: 0.07547985762357712.
image_precision: 64.00%.
image_recall: 93.15%.
image_f1: 75.88%.
Confusion matrix: {"tp": 585, "fp": 329, "tn": 299, "fn": 43}.
Normal-image false-positive rate: 52.39%.

F1-max uses an optimal threshold on test for paper-style reporting; fixed-threshold F1 uses only the validation-derived threshold.
Raw score CSVs, configurations, data fingerprints, validation metrics and run status are archived here. The primary and final checkpoint are the same, so test_final_scores.csv contains both results.
No positive test mask became empty after metric resizing.

Checkpoints remain on the server at `/home/skye/data/Skye/Dinomaly/runs/bridge512-balance-repaired-b16-s1-20261014/` (best_val.pth, final.pth, latest.pth).
This is one seed on a different dataset/test split from Bridge2893; it is not a controlled same-test-set improvement.
