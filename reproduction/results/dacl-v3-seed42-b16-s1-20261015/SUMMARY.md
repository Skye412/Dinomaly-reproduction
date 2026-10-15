# DACL v3 seed-42 Dinomaly baseline

Completed full 10000-step seed-1 run, batch 16. Training began 2026-10-15
14:38:52 CST and completed at approximately 18:03:48 CST (3h25m).
Published training commit: `a1c8073fbd3a002a806b3876e24f10d2a17456be`;
runtime config confirms clean code. Separate three-step smoke passed first.

Training: 413 normal candidates only. Validation: 103 normal / 1284 anomaly.
Test: 55 normal / 920 anomaly. Input full-frame 392x392; pixel metrics 256x256.
Mask union includes 14 damage classes and excludes five object classes.
Best validation-selected checkpoint is step 10000, identical to final.

| Test metric | Percent |
|---|---:|
| Image AUROC | 58.86 |
| Image AP | 95.91 |
| Image F1-max (test oracle) | 97.15 |
| Pixel AUROC | 67.04 |
| Pixel AP | 39.21 |
| Pixel F1-max | 45.34 |
| Pixel AUPRO | 28.46 |

Validation-F1 threshold: 0.11345850676298141, frozen before test.
Test TP=920, FP=54, TN=1, FN=0. Precision=94.46%, recall=100%,
F1=97.15%, normal FPR=98.18%, specificity=1.82%, balanced accuracy=50.91%.
No positive validation/test mask became empty after resizing.

The test anomaly prevalence is 94.36%; predicting every image anomalous gives
97.10% F1. High AP/F1 therefore does not demonstrate strong discrimination.
Image AUROC and the observed false positives show a weak classification
baseline. Pixel metrics describe localization separately; absolute scores on
Bridge512 and DACL are not controlled same-test-set comparisons.

This folder retains raw validation/test score CSVs, metrics, configuration,
dataset fingerprints and final status. No weights are included. Checkpoints
remain under `/home/skye/data/Skye/Dinomaly/runs/dacl-v3-seed42-b16-s1-20261015/`.
