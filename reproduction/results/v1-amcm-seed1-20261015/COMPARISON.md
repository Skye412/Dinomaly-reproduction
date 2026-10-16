# V0 / V1 AMCM seed-1 comparison

Historical 392-input / 256-metric protocol for both versions. No evaluator correction, test selection, or automatic tuning.

| Dataset | Version | I-AUROC % | P-AUROC % | P-AP % | P-F1max % | AUPRO % | Recall % | Normal FP / N | Best step | Wall hours | Peak allocated GiB | Added params |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Bridge512 | V0 | 80.01 | 82.25 | 29.13 | 40.85 | 51.62 | 93.15 | 329 / 628 | 10000 | 2.705 | 8.732 | 0 |
| Bridge512 | V1 | 80.48 | 82.76 | 30.01 | 41.49 | 52.92 | 92.04 | 302 / 628 | 10000 | 2.999 | 9.604 | 2718964 |
| DACL | V0 | 58.86 | 67.04 | 39.21 | 45.34 | 28.46 | 100.00 | 54 / 55 | 10000 | 3.416 | 8.732 | 0 |
| DACL | V1 | 58.59 | 66.95 | 39.16 | 45.26 | 27.05 | 99.67 | 54 / 55 | 10000 | 3.716 | 9.604 | 2718964 |

Bridge512 V1-V0 deltas (percentage points): image_auroc +0.48, pixel_auroc +0.52, pixel_ap +0.88, pixel_f1_max +0.64, pixel_aupro +1.31


DACL V1-V0 deltas (percentage points): image_auroc -0.27, pixel_auroc -0.09, pixel_ap -0.05, pixel_f1_max -0.08, pixel_aupro -1.41

V0 memory peaks are original server log values rounded to 0.001 GiB; V1 retains raw JSON values.
DACL image F1/AP are prevalence sensitive; assess image AUROC, pixel AP/AUPRO and recall with false positives.
A single seed is module-screening evidence, not a stability claim. No further experiments are launched.
