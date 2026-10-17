# V0 re-evaluation with V3 shared z-grid v2

Evaluation only; no smoke or full training. Protocol `v3_shared_z_fpr_256_v2`.

| Dataset | Selected z | Test Image AUROC | Test Pixel AUROC | Pixel AP | AUPRO | Validation actual FPR |
|---|---:|---:|---:|---:|---:|---:|
| bridge512 | 10.0% | 80.1452% | 82.2467% | 29.1303% | 51.6178% | 10.0324% |

### bridge512 z grid

| z | Validation Image AUROC |
|---:|---:|
| 0.10% | 75.1783% |
| 0.25% | 75.3972% |
| 0.50% | 75.5365% |
| 1.00% | 75.6475% |
| 2.00% | 75.7910% |
| 5.00% | 76.0539% |
| 10.00% | 76.1754% |

| dacl | 10.0% | 62.7490% | 67.0428% | 39.2098% | 28.4574% | 10.6796% |

### dacl z grid

| z | Validation Image AUROC |
|---:|---:|
| 0.10% | 55.6793% |
| 0.25% | 56.6661% |
| 0.50% | 57.5439% |
| 1.00% | 58.5314% |
| 2.00% | 59.8002% |
| 5.00% | 61.5741% |
| 10.00% | 63.3223% |

Both datasets selected z=10%, the upper edge of this expanded grid. The grid remains a validation-only choice. Pixel metrics do not change with z; z affects image-level score and checkpoint/z selection during V3.
Bridge fixed z=1% sanity output remains in the earlier v1 directory; this directory is the v2 reference for retention.
No full training was launched.
