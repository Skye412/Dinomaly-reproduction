# Completed reproduction: cvpr2025-b16-s1-20261013

Both multi-class experiments completed 10,000 optimizer steps with exit code 0.
Settings: DINOv2-R ViT-B/14, resize 448 / crop 392, batch 16, seed 1, V100 16GB.
Training source commit: `e7b2315558cabe7cabb0794436d652cd0fbd864a`.

Times (Asia/Shanghai): MVTec 2026-10-13 20:05-22:39; VisA 2026-10-13 22:39 to 2026-10-14 01:16.

## Final metrics

All numbers below are percentages, macro-averaged across categories.

| Dataset | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel AP | Pixel F1-max | AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| mvtec | 99.61 | 99.79 | 99.09 | 98.33 | 68.49 | 68.91 | 94.53 |
| visa | 98.69 | 98.84 | 96.11 | 98.67 | 53.38 | 55.59 | 94.49 |

Raw metrics (0-1 scale), per-category results at steps 5,000 and 10,000, configuration snapshots and queue status are included in this directory. These are single-seed results.

## Model artifacts

Checkpoints remain on the server under:
`/home/skye/data/Skye/Dinomaly/runs/cvpr2025-b16-s1-20261013/{mvtec,visa}/`.
Each dataset has `final.pth`, `latest.pth`, `step_5000.pth` and `step_10000.pth`.
Checkpoints include model, optimizer, scheduler and random states; they are excluded from Git.
