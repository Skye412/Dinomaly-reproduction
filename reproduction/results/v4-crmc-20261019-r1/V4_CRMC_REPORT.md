# V4 CRMC evaluation report

Run: 2026-10-19; published commit `402fbbfbd4f96a66493ef47741b29713ef047e13`.

All three evaluations used frozen V0 checkpoints; no training was performed. Selection used validation Image AUROC then Pixel AP; test scores were not used for selection. The main setting was CLS-context retrieval, K=16, with the selected fusion and top fraction from validation.

## Main results

| Dataset | Selected fusion / α / z | V0 test Image AUROC | V4 test Image AUROC | Δ | V4 Pixel AP | Paired Δ 95% CI |
|---|---|---:|---:|---:|---:|---|
| codebrim512 | weighted_sum / 1.0 / 0.001 | 0.8204 | 0.8571 | +0.0367 | 0.3896 | [+0.0169, +0.0569] |
| dacl | weighted_sum / 1.0 / 1 | 0.7241 | 0.8087 | +0.0846 | 0.3962 | [+0.0480, +0.1232] |
| s2ds512 | weighted_sum / 0.5 / 0.001 | 0.6654 | 0.6811 | +0.0157 | 0.4379 | [-0.0008, +0.0325] |

## Validation retention check

| Dataset | V0 val I-AUROC | V4 val I-AUROC | Δ | V0 val Pixel AP | V4 val Pixel AP | Δ |
|---|---:|---:|---:|---:|---:|---:|
| codebrim512 | 0.7024 | 0.7491 | +0.0468 | 0.1896 | 0.3257 | +0.1361 |
| dacl | 0.7498 | 0.8241 | +0.0743 | 0.3727 | 0.3892 | +0.0165 |
| s2ds512 | 0.7644 | 0.7648 | +0.0003 | 0.5047 | 0.5319 | +0.0272 |

The retention rule is evaluated on CODEBRIM and DACL. Both clear the required validation margins; S2DS is reported only. Reconstruction-only Image AUROC matched the archived V0 exactly for all datasets (maximum score difference 2.98e-08; AUROC delta 0).

## Paired bootstrap

The paired bootstrap uses 2,000 image-stratified resamples, seed 1. It is an image-level interval and does not account for bridge/session clustering or training-seed variation.

- **codebrim512:** test Δ AUROC +0.0367, 95% CI [+0.0169, +0.0569]; validation Δ +0.0468 [+0.0238, +0.0718].
- **dacl:** test Δ AUROC +0.0846, 95% CI [+0.0480, +0.1232]; validation Δ +0.0743 [+0.0462, +0.1042].
- **s2ds512:** test Δ AUROC +0.0157, 95% CI [-0.0008, +0.0325]; validation Δ +0.0003 [-0.0223, +0.0247].

## False-positive attribution

| Dataset | Normal images | Covered | Uncovered | Object-containing normals | Mean top-1% overlap before → after | Enrichment before → after |
|---|---:|---:|---:|---:|---|---|
| codebrim512 | 555 | 0 | 555 | 0 | NA → NA | None → None |
| dacl | 55 | 55 | 0 | 49 | 0.3128660652324775 → 0.2619292158223455 | 3.0302619857089255 → 3.939529739119045 |
| s2ds512 | 73 | 60 | 13 | 3 | 0.021781170483460562 → 0.021450381679389317 | 0.8691782957528681 → 0.8372366765894759 |

S2DS has 13 uncovered normal images because no traceable original annotation crop exists; they are explicitly excluded from object attribution. CODEBRIM has no covered object masks under the configured attribution source, so its attribution values are unavailable. Attribution is descriptive and conditional on annotation coverage.

## Runtime and artifacts

| Dataset | Time | Peak allocated CUDA memory |
|---|---:|---:|
| codebrim512 | 98.3 min | 2.02 GiB |
| dacl | 169.9 min | 2.02 GiB |
| s2ds512 | 48.5 min | 2.02 GiB |

The original failed run is retained separately under `runs/v4-crmc-*-20261018`; it failed before evaluation because server PyTorch 1.12 does not support `argsort(stable=True)`. The rerun used a stable NumPy mergesort compatibility path.

Result JSON, CSV score files, logs, and diagnostic overlays are retained under this directory. Model weights, memory banks, and datasets are excluded.

## Complete test metrics

| Dataset | Method | Image AUROC | Pixel AUROC | Pixel AP | Pixel F1-max | AUPRO |
|---|---|---:|---:|---:|---:|---:|
| codebrim512 | reconstruction_only | 82.04% | 82.88% | 28.06% | 40.09% | 53.33% |
| codebrim512 | main | 85.71% | 87.70% | 38.96% | 47.47% | 58.19% |
| dacl | reconstruction_only | 72.41% | 67.04% | 39.21% | 45.34% | 28.46% |
| dacl | main | 80.87% | 67.73% | 39.62% | 45.82% | 28.95% |
| s2ds512 | reconstruction_only | 66.54% | 87.45% | 40.05% | 50.89% | 49.89% |
| s2ds512 | main | 68.11% | 88.10% | 43.79% | 53.54% | 51.79% |

## Operating points

Thresholds are fitted on validation; actual test FPR and recall are reported below.

| Dataset | Method | Threshold rule | Threshold | Test recall | False positives / normals | Test FPR |
|---|---|---|---:|---:|---|---:|
| codebrim512 | reconstruction_only | primary_fpr10 | 0.091473171 | 23.23% | 40/555 | 7.21% |
| codebrim512 | reconstruction_only | legacy_f1 | 0.074636094 | 82.68% | 157/555 | 28.29% |
| codebrim512 | main | primary_fpr10 | 2.8630488 | 41.73% | 41/555 | 7.39% |
| codebrim512 | main | legacy_f1 | 2.3073206 | 81.89% | 140/555 | 25.23% |
| dacl | reconstruction_only | primary_fpr10 | 0.11178171 | 25.11% | 8/55 | 14.55% |
| dacl | reconstruction_only | legacy_f1 | 0.074358411 | 99.57% | 50/55 | 90.91% |
| dacl | main | primary_fpr10 | 0.69057198 | 48.15% | 7/55 | 12.73% |
| dacl | main | legacy_f1 | -0.41841277 | 99.46% | 40/55 | 72.73% |
| s2ds512 | reconstruction_only | primary_fpr10 | 0.20998054 | 31.55% | 15/73 | 20.55% |
| s2ds512 | reconstruction_only | legacy_f1 | 0.13749 | 93.85% | 64/73 | 87.67% |
| s2ds512 | main | primary_fpr10 | 3.4351436 | 35.29% | 13/73 | 17.81% |
| s2ds512 | main | legacy_f1 | 1.1933945 | 94.92% | 69/73 | 94.52% |

## Prespecified ablations

Each row uses its own validation-selected fusion/z. These are reports, not test-based configuration selection.

| Dataset | Ablation | K | Mode | Alpha | z | Val I-AUROC | Val Pixel AP | Test I-AUROC | Test Pixel AP |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| codebrim512 | main | 16 | weighted_sum | 1.0 | 0.001 | 74.91% | 32.57% | 85.71% | 38.96% |
| codebrim512 | retrieval_4 | 4 | weighted_sum | 1.0 | 0.001 | 75.68% | 31.25% | 84.91% | 38.55% |
| codebrim512 | retrieval_16 | 16 | weighted_sum | 1.0 | 0.001 | 74.91% | 32.57% | 85.71% | 38.96% |
| codebrim512 | retrieval_64 | 64 | weighted_sum | 1.0 | 0.001 | 73.90% | 31.02% | 85.03% | 39.05% |
| codebrim512 | retrieval_global | global | weighted_sum | 0.25 | 0.05 | 70.41% | 20.60% | 82.64% | 30.65% |
| codebrim512 | weighted_sum | 16 | weighted_sum | 1.0 | 0.001 | 74.91% | 32.57% | 85.71% | 38.96% |
| codebrim512 | gated | 16 | gated | None | 0.05 | 72.72% | 24.50% | 84.41% | 34.42% |
| codebrim512 | reconstruction_only | 16 | weighted_sum | 0 | 0.1 | 70.24% | 18.96% | 82.04% | 28.06% |
| codebrim512 | memory_only | 16 | weighted_sum | 1.0 | 0.001 | 74.91% | 32.57% | 85.71% | 38.96% |
| codebrim512 | mixed_sum | 16 | weighted_sum | 0.75 | 0.05 | 74.00% | 28.21% | 86.04% | 37.00% |
| dacl | main | 16 | weighted_sum | 1.0 | 1.0 | 82.41% | 38.92% | 80.87% | 39.62% |
| dacl | retrieval_4 | 4 | weighted_sum | 1.0 | 1.0 | 79.95% | 36.15% | 80.58% | 37.06% |
| dacl | retrieval_16 | 16 | weighted_sum | 1.0 | 1.0 | 82.41% | 38.92% | 80.87% | 39.62% |
| dacl | retrieval_64 | 64 | weighted_sum | 1.0 | 1.0 | 82.22% | 40.49% | 80.35% | 41.46% |
| dacl | retrieval_global | global | weighted_sum | 1.0 | 1.0 | 81.63% | 41.29% | 79.22% | 42.45% |
| dacl | weighted_sum | 16 | weighted_sum | 1.0 | 1.0 | 82.41% | 38.92% | 80.87% | 39.62% |
| dacl | gated | 16 | gated | None | 1.0 | 78.83% | 39.09% | 76.94% | 40.59% |
| dacl | reconstruction_only | 16 | weighted_sum | 0 | 1.0 | 74.98% | 37.27% | 72.41% | 39.21% |
| dacl | memory_only | 16 | weighted_sum | 1.0 | 1.0 | 82.41% | 38.92% | 80.87% | 39.62% |
| dacl | mixed_sum | 16 | weighted_sum | 0.75 | 1.0 | 80.85% | 39.39% | 79.11% | 40.57% |
| s2ds512 | main | 16 | weighted_sum | 0.5 | 0.001 | 76.48% | 53.19% | 68.11% | 43.79% |
| s2ds512 | retrieval_4 | 4 | gated | None | 0.001 | 76.63% | 51.63% | 68.25% | 43.22% |
| s2ds512 | retrieval_16 | 16 | weighted_sum | 0.5 | 0.001 | 76.48% | 53.19% | 68.11% | 43.79% |
| s2ds512 | retrieval_64 | 64 | weighted_sum | 1.0 | 0.001 | 78.23% | 56.07% | 70.27% | 47.87% |
| s2ds512 | retrieval_global | global | weighted_sum | 0.5 | 0.001 | 76.53% | 53.94% | 67.22% | 44.19% |
| s2ds512 | weighted_sum | 16 | weighted_sum | 0.5 | 0.001 | 76.48% | 53.19% | 68.11% | 43.79% |
| s2ds512 | gated | 16 | gated | None | 0.001 | 76.48% | 52.90% | 67.61% | 43.34% |
| s2ds512 | reconstruction_only | 16 | weighted_sum | 0 | 0.001 | 76.44% | 50.47% | 66.54% | 40.05% |
| s2ds512 | memory_only | 16 | weighted_sum | 1.0 | 0.02 | 76.02% | 54.37% | 70.26% | 46.82% |
| s2ds512 | mixed_sum | 16 | weighted_sum | 0.5 | 0.001 | 76.48% | 53.19% | 68.11% | 43.79% |

## Interpretation limits

CODEBRIM and DACL selected alpha=1: the main result is memory-only scoring, not evidence that reconstruction-memory fusion helps on those two datasets. S2DS selected alpha=0.5, but its paired test AUROC interval crosses zero. DACL object-containing normal-image hotspot overlap decreased, while mean per-image enrichment increased (3.03 to 3.94); the attribution evidence therefore does not establish uniform suppression of non-damage object hotspots. S2DS object-containing normals number only three. No additional tuning or experiments were launched.
