# V5 retrieval decoder sequence (2026-10-19)

## Run outcome

The published commit `235b214e92035f36dc08b3e7fea0091e3636b057` completed all three V5 training/evaluation pairs and the V4 supplementary stage. Training used 10,000 steps, batch size 16, seed 1, image size 392. The sequence terminal state is `completed` and retention rule is `false`.

V4 supplementary evaluation completed for S2DS and DACL. CODEBRIM supplementary coreset evaluation was killed by the operating system with SIGKILL; the runner recorded this in `supplement_codebrim512_failure.json` and continued. This failure is separate from the V5 main experiment.

## Main V5 results

| Dataset | V5 validation selection | V5 test Image AUROC | V5 reconstruction-only Image AUROC | V0 reconstruction Image AUROC | reconstruction delta | validation delta vs V4 (Image AUROC / Pixel AP) |
|---|---:|---:|---:|---:|---:|---:|
| CODEBRIM | alpha=1, top 0.1%, step 2000 | 0.7491 | 0.81985 | 0.82041 | -0.00056 | 0.00000 / +0.00000 |
| S2DS | alpha=1, top 2%, step 2000 | 0.6523 | 0.65230 | 0.66537 | -0.01308 | -0.00454 / +0.01177 |
| DACL | alpha=1, top 100%, step 2000 | 0.7201 | 0.72006 | 0.72405 | -0.00399 | 0.00000 / -0.00000 |

The V5 fusion selection used memory-only weighting (`alpha=1`) on all three datasets. CODEBRIM and DACL therefore reproduce the V4 memory-only score at the selected operating point. S2DS is the only dataset where fusion improves pixel AP on validation, while its image AUROC is slightly lower.

## Reconstruction and paired comparisons

The reconstruction-only comparison does not support the intended decoder improvement: all three reconstruction deltas versus V0 are non-positive. The largest drop is S2DS (-0.01308 Image AUROC); CODEBRIM is effectively unchanged (-0.00056), and DACL is slightly lower (-0.00399). The paired image-stratified bootstrap intervals include zero for all three reconstruction comparisons.

For fusion versus V4, the paired test estimate is 0 for CODEBRIM and DACL and +0.02154 for S2DS, with a 95% bootstrap interval of [-0.00088, +0.04509]. The interval crosses zero. Thus V5 does not establish a reliable improvement over V4 under the frozen comparison protocol.

## Crack stratification

The detailed class and area strata are preserved in each downloaded `v5_report.json`. The weakest V5 Crack result is S2DS: Crack Image AUROC 0.46595 and FPR10 recall 0.04317. Its area-stratified FPR10 recall is 0.09756 (<0.1%), 0.05376 (0.1–0.5%), 0.10811 (0.5–1%), 0.19231 (1–5%), and 0.37748 (>=5%). CODEBRIM and DACL class/area strata remain available in their reports for follow-up tables.

## Interpretation

V5 is a valid completed reproduction of the approved queue, but it does not support retaining the retrieval-augmented decoder as the primary method. The training completed cleanly and the selection logic stayed on validation data, yet reconstruction-only performance did not improve over V0 and fusion was not reliably better than V4. The S2DS result suggests a possible pixel-level benefit from memory retrieval, but its image-level change is negative and the paired interval crosses zero. The DACL memory exclusion remains limited to the documented content-group exclusion protocol.

## Archived artifacts

`remote/` contains compact JSON reports, the terminal queue status, the comparison record, and the CODEBRIM supplementary failure record. Model weights, retrieval banks, datasets, and large runtime files were excluded.
