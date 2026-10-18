# V4 CRMC evaluation protocol (review-gated)

## Scope

V4 is evaluation-only on the three completed V0 checkpoints:

- `Bridge512_balance_CODEBRIM` / `codebrim512`
- `Bridge512_balance_S2DS` / `s2ds512`
- `DACL10k_v3_512_split_seed42` / `dacl_v3_seed42`

No training, optimizer update, checkpoint replacement, or test-based selection is
allowed. The V0 checkpoint and preprocessing remain frozen.

## Memory bank

The bank is built from normal training images only. For each image it stores fp16
CPU tensors for the V0 target-layer-9 CLS descriptor and the two fused patch-target
groups. Patch and CLS vectors are cosine-normalized at retrieval time. The bank
metadata records the source commit, sample count, grid and dtype.

## Retrieval and anomaly maps

The main setting is CLS-context retrieval with `K=16`. For each query patch, the
nearest vector is searched exactly in the selected `K x 784` patch set for each
fused group; the two group distances are averaged. Full-bank retrieval and `K=4,64`
are ablations. Retrieval is chunked to bound GPU memory and does not use an
approximate index.

The reconstruction map and memory map are independently standardized with mean and
standard deviation computed from validation normal images. Weighted fusion is
`(1-alpha)*A_rec + alpha*A_mem`, with alpha in `{0,.25,.5,.75,1}`. Gated fusion is
`raw_nonnegative_A_rec * sigmoid(standardized_A_mem)` and is evaluated as a separate family.

## Selection and outputs

The main V4 selection is restricted to the requested main setting (`K=16`,
CLS-context), and jointly selects fusion family, alpha and top-z on validation by
Image AUROC then Pixel AP. K/context alternatives remain recorded as ablations.
The test set is never used for selection. The runner writes `crmc_report.json`,
`test_best_val_v3.json`, `test_scores.csv`, and the fp16 bank metadata/tensors.

Thresholds are the validation-normal 90th percentile (primary FPR10) and validation
maximum-F1 (secondary). Pixel metrics use the shared 256 protocol.

## Diagnostics

DACL non-damage polygons are Bearing, Drainage, EJoint, JTape and PEquipment;
their top-1% normal-score coverage is measured before/after CRMC. The frozen S2DS
manifest does not contain object masks. An audit against the original S2DS directory
matched 60/73 frozen test-normal tiles by source ID and crop coordinates; 13 normal
tiles have no source image in the original directory. Only the matched subset can
be used for vegetation/control-point attribution, and unmatched cases remain
explicitly uncovered. CODEBRIM/S2DS top-20 normal overlays are an optional visual
artifact to generate after the code review; they are not used for selection.

## Required ablation matrix

`K={4,16,64}`, context vs full-bank retrieval, weighted sum vs gated fusion,
reconstruction-only (`alpha=0`), memory-only (`alpha=1`) and the main fused setting.
All rows are evaluation-only and report the same main metrics and stratified output.

## Retention rule

Compare V4 to the archived V0 using CODEBRIM and DACL validation evidence: neither
validation Image AUROC nor Pixel AP may drop by 0.5 percentage points or more, and
at least one dataset/metric must improve by at least 1.0 percentage point. S2DS is
reported only because its validation normal set has 32 images.

User approved execution after gate correction and archived V0 AUROC consistency check. Main and validation paired bootstrap use 2000 image-stratified resamples. Object attribution includes area random baseline, enrichment, and object-containing-only normals. Full-bank search remains chunked CPU-to-GPU exact retrieval; no runtime speed guarantee.
