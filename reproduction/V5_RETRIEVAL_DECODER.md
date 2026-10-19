# V5 retrieval-enhanced Dinomaly (review draft)

V5 keeps Dinomaly as the base model and trains one retrieval cross-attention block between the noisy bottleneck and the original eight-layer decoder. The DINOv2-registers encoder remains frozen. The new block has 2,366,976 trainable parameters (12 heads, 768 channels): linearized ELU-kernel attention, residual projection and LayerNorm.

The retrieval context is built once from training normal images. Each reference contributes the mean of the two V0 fused target groups after 2x2 average pooling: 196 patch tokens per image. K=16 therefore supplies 3,136 context tokens. During training, the provider excludes the query itself and every training row with the same manifest `parent_id` (`group_id` for DACL). A hard assertion checks that no same-parent neighbor leaks. Validation and test retrieve from the complete training-normal bank.

V5 evaluation selects fusion mode, alpha and top fraction on validation only. It reports both the V5 reconstruction-only map and V5 reconstruction plus memory map. The V5 reconstruction-only value is compared with the archived V0 score, but a difference is reported rather than treated as a failure: this is the experiment's main test of whether retrieval changes what Dinomaly learns.

The same published sequence first runs V4 supplementary evaluation on the frozen V0 checkpoints: full-bank memory-only, a deterministic global greedy 10% coreset comparison, and native 512 overlapping block memory (288 block, 224 stride) with validation-selected max/mean stitching. The coreset is explicitly a style comparison using the same DINO features; it is not claimed as a complete reproduction of any external implementation.

All selection and threshold rules remain validation-only. The fixed data profiles, manifest hashes, 392 input, 256 metric resolution, seed 1, batch 16, 10,000 steps, loss, optimizer and scheduler are unchanged. The run plan is `reproduction/plans/v5_retrieval_decoder_20261019.json`.


## Review and launch boundary

This commit is review-only. It does not start a GPU task or install an automation. After user review, launch with `python reproduction/run_v5_sequence.py --approved_commit <full-reviewed-SHA>`. The runner verifies published master/HEAD equality and a clean tracked checkout, preserves independent run directories, logs every subprocess, and stops on the first failure. It never automatically retries.

The approved queue order is CODEBRIM V5 train/eval, S2DS V5 train/eval, DACL V5 train/eval, then all three V4 supplements. Supplement failures are recorded independently and do not invalidate completed V5 runs or block later supplements. Every full training uses fresh reconstruction initialization, not V0 trained decoder weights. Checkpoint/fusion/alpha/z are jointly selected at each scheduled validation; reconstruction-only is reported at the selected checkpoint. Its own separately selected checkpoint is not claimed.

## Limits requiring review

- DACL `parent_id` comes from the frozen manifest's `group_id`: exact content identity, not verified bridge/session ancestry. Self and equal group are excluded, but stronger scene-level exclusion cannot be promised without additional metadata. CODEBRIM/S2DS retain their original parent IDs.
- The two target groups are averaged to one 768-channel reference representation before pooling, maintaining the specified 196 tokens per reference rather than doubling context length.
- The coreset uses global greedy k-center over all normal patches in a seeded Gaussian projection (64 dimensions). Scoring uses full concatenated raw-group Euclidean nearest-neighbor distances. Original PatchCore backbone, local pooling and reweighting are not reproduced; this is a labeled style comparison.
- Multi-scale overlap is averaged to avoid overlap-dependent score inflation; whole/local map fusion (max or mean) and z are selected on validation. Pixel maps follow the shared 256 protocol.
- Global greedy selection of 10% from ~1.52M CODEBRIM patches can be expensive. Together with repeated pixel/AUPRO metrics and multiscale searches, an overnight finish cannot be guaranteed. No random coreset substitute is used.

## Verification performed

Python compilation and CPU unit checks cover linear-attention equivalence to dense kernelized cross-attention, finite gradients for every new parameter, self/parent exclusion, insufficient-neighbor rejection, pooling shape and values, exact deterministic coreset count and overlap stitching. No smoke, small-sample training, formal training or dataset evaluation was launched. The same three CPU unit checks also passed on server PyTorch 1.12 from a temporary review directory, without model training, dataset evaluation or GPU use.
