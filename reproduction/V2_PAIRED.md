# V2: V0 Dinomaly with paired feature denoising

User authorization: 2026-10-16. AMCM stays an archived V1 ablation and is not
part of the default V0 or V2 model. V2 adds no inference parameters. This is a
FoundAD-style training adaptation, not a reproduction of the FoundAD network.

## Source and differences

Reference: ymxlzgy/FoundAD, commit
89ca4dc3709ae517b47eba880a615cb032bcd7aa, foundad/src/train.py and
foundad/src/utils/synthesis.py. Exact source copies and SHA256 are retained in
references/foundad/ for inspection only; license is licenses/FoundAD-LICENSE.txt.
Our paired_denoising.py uses the agreed CutPaste ranges and jitter 0.5,
chooses Normal/Scar per image at equal probability, and samples donors from
that same image. Scar uses expanded bilinear rotation and rectangular paste,
including rotated padding, as in the pinned reference. Full-frame legal paste
locations replace industrial-category foreground heuristics. No masks,
validation/test images or anomalous training images participate in synthesis.

Unlike the reference's normalized input processing, synthesis occurs on raw
[0,1] RGB after deterministic full-frame 392 resize. Clean and context share
coordinates and are normalized once each with V0 ImageNet constants. Every
training step samples one whole clean/perturbed batch with 0.5 probability.
No retries enforce a nonempty perturbation: attempted, effective, unchanged,
changed-pixel and Normal/Scar counts are recorded honestly in paired_stats.json.
The clean target is cloned before synthesis, with a runtime equality assertion
that synthesis did not mutate the clean input. No separate small run is used.

## Clean feature target and unchanged inference

ViTill.encode_tokens factors the original encoder loop without changing its
operations. target_features uses only the frozen encoder and original layer
groups, under no_grad; it never executes bottleneck or decoder. Perturbed
steps compare one context reconstruction to these clean target features.
Clean steps reuse the original forward's detached teacher features and do not
run an extra encoder. Each optimizer update therefore has exactly one batch16
through trainable bottleneck/decoder. The original cosine loose loss, AdamW,
schedule, clipping, feature groups, input size, anomaly score, validation model
selection and historical 256 metric code remain fixed. V2 inference is V0.

## Execution and evidence

Bridge 2211 normals then DACL 413 normal candidates, separately initialized
from the same pretrained encoder, seed1 batch16 and 10000 steps each. Run IDs:
v2-paired-bridge512-b16-s1-20261016 and v2-paired-dacl-b16-s1-20261016.
Each run has a separately published plan commit and immutable clean worktree.
The launch path preserves the existing GPU wait/lock/memory-recheck behavior,
separate logs and checkpoints, error states and no automatic retry. Current
SSH is skye-cong (120.26.32.49:12139) as selected by the user. No other jobs stop.

No independent smoke, subset, model-forward check or short training is run.
Only static Python AST/JSON and code review precede full-run submission.
During full training, pairing/shape/finite-loss and gradient guards stay active.
Extra clean-encoder batches and synchronized elapsed seconds are accumulated
only on perturbed steps, alongside full run wall time and peak CUDA memory.

Raw metrics.jsonl preserves every scheduled validation point; compare_v2.py
exports validation_curves.csv for all three versions and both datasets, plus
V0/V1/V2 tables and paired training-cost records. console.log/log.txt, configs,
raw scores and masks' protocol are archived, excluding weights. Deferred ground
truth resizing fixes must re-evaluate V0/V1/V2 together. No V3 or tuning follows
automatically, whatever the outcome.

## Authorized r1 restart

The initial Bridge attempt failed a raw RGB range assertion after recorded
step 1, before any validation; DACL did not start. It is an implementation
failure record, excluded from comparison. The original cause is unconfirmed.
The r1 runs use fresh seed-1 initialization and new run/sequence IDs ending
`-r1`. The 1e-6 guard checks finite values and range before clamping.
First endpoint events log step, min/max, dtype and element count immediately;
paired_stats records cumulative corrected batches/elements. Severe events
record manifest sample/image identities, Normal/Scar metadata and local
range_failure.pt tensors (never uploaded). The index-carrying dataset wrapper
preserves sample order and random operations. No standalone runtime checks.
Only completed r1 summaries enter comparison; failure stops the sequence.
