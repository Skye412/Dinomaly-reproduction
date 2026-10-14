# CVPR 2025 reproduction

Based on `guojiajeremy/Dinomaly` at commit `1f252be03a918789b19848f0ca37166e7c28dada`.
The original Apache-2.0 license and author attribution are retained.

The multi-class runs use DINOv2-R ViT-B/14, resize 448 / crop 392, seed 1,
batch size 16, and 10,000 optimizer steps per dataset. Architecture, loss,
optimizer and learning-rate schedule follow the upstream implementation.
The upstream evaluation retains mask binarization `gt > 0` and resolution 256.

Changes add device/configuration arguments, finite-loss checks, progress logs,
machine-readable metrics, and atomic checkpoints including optimizer/scheduler
and random states. No automatic resume is implemented. Evaluation occurs at
5,000 and 10,000 steps; `latest.pth` is saved every 1,000 steps, milestone
checkpoints before evaluation, and `final.pth` after the final evaluation.
`colorama==0.4.6` supplies a dependency missing from the upstream requirements.

See `setup/README.md` for the initial environment and dataset audit. Datasets,
pretrained weights, run outputs, and the paper PDF are excluded from Git.

```bash
conda create -n dinomaly python=3.8.12 pip=24.0 -y
conda activate dinomaly
pip install -r requirements.txt
# Run each smoke check before the full suite; the 10,000-step schedule is retained.
python dinomaly_mvtec_uni.py --data_path /home/skye/data/Skye/databases/MVTEC --save_dir runs/smoke --save_name mvtec --smoke_steps 3
python dinomaly_visa_uni.py --data_path /home/skye/data/Skye/databases/VISA_pytorch/1cls --save_dir runs/smoke --save_name visa --smoke_steps 3
# Run on Linux; MVTec completes before VisA starts.
nohup python -u reproduction/run_suite.py --run_id cvpr2025-b16-s1 > suite.log 2>&1 < /dev/null &
```

The encoder downloads `dinov2_vitb14_reg4_pretrain.pth` from Meta's official
server if absent from `backbones/weights/`. The GPU lock protects against a
second suite launched by this runner. `runs/<run_id>/status.json` records queue
state and process IDs. Dataset outputs contain `config.json`, `metrics.jsonl`,
`log.txt`, and checkpoints.

Hardware: Tesla V100 16GB rather than the upstream RTX 3090 24GB. A reduced
batch size, if required, must be recorded as a deviation from the paper.
Completed smoke checks demonstrate execution, not reproduction of final metrics.

Both datasets passed three optimizer steps with batch 16 on 2026-10-13.
Peak allocated CUDA memory was 8.733 GiB in each test. The official register-token
backbone checksum and launch settings are recorded in `launch_record.json`.

Completed run records are archived in `results/cvpr2025-b16-s1-20261013/`.
See that directory's `SUMMARY.md` for the final metrics and server artifact path.

Before every subsequent full experiment, commit and push its code and run plan,
verify the published commit, then synchronize the server and start training.
This also applies to reruns with unchanged code; record a fresh run plan.
The standing user instruction is recorded in the root `AGENTS.md`.

## Bridge2893 baseline

`dinomaly_bridge.py` trains one binary anomaly model using only the 1,014 normal
training images in the frozen Bridge2893 seed-42 split. The 1,018 annotated
training images are excluded. Validation uses 279 images; test uses 582 images.
The backbone, bottleneck, decoder, loss and optimizer follow the upstream
Dinomaly-B implementation. The full image is resized to 392x392 to retain edge
defects, rather than applying the original resize-448/center-crop-392 transform.
This domain adaptation of preprocessing is explicitly recorded in the run plan.

Train for 10,000 steps with batch 16 and seed 1. Validate every 2,000 steps;
select the best checkpoint by validation image AUROC, with pixel AP as a tie-break.
Choose its image-level decision threshold by validation F1, and reuse the fixed
threshold on test. Only after training, evaluate the selected checkpoint and
the final-step checkpoint on test. Test F1-max is also reported as an oracle
metric for comparison with the original paper, separate from the fixed-threshold
test F1. Binary ground truth combines all annotated defect categories.

```bash
python dinomaly_bridge.py --save_name bridge2893-smoke-20261014 --smoke_steps 3
# Publish and synchronize the run plan before launching this full experiment:
python -u dinomaly_bridge.py --save_name bridge2893-b16-s1-20261014
```

See `plans/bridge2893-b16-s1-20261014.json` for the frozen protocol. Three smoke
optimizer steps, evaluation on a normal/defect validation subset, and a complete
checkpoint save/load succeeded on 2026-10-14. These checks are execution evidence,
not final baseline performance. Run artifacts are under
`runs/bridge2893-b16-s1-20261014/`, including status, dataset protocol, validation
and test image scores, best-checkpoint selection, metrics, and final summary.

For an SSH-independent background process, after verifying GitHub HEAD and
synchronizing a clean server checkout, run
`python reproduction/launch_bridge.py --expected_commit <verified-commit>`.
The launcher refuses mismatched commits, dirty worktrees and duplicate run logs.
