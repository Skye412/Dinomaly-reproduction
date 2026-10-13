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
