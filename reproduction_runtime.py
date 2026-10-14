"""Record experiment provenance, metrics and recoverable training checkpoints."""
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

METRIC_NAMES = ['image_auroc', 'image_ap', 'image_f1_max', 'pixel_auroc',
                'pixel_ap', 'pixel_f1_max', 'pixel_aupro']


def add_arguments(parser):
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_iters', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=5000)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--smoke_steps', type=int, default=0,
                        help='Stop after this many optimizer steps without changing the training schedule.')


def record_run(args, encoder_name):
    if min(args.batch_size, args.total_iters, args.eval_interval, args.checkpoint_interval) <= 0:
        raise ValueError('Batch size, iterations and intervals must be positive')
    if args.smoke_steps < 0:
        raise ValueError('smoke_steps must be non-negative')
    folder = Path(args.save_dir) / args.save_name
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), encoder=encoder_name, resize=getattr(args, 'image_size', 448), crop=392,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), torch=torch.__version__,
                  cuda_runtime=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
                  mask_binarization='gt > 0', pixel_metric_resolution=256)
    config['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    config['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    (folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')


def save_checkpoint(args, model, optimizer, scheduler, step, filename='latest.pth'):
    folder = Path(args.save_dir) / args.save_name
    target = folder / filename
    temporary = str(target) + '.tmp'
    torch.save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'args': vars(args),
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}}, temporary)
    os.replace(temporary, target)


def record_metrics(args, step, category, values):
    record = dict(step=step, category=category)
    record.update(zip(METRIC_NAMES, [float(v) for v in values]))
    with (Path(args.save_dir) / args.save_name / 'metrics.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
