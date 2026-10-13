"""Detached, sequential MVTec/VisA runner with durable status and GPU lock."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--data_root', default='/home/skye/data/Skye/databases')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('run_id must be a directory name')
    repo = Path(__file__).resolve().parent.parent
    folder = repo / 'runs' / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    status_file = folder / 'status.json'
    lock = (repo / 'runs/gpu0.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status_file.exists():
        raise RuntimeError('Run already exists; use a new run_id')
    status = dict(run_id=args.run_id, runner_pid=os.getpid(), started_at=now(), state='running',
                  batch_size=args.batch_size, results={})

    def save():
        status['updated_at'] = now()
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(status_file)

    save()
    try:
        for dataset, path in [('mvtec', 'MVTEC'), ('visa', 'VISA_pytorch/1cls')]:
            command = [sys.executable, '-u', 'dinomaly_{}_uni.py'.format(dataset),
                       '--data_path', str(Path(args.data_root) / path),
                       '--save_dir', str(folder), '--save_name', dataset,
                       '--device', 'cuda:0', '--seed', '1', '--batch_size', str(args.batch_size),
                       '--total_iters', '10000', '--eval_interval', '5000', '--checkpoint_interval', '1000']
            status['current_dataset'] = dataset
            status['results'][dataset] = dict(state='running', started_at=now(), command=command)
            with (folder / (dataset + '-console.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=str(repo), stdout=log, stderr=subprocess.STDOUT)
                status['child_pid'] = child.pid
                save()
                code = child.wait()
            status['results'][dataset].update(state='completed' if code == 0 else 'failed',
                                               exit_code=code, finished_at=now())
            save()
            if code != 0:
                status['state'] = 'failed'
                save()
                return code
        status['state'] = 'completed'
        status['finished_at'] = now()
        save()
        return 0
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
