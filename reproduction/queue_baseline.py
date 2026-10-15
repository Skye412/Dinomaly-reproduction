"""Wait without disturbing other GPU jobs; smoke and execute the published baseline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['expected_commit', 'run_id', 'data_path', 'dataset_profile', 'split_view']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--smoke_first', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    folder = repo / 'runs' / args.run_id
    status = dict(pid=os.getpid(), commit=args.expected_commit, run_id=args.run_id,
                  state='waiting_for_gpu', submitted_at=time.time(), required_free_mib=12288)

    def update(**kwargs):
        status.update(kwargs, updated_at=time.time())
        temporary = folder / 'queue_status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(folder / 'queue_status.json')

    def verify():
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo), text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(repo), text=True).strip()
        if head != args.expected_commit or dirty:
            raise RuntimeError('Queued checkout changed; refusing unpublished or dirty experiment')

    def wait_gpu():
        while True:
            verify()
            output = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                              '--format=csv,noheader,nounits'], text=True)
            free = int(output.strip())
            update(state='waiting_for_gpu', free_mib=free)
            if free >= 12288:
                return
            time.sleep(30)

    command = [sys.executable, '-u', 'dinomaly_bridge.py', '--data_path', args.data_path,
               '--dataset_profile', args.dataset_profile, '--split_view', args.split_view,
               '--batch_size', '16', '--seed', '1', '--total_iters', '10000',
               '--eval_interval', '2000', '--checkpoint_interval', '1000']
    try:
        update()
        wait_gpu()
        if args.smoke_first:
            smoke_id = args.run_id + '-smoke'
            smoke_dir = repo / 'runs' / smoke_id
            smoke_dir.mkdir(parents=True, exist_ok=True)
            update(state='smoke_running')
            with (smoke_dir / 'console.log').open('x') as log:
                subprocess.run(command + ['--save_name', smoke_id, '--smoke_steps', '3'],
                               cwd=str(repo), stdout=log, stderr=subprocess.STDOUT, check=True)
            smoke_status = json.loads((smoke_dir / 'status.json').read_text())
            if smoke_status['state'] != 'completed' or smoke_status['mode'] != 'smoke':
                raise RuntimeError('Smoke did not complete successfully')
            update(smoke_passed=True)
            wait_gpu()
        verify()
        update(state='training_running', launched_at=time.time())
        with (folder / 'training_console.log').open('x') as log:
            subprocess.run(command + ['--save_name', args.run_id], cwd=str(repo), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        result = json.loads((folder / 'status.json').read_text())
        if result['state'] != 'completed':
            raise RuntimeError('Trainer exited without completed status')
        archive = repo / 'reproduction' / 'results' / args.run_id
        archive.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            if path.suffix in {'.json', '.jsonl', '.csv'} and path.name != 'queue_status.json':
                shutil.copy2(str(path), str(archive / path.name))
        update(state='completed', finished_at=time.time())
    except BaseException as exc:
        update(state='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
