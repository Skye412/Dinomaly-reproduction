"""Full V1 runs only: immutable published checkouts, Bridge then independent DACL."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def git(repo, *args):
    return subprocess.check_output(['git'] + list(args), cwd=str(repo), text=True).strip()


def verify(repo, commit):
    if git(repo, 'rev-parse', 'HEAD') != commit or git(repo, 'status', '--porcelain'):
        raise RuntimeError('Expected clean published checkout: ' + str(repo))


def archive_run(checkout, run_id, archive_root):
    target = archive_root / run_id
    target.mkdir(parents=True, exist_ok=True)
    for source in (checkout / 'runs' / run_id).iterdir():
        if source.suffix in {'.json', '.jsonl', '.csv', '.log', '.txt'}:
            shutil.copy2(str(source), str(target / source.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge_commit', required=True)
    parser.add_argument('--dacl_commit', required=True)
    parser.add_argument('--sequence_id', default='v1-amcm-seed1-20261015-launchfix1')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    folder = repo / 'runs' / args.sequence_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'sequence.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'status.json').exists():
        raise RuntimeError('Sequence already exists; refusing a duplicate launch')
    status = dict(pid=os.getpid(), state='queued', submitted_at=time.time(),
                  smoke_tests=False, tuning_restarts=False)

    def update(**kwargs):
        status.update(kwargs, updated_at=time.time())
        tmp = folder / 'status.tmp'
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(folder / 'status.json')
        print(json.dumps(dict(timestamp=status['updated_at'], **kwargs)), flush=True)

    runs = [dict(run_id='v1-amcm-bridge512-b16-s1-20261015', profile='bridge512_balance_repaired',
                 dataset='Bridge512_balance', commit=args.bridge_commit),
            dict(run_id='v1-amcm-dacl-b16-s1-20261015', profile='dacl_v3_seed42',
                 dataset='DACL10k_v3_512_split_seed42', commit=args.dacl_commit)]
    try:
        update(runs=runs)
        # Both refs are fetched from the locally verified published Git bundle.
        # Isolated worktrees keep the second run clean when the first is archived.
        for run in runs:
            checkout = folder / ('checkout-' + run['profile'])
            subprocess.run(['git', 'worktree', 'add', '--detach', str(checkout), run['commit']],
                           cwd=str(repo), check=True)
            (checkout / 'backbones').mkdir(exist_ok=True)
            (checkout / 'backbones' / 'weights').symlink_to(repo / 'backbones' / 'weights', target_is_directory=True)
            (checkout / 'runs').mkdir(exist_ok=True)
            # All our runners share the same device lock, including legacy V0.
            (checkout / 'runs' / 'gpu0.lock').symlink_to(repo / 'runs' / 'gpu0.lock')
            verify(checkout, run['commit'])
        for index, run in enumerate(runs):
            checkout = folder / ('checkout-' + run['profile'])
            run_folder = checkout / 'runs' / run['run_id']
            run_folder.mkdir()
            plan = json.loads((checkout / 'reproduction' / 'plans' / (run['run_id'] + '.json')).read_text())
            assert plan['publication_stage'] == 'ready' and plan['smoke_steps'] == 0
            assert plan['training']['optimizer_steps'] == 10000
            expected_hashes = plan['manifest_sha256']
            module_hash = hashlib.sha256((checkout / 'models' / 'amcm_official.py').read_bytes()).hexdigest()
            if module_hash != plan['amcm']['source_sha256']:
                raise RuntimeError('Pinned official AMCM source changed')
            data = Path('/home/skye/data/Skye/databases') / run['dataset']
            while True:
                verify(checkout, run['commit'])
                for split, expected in expected_hashes.items():
                    actual = hashlib.sha256((data / (split + '_manifest.csv')).read_bytes()).hexdigest()
                    if actual != expected:
                        raise RuntimeError('Frozen manifest changed: ' + str(data))
                free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                                    '--format=csv,noheader,nounits'], text=True).strip())
                update(state='waiting_for_gpu', current_run=run['run_id'], free_mib=free, required_free_mib=14500)
                if free >= 14500:
                    break
                time.sleep(30)
            verify(checkout, run['commit'])
            command = [sys.executable, '-u', 'dinomaly_bridge.py', '--model_variant', 'v1_amcm',
                       '--data_path', str(data), '--dataset_profile', run['profile'], '--split_view', 'original',
                       '--save_name', run['run_id'], '--batch_size', '16', '--seed', '1',
                       '--total_iters', '10000', '--eval_interval', '2000', '--checkpoint_interval', '1000']
            (run_folder / 'launch.json').write_text(json.dumps(dict(command=command, published_commit=run['commit'],
                                                                  order=index + 1, full_run=True), indent=2))
            update(state='training_process_running', command=command, current_commit=run['commit'])
            with (run_folder / 'console.log').open('x') as log:
                process = subprocess.Popen(command, cwd=str(checkout), stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                update(training_pid=process.pid)
                code = process.wait()
            archive_run(checkout, run['run_id'], repo / 'reproduction' / 'results')
            if code:
                raise RuntimeError('Full training failed with code {}; no retry or next dataset'.format(code))
            result = json.loads((run_folder / 'status.json').read_text())
            if result['state'] != 'completed' or result['step'] != 10000 or result['mode'] != 'full':
                raise RuntimeError('Incomplete full run; refusing next dataset')
            update(completed_runs=[r['run_id'] for r in runs[:index + 1]])
        subprocess.run([sys.executable, 'reproduction/compare_v1.py', '--results_root',
                        str(repo / 'reproduction' / 'results')], cwd=str(repo), check=True)
        update(state='completed', finished_at=time.time())
    except BaseException as exc:
        update(state='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
