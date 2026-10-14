"""Detach the Bridge2893 baseline only from the expected clean, published checkout."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected_commit', required=True,
                        help='Commit already verified against the GitHub branch by the caller.')
    parser.add_argument('--run_id', default='bridge2893-b16-s1-20261014')
    parser.add_argument('--data_path', default='/home/skye/data/Skye/databases/Bridge2893_split_seed42')
    parser.add_argument('--dataset_profile', choices=['bridge2893','bridge512_balance'], default='bridge2893')
    parser.add_argument('--split_view', choices=['original','deduplicated'], default='original')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('run_id must be a directory name')
    repo = Path(__file__).resolve().parent.parent
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo), text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError('Server commit does not match the verified published commit')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(repo), text=True).strip():
        raise RuntimeError('Server checkout is dirty; synchronize the published code first')
    folder = repo / 'runs' / args.run_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'status.json').exists():
        raise RuntimeError('Run already has status; do not launch twice')
    command = [sys.executable, '-u', 'dinomaly_bridge.py', '--save_name', args.run_id,
               '--data_path', args.data_path, '--dataset_profile', args.dataset_profile,
               '--split_view', args.split_view,
               '--batch_size', '16', '--seed', '1', '--total_iters', '10000',
               '--eval_interval', '2000', '--checkpoint_interval', '1000']
    with (folder / 'console.log').open('x') as log:
        process = subprocess.Popen(command, cwd=str(repo), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    print('Launched PID {} from commit {}. Log: {}'.format(process.pid, head, folder / 'console.log'))


if __name__ == '__main__':
    main()
