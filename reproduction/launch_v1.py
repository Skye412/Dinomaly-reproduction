"""Detach the approved full-only AMCM sequence after publication verification."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge_commit', required=True)
    parser.add_argument('--dacl_commit', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo), text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(repo), text=True).strip()
    if head != args.dacl_commit or dirty:
        raise RuntimeError('Synchronize clean published sequence checkout before submission')
    for commit in [args.bridge_commit, args.dacl_commit]:
        subprocess.run(['git', 'cat-file', '-e', commit + '^{commit}'], cwd=str(repo), check=True)
    folder = repo / 'runs' / 'v1-amcm-seed1-20261015'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'status.json').exists():
        raise RuntimeError('Sequence already submitted')
    command = [sys.executable, '-u', 'reproduction/run_v1_sequence.py',
               '--bridge_commit', args.bridge_commit, '--dacl_commit', args.dacl_commit]
    with (folder / 'console.log').open('x') as log:
        process = subprocess.Popen(command, cwd=str(repo), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    print('Full-only V1 Bridge -> DACL queue PID {}; status: {}'.format(process.pid, folder / 'status.json'))


if __name__ == '__main__':
    main()
