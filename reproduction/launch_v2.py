"""Detach the approved full-only paired-denoising sequence after publication verification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge_commit', required=True)
    parser.add_argument('--dacl_commit', required=True)
    parser.add_argument('--sequence_id', default='v2-paired-seed1-20261016')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo), text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(repo), text=True).strip()
    if head != args.dacl_commit or dirty:
        raise RuntimeError('Synchronize clean published sequence checkout before submission')
    for commit in [args.bridge_commit, args.dacl_commit]:
        subprocess.run(['git', 'cat-file', '-e', commit + '^{commit}'], cwd=str(repo), check=True)
    if Path(args.sequence_id).name != args.sequence_id or args.sequence_id in {'.', '..'}:
        raise ValueError('sequence_id must be a directory name')
    # Refuse a second live queue, even if submitted with a different sequence ID.
    for previous in (repo / 'runs').glob('v2-paired-*/status.json'):
        record = json.loads(previous.read_text())
        if record['state'] not in {'completed', 'failed', 'superseded_before_training'}:
            try:
                os.kill(record['pid'], 0)
            except ProcessLookupError:
                continue
            raise RuntimeError('Another active V2 queue exists: ' + str(previous))
    folder = repo / 'runs' / args.sequence_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'status.json').exists():
        raise RuntimeError('Sequence already submitted')
    command = [sys.executable, '-u', 'reproduction/run_v2_sequence.py',
               '--bridge_commit', args.bridge_commit, '--dacl_commit', args.dacl_commit,
               '--sequence_id', args.sequence_id]
    with (folder / 'console.log').open('x') as log:
        process = subprocess.Popen(command, cwd=str(repo), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    print('Full-only V2 Bridge -> DACL queue PID {}; status: {}'.format(process.pid, folder / 'status.json'))


if __name__ == '__main__':
    main()
