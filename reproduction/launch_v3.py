import argparse,json,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--sequence_id',default='v3-full-seed1-20261017');a=ap.parse_args();repo=Path(__file__).resolve().parent.parent;head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();dirty=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip();
if head!=a.commit or dirty:raise RuntimeError('Need clean published commit')
seq=repo/'runs'/a.sequence_id;seq.mkdir(parents=True,exist_ok=True)
if (seq/'status.json').exists():raise RuntimeError('Sequence exists')
cmd=[sys.executable,'-u','reproduction/run_v3_sequence.py','--commit',a.commit,'--sequence_id',a.sequence_id]
with (seq/'console.log').open('x') as log:p=subprocess.Popen(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print('V3 full queue PID',p.pid,'status',seq/'status.json')
