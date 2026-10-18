"""Run approved CRMC evaluations sequentially and stop on first failure."""
import json, os, subprocess, sys, time, traceback
from pathlib import Path
root=Path(__file__).resolve().parents[1]
plan=json.loads((root/'reproduction/plans/v4_crmc_evaluation_20261018.json').read_text())
out=root/'runs'/plan['sequence_id'];out.mkdir(exist_ok=False,parents=True)
state=dict(pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
def update(**kw):
 state.update(kw,updated_at=time.time());(out/'status.json').write_text(json.dumps(state,indent=2))
try:
 for profile,run in plan['runs'].items():
  update(state='evaluating',dataset=profile,command=run['command'])
  with (out/(profile+'-console.log')).open('x') as log:
   subprocess.run([sys.executable]+run['command'].split()[1:],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
  if json.loads((root/run['output']/'status.json').read_text())['state']!='completed':raise RuntimeError('Incomplete evaluation')
 update(state='completed')
except BaseException as e:
 update(state='failed',error=repr(e),traceback=traceback.format_exc());raise
