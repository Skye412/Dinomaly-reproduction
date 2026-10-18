"""Approved sequential baseline execution; stop on any failure, never retry."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'reproduction/plans/three_dataset_v0_review_20261018.json'
OUT = ROOT / 'runs/three-v0-sequence-20261018'
def main():
    OUT.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), state='starting', commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def update(**kw):
        state.update(kw,updated_at=time.time())
        (OUT/'status.json').write_text(json.dumps(state,indent=2))
    try:
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Execution checkout must be clean')
        plan=json.loads(PLAN.read_text())
        sys.path.insert(0,str(ROOT))
        from reproduction_gpu import acquire_gpu_lock
        for i,task in enumerate(plan['sequence']):
            update(state='running',task=i,command=task['command'])
            cmd=[sys.executable,'-u']+task['command'].split()[1:]
            lock=None
            if task['kind']!='full_training':
                lock=acquire_gpu_lock(ROOT/'runs/gpu0.lock',lambda **kw:update(**kw),14500)
            try:
                with (OUT/('task%d-console.log'%i)).open('x') as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:
                if lock: lock.close()
            if task['kind']=='full_training':
                status=json.loads((ROOT/'runs'/task['run_id']/'status.json').read_text())
                if status['state']!='completed' or status['step']!=10000:
                    raise RuntimeError('Full training incomplete')
        update(state='completed')
    except BaseException as exc:
        update(state='failed',error=repr(exc));raise
if __name__=='__main__':main()
