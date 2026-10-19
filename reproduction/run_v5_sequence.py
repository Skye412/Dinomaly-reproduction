"""Approved-sequence runner for V5 training/evaluation followed by non-blocking V4 supplements."""
import json, subprocess, sys, time, traceback
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--approved_commit', required=True, help='Exact reviewed and published HEAD; user approval required')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
plan=json.loads((root/'reproduction/plans/v5_retrieval_decoder_20261019.json').read_text(encoding='utf-8'))
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if head != args.approved_commit:
    raise RuntimeError('HEAD differs from approved commit')
if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
    raise RuntimeError('Execution checkout must be clean')
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/master'],cwd=root,text=True).split()[0]
if remote != head:
    raise RuntimeError('Published remote commit differs from execution HEAD')
runroot=root/'runs'/plan['sequence_id']; runroot.mkdir(parents=True,exist_ok=False)
state={'pid':__import__('os').getpid(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
def update(**kw): state.update(kw,updated_at=time.time()); (runroot/'status.json').write_text(json.dumps(state,indent=2))
def call(cmd,label):
 update(state='running',phase=label,command=cmd)
 with (runroot/(label+'-console.log')).open('x',encoding='utf-8') as log:
  subprocess.run(cmd,cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
try:
    # V5 is the primary experiment. Run all three train/eval pairs first;
    # supplementary V4 work is explicitly non-blocking and runs afterward.
    for name,d in plan['datasets'].items():
        v5=root/'runs'/d['v5_run']
        train=['dinomaly_bridge.py','--dataset_profile',d['profile'],'--data_path',d['data_path'],'--save_name',d['v5_run'],'--model_variant','v5_retrieval','--image_size','392','--v3_evaluation','--seed','1','--batch_size','16','--total_iters','10000','--eval_interval','2000']
        call([sys.executable]+train,f'train_{name}')
        call([sys.executable,'v5_evaluate.py','--config',str(v5/'config.json'),'--checkpoint',str(v5/'best_val.pth'),'--bank_dir',str(v5/'retrieval_bank'),'--baseline_dir',d['baseline_dir'],'--v4_reference',d['v4_reference'],'--output',str(root/'runs'/f'{d["v5_run"]}-eval')],f'evaluate_{name}')
    supplemental={}
    for name,d in plan['datasets'].items():
        v0=root/'runs'/d['v0_run']; supp=root/'runs'/f'v4-supplement-{name}-20261019'
        try:
            call([sys.executable,'crmc_evaluate.py','--config',str(v0/'config.json'),'--checkpoint',str(v0/'best_val.pth'),'--baseline_dir',d['baseline_dir'],'--output',str(supp),'--supplementary','--v4_run',d['v4_run']],f'supplement_{name}')
            supplemental[name]=dict(state='completed',output=str(supp))
        except Exception as exc:
            supplemental[name]=dict(state='failed',error=repr(exc),traceback=traceback.format_exc())
            (runroot/(f'supplement_{name}_failure.json')).write_text(json.dumps(supplemental[name],indent=2))
            update(state='supplement_failed_continuing',dataset=name,supplemental=supplemental)
    reports={name:json.loads((root/'runs'/f'{d["v5_run"]}-eval'/'v5_report.json').read_text()) for name,d in plan['datasets'].items()}
    deltas=[v for name in ('codebrim512','dacl_v3_seed42') for v in reports[name]['validation_delta_vs_v4'].values()]
    retained=all(x>-.005 for x in deltas) and any(x>=.01 for x in deltas)
    (runroot/'comparison.json').write_text(json.dumps(dict(retention_passed=retained,datasets=reports),indent=2))
    update(state='completed',retention_passed=retained,supplemental=supplemental)
except BaseException as e:
    update(state='failed',error=repr(e),traceback=traceback.format_exc()); raise
