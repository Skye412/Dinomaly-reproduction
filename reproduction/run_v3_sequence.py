"""Full V3 Bridge then DACL sequence; no retries; V0 reevaluation follows both runs."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def git(repo,*args): return subprocess.check_output(['git']+list(args),cwd=str(repo),text=True).strip()
def verify(repo,commit):
 if git(repo,'rev-parse','HEAD')!=commit or git(repo,'status','--porcelain'): raise RuntimeError('Published clean checkout changed')
def archive(checkout,run_id,out):
 target=out/run_id;target.mkdir(parents=True,exist_ok=True)
 for p in (checkout/'runs'/run_id).iterdir():
  if p.suffix in {'.json','.jsonl','.csv','.log','.txt'}:shutil.copy2(p,target/p.name)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--sequence_id',default='v3-full-seed1-20261017');a=ap.parse_args();repo=Path(__file__).resolve().parent.parent;seq=repo/'runs'/a.sequence_id;seq.mkdir(parents=True,exist_ok=True);lock=(seq/'sequence.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (seq/'status.json').exists():raise RuntimeError('Duplicate sequence')
 status={'pid':os.getpid(),'state':'queued','full_training':True,'smoke_tests':False,'tuning_restarts':False,'commit':a.commit,'runs':[]}
 def update(**kw):status.update(kw,updated_at=time.time());(seq/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(kw),flush=True)
 runs=[('bridge512','bridge512_balance_repaired','/home/skye/data/Skye/databases/Bridge512_balance','v3-bridge512-b16-s1-20261016'),('dacl','dacl_v3_seed42','/home/skye/data/Skye/databases/DACL10k_v3_512_split_seed42','v3-dacl-b16-s1-20261016')]
 try:
  update(runs=[{'dataset':x[0],'profile':x[1],'run_id':x[3]} for x in runs])
  for name,profile,data,runid in runs:
   co=seq/('checkout-'+name);subprocess.run(['git','worktree','add','--detach',str(co),a.commit],cwd=repo,check=True);(co/'backbones').mkdir(exist_ok=True);(co/'backbones'/'weights').symlink_to(repo/'backbones'/'weights',target_is_directory=True);(co/'runs').mkdir(exist_ok=True);(co/'runs'/'gpu0.lock').symlink_to(repo/'runs'/'gpu0.lock');verify(co,a.commit)
  for index,(name,profile,data,runid) in enumerate(runs):
   co=seq/('checkout-'+name);plan=json.loads((co/'reproduction'/'plans'/('v3-'+name+'-b16-s1-20261016.json')).read_text(encoding='utf-8'));assert plan['status']=='launch_commit_awaiting_review';rf=co/'runs'/runid;rf.mkdir();
   while True:
    verify(co,a.commit);free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());update(state='waiting_for_gpu',current_run=runid,free_mib=free,required_free_mib=14500)
    if free>=14500:break
    time.sleep(30)
   cmd=[sys.executable,'-u','dinomaly_bridge.py','--data_path',data,'--dataset_profile',profile,'--save_name',runid,'--model_variant','v3','--image_size','448','--context_recenter','--train_augment','--v3_evaluation','--batch_size','16','--seed','1','--total_iters','10000','--eval_interval','2000','--checkpoint_interval','1000'];(rf/'launch.json').write_text(json.dumps({'command':cmd,'published_commit':a.commit,'full_run':True},indent=2));update(state='training_process_running',current_run=runid,current_commit=a.commit);log=(rf/'console.log').open('x');p=subprocess.Popen(cmd,cwd=co,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);status['training_pid']=p.pid;status['updated_at']=time.time();(seq/'status.json').write_text(json.dumps(status,indent=2));code=p.wait();log.close();
   if code: raise RuntimeError('V3 full run failed code {}'.format(code))
   result=json.loads((rf/'status.json').read_text());
   if result.get('state')!='completed' or result.get('step')!=10000 or result.get('mode')!='full':raise RuntimeError('Incomplete V3 run')
   archive(co,runid,repo/'reproduction'/'results');update(completed_runs=[r[3] for r in runs[:index+1]])
  update(state='completed_training',message='Both V3 full runs completed; V0 reevaluation remains queued as a separate post-training step')
 except BaseException as e:update(state='failed',error=str(e));raise
if __name__=='__main__':main()
