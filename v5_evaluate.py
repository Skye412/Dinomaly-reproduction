"""Evaluation for a trained V5 retrieval decoder; selection is validation-only."""
import argparse, csv, json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from bridge_dataset import check_split_protocol, dataset_class, load_profile
from dinomaly_bridge import build_model
from crmc import load_bank, sha256_file
from reproduction_gpu import acquire_gpu_lock
from v5_evaluation import V5Evaluator



def main(args):
    cfg=json.loads(args.config.read_text(encoding='utf-8')); args.output.mkdir(parents=True,exist_ok=False)
    state={"pid":os.getpid()}
    def notify(**kw):
        state.update(kw,updated_at=time.time()); (args.output/'status.json').write_text(json.dumps(state,indent=2))
        with (args.output/'console.log').open('a') as f:f.write(json.dumps(kw)+'\n')
    lock=None
    try:
        if cfg.get('model_variant')!='v5_retrieval': raise ValueError('V5 evaluator requires model_variant=v5_retrieval')
        if cfg.get('image_size')!=392 or not cfg.get('v3_evaluation') or cfg.get('context_recenter') or cfg.get('encoder_fp16'):
            raise ValueError('V5 evaluator requires frozen V0 392 FP32 protocol')
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(): raise RuntimeError('Execution code must be clean')
        notify(state='waiting_for_gpu',git_commit=head); lock=acquire_gpu_lock(Path('runs/gpu0.lock'),notify,14500)
        meta,exclusions,_=load_profile(cfg['dataset_profile'],cfg.get('split_view','original'));check_split_protocol(cfg['data_path'],cfg['dataset_profile'],cfg.get('split_view','original'))
        D=dataset_class(cfg['dataset_profile']); train=D(cfg['data_path'],'train',normal_only=True,image_size=392,exclude_samples=exclusions['train'])
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
        model,_=build_model(args.device,'v5_retrieval');ck=torch.load(args.checkpoint,map_location='cpu');model.load_state_dict(ck['model'],strict=True);step=int(ck['step']);del ck
        bank=load_bank(args.bank_dir)
        if bank[0]['provenance']['manifest_sha256'] != meta['manifest_sha256']: raise ValueError('Bank manifest mismatch')
        from retrieval_v5 import attach_retrieval;attach_retrieval(model,bank,train.rows,args.device);model.eval();model.requires_grad_(False)
        provenance=dict(protocol='v5_retrieval_decoder_eval_v1',git_commit=head,checkpoint_sha256=sha256_file(args.checkpoint),bank_sha256=bank[0]['file_sha256'],manifest_sha256=meta['manifest_sha256'],step=step)
        (args.output/'config.json').write_text(json.dumps(dict(provenance,original_config=cfg),indent=2))
        def loader(split):return DataLoader(D(cfg['data_path'],split,image_size=392,mask_size=392,exclude_samples=exclusions[split]),batch_size=cfg['batch_size'],shuffle=False,num_workers=4)
        evaluator=V5Evaluator(args,args.output);notify(state='validation');_,vcal=evaluator.validation(model,loader('val'),step);notify(state='test');_,tcal=evaluator.test(model,loader('test'),'best_val',step)
        baseline=Path(args.baseline_dir); baseline_summary=json.loads((baseline/'summary.json').read_text())
        v0_scores=list(csv.DictReader((baseline/'test_best_val_scores.csv').open())); v5_rec=list(csv.DictReader((args.output/'test_best_val_reconstruction_only_scores.csv').open()))
        if [x['image_path'] for x in v0_scores] != [x['image_path'] for x in v5_rec] or [x['label'] for x in v0_scores] != [x['label'] for x in v5_rec]: raise ValueError('V0/V5 sample identity mismatch')
        from sklearn.metrics import roc_auc_score
        y=np.array([int(x['label']) for x in v5_rec]); a=np.array([float(x['score']) for x in v0_scores]); b=np.array([float(x['score']) for x in v5_rec])
        comparison=dict(v0_image_auroc=float(roc_auc_score(y,a)),v5_reconstruction_image_auroc=float(roc_auc_score(y,b)),delta=float(roc_auc_score(y,b)-roc_auc_score(y,a)),v0_expected=float(roc_auc_score(y,a)))
        from v3_evaluation import bootstrap_auc
        fusion=list(csv.DictReader((args.output/'test_best_val_scores.csv').open()))
        fused_scores=np.array([float(x['score']) for x in fusion])
        reference=Path(args.v4_reference)
        v4rows=list(csv.DictReader((reference/'test_best_val_scores.csv').open()))
        if [x['image_path'] for x in v4rows]!=[x['image_path'] for x in fusion] or [x['label'] for x in v4rows]!=[x['label'] for x in fusion]: raise ValueError('V4/V5 paired identities differ')
        v4scores=np.array([float(x['score']) for x in v4rows])
        v4=json.loads((reference/'crmc_report.json').read_text())
        vmetrics=vcal['selection'];old=v4['main']['validation_metrics']
        changes={k:float(vmetrics[k]-old[k]) for k in ('image_auroc','pixel_ap')}
        report=dict(provenance=provenance,validation=vcal,test=tcal,reconstruction_comparison=comparison,
            validation_delta_vs_v4=changes,
            paired_test_delta=dict(reconstruction_vs_v0=bootstrap_auc(y,b,2000,cfg['seed'],a),
                fusion_vs_v4=bootstrap_auc(y,fused_scores,2000,cfg['seed'],v4scores)),
            note='V5 reconstruction is expected to differ from V0; this is a report, not a pass/fail consistency gate')
        (args.output/'v5_report.json').write_text(json.dumps(report,indent=2));notify(state='completed')
    except BaseException as e: notify(state='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if lock is not None:lock.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank_dir',type=Path,required=True);p.add_argument('--baseline_dir',type=Path,required=True);p.add_argument('--v4_reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');main(p.parse_args())

