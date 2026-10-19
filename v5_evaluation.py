"""V5 checkpoint+fusion+z validation selection with explicit reconstruction report."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from crmc import forward_with_context, nearest_memory_maps, prepare_map, fuse_maps
from crmc_evaluate import FRACTIONS, ALPHAS, normal_stats, calibration, stats, op, export_scores, write_json
from reproduction_runtime import METRIC_NAMES
from stratified_evaluation import stratified_metrics
from utils import get_gaussian_kernel

def top_scores(amap, fractions):
    flat=np.sort(amap.reshape(len(amap),-1),axis=1)[:,::-1]
    return {z:flat[:,:max(1,int(flat.shape[1]*z))].mean(1) for z in fractions}


class V5Evaluator:
    def __init__(self,args,folder):
        self.args,self.folder=args,Path(folder)
        self.selections={}

    def collect(self,model,loader):
        model.eval(); rec=[]; mem=[]; gt=[]; labels=[]; paths=[]
        provider=model.retrieval_provider
        kernel=get_gaussian_kernel(5,4).to(self.args.device)
        with torch.no_grad():
            for images,masks,y,names in loader:
                cls,groups,raw=forward_with_context(model,images.to(self.args.device))
                memory,_=nearest_memory_maps(cls,groups,provider.bank[1],provider.bank[2],ks=(16,),include_global=False)
                rec.append(prepare_map(raw,kernel)[:,0].cpu().numpy())
                mem.append(prepare_map(memory['16'],kernel)[:,0].cpu().numpy())
                gt.append(F.interpolate(masks.float(),size=256,mode='nearest')[:,0].bool().numpy())
                labels.extend(y.tolist());paths.extend(names)
        return np.concatenate(rec),np.concatenate(mem),np.concatenate(gt),np.asarray(labels),paths

    def validation(self,model,loader,step):
        r,m,gt,y,paths=self.collect(model,loader)
        rm,rs=normal_stats(r,y);mm,ms=normal_stats(m,y)
        norm=dict(rec_mean=rm,rec_std=rs,mem_mean=mm,mem_std=ms)
        candidates=[]
        for mode in ('weighted_sum','gated'):
            for alpha in (ALPHAS if mode=='weighted_sum' else (None,)):
                maps=fuse_maps(r,m,norm,mode,alpha);pap=float(average_precision_score(gt.ravel(),maps.ravel()))
                for z,scores in top_scores(maps,FRACTIONS).items():
                    candidates.append(dict(mode=mode,alpha=alpha,top_fraction=z,image_auroc=float(roc_auc_score(y,scores)),pixel_ap=pap))
        key=lambda x:(x['image_auroc'],x['pixel_ap'])
        selected=max(candidates,key=key)
        reconstruction=max([c for c in candidates if c['mode']=='weighted_sum' and c['alpha']==0],key=key)
        maps=fuse_maps(r,m,norm,selected['mode'],selected['alpha']); scores=top_scores(maps,(selected['top_fraction'],))[selected['top_fraction']]
        fpr,legacy=calibration(y,scores)
        rscores=top_scores(r,(reconstruction['top_fraction'],))[reconstruction['top_fraction']];rfpr,rlegacy=calibration(y,rscores)
        selected=dict(selected,step=step,normalization=norm,threshold=legacy,fpr_threshold=fpr,
            reconstruction=dict(reconstruction,threshold=rlegacy,fpr_threshold=rfpr))
        self.selections[step]=selected
        record=dict(step=step,candidates=candidates,selected=selected)
        with (self.folder/'validation_z.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        metrics=stats(y,scores,maps,gt)
        export_scores(self.folder,f'val_{step}',paths,y,scores,fpr,legacy)
        return [metrics[k] for k in METRIC_NAMES],dict(threshold=legacy,top_fraction=selected['top_fraction'],primary=op(y,scores,fpr),legacy=op(y,scores,legacy),selection=selected)

    def test(self,model,loader,name,step):
        r,m,gt,y,paths=self.collect(model,loader);sel=self.selections[step]
        recsel=sel['reconstruction'];reports={}
        for kind,setting in [('fusion',sel),('reconstruction_only',recsel)]:
            maps=(r if kind=='reconstruction_only' else fuse_maps(r,m,sel['normalization'],sel['mode'],sel['alpha']))
            scores=top_scores(maps,(setting['top_fraction'],))[setting['top_fraction']]
            metrics=stats(y,scores,maps,gt);primary=op(y,scores,setting['fpr_threshold']);legacy=op(y,scores,setting['threshold'])
            result=dict(protocol='v5_crmc_validation_selection_v1',step=step,selection=setting,
                metrics=metrics,primary_fpr10=primary,legacy_f1=legacy,
                stratified=stratified_metrics(loader.dataset,[dict(image_path=p) for p in paths],y,scores,setting['fpr_threshold']),
                positive_images_with_empty_resized_mask=int(((y==1)&~gt.reshape(len(y),-1).any(1)).sum()))
            suffix='' if kind=='fusion' else '_reconstruction_only'
            write_json(self.folder/f'test_{name}{suffix}_v3.json',result)
            export_scores(self.folder,f'test_{name}{suffix}',paths,y,scores,setting['fpr_threshold'],setting['threshold'])
            reports[kind]=result
        main=reports['fusion']
        return [main['metrics'][k] for k in METRIC_NAMES],dict(primary_fpr10=main['primary_fpr10'],legacy_f1=main['legacy_f1'],reconstruction_only=reports['reconstruction_only'])
