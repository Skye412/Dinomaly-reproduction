"""Prespecified V4 supplementary comparisons; no training or test selection."""
import json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from crmc import extract_crmc_features, load_bank, nearest_memory_maps, prepare_map, sha256_file

BOXES = ((0, 0, 288, 288), (224, 0, 512, 288), (0, 224, 288, 512), (224, 224, 512, 512))


def original_path(ds, row):
    return ds.root / row['image_path'] if 'annotation_path' in row else ds.root / ds.split / row['image_path']


class BlockDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        from dataset import get_data_transforms
        self.transform, _ = get_data_transforms(392, 392)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        with Image.open(original_path(self.ds, self.ds.rows[i])) as im:
            im = im.convert('RGB')
            if im.size != (512, 512):
                raise ValueError('Multi-scale requires native 512 tiles')
            return torch.stack([self.transform(im.crop(box)) for box in BOXES])


def coreset_indices(groups, device, seed=1, fraction=.1, chunk=16384, projection=64):
    """Global greedy k-center in a seeded projection; all train patches eligible.

    Projected distances select the coreset; full original Euclidean features are
    used for scoring. This is PatchCore-style, without its original reweighting.
    """
    flat=[g.reshape(-1,g.shape[-1]) for g in groups]
    n=len(flat[0]);target=max(1,int(n*fraction))
    generator=torch.Generator().manual_seed(seed)
    proj=torch.randn(flat[0].shape[-1]*len(groups),projection,generator=generator).to(device)/projection**.5
    with torch.no_grad():
        x=torch.cat([torch.cat([g[i:i+chunk].to(device).float() for g in flat],1)@proj for i in range(0,n,chunk)])
        closest=torch.full((n,),float('inf'),device=device)
        used=torch.zeros(n,dtype=torch.bool,device=device)
        pick=int((x-x.mean(0)).square().sum(1).argmax());selected=[]
        for _ in range(target):
            selected.append(pick);used[pick]=True;center=x[pick].clone()
            for start in range(0,n,chunk):
                end=min(start+chunk,n)
                closest[start:end]=torch.minimum(closest[start:end],(x[start:end]-center).square().sum(1))
            closest[used]=-1;pick=int(closest.argmax())
    if len(set(selected))!=target:raise AssertionError('Duplicate coreset indices')
    return np.asarray(selected,dtype=np.int64)


def patchcore_map(query_groups, banks, device, chunk=4096):
    """Euclidean nearest neighbor on concatenated raw DINO fusion groups.

    No original PatchCore reweighting/backbone: deliberately a style comparison.
    """
    b,n,c=query_groups[0].shape
    q=torch.cat(query_groups,-1).float().reshape(b*n,-1)
    best=torch.full((len(q),),float('inf'),device=device)
    qnorm=q.square().sum(1)
    for start in range(0,len(banks[0]),chunk):
        ref=torch.cat([g[start:start+chunk].to(device).float() for g in banks],-1)
        for row in range(0,len(q),128):
            qq=q[row:row+128]
            dist=(qnorm[row:row+128,None]+ref.square().sum(1)[None,:]-2*(qq@ref.T)).clamp_min_(0)
            best[row:row+128]=torch.minimum(best[row:row+128],dist.min(1).values)
    side=int(n**.5)
    return best.sqrt().reshape(b,1,side,side)


def block_memory(model, ds, device, output, provenance):
    cls_parts=[]; patch_parts=[[],[]]
    with torch.no_grad():
        for tiles in DataLoader(BlockDataset(ds),batch_size=4,shuffle=False,num_workers=4):
            cls,groups,_=extract_crmc_features(model,tiles.flatten(0,1).to(device))
            cls_parts.append(cls.cpu().half())
            for dest,g in zip(patch_parts,groups): dest.append(g.cpu().half())
    cls=torch.cat(cls_parts); groups=[torch.cat(g) for g in patch_parts]
    output.mkdir(exist_ok=False)
    torch.save(cls,output/'cls.fp16.pt')
    for i,g in enumerate(groups): torch.save(g,output/f'patch_group{i}.fp16.pt')
    metadata=dict(samples=len(cls),boxes=BOXES,source_samples=[r['sample'] for r in ds.rows],
                  normal_only=True,source_split='train',provenance=provenance,
                  file_sha256={p.name:sha256_file(p) for p in output.glob('*.pt')})
    (output/'bank.json').write_text(json.dumps(metadata,indent=2))
    return metadata,cls,groups


def stitch(maps):
    """Average overlapping 288 blocks; whole/local fusion is selected separately."""
    b=maps.shape[0]; out=torch.zeros(b,1,512,512,device=maps.device); counts=torch.zeros_like(out)
    for i,(x,y,x2,y2) in enumerate(BOXES):
        local=F.interpolate(maps[:,i],size=(288,288),mode='bilinear',align_corners=True)
        out[:,:,y:y2,x:x2]+=local; counts[:,:,y:y2,x:x2]+=1
    if not (counts>0).all(): raise AssertionError('Uncovered pixels')
    return out/counts


def collect(model, ds, bank, block_bank, coreset, device, kernel, notify):
    from crmc import forward_with_context
    outputs={'patchcore_10pct':[],'context16':[],'local16':[]}
    paths=[]; labels=[]; masks=[]
    tiles=BlockDataset(ds)
    with torch.no_grad():
        for i,(images,gt,y,names) in enumerate(DataLoader(ds,batch_size=4,shuffle=False,num_workers=4)):
            _,groups,_=forward_with_context(model,images.to(device))
            pc=patchcore_map(groups,coreset,device)
            # Whole-scene map is identical to V4 K=16 processing.
            cls,_,_=extract_crmc_features(model,images.to(device))
            full,_=nearest_memory_maps(cls,groups,bank[1],bank[2],ks=(16,),include_global=False)
            blocks=torch.stack([tiles[j] for j in range(i*4,min((i+1)*4,len(ds)))]).flatten(0,1).to(device)
            bcls,bgroups,_=extract_crmc_features(model,blocks)
            local,_=nearest_memory_maps(bcls,bgroups,block_bank[1],block_bank[2],ks=(16,),include_global=False)
            rawlocal=stitch(local['16'].reshape(len(images),4,1,28,28))
            # Stitch first; resize to392 ->256 and smooth once, consistent support.
            for name,amap in [('patchcore_10pct',pc),('context16',full['16']),('local16',rawlocal)]:
                if name=='local16': amap=F.interpolate(amap,size=392,mode='bilinear',align_corners=True)
                outputs[name].append(prepare_map(amap,kernel)[:,0].cpu().numpy())
            paths.extend(names); labels.extend(y.tolist()); masks.append(F.interpolate(gt.float(),size=256,mode='nearest')[:,0].bool().numpy())
            notify(phase='supplement_'+ds.split,samples=len(paths))
    return {k:np.concatenate(v) for k,v in outputs.items()},np.concatenate(masks),np.asarray(labels),paths


def main(args):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from bridge_dataset import check_split_protocol, dataset_class, load_profile
    from dinomaly_bridge import build_model
    from crmc_evaluate import write_json, top_score, stats, calibration, op, export_scores, FRACTIONS
    from stratified_evaluation import stratified_metrics
    from reproduction_gpu import acquire_gpu_lock
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False); state=dict(pid=os.getpid())
    def notify(**kw):
        state.update(kw,updated_at=time.time()); write_json(args.output/'status.json',state)
        with (args.output/'console.log').open('a') as f: f.write(json.dumps(kw)+'\n')
    lock=None
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
            raise RuntimeError('Execution code must be clean')
        notify(state='waiting_for_gpu',git_commit=head)
        lock=acquire_gpu_lock(Path('runs/gpu0.lock'),notify,14500)
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
        profile=cfg['dataset_profile'];check_split_protocol(cfg['data_path'],profile,cfg.get('split_view','original'))
        if cfg.get('model_variant','v0')!='v0' or cfg['image_size']!=392: raise ValueError('V4 supplement requires V0 392')
        meta,exclusions,_=load_profile(profile,cfg.get('split_view','original'));D=dataset_class(profile)
        model,_=build_model(args.device,'v0');ck=torch.load(args.checkpoint,map_location='cpu');model.load_state_dict(ck['model'],strict=True);del ck
        model.eval();model.requires_grad_(False)
        source=Path(args.v4_run);bank=load_bank(source/'memory_bank')
        if bank[0]['provenance']['manifest_sha256']!=meta['manifest_sha256']:raise ValueError('Source bank mismatch')
        train=D(cfg['data_path'],'train',normal_only=True,image_size=392,exclude_samples=exclusions['train'])
        if bank[0]['sample_ids']!=[r['sample'] for r in train.rows]:raise ValueError('Source bank order mismatch')
        provenance=dict(commit=head,manifest_sha256=meta['manifest_sha256'],checkpoint_sha256=sha256_file(args.checkpoint),
            cfg=cfg,source_v4=str(source),external_baselines='style comparisons; not full official reproductions')
        write_json(args.output/'config.json',provenance)
        notify(state='coreset')
        ids=coreset_indices(bank[2],args.device,cfg['seed']);np.save(args.output/'coreset_indices.npy',ids)
        cores=[g.reshape(-1,g.shape[-1])[ids].contiguous() for g in bank[2]]
        write_json(args.output/'coreset.json',dict(fraction=.1,count=len(ids),total=len(bank[2][0])*784,
            algorithm='global greedy k-center in seeded Gaussian projection64; every training patch eligible',
            seed=cfg['seed'],distance='raw concatenated target groups, Euclidean',patch_reweighting=False))
        notify(state='building_block_bank')
        blocks=block_memory(model,train,args.device,args.output/'block_memory_bank',provenance)
        kernel=__import__('utils').get_gaussian_kernel(5,4).to(args.device)
        def dataset(split):return D(cfg['data_path'],split,image_size=392,mask_size=392,exclude_samples=exclusions[split])
        notify(state='validation');val=collect(model,dataset('val'),bank,blocks,cores,args.device,kernel,notify)
        vcache=np.load(source/'val_maps.npz');
        if val[3]!=vcache['paths'].tolist() or not np.array_equal(val[2],vcache['labels']): raise ValueError('V4 cached validation identity mismatch')
        val[0]['full_bank_memory_only']=vcache['global'].copy();vcache.close()
        # Whole/local maps have the same cosine-distance scale; max/mean are raw-map fusion.
        for mode in ('max','mean'):
            val[0]['multiscale_'+mode]=(np.maximum(val[0]['context16'],val[0]['local16']) if mode=='max' else (val[0]['context16']+val[0]['local16'])*.5)
        names=['full_bank_memory_only','patchcore_10pct','multiscale_max','multiscale_mean']; selections={}; candidates=[]
        for name in names:
            vm=val[0][name];pap=float(average_precision_score(val[1].ravel(),vm.ravel()))
            rows=[dict(method=name,z=z,image_auroc=float(roc_auc_score(val[2],top_score(vm,z))),pixel_ap=pap) for z in FRACTIONS]
            selections[name]=max(rows,key=lambda r:(r['image_auroc'],r['pixel_ap']));candidates+=rows
        selections['multiscale_selected']=max([selections['multiscale_max'],selections['multiscale_mean']],key=lambda r:(r['image_auroc'],r['pixel_ap']))
        write_json(args.output/'best_selection.json',selections)
        (args.output/'validation_z.jsonl').write_text(json.dumps(dict(candidates=candidates,selected=selections))+'\n')
        # All settings fixed before any test inference/cache load.
        notify(state='test');testds=dataset('test');test=collect(model,testds,bank,blocks,cores,args.device,kernel,notify)
        tcache=np.load(source/'test_maps.npz')
        if test[3]!=tcache['paths'].tolist() or not np.array_equal(test[2],tcache['labels']):raise ValueError('V4 cached test identity mismatch')
        test[0]['full_bank_memory_only']=tcache['global'].copy();tcache.close()
        for mode in ('max','mean'):test[0]['multiscale_'+mode]=(np.maximum(test[0]['context16'],test[0]['local16']) if mode=='max' else (test[0]['context16']+test[0]['local16'])*.5)
        reports={}
        for name,sel in selections.items():
            method=sel['method'];vm=val[0][method];tm=test[0][method];vs=top_score(vm,sel['z']);ts=top_score(tm,sel['z']);fpr,legacy=calibration(val[2],vs)
            out=args.output/name;out.mkdir()
            result=dict(protocol='crmc_supplement_v1',selection=sel,validation_metrics=stats(val[2],vs,vm,val[1]),metrics=stats(test[2],ts,tm,test[1]),
                primary_fpr10=op(test[2],ts,fpr),legacy_f1=op(test[2],ts,legacy),validation_fpr10=op(val[2],vs,fpr),
                stratified=stratified_metrics(testds,[dict(image_path=p) for p in test[3]],test[2],ts,fpr),
                positive_images_with_empty_resized_mask=int(((test[2]==1)&~test[1].reshape(len(test[2]),-1).any(1)).sum()))
            export_scores(out,'validation',val[3],val[2],vs,fpr,legacy);export_scores(out,'test_best_val',test[3],test[2],ts,fpr,legacy)
            write_json(out/'test_best_val_v3.json',result);reports[name]=result;notify(phase='report',method=name)
        write_json(args.output/'supplement_report.json',dict(provenance=provenance,results=reports))
        notify(state='completed')
    except BaseException as e:
        notify(state='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if lock is not None:lock.close()
