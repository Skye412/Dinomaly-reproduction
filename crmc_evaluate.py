"""CRMC evaluation-only entry point for the three frozen V0 datasets."""
import argparse,csv,json,subprocess,time,traceback
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
from bridge_dataset import check_split_protocol,dataset_class,load_profile
from dinomaly_bridge import build_model
from crmc import build_memory_bank,load_bank,forward_with_context,nearest_memory_maps,prepare_map,fuse_maps,sha256_file
from crmc_diagnostics import diagnostics
from stratified_evaluation import stratified_metrics
from v3_evaluation import bootstrap_auc
from utils import compute_pro,f1_score_max,get_gaussian_kernel

FRACTIONS=(.001,.0025,.005,.01,.02,.05,.1,.2,.5,1.);ALPHAS=(0,.25,.5,.75,1.)

def op(y,s,t):
 p=s>=t;tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());tn=int(((y==0)&~p).sum());fn=int(((y==1)&~p).sum());r=tp/max(tp+fn,1);pr=tp/max(tp+fp,1);sp=tn/max(tn+fp,1)
 return dict(threshold=float(t),image_precision=pr,image_recall=r,image_f1=2*pr*r/max(pr+r,1e-12),tp=tp,fp=fp,tn=tn,fn=fn,normal=tn+fp,anomaly=tp+fn,samples=len(y),false_positive_rate=fp/max(tn+fp,1),specificity=sp,balanced_accuracy=(r+sp)/2)

def stats(labels,scores,amap,masks):
 return dict(image_auroc=float(roc_auc_score(labels,scores)),image_ap=float(average_precision_score(labels,scores)),image_f1_max=float(f1_score_max(labels,scores)),pixel_auroc=float(roc_auc_score(masks.reshape(-1),amap.reshape(-1))),pixel_ap=float(average_precision_score(masks.reshape(-1),amap.reshape(-1))),pixel_f1_max=float(f1_score_max(masks.reshape(-1),amap.reshape(-1))),pixel_aupro=float(compute_pro(masks.astype(bool),amap)))

def normal_stats(x,y):
 n=x[y==0];return float(n.mean()),float(n.std()+1e-8)

def run(model, loader, bank, device, kernel, folder, notify):
    # Process/retrieve one batch at a time; do not retain a second feature bank.
    rec, masks, labels, paths = [], [], [], []
    mem = {str(k): [] for k in (4, 16, 64)}
    mem['global'] = []
    references = []
    with torch.no_grad():
        for batch_index, (images, gt, label, names) in enumerate(loader):
            images = images.to(device)
            cls, groups, raw = forward_with_context(model, images)
            rec.append(prepare_map(raw, kernel)[:, 0].cpu().numpy())
            maps, selected_ids = nearest_memory_maps(
                cls, groups, bank[1], bank[2], (4, 16, 64), True)
            for key, val in maps.items():
                mem[key].append(prepare_map(val, kernel)[:, 0].cpu().numpy())
            masks.append(torch.nn.functional.interpolate(
                gt.float(), size=256, mode='nearest')[:, 0].bool().numpy())
            labels.extend(label.tolist())
            paths.extend(names)
            references.extend(dict(image_path=name, normal_bank_indices=ids)
                              for name, ids in zip(names, selected_ids))
            notify(phase='retrieving_' + loader.dataset.split,
                   batch=batch_index + 1, samples=len(paths))
    result = (np.concatenate(rec), {k: np.concatenate(v) for k, v in mem.items()},
              np.concatenate(masks), np.asarray(labels), paths)
    # Runtime maps are useful for audit/diagnostics, and stay outside Git.
    np.savez(folder / (loader.dataset.split + '_maps.npz'), rec=result[0],
             gt=result[2], labels=result[3], paths=np.asarray(paths), **result[1])
    write_json(folder / (loader.dataset.split + '_retrieval.json'), references)
    return result


def top_score(amap,z):
 n=max(1,int(amap.shape[-1]*amap.shape[-1]*z));return np.sort(amap.reshape(len(amap),-1),axis=1)[:,::-1][:,:n].mean(1)

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def selection_key(row):
    return row['image_auroc'], row['pixel_ap']


def select_candidates(val):
    rm, rs = normal_stats(val[0], val[3])
    candidates, normalizations = [], {}
    for key in ('4', '16', '64', 'global'):
        mm, ms = normal_stats(val[1][key], val[3])
        norm = dict(rec_mean=rm, rec_std=rs, mem_mean=mm, mem_std=ms)
        normalizations[key] = norm
        for mode in ('weighted_sum', 'gated'):
            for alpha in (ALPHAS if mode == 'weighted_sum' else (None,)):
                amap = fuse_maps(val[0], val[1][key], norm, mode, alpha)
                pixel_ap = float(average_precision_score(val[2].ravel(), amap.ravel()))
                # Sort each candidate map once for all ten z values.
                sorted_maps = np.sort(amap.reshape(len(amap), -1), axis=1)[:, ::-1]
                for z in FRACTIONS:
                    scores = sorted_maps[:, :max(1, int(sorted_maps.shape[1] * z))].mean(1)
                    candidates.append(dict(k=key, mode=mode, alpha=alpha, top_fraction=z,
                                           image_auroc=float(roc_auc_score(val[3], scores)),
                                           pixel_ap=pixel_ap))
    return candidates, normalizations


def named_selections(candidates):
    def best(predicate):
        return max((c for c in candidates if predicate(c)), key=selection_key)
    selected = {'main': best(lambda c: c['k'] == '16')}
    for k in ('4', '16', '64', 'global'):
        selected['retrieval_' + k] = best(lambda c: c['k'] == k)
    selected['weighted_sum'] = best(lambda c: c['k'] == '16' and c['mode'] == 'weighted_sum')
    selected['gated'] = best(lambda c: c['k'] == '16' and c['mode'] == 'gated')
    selected['reconstruction_only'] = best(lambda c: c['k'] == '16' and c['mode'] == 'weighted_sum' and c['alpha'] == 0)
    selected['memory_only'] = best(lambda c: c['k'] == '16' and c['mode'] == 'weighted_sum' and c['alpha'] == 1)
    selected['mixed_sum'] = best(lambda c: c['k'] == '16' and c['mode'] == 'weighted_sum' and c['alpha'] in (.25, .5, .75))
    return selected


def calibration(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(np.percentile(scores[y == 0], 90)), float(thresholds[np.argmax(f1)])


def export_scores(folder, split, paths, y, scores, fpr10, legacy):
    for suffix, threshold in [('scores.csv', legacy), ('fpr10_scores.csv', fpr10)]:
        with (folder / (split + '_' + suffix)).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label', 'score', 'prediction_at_validation_threshold'])
            writer.writerows((p, int(v), float(s), int(s >= threshold))
                             for p, v, s in zip(paths, y, scores))


def main(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    allowed = {'codebrim512', 's2ds512', 'dacl_v3_seed42'}
    if cfg['dataset_profile'] not in allowed or cfg.get('model_variant', 'v0') != 'v0':
        raise ValueError('CRMC requires one of the three frozen V0 benchmarks')
    if cfg['image_size'] != 392 or cfg.get('context_recenter') or cfg.get('encoder_fp16'):
        raise ValueError('Requires original FP32, uncentered, 392-input V0')
    if args.output.exists():
        raise ValueError('Use a new output directory; no automatic retry/resume')
    args.output.mkdir(parents=True)
    started = time.time()
    state = dict(pid=__import__('os').getpid(), state='starting')
    def notify(**kw):
        state.update(kw, updated_at=time.time())
        write_json(args.output / 'status.json', state)
        with (args.output / 'console.log').open('a', encoding='utf-8') as f:
            f.write(json.dumps(kw) + '\n')
    lock = None
    try:
        from reproduction_gpu import acquire_gpu_lock
        root = Path(__file__).resolve().parent
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=str(root), text=True).strip())
        if dirty:
            raise RuntimeError('Tracked execution checkout must be clean')
        notify(state='waiting_for_gpu', git_commit=head)
        (root / 'runs').mkdir(exist_ok=True)
        lock = acquire_gpu_lock(root / 'runs/gpu0.lock', notify, 14500)
        check_split_protocol(cfg['data_path'], cfg['dataset_profile'], cfg.get('split_view', 'original'))
        metadata, exclusions, _ = load_profile(cfg['dataset_profile'], cfg.get('split_view', 'original'))
        provenance = dict(git_commit=head, checkpoint_sha256=sha256_file(args.checkpoint),
                          original_config=cfg, manifest_sha256=metadata['manifest_sha256'])
        write_json(args.output / 'config.json', dict(provenance, training=False,
                   main_k=16, retrieval='exact cosine; CPU fp16 raw bank, FP32 compute',
                   source_object_root=str(args.s2ds_source_root)))
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        model, _ = build_model(args.device, 'v0')
        ck = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ck['model'], strict=True)
        step = int(ck['step']); del ck
        model.eval(); model.requires_grad_(False)
        DatasetClass = dataset_class(cfg['dataset_profile'])
        train = DatasetClass(cfg['data_path'], 'train', normal_only=True,
                             image_size=392, exclude_samples=exclusions['train'])
        notify(state='building_bank')
        bank = build_memory_bank(model, DataLoader(train, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4), args.device,
                                 args.output / 'memory_bank', provenance)
        def loader(split):
            return DataLoader(DatasetClass(cfg['data_path'], split, image_size=392,
                              mask_size=392, exclude_samples=exclusions[split]),
                              batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
        kernel = get_gaussian_kernel(5, 4).to(args.device)
        notify(state='validation')
        val = run(model, loader('val'), bank, args.device, kernel, args.output, notify)
        candidates, normalizations = select_candidates(val)
        selections = named_selections(candidates)
        write_json(args.output / 'best_selection.json', dict(step=step, selected=selections['main']))
        write_json(args.output / 'normalization.json', normalizations)
        with (args.output / 'validation_z.jsonl').open('w', encoding='utf-8') as f:
            f.write(json.dumps(dict(step=step, candidates=candidates, selected=selections['main'])) + '\n')
        # All main/ablation choices are frozen on disk BEFORE test inference.
        write_json(args.output / 'ablation_selections.json', selections)
        notify(state='test')
        test = run(model, loader('test'), bank, args.device, kernel, args.output, notify)
        reports = {}; cached = {}; main_map = None
        test_dataset = loader('test').dataset
        for name, sel in selections.items():
            signature = tuple(sel[k] for k in ('k', 'mode', 'alpha', 'top_fraction'))
            if signature in cached:
                reports[name] = cached[signature]
                continue
            folder = args.output if name == 'main' else args.output / 'ablations' / name
            folder.mkdir(parents=True, exist_ok=True)
            norm = normalizations[sel['k']]
            vm = fuse_maps(val[0], val[1][sel['k']], norm, sel['mode'], sel['alpha'])
            tm = fuse_maps(test[0], test[1][sel['k']], norm, sel['mode'], sel['alpha'])
            vs, ts = top_score(vm, sel['top_fraction']), top_score(tm, sel['top_fraction'])
            fpr10, legacy = calibration(val[3], vs)
            rows = [dict(image_path=p) for p in test[4]]
            empty = int(((test[3] == 1) & ~test[2].reshape(len(test[3]), -1).any(1)).sum())
            result = dict(protocol='crmc_eval_v1', model_variant='v0_plus_crmc',
                          step=step, selected=sel, validation_metrics=stats(val[3], vs, vm, val[2]),
                          metrics=stats(test[3], ts, tm, test[2]),
                          primary_fpr10=op(test[3], ts, fpr10), legacy_f1=op(test[3], ts, legacy),
                          validation_fpr10=op(val[3], vs, fpr10),
                          top_fraction=sel['top_fraction'], pixel_metric_resolution=256,
                          positive_images_with_empty_resized_mask=empty,
                          stratified=stratified_metrics(test_dataset, rows, test[3], ts, fpr10))
            export_scores(folder, 'test_best_val', test[4], test[3], ts, fpr10, legacy)
            export_scores(folder, 'validation', val[4], val[3], vs, fpr10, legacy)
            write_json(folder / 'test_best_val_v3.json', result)
            write_json(folder / 'summary.json', result)
            cached[signature] = result; reports[name] = result
            if name == 'main':
                main_map = tm; main_scores = ts
        raw = selections['reconstruction_only']
        baseline_folder = Path(args.baseline_dir)
        baseline_summary = json.loads((baseline_folder / 'summary.json').read_text(encoding='utf-8'))
        expected = (baseline_summary['test_best_val']['metrics']['image_auroc']
                    if 'test_best_val' in baseline_summary else baseline_summary['metrics'][0])
        observed = reports['reconstruction_only']['metrics']['image_auroc']
        archived = list(csv.DictReader((baseline_folder / 'test_best_val_scores.csv').open()))
        if [r['image_path'] for r in archived] != test[4] or [int(r['label']) for r in archived] != test[3].tolist():
            raise RuntimeError('V0 archived sample paths/labels do not match')
        baseline_test = top_score(test[0], raw['top_fraction'])
        archived_scores = np.array([float(r['score']) for r in archived])
        baseline_check = dict(expected_image_auroc=expected, observed_image_auroc=observed,
                              absolute_delta=abs(expected-observed),
                              maximum_score_difference=float(np.max(np.abs(baseline_test-archived_scores))),
                              passed=abs(expected-observed) <= 1e-12)
        write_json(args.output / 'v0_consistency.json', baseline_check)
        if not baseline_check['passed']:
            write_json(args.output / 'crmc_report.json', dict(state='failed', v0_consistency=baseline_check))
            raise RuntimeError('Reconstruction-only AUROC differs from archived V0')
        main_val_map = fuse_maps(val[0], val[1]['16'], normalizations['16'],
                                 selections['main']['mode'], selections['main']['alpha'])
        paired = dict(test=bootstrap_auc(test[3], main_scores, 2000, cfg['seed'], baseline_test),
                      validation=bootstrap_auc(val[3], top_score(main_val_map, selections['main']['top_fraction']),
                                               2000, cfg['seed'], top_score(val[0], raw['top_fraction'])))
        baseline_scores = top_score(test[0], raw['top_fraction'])
        norm = normalizations['16']
        before_display = ((test[0] - norm['rec_mean']) / norm['rec_std']
                          if selections['main']['mode'] == 'weighted_sum' else test[0])
        after_display = main_map
        if selections['main']['mode'] == 'weighted_sum' and selections['main']['alpha'] == 0:
            after_display = before_display
        notify(state='diagnostics')
        diag = diagnostics(test_dataset, cfg['dataset_profile'], test[4], test[3],
                           before_display, after_display, baseline_scores, main_scores,
                           args.output / 'diagnostics', args.s2ds_source_root)
        write_json(args.output / 'crmc_report.json', dict(provenance=provenance,
                   main=reports['main'], ablations=reports, diagnostics=diag,
                   v0_consistency=baseline_check, paired_auc_delta_ci95=paired,
                   elapsed_seconds=time.time() - started,
                   peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                   note='Fixed V0 checkpoint; no training; all selection on validation'))
        notify(state='completed', elapsed_seconds=time.time() - started)
    except BaseException as exc:
        notify(state='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--baseline_dir', type=Path, required=True)
    p.add_argument('--s2ds_source_root', type=Path, default='/home/skye/data/Skye/databases/S2DS')
    main(p.parse_args())
