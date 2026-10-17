"""Summarize completed V3 and protocol-v3 V0 records; never runs inference/training."""
import argparse
import json
from pathlib import Path

PROTOCOL = 'v3_shared_z_fpr_256_v3'
GRID = [.001,.0025,.005,.01,.02,.05,.1,.2,.5,1.]
KEYS = ['image_auroc','image_ap','image_f1_max','pixel_auroc','pixel_ap','pixel_f1_max','pixel_aupro']

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results_root',type=Path,required=True)
    ap.add_argument('--v0_root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--bootstrap_samples',type=int,default=2000)
    ap.add_argument('--bootstrap_seed',type=int,default=1)
    args=ap.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError('bootstrap_samples must be positive')
    from reevaluate_v3 import paired_comparison
    records=[]; deltas=[]
    for ds in ['bridge512','dacl']:
        folder=args.results_root/('v3-'+ds+'-b16-s1-20261016')
        status=read(folder/'status.json'); summary=read(folder/'summary.json')
        if status['state']!='completed' or status['step']!=10000 or status['mode']!='full':
            raise ValueError('V3 full run incomplete: '+ds)
        points=[json.loads(line) for line in (folder/'validation_z.jsonl').read_text(encoding='utf-8').splitlines()]
        if [p['step'] for p in points]!=[2000,4000,6000,8000,10000]:
            raise ValueError('Scheduled validation points incomplete: '+ds)
        for point in points:
            if [c['top_fraction'] for c in point['candidates']]!=GRID:
                raise ValueError('Validation z grid differs')
            expected = max(point['candidates'], key=lambda c:(c['metrics'][0],c['metrics'][4]))
            if point['selected']['top_fraction'] != expected['top_fraction'] or point['selected']['metrics'] != expected['metrics']:
                raise ValueError('Validation z selection inconsistent')
        chosen=max(points,key=lambda p:(p['selected']['metrics'][0],p['selected']['metrics'][4]))['selected']
        selection=read(folder/'best_selection.json')
        if chosen['step']!=summary['best_validation_step'] or chosen['step']!=selection['step']:
            raise ValueError('Validation checkpoint selection inconsistent')
        ref=args.v0_root/('v0-'+ds); baseline=read(ref/'reevaluation.json')
        if baseline['protocol']!=PROTOCOL:
            raise ValueError('Wrong V0 reference protocol')
        z=read(folder/('test_best_val_v3.json' if (folder/'test_best_val_v3.json').exists() else 'test_final_v3.json'))
        if z['step']!=chosen['step'] or z['top_fraction']!=chosen['top_fraction']:
            raise ValueError('Test evaluated a different checkpoint/z')
        candidate_csv = folder/'test_best_val_scores.csv'
        if not candidate_csv.exists():
            if chosen['step'] != summary['final_step']:
                raise ValueError('Missing selected checkpoint scores')
            candidate_csv = folder/'test_final_scores.csv'
        paired = paired_comparison(ref/'test_best_val_scores.csv',
            candidate_csv, args.bootstrap_samples, args.bootstrap_seed)
        differences={key:chosen['metrics'][idx]-baseline['validation_selection']['metrics'][idx]
                     for key,idx in [('image_auroc',0),('pixel_ap',4)]}
        deltas.extend(differences.values())
        records.append(dict(dataset=ds,paired_auc_delta_ci95=paired,
            bootstrap_samples=args.bootstrap_samples,bootstrap_seed=args.bootstrap_seed,v0_test=dict(zip(KEYS,baseline['metrics'])),
            v3_test=summary['test_best_val']['metrics'],v0_validation=baseline['validation_selection'],
            v3_validation=chosen,validation_curve=[p['selected'] for p in points],validation_delta=differences,
            v0_operating_points={k:baseline['calibrated'][k] for k in ['primary_fpr10','legacy_f1']},
            v3_operating_points={k:z[k] for k in ['primary_fpr10','legacy_f1']},
            pixel_random_ap=z['pixel_anomaly_prevalence'],auc_ci95=z['image_auc_ci95'],
            best_step=chosen['step'],top_fraction=chosen['top_fraction'],
            training_seconds=summary['training_seconds'],total_seconds=summary['total_seconds'],
            peak_allocated_gib=summary['peak_cuda_allocated_gib'],
            peak_reserved_gib=summary['peak_cuda_reserved_gib'],
            added_parameters=summary['model_info']['added_parameters']))
    report=dict(protocol=PROTOCOL,records=records,
        retention=dict(all_deltas_ge_minus_0_005=all(d>=-.005 for d in deltas),
                       any_delta_ge_0_01=any(d>=.01 for d in deltas),
                       passed=all(d>=-.005 for d in deltas) and any(d>=.01 for d in deltas)),
        limitations=['One training seed only','V0 z selected at frozen best_val; V3 checkpoint/z jointly selected',
                     '392 V0 vs 448 V3 is bundled training/input change',
                     'Image bootstrap does not model training-seed or bridge/session dependence'])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# V0 / V3 completed comparison','', '| Dataset | Model | I-AUROC | P-AUROC | P-AP | P-F1max | AUPRO |',
           '|---|---|---:|---:|---:|---:|---:|']
    for record in records:
        for name,key in [('V0','v0_test'),('V3','v3_test')]:
            lines.append('| '+record['dataset']+' | '+name+' | '+' | '.join('{:.4f}'.format(record[key][k]*100)
                for k in ['image_auroc','pixel_auroc','pixel_ap','pixel_f1_max','pixel_aupro'])+' |')
    lines+=['','Retention rule passed: '+str(report['retention']['passed']),
            'Thresholds/checkpoints selected on validation only; percentages shown above. No automatic next experiment.']
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
