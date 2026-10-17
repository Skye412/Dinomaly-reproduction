"""Explicit opt-in shared V3 evaluation of an existing V0/V1/V2/V3 checkpoint.
Never trains, never selects a checkpoint on test, never modifies source results.
Run from repository root. Requires original config and validation-selected checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.utils.data import DataLoader
from bridge_dataset import dataset_class, check_split_protocol, load_profile
from dinomaly_bridge import build_model
from v3_evaluation import V3Evaluator, bootstrap_auc
import csv


def paired_comparison(reference_csv, candidate_csv, repeats, seed):
    reference_csv = Path(reference_csv)
    provenance_path = reference_csv.parent / 'reevaluation.json'
    if reference_csv.name != 'test_best_val_scores.csv' or not provenance_path.is_file():
        raise ValueError('Use shared-protocol V0 reevaluation/test_best_val_scores.csv, not historical CSV')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance.get('protocol') != 'v3_shared_z_fpr_256_v1':
        raise ValueError('Reference was not produced by shared V3 reevaluation protocol')
    if provenance.get('original_config', {}).get('model_variant', 'v0') != 'v0':
        raise ValueError('Paired reference must be reevaluated V0')
    def rows(path):
        records = list(csv.DictReader(Path(path).open(newline='')))
        # Native recorded paths must match. No basename-based guesses or dropped rows.
        result = {r['image_path']: r for r in records}
        if len(result) != len(records):
            raise ValueError('Duplicate image identities')
        return result
    a, b = rows(reference_csv), rows(candidate_csv)
    if set(a) != set(b):
        raise ValueError('Paired comparison requires identical test images/paths')
    names = sorted(a)
    if any(a[k]['label'] != b[k]['label'] for k in names):
        raise ValueError('Paired labels differ')
    y = np.array([int(a[k]['label']) for k in names])
    return bootstrap_auc(y, np.array([float(b[k]['score']) for k in names]), repeats, seed,
                         baseline=np.array([float(a[k]['score']) for k in names]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bootstrap_samples', type=int, default=2000)
    parser.add_argument('--native_512', action='store_true')
    parser.add_argument('--paired_reference_csv', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; original results must remain untouched')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.bootstrap_samples <= 0:
        raise ValueError('bootstrap_samples must be positive')
    check_split_protocol(config['data_path'], config['dataset_profile'], config.get('split_view','original'))
    _, exclusions, _ = load_profile(config['dataset_profile'], config.get('split_view','original'))
    model, _ = build_model(args.device, config.get('model_variant','v0'))
    for attr, key in [('context_recenter','context_recenter'), ('encoder_fp16','encoder_fp16'),
                      ('reconstruction_fp32','fp32_reconstruction')]:
        setattr(model, attr, config.get(key, False))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    step = int(checkpoint['step'])
    args.seed = config['seed']
    args.output.mkdir(parents=True)
    evaluator = V3Evaluator(args, args.output)
    cls = dataset_class(config['dataset_profile'])
    def loader(split, mask_size):
        return DataLoader(cls(config['data_path'], split, image_size=config['image_size'],
                          mask_size=mask_size, exclude_samples=exclusions[split]),
                          batch_size=config['batch_size'], num_workers=4)
    evaluator.validation(model, loader('val',392), step)
    metrics, calibrated = evaluator.test(model, loader('test',392), 'best_val', step)
    report = dict(protocol='v3_shared_z_fpr_256_v1',
                  validation_selection=evaluator.selections[step], original_config=config, checkpoint=str(args.checkpoint.resolve()), step=step,
                  metrics=metrics, calibrated=calibrated,
                  note='Fixed supplied checkpoint; z/thresholds calibrated only on validation; no training')
    if args.paired_reference_csv:
        report['paired_auc_delta_ci95'] = paired_comparison(args.paired_reference_csv,
            args.output / 'test_best_val_scores.csv', args.bootstrap_samples, args.seed)
    (args.output/'reevaluation.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    if args.native_512:
        evaluator.native_test(model, loader('test',512), step)


if __name__ == '__main__':
    main()
