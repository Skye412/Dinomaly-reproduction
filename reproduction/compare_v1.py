"""Summarize already completed V0/V1 runs without model inference or retuning."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', type=Path, default=Path('reproduction/results'))
    args = parser.parse_args()
    pairs = [('Bridge512', 'bridge512-balance-repaired-b16-s1-20261014', 'v1-amcm-bridge512-b16-s1-20261015'),
             ('DACL', 'dacl-v3-seed42-b16-s1-20261015', 'v1-amcm-dacl-b16-s1-20261015')]
    keys = ['image_auroc', 'pixel_auroc', 'pixel_ap', 'pixel_f1_max', 'pixel_aupro']
    records = []
    deltas = []
    lines = ['# V0 / V1 AMCM seed-1 comparison', '',
             'Historical 392-input / 256-metric protocol for both versions. No evaluator correction, test selection, or automatic tuning.', '',
             '| Dataset | Version | I-AUROC % | P-AUROC % | P-AP % | P-F1max % | AUPRO % | Recall % | Normal FP / N | Best step | Wall hours | Peak allocated GiB | Added params |',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for name, baseline_id, variant_id in pairs:
        comparison = {}
        for version, run_id in [('V0', baseline_id), ('V1', variant_id)]:
            folder = args.results_root / run_id
            summary = json.loads((folder / 'summary.json').read_text())
            status = json.loads((folder / 'status.json').read_text())
            chosen = summary['test_best_val']
            cal = chosen['calibrated']
            scores = folder / 'test_best_val_scores.csv'
            if not scores.exists():
                assert summary['best_validation_step'] == summary['final_step']
                scores = folder / 'test_final_scores.csv'
            rows = list(csv.DictReader(scores.open(newline='')))
            fp = sum(int(r['label']) == 0 and int(r['prediction_at_validation_threshold']) == 1 for r in rows)
            normal = sum(int(r['label']) == 0 for r in rows)
            duration = status['finished_at'] - status['started_at']
            peak = summary.get('peak_cuda_allocated_gib')
            if peak is None:
                runtime_file = Path(__file__).resolve().parent / 'baselines' / 'V0_RUNTIME.json'
                runtime = json.loads(runtime_file.read_text())
                peak = runtime[run_id]['peak_allocated_gib_logged']
            record = dict(dataset=name, version=version, run_id=run_id, metrics=chosen['metrics'],
                          recall=cal['image_recall'], normal_false_positives=fp, normal_images=normal,
                          best_step=summary['best_validation_step'], total_seconds=duration,
                          peak_cuda_allocated_gib=peak,
                          added_parameters=summary.get('model_info', {}).get('added_parameters', 0))
            records.append(record)
            comparison[version] = chosen['metrics']
            values = ['{:.2f}'.format(chosen['metrics'][k] * 100) for k in keys]
            lines.append('| {} | {} | {} | {:.2f} | {} / {} | {} | {:.3f} | {} | {} |'.format(
                name, version, ' | '.join(values), cal['image_recall'] * 100, fp, normal,
                summary['best_validation_step'], duration / 3600,
                '{:.3f}'.format(peak) if peak is not None else 'not archived', record['added_parameters']))
        deltas += ['', '{} V1-V0 deltas (percentage points): {}'.format(name, ', '.join(
            '{} {:+.2f}'.format(k, 100 * (comparison['V1'][k] - comparison['V0'][k])) for k in keys)), '']
    lines += deltas + ['V0 memory peaks are original server log values rounded to 0.001 GiB; V1 retains raw JSON values.',
              'DACL image F1/AP are prevalence sensitive; assess image AUROC, pixel AP/AUPRO and recall with false positives.',
              'A single seed is module-screening evidence, not a stability claim. No further experiments are launched.']
    out = args.results_root / 'v1-amcm-seed1-20261015'
    out.mkdir(exist_ok=True)
    (out / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    (out / 'comparison.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
