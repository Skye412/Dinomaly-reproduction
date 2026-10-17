"""Validation-only z/threshold selection; fixed test protocol and bootstrap CIs."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from bridge_evaluation import evaluate
from utils import f1_score_max


def operating_point(y, scores, threshold):
    prediction = scores >= threshold
    tp = int(((y == 1) & prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return dict(threshold=float(threshold), image_precision=precision, image_recall=recall,
                image_f1=2 * precision * recall / max(precision + recall, 1e-12),
                tp=tp, fp=fp, tn=tn, fn=fn, normal=tn+fp, anomaly=tp+fn, samples=len(y),
                false_positive_rate=fp / max(tn+fp, 1), specificity=specificity,
                balanced_accuracy=(recall + specificity)/2)


def bootstrap_auc(y, scores, repeats=2000, seed=1, baseline=None):
    rng = np.random.RandomState(seed)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    values = []
    for _ in range(repeats):
        index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        value = roc_auc_score(y[index], scores[index])
        if baseline is not None:
            value -= roc_auc_score(y[index], baseline[index])
        values.append(value)
    return dict(estimate=float(roc_auc_score(y, scores) - (roc_auc_score(y, baseline) if baseline is not None else 0)),
                lower=float(np.percentile(values, 2.5)), upper=float(np.percentile(values, 97.5)),
                repeats=repeats, seed=seed, method='image-stratified paired bootstrap' if baseline is not None else 'image-stratified bootstrap',
                unit='image; not bridge/session; not across training seeds')


class V3Evaluator:
    fractions = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)

    def __init__(self, args, folder):
        self.args, self.folder = args, folder
        self.selections = {}

    def validation(self, model, loader, step):
        # One model pass. Pixel maps/metrics shared by all z; never one forward per fraction.
        metrics, _, y, _, gt, maps = evaluate(model, loader, self.args.device,
            self.folder / ('val_scores_{}.csv'.format(step)), return_scores=True)
        original_rows = list(csv.DictReader((self.folder / ('val_scores_{}.csv'.format(step))).open()))
        sorted_maps = np.sort(maps.reshape(len(y), -1), axis=1)[:, ::-1]
        candidates = []
        for fraction in self.fractions:
            scores = sorted_maps[:, :max(1, int(maps.shape[-1]**2*fraction))].mean(1)
            precision, recall, thresholds = precision_recall_curve(y, scores)
            f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
            f1_threshold = float(thresholds[np.argmax(f1)])
            fpr_threshold = float(np.percentile(scores[y == 0], 90))
            candidate_metrics = list(metrics)
            candidate_metrics[:3] = [roc_auc_score(y, scores), average_precision_score(y, scores), f1_score_max(y, scores)]
            candidate = dict(step=step, top_fraction=fraction, metrics=candidate_metrics,
                             threshold=f1_threshold, fpr_threshold=fpr_threshold,
                             validation_fpr=operating_point(y, scores, fpr_threshold),
                             validation_f1=operating_point(y, scores, f1_threshold))
            candidates.append(candidate)
            path = self.folder / ('val_scores_{}_z{}.csv'.format(step, fraction))
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['image_path','label','score','prediction_at_validation_threshold'])
                writer.writerows((row['image_path'], int(label), float(score), int(score >= f1_threshold))
                                 for row, label, score in zip(original_rows, y, scores))
        # Stable ties retain first z, then earliest checkpoint (outer trainer >).
        selected = max(candidates, key=lambda c: (c['metrics'][0], c['metrics'][4]))
        self.selections[step] = selected
        with (self.folder / 'validation_z.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(step=step, candidates=candidates, selected=selected)) + '\n')
        return selected['metrics'], dict(threshold=selected['threshold'], top_fraction=selected['top_fraction'],
                                        primary=selected['validation_fpr'], legacy=selected['validation_f1'])

    def test(self, model, loader, name, step):
        selected = self.selections[step]
        path = self.folder / ('test_{}_scores.csv'.format(name))
        metrics, legacy, y, scores, gt, _ = evaluate(model, loader, self.args.device, path,
            threshold=selected['threshold'], top_fraction=selected['top_fraction'], return_scores=True)
        primary = operating_point(y, scores, selected['fpr_threshold'])
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        with (self.folder / ('test_{}_fpr10_scores.csv'.format(name))).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image_path','label','score','prediction_at_validation_fpr10_threshold'])
            writer.writerows((row['image_path'], int(label), float(score), int(score >= selected['fpr_threshold']))
                             for row, label, score in zip(rows, y, scores))
        details = dict(step=step, top_fraction=selected['top_fraction'],
            primary_fpr10=primary, legacy_f1=legacy,
            pixel_anomaly_prevalence=float(gt.mean()), pixel_metric_resolution=256,
            image_auc_ci95=bootstrap_auc(y, scores, self.args.bootstrap_samples, self.args.seed),
            threshold_note='90th normal-validation percentile, >= comparison; empirical FPR need not equal 10%')
        (self.folder / ('test_{}_v3.json'.format(name))).write_text(json.dumps(details, indent=2))
        return metrics, dict(primary_fpr10=primary, legacy_f1=legacy, **legacy)

    def native_test(self, model, loader, step):
        selected = self.selections[step]
        values, _, _, _, gt, _ = evaluate(model, loader, self.args.device,
            self.folder / 'test_best_val_native512_scores.csv',
            threshold=selected['threshold'], top_fraction=selected['top_fraction'],
            return_scores=True, metric_size=512)
        # Image thresholds were calibrated at 256 and are deliberately not reported here.
        from reproduction_runtime import METRIC_NAMES
        record = dict(step=step, metric_resolution=512, mask_resolution=512,
                      pixel_anomaly_prevalence=float(gt.mean()),
                      metrics=dict(zip(METRIC_NAMES[3:], values[3:])),
                      note='Pixel-only supplementary protocol; no checkpoint/threshold selection; compare versions using same evaluator')
        (self.folder / 'test_best_val_native512.json').write_text(json.dumps(record, indent=2))
