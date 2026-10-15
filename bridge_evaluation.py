"""Upstream Dinomaly metrics, plus validation-calibrated image threshold reporting."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_recall_fscore_support
from utils import cal_anomaly_maps, get_gaussian_kernel, compute_pro, f1_score_max


def evaluate(model, loader, device, output_csv, threshold=None):
    model.eval()
    masks, maps, labels, scores, paths = [], [], [], [], []
    kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
        for image, gt, label, names in loader:
            en, de = model(image.to(device))
            amap, _ = cal_anomaly_maps(en, de, image.shape[-1])
            amap = F.interpolate(amap, size=256, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=256, mode='nearest').bool()
            amap = kernel(amap)
            score = torch.sort(amap.flatten(1), dim=1, descending=True)[0][:, :int(256 * 256 * 0.01)].mean(1)
            masks.append(gt[:, 0].cpu().numpy())
            maps.append(amap[:, 0].cpu().numpy())
            labels.extend(label.tolist())
            scores.extend(score.cpu().tolist())
            paths.extend(names)
    gt, pr = np.concatenate(masks), np.concatenate(maps)
    y, s = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(pr).all() or not np.isfinite(s).all():
        raise RuntimeError('Non-finite evaluation scores')
    metrics = [roc_auc_score(y, s), average_precision_score(y, s), f1_score_max(y, s),
               roc_auc_score(gt.ravel(), pr.ravel()), average_precision_score(gt.ravel(), pr.ravel()),
               f1_score_max(gt.ravel(), pr.ravel()), compute_pro(gt, pr)]
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y, s)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        threshold = float(thresholds[np.argmax(f1)])
    predictions = s >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
    calibrated = dict(threshold=float(threshold), image_precision=float(precision), image_recall=float(recall),
                      image_f1=float(f1), samples=len(y), normal=int((y == 0).sum()), anomaly=int((y == 1).sum()),
                      positive_images_with_empty_resized_mask=int(((y == 1) & ~gt.reshape(len(y), -1).any(1)).sum()))
    tp = int(((y == 1) & predictions).sum())
    fp = int(((y == 0) & predictions).sum())
    tn = int(((y == 0) & ~predictions).sum())
    fn = int(((y == 1) & ~predictions).sum())
    specificity = tn / max(tn + fp, 1)
    calibrated.update(tp=tp, fp=fp, tn=tn, fn=fn,
                      false_positive_rate=fp / max(tn + fp, 1), specificity=specificity,
                      balanced_accuracy=(float(recall) + specificity) / 2)
    with Path(output_csv).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'score', 'prediction_at_validation_threshold'])
        writer.writerows(zip(paths, y.tolist(), s.tolist(), predictions.astype(int).tolist()))
    return metrics, calibrated
