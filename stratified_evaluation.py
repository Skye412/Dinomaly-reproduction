"""Native-mask test strata. Never used for checkpoint or threshold selection."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

PALETTE = {(255, 0, 0): 'Crack', (0, 255, 0): 'Spalling',
           (0, 0, 255): 'Corrosion', (255, 255, 0): 'Efflorescence'}
BINS = [('lt_0.1pct', 0, .001), ('0.1_0.5pct', .001, .005),
        ('0.5_1pct', .005, .01), ('1_5pct', .01, .05), ('ge_5pct', .05, float('inf'))]


def native_metadata(dataset):
    """Read original masks, never the 392/256 evaluation masks."""
    result = {}
    dacl = hasattr(dataset, 'rows') and bool(dataset.rows) and 'annotation_path' in dataset.rows[0]
    colored = dataset.root.name in {'Bridge512_balance_CODEBRIM', 'Bridge512_balance_S2DS'}
    for row in dataset.rows:
        label = int(row['category'] == 'annotated')
        path = dataset.root / row['image_path'] if dacl else dataset.root / dataset.split / row['image_path']
        classes = []; foreground = None
        if label:
            if dacl:
                from dacl_dataset import damage_mask
                annotation = json.loads((dataset.root / row['annotation_path']).read_text(encoding='utf-8'))
                foreground = np.asarray(damage_mask(annotation)) != 0
            else:
                with Image.open(dataset.root / dataset.split / row['mask_path']) as im:
                    rgb = np.asarray(im.convert('RGB'))
                foreground = np.any(rgb != 0, axis=2)
                if colored:
                    colors = {tuple(c) for c in np.unique(rgb.reshape(-1, 3), axis=0)} - {(0, 0, 0)}
                    if colors - set(PALETTE):
                        raise ValueError('Unknown defect mask colors: {}'.format(colors - set(PALETTE)))
                    classes = sorted(PALETTE[c] for c in colors)
            if foreground.shape != (512, 512):
                raise ValueError('Native strata require 512x512 masks: {}'.format(path))
        result[str(path)] = dict(label=label, area_fraction=float(foreground.mean()) if label else 0.0,
                                 defect_pixels=int(foreground.sum()) if label else 0,
                                 classes=classes, native_mask_resolution=512)
    return result, colored


def stratified_metrics(dataset, score_rows, y, scores, threshold):
    metadata, colored = native_metadata(dataset)
    if len(score_rows) != len(y) or len(y) != len(dataset):
        raise ValueError('Strata require complete test dataset and aligned scores')
    records = []
    for row, label, score in zip(score_rows, y, scores):
        item = metadata[row['image_path']]
        if item['label'] != int(label):
            raise ValueError('Native metadata/score label mismatch')
        records.append(dict(item, image_path=row['image_path'], score=float(score)))
    def measure(selected):
        index = np.array([r['label'] == 0 or (r['label'] == 1 and selected(r)) for r in records])
        labels = np.asarray(y)[index]; values = np.asarray(scores)[index]
        npos = int((labels == 1).sum()); nneg = int((labels == 0).sum())
        return dict(normal=nneg, anomaly=npos,
                    image_auroc=float(roc_auc_score(labels, values)) if npos and nneg else None,
                    recall_fpr10=float((values[labels == 1] >= threshold).mean()) if npos else None,
                    false_positive_normal=int((values[labels == 0] >= threshold).sum()),
                    threshold=float(threshold), undefined_reason=None if npos and nneg else 'empty positive or normal stratum')
    return dict(protocol='native512_test_strata_v1',
                definition='All normal test images plus relevant anomalous images; fixed validation-normal FPR10 threshold; lower-inclusive upper-exclusive area bins',
                area_bins={name: measure(lambda r, lo=lo, hi=hi: lo <= r['area_fraction'] < hi) for name, lo, hi in BINS},
                classes={cls: measure(lambda r, cls=cls: cls in r['classes']) for cls in PALETTE.values()} if colored else {},
                per_image=records)
