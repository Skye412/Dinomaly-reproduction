"""Audit existing datasets without modifying them; compare VisA official split."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path('/home/skye/data/Skye')
REPO = ROOT / 'Dinomaly'
OUT = REPO / 'reproduction'
OUT.mkdir(exist_ok=True)
errors = []
report = {'sources': {
    'visa_split': 'https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv',
    'mvtec': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
    'paper': 'CVPR 2025 Dinomaly, section 4.1'},
    'limitations': 'Structural/split/integrity audit, not byte-for-byte verification against official image archive.'}

def images(folder):
    return sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'})

cached = OUT / 'spot-diff-official/split_csv/1cls.csv'
if not cached.is_file():
    cached = OUT / 'official_visa_1cls.csv'
data = cached.read_bytes() if cached.is_file() else urlopen(report['sources']['visa_split'], timeout=60).read()
(OUT / 'official_visa_1cls.csv').write_bytes(data)
official = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
visa = ROOT / 'databases/VISA'
report['visa_split_sha256'] = hashlib.sha256(data).hexdigest()
report['visa_local_split_matches_official'] = rows(visa / 'split_csv/1cls.csv') == official
report['dinomaly_split_matches_official'] = rows(REPO / 'prepare_data/split_csv/1cls.csv') == official
if not all([report['visa_local_split_matches_official'], report['dinomaly_split_matches_official']]):
    errors.append('VisA split differs from official')
expected_images = {r['image'] for r in official}
expected_masks = {r['mask'] for r in official if r['mask']}
actual_images = {p.relative_to(visa).as_posix() for p in images(visa) if '/Images/' in p.as_posix()}
actual_masks = {p.relative_to(visa).as_posix() for p in images(visa) if '/Masks/' in p.as_posix()}
for label, expected, actual in [('images', expected_images, actual_images), ('masks', expected_masks, actual_masks)]:
    report['visa_' + label + '_missing'] = sorted(expected - actual)
    report['visa_' + label + '_extra'] = sorted(actual - expected)
    if expected != actual:
        errors.append('VisA ' + label + ' inventory mismatch')
report['visa'] = {}
for category in sorted({r['object'] for r in official}):
    subset = [r for r in official if r['object'] == category]
    counts = Counter((r['split'], r['label']) for r in subset)
    report['visa'][category] = {'train_good': counts['train', 'normal'], 'test_good': counts['test', 'normal'], 'test_bad': counts['test', 'anomaly'], 'masks': sum(bool(r['mask']) for r in subset)}
report['visa_totals'] = {k: sum(v[k] for v in report['visa'].values()) for k in ['train_good', 'test_good', 'test_bad', 'masks']}
if report['visa_totals'] != dict(train_good=8659, test_good=962, test_bad=1200, masks=1200):
    errors.append('VisA totals differ from Dinomaly paper')

# Published standard MVTec-AD per-class counts: train normal, test normal, test anomalous.
expected_mvtec = {'bottle': (209,20,63), 'cable': (224,58,92), 'capsule': (219,23,109),
    'carpet': (280,28,89), 'grid': (264,21,57), 'hazelnut': (391,40,70),
    'leather': (245,32,92), 'metal_nut': (220,22,93), 'pill': (267,26,141),
    'screw': (320,41,119), 'tile': (230,33,84), 'toothbrush': (60,12,30),
    'transistor': (213,60,40), 'wood': (247,19,60), 'zipper': (240,32,119)}
mvtec = ROOT / 'databases/MVTEC'
report['mvtec'] = {}
all_pairs = []
for category, expected in sorted(expected_mvtec.items()):
    folder = mvtec / category
    train = images(folder / 'train/good')
    good = images(folder / 'test/good')
    bad = [p for p in images(folder / 'test') if p.parent.name != 'good']
    masks = images(folder / 'ground_truth')
    counts = (len(train), len(good), len(bad))
    report['mvtec'][category] = dict(train_good=counts[0], test_good=counts[1], test_bad=counts[2], masks=len(masks), counts_match=counts == expected)
    if counts != expected or len(masks) != len(bad):
        errors.append('MVTec counts mismatch: ' + category)
    if set(images(folder / 'train')) != set(train):
        errors.append('Non-normal MVTec training images: ' + category)
    paired = set()
    for p in bad:
        mask = folder / 'ground_truth' / p.parent.name / (p.stem + '_mask.png')
        if not mask.is_file():
            errors.append('Missing mask: ' + str(mask))
        else:
            paired.add(mask)
            all_pairs.append((p, mask))
    if set(masks) != paired:
        errors.append('Extra/unpaired MVTec masks: ' + category)
report['mvtec_totals'] = {k: sum(v[k] for v in report['mvtec'].values()) for k in ['train_good', 'test_good', 'test_bad', 'masks']}
for r in official:
    if r['mask']:
        all_pairs.append((visa / r['image'], visa / r['mask']))

# Verify headers and PNG CRCs for every image; fully decode each mask and check dimensions.
from PIL import Image
import numpy as np
checked = 0
for p in images(mvtec) + images(visa):
    try:
        with Image.open(p) as im:
            im.verify()
        checked += 1
    except Exception as exc:
        errors.append('Unreadable image: {}: {}'.format(p, exc))
for p, m in all_pairs:
    try:
        with Image.open(p) as im, Image.open(m) as mask:
            if im.size != mask.size:
                errors.append('Mask dimension mismatch: ' + str(m))
            if not np.asarray(mask).any():
                errors.append('Empty anomaly mask: ' + str(m))
    except Exception as exc:
        errors.append('Invalid mask pair: {}: {}'.format(m, exc))
report['verified_image_and_mask_files'] = checked
report['mask_pairs_checked'] = len(all_pairs)
report['errors'] = errors
report['passed'] = not errors
(OUT / 'dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
raise SystemExit(bool(errors))
