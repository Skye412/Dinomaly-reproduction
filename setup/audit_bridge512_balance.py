"""Read-only verification of Bridge512_balance manifests, RGB masks and split integrity."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/home/skye/data/Skye/databases/Bridge512_balance')
OUT = Path('/home/skye/data/Skye/Dinomaly/reproduction/audits/bridge512-balance-20261014')
COLORS = {'Crack': (255,0,0), 'Spalling': (0,255,0), 'Corrosion': (0,0,255), 'Efflorescence': (255,255,0)}
OUT.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


report = dict(root=str(ROOT), splits={}, errors=[], warnings=[])
errors = report['errors']
hashes = {}
for line in (ROOT / 'checksums.sha256').read_text().splitlines():
    expected, rel = line.split(None, 1)
    rel = rel.strip().lstrip('*')
    p = ROOT / rel
    if not p.is_file():
        errors.append('Missing checksum file: ' + rel)
        continue
    hashes[rel] = digest(p)
    if hashes[rel] != expected:
        errors.append('Checksum mismatch: ' + rel)
report['checksum_files_verified'] = len(hashes)
print('Checksum files verified:', len(hashes), flush=True)
parents, sample_ids, image_hashes = {}, {}, {}
allowed = {0, 0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00}
for split in ['train','val','test']:
    with (ROOT / (split + '_manifest.csv')).open(newline='') as f:
        rows = list(csv.DictReader(f))
    stats = dict(images=len(rows), categories=dict(Counter(r['category'] for r in rows)),
                 parents=len({r['parent_id'] for r in rows}), sources=dict(Counter(r['source'] for r in rows)),
                 class_images=Counter(), class_pixels=Counter())
    expected_files = set()
    for number, r in enumerate(rows, 1):
        image_rel = split + '/' + r['image_path']
        expected_files.add(image_rel)
        if r['assigned_split'] != split:
            errors.append('Assigned split mismatch: ' + r['sample'])
        for mapping, key, kind in [(parents,r['parent_id'],'parent'), (sample_ids,r['sample'],'sample')]:
            if key in mapping and mapping[key] != split:
                errors.append('Cross-split {} leakage: {}'.format(kind,key))
            mapping[key] = split
        try:
            h = hashes.get(image_rel) or digest(ROOT / image_rel)
            if h in image_hashes and image_hashes[h] != split:
                errors.append('Identical image across splits: ' + image_rel)
            image_hashes[h] = split
            with Image.open(ROOT / image_rel) as image:
                image.load()
                if image.size != (512,512):
                    errors.append('Unexpected image size: ' + image_rel)
            if r['category'] == 'annotated':
                mask_rel = split + '/' + r['mask_path']
                expected_files.add(mask_rel)
                with Image.open(ROOT / mask_rel) as mask:
                    a = np.asarray(mask.convert('RGB'))
                    if mask.size != (512,512):
                        errors.append('Unexpected mask size: ' + mask_rel)
                packed = (a[:,:,0].astype('uint32') << 16) | (a[:,:,1].astype('uint32') << 8) | a[:,:,2]
                unknown = set(np.unique(packed).tolist()) - allowed
                if unknown:
                    errors.append('Unexpected mask colors: ' + mask_rel)
                present = set()
                for name, color in COLORS.items():
                    code = (color[0] << 16) | (color[1] << 8) | color[2]
                    pixels = int(np.count_nonzero(packed == code))
                    if pixels:
                        present.add(name)
                        stats['class_images'][name] += 1
                    stats['class_pixels'][name] += pixels
                    if pixels != int(r[name + '_pixels']):
                        errors.append('Mask/manifest pixel count mismatch: {} {}'.format(mask_rel,name))
                if not present or present != set(filter(None, r['labels'].split(','))):
                    errors.append('Mask/manifest labels mismatch: ' + mask_rel)
            elif r['category'] != 'normal' or r['labels']:
                errors.append('Invalid normal/category labels: ' + r['sample'])
        except Exception as exc:
            errors.append('Invalid sample {}: {}'.format(r['sample'],exc))
        if number % 1000 == 0:
            print(split, number, '/', len(rows), flush=True)
    actual_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / split).rglob('*')
                    if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
    if expected_files != actual_files:
        errors.append('Image/mask file inventory differs from manifest: ' + split)
    report['splits'][split] = stats
    print(split, 'verified', flush=True)
reference = json.loads((ROOT / 'split_audit.json').read_text())
for split, stats in report['splits'].items():
    expected = reference['splits'][split]
    for field in ['images','parents','sources','class_images','class_pixels']:
        if stats[field] != expected[field]:
            errors.append('Split audit field mismatch: {} {}'.format(split,field))
    if stats['categories'] != dict(normal=expected['normal'], annotated=expected['annotated']):
        errors.append('Split category counts mismatch: ' + split)
report['manifest_sha256'] = {s: hashes[s + '_manifest.csv'] for s in ['train','val','test']}
report['distinct_images'] = len(image_hashes)
report['distinct_parents'] = len(parents)
report['passed'] = not errors
(OUT / 'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
raise SystemExit(bool(errors))
