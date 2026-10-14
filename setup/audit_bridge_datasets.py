"""Read-only audits against frozen official metadata; original datasets are untouched."""
import csv
import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path('/home/skye/data/Skye/Dinomaly')
ROOT = Path('/home/skye/data/Skye/databases')
OUT = REPO / 'reproduction/audits/bridge-datasets-20261014'
OFFICIAL = OUT / 'official'
OUT.mkdir(parents=True, exist_ok=True)


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, obj):
    (OUT / (name + '.json')).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    print(name, json.dumps(obj, ensure_ascii=True)[:1800], flush=True)


def audit_dacl():
    root = ROOT / 'dacl10k_v3_512'
    metadata = json.loads((OFFICIAL / 'dacl_XL7ZRG_metadata.json').read_text())
    files = {f['dataFile']['filename']: f['dataFile'] for f in metadata['data']['latestVersion']['files']}
    report = {'official_dataset': 'https://doi.org/10.60776/XL7ZRG', 'archives': {}, 'errors': []}
    errors = report['errors']
    for name in ['dacl10k_v3_images_512.zip', 'dacl10k_v3_devphase_annotations_512.zip']:
        p = root / name
        expected = files[name]
        actual = digest(p, 'md5')
        info = {'size': p.stat().st_size, 'expected_size': expected['filesize'],
                'md5': actual, 'official_md5': expected['md5'], 'matches_official': actual == expected['md5']}
        if not info['matches_official'] or info['size'] != info['expected_size']:
            errors.append('Official archive mismatch: ' + name)
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            info['zip_crc_error'] = bad
            checked = 0
            missing, modified = [], []
            for member in z.infolist():
                if member.is_dir():
                    continue
                target = root / 'extracted' / member.filename
                if not target.is_file():
                    missing.append(member.filename)
                elif digest(target) != hashlib.sha256(z.read(member)).hexdigest():
                    modified.append(member.filename)
                checked += 1
            info.update(extracted_files_checked=checked, missing=missing, modified=modified)
            if bad or missing or modified:
                errors.append('Extraction mismatch: ' + name)
        report['archives'][name] = info
    damage_labels = set('Crack ACrack Wetspot Efflorescence Rust Rockpocket Hollowareas Cavity Spalling Graffiti Weathering Restformwork ExposedRebars'.split())
    object_labels = set('Bearing EJoint Drainage PEquipment JTape WConccor'.split())
    report['image_splits'] = {}
    image_index = {}
    for split in ['train', 'valid', 'testdev', 'test']:
        paths = sorted((root / 'extracted/images' / split).glob('*.jpg'))
        report['image_splits'][split] = len(paths)
        for p in paths:
            image_index[split, p.name] = p
            try:
                with Image.open(p) as im:
                    if im.size != (512, 512):
                        errors.append('Unexpected resolution: ' + str(p))
                    im.load()
            except Exception as exc:
                errors.append('Unreadable image: {}: {}'.format(p, exc))
    report['annotation_splits'] = {}
    for split in ['train', 'validation']:
        paths = sorted((root / 'extracted/dacl10k_v3_devphase_annotations_512' / split).glob('*.json'))
        stats = dict(files=len(paths), no_annotated_damage=0, no_selected_four_defects=0, empty_shapes=0)
        labels = Counter()
        for p in paths:
            obj = json.loads(p.read_text())
            image_split = 'valid' if split == 'validation' else split
            if (image_split, obj['imageName']) not in image_index:
                errors.append('Annotation image missing: ' + str(p))
            if (obj['imageWidth'], obj['imageHeight']) != (512, 512):
                errors.append('Annotation resolution mismatch: ' + str(p))
            if str(obj.get('version')) != '3':
                errors.append('Annotation version mismatch: ' + str(p))
            present = {s['label'] for s in obj['shapes']}
            labels.update(present)
            stats['empty_shapes'] += not bool(obj['shapes'])
            stats['no_annotated_damage'] += not bool(present & damage_labels)
            stats['no_selected_four_defects'] += not bool(present & {'Crack', 'Spalling', 'Rust', 'Efflorescence'})
            for shape in obj['shapes']:
                points = shape['points']
                if len(points) < 3 or any(len(v) != 2 or any(not math.isfinite(float(x)) for x in v) for v in points):
                    errors.append('Invalid polygon: ' + str(p))
            unknown = present - damage_labels - object_labels
            if unknown:
                errors.append('Unknown annotation labels: {} {}'.format(p, sorted(unknown)))
        stats['images_per_label'] = dict(labels)
        report['annotation_splits'][split] = stats
    report['public_test_annotations_present'] = False
    report['passed'] = not errors
    save('dacl_audit', report)


def audit_bridge():
    root = ROOT / 'Bridge2893_split_seed42'
    report = {'path': str(root), 'errors': [], 'splits': {}, 'checksums': {}}
    errors = report['errors']
    for line in (root / 'checksums.sha256').read_text().splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.strip().lstrip('*')
        matched = digest(root / rel) == expected
        report['checksums'][rel] = matched
        if not matched:
            errors.append('Metadata checksum mismatch: ' + rel)
    parents, image_hashes, samples = {}, {}, {}
    unique_mask_values = Counter()
    for split in ['train', 'val', 'test']:
        with (root / (split + '_manifest.csv')).open(newline='') as f:
            rows = list(csv.DictReader(f))
        stats = dict(samples=len(rows), categories=dict(Counter(r['category'] for r in rows)),
                     sources=dict(Counter(r['source'] for r in rows)), parents=len({r['parent_id'] for r in rows}))
        for r in rows:
            p = root / split / r['image_path']
            if r['assigned_split'] != split:
                errors.append('Manifest assigned split mismatch: ' + r['sample'])
            for mapping, key, label in [(parents, r['parent_id'], 'parent'), (samples, r['sample'], 'sample')]:
                if key in mapping and mapping[key] != split:
                    errors.append('Cross-split {} leakage: {}'.format(label, key))
                mapping[key] = split
            try:
                h = digest(p)
                if h in image_hashes and image_hashes[h] != split:
                    errors.append('Identical image across splits: ' + str(p))
                image_hashes[h] = split
                with Image.open(p) as image:
                    image.load()
                    size = image.size
                if r['mask_path']:
                    m = root / split / r['mask_path']
                    with Image.open(m) as mask:
                        a = np.asarray(mask)
                        if mask.size != size:
                            errors.append('Mask dimensions differ: ' + str(m))
                        unique_mask_values.update(np.unique(a).tolist())
                        if r['category'] == 'normal' and a.any():
                            errors.append('Normal image has nonempty mask: ' + str(m))
                        if r['category'] == 'annotated' and not a.any():
                            errors.append('Annotated image has empty mask: ' + str(m))
            except Exception as exc:
                errors.append('Invalid file {}: {}'.format(p, exc))
        report['splits'][split] = stats
    report['mask_values_file_counts'] = dict(unique_mask_values)
    report['distinct_image_hashes'] = len(image_hashes)
    report['distinct_parents'] = len(parents)
    report['passed'] = not errors
    save('bridge2893_audit', report)


def audit_cif():
    import pyarrow.parquet as pq
    root = ROOT / 'cif'
    official = json.loads((OFFICIAL / 'cif_repo_info.json').read_text())
    expected = {f['rfilename']: f for f in official['siblings'] if f['rfilename'].endswith('.parquet') and '_full-' in f['rfilename']}
    actual = {p.relative_to(root).as_posix() for p in (root / 'data').glob('*.parquet')}
    report = {'official_repository': 'https://huggingface.co/datasets/ibm-research/cif-dataset',
              'official_revision': official['sha'], 'scope': 'full-resolution splits only',
              'missing': sorted(set(expected) - actual), 'extra': sorted(actual - set(expected)),
              'files': [], 'splits': {}, 'errors': [], 'id_overlap_notes': []}
    stats = defaultdict(lambda: {'shards': 0, 'rows': 0, 'empty_objects': 0, 'instances_per_category': Counter()})
    names, ids = {}, {}
    start = time.monotonic()
    for number, (rel, info) in enumerate(sorted(expected.items()), 1):
        path = root / rel
        if not path.is_file():
            continue
        h = digest(path)
        matched = h == info['lfs']['sha256'] and path.stat().st_size == info['size']
        record = dict(path=rel, sha256=h, expected_sha256=info['lfs']['sha256'],
                      bytes=path.stat().st_size, matches_official=matched)
        if not matched:
            report['errors'].append('Official file mismatch: ' + rel)
        split = path.name.split('-')[0]
        try:
            f = pq.ParquetFile(path)
            record['rows'] = f.metadata.num_rows
            stats[split]['shards'] += 1
            stats[split]['rows'] += f.metadata.num_rows
            table = f.read(columns=['image_id', 'file_name', 'width', 'height', 'objects', 'tile_row'])
            for row in table.to_pylist():
                for mapping, value, label in [(names, row['file_name'], 'filename'), (ids, row['image_id'], 'image_id')]:
                    if value in mapping and mapping[value] != split:
                        if label == 'image_id':
                            report['id_overlap_notes'].append('Cross-split image_id reuse: {}'.format(value))
                        else:
                            report['errors'].append('Cross-split filename overlap: {}'.format(value))
                    mapping[value] = split
                if row['tile_row'] is not None or min(row['width'], row['height']) <= 0:
                    report['errors'].append('Unexpected full image metadata: ' + str(row['file_name']))
                objects = row['objects']
                categories = objects['category_id']
                stats[split]['empty_objects'] += not bool(categories)
                stats[split]['instances_per_category'].update(categories)
                if any(c not in range(1, 7) for c in categories):
                    report['errors'].append('Unknown CiF category: ' + str(row['file_name']))
                if any(len(objects[k]) != len(categories) for k in ['id','bbox','area','iscrowd','segmentation']):
                    report['errors'].append('Inconsistent annotation lengths: ' + str(row['file_name']))
        except Exception as exc:
            report['errors'].append('Parquet read error: {} {}'.format(rel, exc))
        report['files'].append(record)
        if number % 20 == 0 or number == len(expected):
            print('CiF SHA256 checked {}/{}; elapsed {:.0f}s'.format(number, len(expected), time.monotonic() - start), flush=True)
            report['splits'] = dict(stats)
            save_progress = dict(checked=number, total=len(expected), errors=report['errors'][-5:])
            (OUT / 'cif_progress.json').write_text(json.dumps(save_progress))
    report['splits'] = dict(stats)
    report['official_files_match'] = len(report['files']) == len(expected) and all(f['matches_official'] for f in report['files']) and not report['missing'] and not report['extra']
    report['image_id_scope_note'] = 'IDs are inspected separately from filenames; reused numeric IDs across splits do not prove image duplication.'
    report['passed'] = not (report['errors'] or report['missing'] or report['extra'])
    save('cif_audit', report)


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if task in ['all', 'dacl']:
        audit_dacl()
    if task in ['all', 'bridge']:
        audit_bridge()
    if task in ['all', 'cif']:
        audit_cif()
