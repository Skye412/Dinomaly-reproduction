"""Independently audit the DACL v3 seed-42 split without changing source data."""
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path('/home/skye/data/Skye/databases/DACL10k_v3_512_split_seed42')
OUT = Path('/home/skye/data/Skye/Dinomaly/reproduction/audits/dacl-split-seed42-20261015')
OUT.mkdir(parents=True, exist_ok=True)
protocol = json.loads((ROOT/'split_protocol.json').read_text())
reference = json.loads((ROOT/'split_audit.json').read_text())
source = Path(protocol['source']['root'])
errors = []
hashes = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


for line in (ROOT/'checksums.sha256').read_text().splitlines():
    expected, rel = line.split(None,1)
    rel = rel.strip().lstrip('*')
    path = ROOT/rel
    if not path.is_file():
        errors.append('Missing checksum file: '+rel)
        continue
    hashes[rel] = sha(path)
    if hashes[rel] != expected:
        errors.append('Checksum mismatch: '+rel)
print('Checksum files verified:',len(hashes),flush=True)
inventory = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='checksums.sha256'}
if inventory != set(hashes):
    errors.append('Checksum inventory differs from actual files')
with (ROOT/'parent_mapping.csv').open(newline='') as f:
    mapping_rows = list(csv.DictReader(f))
mapping = {r['image_path']:r for r in mapping_rows}
if len(mapping)!=len(mapping_rows):
    errors.append('Duplicate image paths in parent_mapping')
anomaly_labels = set(protocol['anomaly_policy']['anomaly_1_if_any_of_these_14_labels'])
known_labels = set(protocol['anomaly_policy']['all_source_labels'])
report = dict(root=str(ROOT),source_root=str(source),splits={},anomaly_labels=sorted(anomaly_labels),errors=errors)
groups = {'group_id':defaultdict(list),'image_sha256':defaultdict(list),'rgb_pixel_sha256':defaultdict(list)}
seen_paths = set()
paired_payload = set()
for split in ['train','val','test']:
    with (ROOT/(split+'_manifest.csv')).open(newline='') as f:
        rows = list(csv.DictReader(f))
    stats = dict(samples=len(rows),normal_candidates=0,anomaly_positives=0,source_splits=Counter(),class_presence_marginals=Counter())
    for number,row in enumerate(rows,1):
        image_rel, ann_rel = row['image_path'],row['annotation_path']
        paired_payload.update([image_rel,ann_rel])
        if image_rel in seen_paths:
            errors.append('Duplicate manifest image path: '+image_rel)
        seen_paths.add(image_rel)
        if row['assigned_split']!=split or not image_rel.startswith(split+'/') or not ann_rel.startswith(split+'/'):
            errors.append('Manifest split mismatch: '+image_rel)
        if hashes.get(image_rel)!=row['image_sha256'] or hashes.get(ann_rel)!=row['annotation_sha256']:
            errors.append('Manifest file hash mismatch: '+image_rel)
        m = mapping[image_rel]
        if any(m[k]!=v for k,v in row.items()):
            errors.append('parent_mapping/manifest mismatch: '+image_rel)
        if sha(source/m['source_image_path'])!=row['image_sha256'] or sha(source/m['source_annotation_path'])!=row['annotation_sha256']:
            errors.append('Source copy mismatch: '+image_rel)
        annotation = json.loads((ROOT/ann_rel).read_text())
        if annotation['imageName']!=Path(image_rel).name or str(annotation.get('version'))!='3':
            errors.append('Annotation pairing/version mismatch: '+ann_rel)
        with Image.open(ROOT/image_rel) as image:
            image = image.convert('RGB')
            if image.size!=(512,512) or (annotation['imageWidth'],annotation['imageHeight'])!=image.size:
                errors.append('Dimensions mismatch: '+image_rel)
            prefix='RGB:{}x{}\0'.format(*image.size).encode('ascii')
            rgb_hash=hashlib.sha256(prefix+image.tobytes()).hexdigest()
        if rgb_hash!=m['rgb_pixel_sha256']:
            errors.append('Decoded RGB hash mismatch: '+image_rel)
        labels={shape['label'] for shape in annotation['shapes']}
        if labels!=set(filter(None,row['labels'].split(';'))) or labels-known_labels:
            errors.append('Annotation labels mismatch: '+ann_rel)
        anomaly=int(bool(labels&anomaly_labels))
        if int(row['anomaly'])!=anomaly or int(row['normal_candidate'])!=1-anomaly:
            errors.append('Anomaly policy mismatch: '+ann_rel)
        for shape in annotation['shapes']:
            points=shape['points']
            if len(points)<3 or any(len(p)!=2 or any(not math.isfinite(float(v)) for v in p) for p in points):
                errors.append('Invalid polygon: '+ann_rel)
        stats['class_presence_marginals'].update(labels)
        stats['anomaly_positives']+=anomaly
        stats['normal_candidates']+=1-anomaly
        stats['source_splits'][row['source_split']]+=1
        for key,value in [('group_id',row['group_id']),('image_sha256',row['image_sha256']),('rgb_pixel_sha256',rgb_hash)]:
            groups[key][value].append(dict(split=split,image_path=image_rel,annotation_path=ann_rel,anomaly=anomaly))
        if number%1000==0:
            print(split,number,'/',len(rows),flush=True)
    for key in ['samples','normal_candidates','anomaly_positives','class_presence_marginals']:
        if stats[key]!=reference['split_statistics'][split][key]:
            errors.append('Split statistics mismatch: {} {}'.format(split,key))
    if split=='test' and set(stats['source_splits'])!={'validation'}:
        errors.append('Test set is not exclusively official validation')
    if split in ['train','val'] and set(stats['source_splits'])!={'officialtrain'}:
        errors.append('Train/val contains non-officialtrain source')
    report['splits'][split]=stats
payload_files = {p.relative_to(ROOT).as_posix() for s in ['train','val','test'] for p in (ROOT/s).rglob('*') if p.is_file()}
if paired_payload != payload_files:
    errors.append('Payload inventory differs from manifest')
report['cross_split_overlap_groups']={key:sum(len({r['split'] for r in records})>1 for records in values.values()) for key,values in groups.items()}
if any(report['cross_split_overlap_groups'].values()):
    errors.append('Cross-split content/group leakage')
report['within_split_image_duplicate_groups']={h:records for h,records in groups['image_sha256'].items() if len(records)>1}
report['within_split_rgb_duplicate_groups']={h:records for h,records in groups['rgb_pixel_sha256'].items() if len(records)>1}
report['checksum_files_verified']=len(hashes)
report['source_pairs_verified']=len(seen_paths)
report['manifest_sha256']={s:hashes[s+'_manifest.csv'] for s in ['train','val','test']}
report['source_donors']={d['filename']:sha(source/d['filename'])==d['sha256'] for d in protocol['source']['donors']}
if not all(report['source_donors'].values()):
    errors.append('Source archive fingerprint mismatch')
report['passed']=not errors
report['limitations']=['Normal candidates are annotation-defined, not certified healthy.',
                       'Content groups do not certify bridge/session independence or absence of near duplicates.',
                       'Mask rasterization is needed before pixel evaluation; files contain raw polygons.']
(OUT/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
raise SystemExit(bool(errors))
