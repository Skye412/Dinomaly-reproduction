"""Derive optional non-destructive cross-split exclusions from verified checksum manifest."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path('/home/skye/data/Skye/databases/Bridge512_balance')
out = Path('/home/skye/data/Skye/Dinomaly/reproduction/audits/bridge512-balance-20261014')
hashes = {}
for line in (root/'checksums.sha256').read_text().splitlines():
    value, path = line.split(None,1)
    hashes[path.strip().lstrip('*')] = value
groups = defaultdict(list)
rows_by_split = {}
for split in ['train','val','test']:
    with (root/(split+'_manifest.csv')).open(newline='') as f:
        rows_by_split[split] = list(csv.DictReader(f))
    for row in rows_by_split[split]:
        groups[hashes[split+'/'+row['image_path']]].append(dict(split=split,sample=row['sample'],
                                                             image_path=row['image_path'],category=row['category']))
overlap = {h: records for h,records in groups.items() if len({r['split'] for r in records})>1}
excluded = {s: [] for s in ['train','val','test']}
already_kept = set()
counts = {}
for split in ['test','val','train']:
    kept_hashes = set()
    kept_rows = []
    for row in rows_by_split[split]:
        h = hashes[split+'/'+row['image_path']]
        if h in already_kept:
            excluded[split].append(row['sample'])
        else:
            kept_rows.append(row)
            kept_hashes.add(h)
    already_kept.update(kept_hashes)
    counts[split] = dict(Counter(r['category'] for r in kept_rows))
report = dict(groups=overlap, cross_split_hash_groups=len(overlap),
              duplicate_categories=dict(Counter(r['category'] for records in overlap.values() for r in records)),
              exclusion_policy='test unchanged; exclude val overlap with test; exclude train overlap with val or test; retain within-split duplicates',
              exclude_samples=excluded, view_categories=counts)
(out/'duplicate_groups.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['groups','exclude_samples']},indent=2))
print('Excluded counts', {k:len(v) for k,v in excluded.items()})
