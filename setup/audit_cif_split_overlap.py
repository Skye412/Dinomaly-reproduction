"""Verify whether repeated full-resolution filenames across official splits share image bytes."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/home/skye/data/Skye/databases/cif/data')
out = Path('/home/skye/data/Skye/Dinomaly/reproduction/audits/bridge-datasets-20261014')
locations = defaultdict(list)
for path in sorted(root.glob('*_full-*.parquet')):
    split = path.name.split('-')[0]
    for index, name in enumerate(pq.read_table(path, columns=['file_name']).column('file_name').to_pylist()):
        locations[name].append({'split': split, 'shard': path.name, 'row': index})
overlaps = {name: records for name, records in locations.items() if len({r['split'] for r in records}) > 1}
by_shard = defaultdict(dict)
for name, records in overlaps.items():
    for record in records:
        by_shard[record['shard']][record['row']] = (name, record)
for number, (shard, selected) in enumerate(sorted(by_shard.items()), 1):
    table = pq.read_table(root / shard, columns=['image', 'objects'])
    for index, (name, record) in selected.items():
        image = table.column('image')[index].as_py()
        raw = image['bytes']
        record['image_sha256'] = hashlib.sha256(raw).hexdigest()
        record['image_bytes'] = len(raw)
        objects = table.column('objects')[index].as_py()
        record['annotation_sha256'] = hashlib.sha256(json.dumps(objects, sort_keys=True).encode()).hexdigest()
    if number % 20 == 0:
        print('Duplicate image content checked in {}/{} shards'.format(number, len(by_shard)), flush=True)
identical = {name: records for name, records in overlaps.items() if len({r['image_sha256'] for r in records}) == 1}
report = {'cross_split_repeated_filenames': len(overlaps),
          'cross_split_identical_image_filename_groups': len(identical),
          'overlap_records': overlaps,
          'interpretation': 'Same image bytes across official splits are data leakage candidates even though every shard matches the official repository.'}
(out / 'cif_split_overlap.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'overlap_records'}, indent=2), flush=True)
