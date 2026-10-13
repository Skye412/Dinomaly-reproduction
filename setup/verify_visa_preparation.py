import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path('/home/skye/data/Skye')
source = root / 'databases/VISA'
target = root / 'databases/VISA_pytorch/1cls'
errors = []
with (root / 'Dinomaly/reproduction/official_visa_1cls.csv').open() as f:
    rows = list(csv.DictReader(f))
expected = set()
def digest(p):
    return hashlib.sha256(p.read_bytes()).digest()
for r in rows:
    label = 'good' if r['label'] == 'normal' else 'bad'
    dst = target / r['object'] / r['split'] / label / Path(r['image']).name
    expected.add(dst)
    if not dst.is_file() or digest(source / r['image']) != digest(dst):
        errors.append(str(dst))
    if r['mask']:
        m = target / r['object'] / 'ground_truth' / label / Path(r['mask']).name
        expected.add(m)
        if not m.is_file():
            errors.append(str(m))
        else:
            with Image.open(source / r['mask']) as original, Image.open(m) as prepared:
                if not np.array_equal((np.array(original) != 0).astype('uint8') * 255, np.array(prepared)):
                    errors.append(str(m))
actual = {p for p in target.rglob('*') if p.is_file()}
errors.extend(str(p) for p in actual - expected)
report = {'passed': not errors, 'images_sha256_verified': len(rows), 'binary_masks_verified': sum(bool(r['mask']) for r in rows), 'errors': errors}
Path('reproduction/visa_preparation_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(bool(errors))
