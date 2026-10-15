"""CPU preflight for DACL polygon semantics and the actual frozen datasets."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from bridge_dataset import check_split_protocol, dataset_class
from dacl_dataset import DaclDataset, damage_mask

torch.set_num_threads(1)
root = '/home/skye/data/Skye/databases/DACL10k_v3_512_split_seed42'
fixture = dict(imageWidth=16, imageHeight=16, shapes=[
    dict(label='Bearing', points=[[0,0],[5,0],[5,5],[0,5]]),
    dict(label='Crack', points=[[8,8],[12,8],[12,12],[8,12]]),
    dict(label='WConccor', points=[[10,10],[14,10],[14,14],[10,14]])])
mask = np.asarray(damage_mask(fixture))
assert mask[2,2] == 0 and mask[9,9] == 255 and mask[13,13] == 255
assert mask[11,11] == 255 and mask[7,7] == 0
assert dataset_class('dacl_v3_seed42') is DaclDataset
protocol = check_split_protocol(root, 'dacl_v3_seed42')
bridge = check_split_protocol('/home/skye/data/Skye/databases/Bridge512_balance', 'bridge512_balance_repaired')
train = DaclDataset(root, 'train', normal_only=True)
assert len(train) == 413 and all(row['anomaly'] == '0' for row in train.rows)
empty = {}
for split in ['val', 'test']:
    data = DaclDataset(root, split)
    count = 0
    for index in range(len(data)):
        image, mask, label, path = data[index]
        assert image.shape == (3,392,392) and mask.shape == (1,392,392)
        assert torch.isfinite(image).all() and torch.isfinite(mask).all()
        metric_mask = torch.nn.functional.interpolate(mask[None], size=256, mode='nearest').bool()
        if label and not metric_mask.any():
            count += 1
        if not label:
            assert not mask.any()
    empty[split] = count
    print(split, 'verified', len(data), 'empty positive masks', count, flush=True)
result = dict(passed=True, device='cpu', dacl=protocol, bridge_protocol_unchanged=bridge,
              normal_training_images=len(train), empty_positive_metric_masks=empty,
              polygon_fixture='damage union includes WConccor, excludes object class and retains overlap',
              gpu_training_smoke='not run by this CPU check')
out = Path('/home/skye/data/Skye/Dinomaly/runs/dacl-v3-seed42-cpu-preflight-20261015.json')
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
