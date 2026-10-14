"""Bridge2893 manifest loader: train exclusively on the existing normal split."""
import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from dataset import get_data_transforms


class BridgeDataset(Dataset):
    def __init__(self, root, split, normal_only=False, image_size=392):
        self.root = Path(root)
        self.split = split
        self.normal_only = normal_only
        with (self.root / (split + '_manifest.csv')).open(newline='') as f:
            rows = list(csv.DictReader(f))
        if any(r['assigned_split'] != split or r['category'] not in {'normal', 'annotated'} for r in rows):
            raise ValueError('Invalid manifest split/category')
        self.rows = [r for r in rows if r['category'] == 'normal'] if normal_only else rows
        self.transform, self.mask_transform = get_data_transforms(image_size, 392)
        for r in self.rows:
            if not (self.root / split / r['image_path']).is_file():
                raise FileNotFoundError(r['image_path'])
            if r['category'] == 'annotated' and not (self.root / split / r['mask_path']).is_file():
                raise FileNotFoundError(r['mask_path'])

    def __len__(self):
        return len(self.rows)

    def summary(self):
        return dict(split=self.split, samples=len(self),
                    categories=dict(Counter(r['category'] for r in self.rows)),
                    sources=dict(Counter(r['source'] for r in self.rows)),
                    parents=len({r['parent_id'] for r in self.rows}))

    def __getitem__(self, index):
        r = self.rows[index]
        path = self.root / self.split / r['image_path']
        with Image.open(path) as image:
            image = self.transform(image.convert('RGB'))
        label = int(r['category'] == 'annotated')
        if self.normal_only:
            if label:
                raise RuntimeError('Anomaly entered normal-only training')
            return image, label
        if label:
            with Image.open(self.root / self.split / r['mask_path']) as mask:
                mask = self.mask_transform(mask.convert('L'))
        else:
            mask = torch.zeros(1, 392, 392)
        return image, mask, label, str(path)


def check_split_protocol(root):
    groups = []
    summaries = {}
    for split in ['train', 'val', 'test']:
        data = BridgeDataset(root, split)
        groups.append({r['parent_id'] for r in data.rows})
        summaries[split] = data.summary()
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Parent groups overlap across splits')
    expected = {'train': {'normal': 1014, 'annotated': 1018},
                'val': {'normal': 144, 'annotated': 135},
                'test': {'normal': 289, 'annotated': 293}}
    if any(summaries[s]['categories'] != expected[s] for s in expected):
        raise RuntimeError('Dataset counts changed since audit')
    return summaries
