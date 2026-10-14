"""Bridge manifest loader: train exclusively on the existing normal split."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from dataset import get_data_transforms


class BridgeDataset(Dataset):
    def __init__(self, root, split, normal_only=False, image_size=392, exclude_samples=()):
        self.root = Path(root)
        self.split = split
        self.normal_only = normal_only
        with (self.root / (split + '_manifest.csv')).open(newline='') as f:
            rows = list(csv.DictReader(f))
        if any(r['assigned_split'] != split or r['category'] not in {'normal', 'annotated'} for r in rows):
            raise ValueError('Invalid manifest split/category')
        self.original_categories = dict(Counter(r['category'] for r in rows))
        exclude_samples = set(exclude_samples)
        if exclude_samples - {r['sample'] for r in rows}:
            raise ValueError('Excluded sample is not in the manifest')
        rows = [r for r in rows if r['sample'] not in exclude_samples]
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
                # Collapse RGB class colors before resizing, so all classes have equal support.
                foreground = np.any(np.asarray(mask.convert('RGB')) != 0, axis=2)
                mask = self.mask_transform(Image.fromarray(foreground.astype('uint8') * 255))
        else:
            mask = torch.zeros(1, 392, 392)
        return image, mask, label, str(path)


def load_profile(profile, split_view='original'):
    profile_file = Path(__file__).resolve().parent / 'reproduction/dataset_profiles' / (profile + '.json')
    metadata = json.loads(profile_file.read_text())
    if split_view == 'original':
        return metadata, {s: [] for s in ['train','val','test']}, metadata['categories']
    view = metadata['views'][split_view]
    return metadata, view['exclude_samples'], view['categories']


def check_split_protocol(root, profile='bridge2893', split_view='original'):
    expected_profile, exclusions, expected = load_profile(profile, split_view)
    groups = []
    summaries = {}
    for split in ['train', 'val', 'test']:
        data = BridgeDataset(root, split, exclude_samples=exclusions[split])
        if data.original_categories != expected_profile['categories'][split]:
            raise RuntimeError('Original dataset counts changed since audit')
        groups.append({r['parent_id'] for r in data.rows})
        summaries[split] = data.summary()
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Parent groups overlap across splits')
    if any(summaries[s]['categories'] != expected[s] for s in expected):
        raise RuntimeError('Dataset counts changed since audit')
    for split, expected_hash in expected_profile['manifest_sha256'].items():
        actual_hash = hashlib.sha256((Path(root) / (split + '_manifest.csv')).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError('Dataset manifest changed since audit: ' + split)
    return summaries
