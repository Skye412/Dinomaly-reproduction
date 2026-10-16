"""DACL v3 manifest loader with native-resolution union of 14 damage polygons."""
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from dataset import get_data_transforms

ANOMALY_LABELS = {'ACrack','Cavity','Crack','Efflorescence','ExposedRebars','Graffiti',
                  'Hollowareas','Restformwork','Rockpocket','Rust','Spalling','WConccor','Weathering','Wetspot'}


def damage_mask(annotation):
    mask = Image.new('L', (annotation['imageWidth'], annotation['imageHeight']), 0)
    draw = ImageDraw.Draw(mask)
    for shape in annotation['shapes']:
        if shape['label'] in ANOMALY_LABELS:
            draw.polygon([tuple(point) for point in shape['points']], fill=255)
    return mask


class DaclDataset(Dataset):
    def __init__(self, root, split, normal_only=False, image_size=392, exclude_samples=(), mask_size=None):
        self.root, self.split, self.normal_only = Path(root), split, normal_only
        self.transform, self.mask_transform = get_data_transforms(image_size, image_size)
        if mask_size is not None:
            _, self.mask_transform = get_data_transforms(mask_size, mask_size)
        self.mask_size = mask_size or image_size
        with (self.root/(split+'_manifest.csv')).open(newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row['assigned_split'] != split or row['anomaly'] not in {'0','1'}:
                raise ValueError('Invalid DACL manifest split/label')
            labels = set(filter(None, row['labels'].split(';')))
            if int(row['anomaly']) != int(bool(labels & ANOMALY_LABELS)):
                raise ValueError('DACL anomaly policy mismatch')
            row.update(category='annotated' if int(row['anomaly']) else 'normal',
                       parent_id=row['group_id'], source=row['source_split'])
            if not (self.root/row['image_path']).is_file() or not (self.root/row['annotation_path']).is_file():
                raise FileNotFoundError(row['image_path'])
        self.original_categories = dict(Counter(r['category'] for r in rows))
        excluded = set(exclude_samples)
        if excluded - {r['sample'] for r in rows}:
            raise ValueError('Unknown excluded DACL sample')
        self.rows = [r for r in rows if r['sample'] not in excluded and (not normal_only or r['category']=='normal')]

    def __len__(self):
        return len(self.rows)

    def summary(self):
        return dict(split=self.split, samples=len(self), categories=dict(Counter(r['category'] for r in self.rows)),
                    sources=dict(Counter(r['source'] for r in self.rows)),
                    content_groups=len({r['group_id'] for r in self.rows}),
                    group_unit='exact JPEG/RGB content, not bridge/session')

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root/row['image_path']
        with Image.open(path) as im:
            image = self.transform(im.convert('RGB'))
        label = int(row['anomaly'])
        if self.normal_only:
            if label:
                raise RuntimeError('Anomaly entered DACL normal-only training')
            return image, label
        if label:
            annotation = json.loads((self.root/row['annotation_path']).read_text())
            if annotation['imageName'] != path.name:
                raise ValueError('DACL imageName pairing mismatch')
            present = {shape['label'] for shape in annotation['shapes']}
            if present != set(filter(None, row['labels'].split(';'))):
                raise ValueError('DACL polygon labels changed since manifest creation')
            mask = self.mask_transform(damage_mask(annotation))
        else:
            mask = torch.zeros(1,self.mask_size,self.mask_size)
        return image, mask, label, str(path)
