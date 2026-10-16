"""FoundAD-style training adaptation: raw-RGB same-image CutPaste, clean targets."""
import json
import math
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


def raw_transform():
    return transforms.Compose([transforms.Resize((392,392)), transforms.ToTensor()])


class IndexedTrainingDataset(torch.utils.data.Dataset):
    """Preserve sampler order while carrying original manifest identities."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


class PairedInput:
    def __init__(self, folder, logger):
        self.folder, self.logger = folder, logger
        self.jitter = transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5)
        self.normalize = transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        self.stats = dict(clean_batches=0, perturbed_batches=0, attempted_images=0,
                          effective_images=0, unchanged_images=0, normal_images=0, scar_images=0,
                          changed_pixels=0, extra_encoder_batches=0, extra_encoder_seconds=0., range_corrected_batches=0,
                          range_corrected_elements=0, first_range_event=None)

    def __call__(self, batch, step, samples):
        assert batch.ndim == 4 and batch.shape[1:] == (3,392,392)
        assert torch.isfinite(batch).all() and batch.min() >= 0 and batch.max() <= 1
        # Clone targets before synthesis and never write to batch or clean.
        clean = self.normalize(batch.clone())
        if random.random() < .5:
            self.stats['clean_batches'] += 1
            return clean, clean, False
        self.stats['perturbed_batches'] += 1
        augmented = batch.clone()
        synthesis = []
        for i, image in enumerate(batch):
            _, h, w = image.shape
            scar = random.random() >= .5
            if scar:
                cw, ch = int(random.uniform(2,16)), int(random.uniform(10,25))
                self.stats['scar_images'] += 1
            else:
                area = random.uniform(.02,.25) * h * w
                aspect = random.uniform(.3, 1/.3)
                cw, ch = round(math.sqrt(area*aspect)), round(math.sqrt(area/aspect))
                self.stats['normal_images'] += 1
            assert 0 < cw <= w and 0 < ch <= h
            fx, fy = random.randint(0,w-cw), random.randint(0,h-ch)
            patch = self.jitter(image[:,fy:fy+ch,fx:fx+cw].clone())
            if scar:
                patch = TF.rotate(patch, random.uniform(-45,45),
                                  interpolation=TF.InterpolationMode.BILINEAR, expand=True)
            ph, pw = patch.shape[-2:]
            tx, ty = random.randint(0,w-pw), random.randint(0,h-ph)
            synthesis.append(dict(sample=samples[i], type='Scar' if scar else 'Normal',
                                  donor_xy=[fx, fy], donor_wh=[cw, ch], paste_xy=[tx, ty],
                                  patch_hw=[ph, pw]))
            augmented[i,:,ty:ty+ph,tx:tx+pw] = patch
            changed = int((augmented[i] != image).any(0).sum())
            self.stats['attempted_images'] += 1
            self.stats['effective_images'] += int(changed > 0)
            self.stats['unchanged_images'] += int(changed == 0)
            self.stats['changed_pixels'] += changed
        assert torch.equal(clean, self.normalize(batch.clone())), 'Clean target modified during synthesis'
        # Floating RGB jitter/rotation may overshoot an endpoint by roundoff.
        # Reject material violations before restoring the required RGB range.
        low, high = float(augmented.min()), float(augmented.max())
        finite = bool(torch.isfinite(augmented).all())
        outside = (augmented < 0) | (augmented > 1)
        count = int(outside.sum())
        severe = not finite or low < -1e-6 or high > 1 + 1e-6
        if severe or count:
            event = dict(step=step, minimum=low, maximum=high, dtype=str(augmented.dtype),
                         outside_elements=count, finite=finite, severe=severe,
                         corrected_batches=self.stats['range_corrected_batches'] + int(not severe))
            if severe:
                bad = (outside | ~torch.isfinite(augmented)).flatten(1).any(1)
                event['samples'] = [meta for meta, invalid in zip(synthesis, bad.tolist()) if invalid]
                # Preserve the failing batch and synthesis context; excluded from Git.
                torch.save(dict(raw=batch, augmented=augmented, synthesis=synthesis, event=event),
                           self.folder / 'range_failure.pt')
            if severe or self.stats['first_range_event'] is None:
                with (self.folder / 'range_events.jsonl').open('a') as handle:
                    handle.write(json.dumps(event) + '\n')
                self.logger.info('Synthesized RGB range event: %s', event)
            if severe:
                self.save(self.folder / 'paired_stats.json')
                raise RuntimeError('Synthesized RGB range failure: ' + json.dumps(event))
            if self.stats['first_range_event'] is None:
                self.stats['first_range_event'] = event
            self.stats['range_corrected_batches'] += 1
            self.stats['range_corrected_elements'] += count
        augmented.clamp_(0, 1)
        return clean, self.normalize(augmented), True

    def save(self, path):
        path.write_text(json.dumps(self.stats, indent=2))
