"""Frozen V0 feature memory. No training, approximate index, or test fitting."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_crmc_features(model, images):
    tokens = model.encode_tokens(images)
    prefix = 1 + model.encoder.num_register_tokens
    n = tokens[-1].shape[1] - prefix
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError('Non-square patch grid')
    groups = [model.fuse_feature([tokens[i] for i in ids])[:, prefix:].contiguous()
              for ids in model.fuse_layer_encoder]
    return tokens[-1][:, 0].float(), [g.float() for g in groups], side


def forward_with_context(model, images):
    """Use the original V0 forward verbatim; capture same-forward layer-9 CLS."""
    from utils import cal_anomaly_maps
    captured = []
    def capture(module, inputs, output):
        captured.append(output[:, 0].detach().float())
    handle = model.encoder.blocks[model.target_layers[-1]].register_forward_hook(capture)
    try:
        en, de = model(images)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected one final-target-layer CLS')
    rec, _ = cal_anomaly_maps(en, de, images.shape[-1])
    return captured[0], [g.flatten(2).transpose(1, 2).contiguous() for g in en], rec


def build_memory_bank(model, loader, device, output_dir, provenance):
    """Store raw targets fp16; normalize only after loading for cosine search."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    ds = loader.dataset
    if not ds.normal_only or any(r['category'] != 'normal' for r in ds.rows):
        raise ValueError('Memory must contain normal training images only')
    if ds.split != 'train' or loader.drop_last:
        raise ValueError('Memory must cover all normal training images')
    count, grid = len(ds), None
    cls_bank, groups = None, None
    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            cls, patches, side = extract_crmc_features(model, batch[0].to(device))
            if cls_bank is None:
                grid = side
                cls_bank = torch.empty(count, cls.shape[-1], dtype=torch.float16)
                groups = [torch.empty(count, p.shape[1], p.shape[2], dtype=torch.float16)
                          for p in patches]
            if side != grid or len(groups) != 2:
                raise ValueError('Expected two fixed V0 fusion groups')
            stop = cursor + len(cls)
            cls_bank[cursor:stop] = cls.cpu().half()
            for bank, p in zip(groups, patches):
                bank[cursor:stop] = p.cpu().half()
            cursor = stop
    if cursor != count or cursor == 0:
        raise ValueError('Incomplete memory bank')
    files = [('cls.fp16.pt', cls_bank)] + [
        ('patch_group%d.fp16.pt' % i, p) for i, p in enumerate(groups)]
    hashes = {}
    for name, data in files:
        torch.save(data, output_dir / name)
        hashes[name] = sha256_file(output_dir / name)
    metadata = dict(protocol='crmc_bank_v1', provenance=provenance, samples=count,
                    patch_grid=[grid, grid], channels=cls_bank.shape[-1],
                    dtype='float16', storage='raw uncentered CLS and fused patch targets',
                    target_layer=model.target_layers[-1], groups=model.fuse_layer_encoder,
                    normal_only=True, sample_ids=[r['sample'] for r in ds.rows],
                    file_sha256=hashes)
    (output_dir / 'bank.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata, cls_bank, groups


def nearest_memory_maps(query_cls, query_groups, cls_bank, patch_banks, ks=(4, 16, 64),
                        include_global=True, candidate_chunk=4096, row_chunk=128):
    """Exact FP32 cosine search; global bank scanned once, K contexts reused.

    CPU fp16 bank, bounded GPU reference chunks. Tied CLS similarities retain
    bank order. K is not silently reduced if insufficient normal images exist.
    """
    batch, n, channels = query_groups[0].shape
    side = math.isqrt(n)
    if side * side != n or len(query_groups) != len(patch_banks):
        raise ValueError('Invalid patch grid/groups')
    if not ks or min(ks) <= 0 or max(ks) > len(cls_bank):
        raise ValueError('K outside bank size')
    device = query_cls.device
    descriptors = F.normalize(cls_bank.to(device).float(), dim=-1)
    similarities = F.normalize(query_cls.float(), dim=-1) @ descriptors.T
    # NumPy stable sorting preserves bank order on ties on older PyTorch too.
    ranking = torch.from_numpy(np.argsort(
        -similarities.detach().cpu().numpy(), axis=1, kind="mergesort"))
    accumulated = {str(k): [] for k in ks}
    if include_global:
        accumulated['global'] = []
    for query, bank in zip(query_groups, patch_banks):
        grouped = {key: [] for key in accumulated}
        for bi in range(batch):
            q = F.normalize(query[bi].float(), dim=-1)
            best = torch.full((n,), -float('inf'), device=device)
            previous = 0
            for k in sorted(ks):
                refs = bank[ranking[bi, previous:k]].reshape(-1, channels)
                best = _scan(q, refs, best, candidate_chunk, row_chunk)
                grouped[str(k)].append(1 - best.clone())
                previous = k
            if include_global:
                best_all = _scan(q, bank.reshape(-1, channels),
                                 torch.full_like(best, -float('inf')),
                                 candidate_chunk, row_chunk)
                grouped['global'].append(1 - best_all)
        for key in accumulated:
            accumulated[key].append(torch.stack(grouped[key]).reshape(batch, 1, side, side))
    maps = {key: torch.stack(values).mean(0) for key, values in accumulated.items()}
    if not all(torch.isfinite(v).all() for v in maps.values()):
        raise RuntimeError('Non-finite memory map')
    return maps, ranking[:, :max(ks)].tolist()


def _scan(query, refs, best, candidate_chunk, row_chunk):
    for start in range(0, len(refs), candidate_chunk):
        ref = F.normalize(refs[start:start + candidate_chunk].to(query.device).float(), dim=-1)
        for row in range(0, len(query), row_chunk):
            end = row + row_chunk
            best[row:end] = torch.maximum(best[row:end], (query[row:end] @ ref.T).max(1).values)
    return best


def prepare_map(amap, kernel):
    # Identical interpolation/smoothing order to the 392-input V0 evaluator.
    if amap.shape[-2:] != (392, 392):
        amap = F.interpolate(amap, size=392, mode='bilinear', align_corners=True)
    amap = F.interpolate(amap, size=256, mode='bilinear', align_corners=False)
    return kernel(amap)


def fuse_maps(rec, mem, stats, mode, alpha=None):
    if mode == 'raw' or (mode == 'weighted_sum' and alpha == 0):
        # Exact V0 fallback. Affine z-score is rank-equivalent but can move ties.
        return np.asarray(rec, dtype=np.float32)
    r = (np.asarray(rec) - stats['rec_mean']) / stats['rec_std']
    m = (np.asarray(mem) - stats['mem_mean']) / stats['mem_std']
    if mode == 'weighted_sum':
        result = (1 - alpha) * r + alpha * m
    elif mode == 'gated':
        from scipy.special import expit
        result = np.asarray(rec) * expit(m)
    else:
        raise ValueError(mode)
    if not np.isfinite(result).all():
        raise RuntimeError('Non-finite fused maps')
    return result.astype(np.float32)


def top_scores(amaps, fractions):
    # Descending order and floor pixel count follow V3Evaluator exactly.
    flat = np.sort(amaps.reshape(len(amaps), -1), axis=1)[:, ::-1]
    return {z: flat[:, :max(1, int(flat.shape[1] * z))].mean(1) for z in fractions}


def load_bank(bank_dir):
    bank_dir = Path(bank_dir)
    metadata = json.loads((bank_dir / 'bank.json').read_text(encoding='utf-8'))
    for name, digest in metadata['file_sha256'].items():
        if sha256_file(bank_dir / name) != digest:
            raise ValueError('Corrupt memory bank: ' + name)
    cls = torch.load(bank_dir / 'cls.fp16.pt', map_location='cpu')
    groups = [torch.load(bank_dir / ('patch_group%d.fp16.pt' % i), map_location='cpu')
              for i in range(2)]
    return metadata, cls, groups
