"""Normal-only Dinomaly baseline on frozen bridge dataset splits."""
import argparse
import fcntl
import hashlib
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from bridge_dataset import dataset_class, check_split_protocol, load_profile
from bridge_evaluation import evaluate
from dinomaly_mvtec_uni import setup_seed, get_logger
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block, bMlp, LinearAttention2
from dinov1.utils import trunc_normal_
from optimizers import StableAdamW
from utils import global_cosine_hm_percent, WarmCosineScheduler
from reproduction_runtime import add_arguments, record_run, save_checkpoint, record_metrics, METRIC_NAMES


def build_model(device, variant='v0'):
    encoder = vit_encoder.load('dinov2reg_vit_base_14')
    bottleneck = nn.ModuleList([bMlp(768, 768 * 4, 768, drop=0.2)])
    decoder = nn.ModuleList([Block(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2) for _ in range(8)])
    model = ViTill(encoder, bottleneck, decoder, target_layers=list(range(2, 10)), mask_neighbor_size=0,
                  fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]], fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]]).to(device)
    trainable = nn.ModuleList([bottleneck, decoder])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    # Initialize all original reconstruction layers exactly as in V0 before
    # constructing the added module, preserving their seed-1 initialization.
    if variant == 'v1_amcm':
        from models.amcm_tokens import AMCMTokenAdapter
        # Official initialization must not shift the V0 DataLoader RNG stream.
        with torch.random.fork_rng(devices=[]):
            adapter = AMCMTokenAdapter(encoder)
        bottleneck.append(adapter.to(device))
    encoder.requires_grad_(False)
    return model, trainable


def main(args):
    folder = Path(args.save_dir) / args.save_name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (Path(args.save_dir) / 'gpu0.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'status.json').exists():
        raise RuntimeError('Run already exists; choose a new save_name')
    logger = get_logger(args.save_name, str(folder))
    status = dict(state='running', pid=os.getpid(), step=0, mode='smoke' if args.smoke_steps else 'full',
                  started_at=time.time())
    def state(**updates):
        status.update(updates, updated_at=time.time())
        temporary = folder / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(folder / 'status.json')
    state()
    try:
        setup_seed(args.seed)
        protocol = check_split_protocol(args.data_path, args.dataset_profile, args.split_view)
        _, exclusions, _ = load_profile(args.dataset_profile, args.split_view)
        DatasetClass = dataset_class(args.dataset_profile)
        record_run(args, 'dinov2reg_vit_base_14')
        manifests = {s: hashlib.sha256((Path(args.data_path) / (s + '_manifest.csv')).read_bytes()).hexdigest()
                     for s in ['train','val','test']}
        (folder / 'dataset_protocol.json').write_text(json.dumps(dict(splits=protocol, manifest_sha256=manifests,
                    dataset_profile=args.dataset_profile,
                    split_view=args.split_view, excluded_samples=exclusions,
                    training_samples=protocol['train']['categories']['normal'],
                    mask_conversion=('union of 14 native-resolution damage polygons; exclude five object labels'
                                     if args.dataset_profile == 'dacl_v3_seed42' else 'union of nonzero RGB pixels before resizing'),
                    train_filter='category == normal', input='full-frame resize to 392x392',
                    validation_selection='max image AUROC; tie-break by pixel AP', test_usage='after training only'), indent=2))
        train = DatasetClass(args.data_path, 'train', normal_only=True, image_size=args.image_size, exclude_samples=exclusions['train'])
        val = DatasetClass(args.data_path, 'val', image_size=args.image_size, exclude_samples=exclusions['val'])
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
        if not len(train_loader):
            raise ValueError('Training loader is empty')
        if args.model_variant == 'v1_amcm' and args.smoke_steps:
            raise ValueError('V1 is authorized for full runs only; smoke_steps must be zero')
        model, trainable = build_model(args.device, args.model_variant)
        optimizer = StableAdamW([{'params': trainable.parameters()}], lr=2e-3, betas=(0.9,0.999),
                                weight_decay=1e-4, amsgrad=True, eps=1e-10)
        scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4,
                                       total_iters=args.total_iters, warmup_iters=100)
        added = model.bottleneck[-1] if args.model_variant == 'v1_amcm' else None
        optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert {id(p) for p in model.parameters() if p.requires_grad} == optimized
        if added is not None:
            assert all(id(p) in optimized for p in added.parameters())
        model_info = dict(variant=args.model_variant,
                          added_parameters=sum(p.numel() for p in added.parameters()) if added else 0,
                          trainable_parameters=sum(p.numel() for p in trainable.parameters()),
                          encoder_frozen=not any(p.requires_grad for p in model.encoder.parameters()),
                          initialization='fresh pretrained encoder; original V0 reconstruction initialization then official AMCM defaults',
                          amcm=dict(dim=768, sa_grid_size=5, sa_reduction=2, multiscale_kernels=[5,7,9,11]) if added else None)
        (folder / 'model_info.json').write_text(json.dumps(model_info, indent=2))
        logger.info('Model provenance: %s', model_info)
        torch.cuda.reset_peak_memory_stats()
        logger.info('Normal-only train=%d; val=%d; test=%d; batch=%d', len(train), len(val),
                    protocol['test']['samples'], args.batch_size)
        best_key = (-float('inf'), -float('inf'))
        best_step, best_threshold = 0, None
        last_threshold = None
        step = 0
        while step < args.total_iters:
            model.train()
            for image, label in train_loader:
                if label.any():
                    raise RuntimeError('Anomaly label encountered in training')
                en, de = model(image.to(args.device))
                loss = global_cosine_hm_percent(en, de, p=min(0.9 * step / 1000, 0.9), factor=0.1)
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Non-finite loss at step {}'.format(step + 1))
                optimizer.zero_grad()
                loss.backward()
                if step == 0 and added is not None:
                    assert all(p.grad is not None for p in added.parameters() if p.requires_grad), 'AMCM missing gradients'
                grad_norm = nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                if not torch.isfinite(grad_norm).item():
                    raise RuntimeError('Non-finite gradient norm at step {}'.format(step + 1))
                optimizer.step()
                scheduler.step()
                step += 1
                if step == 1 or step % 100 == 0:
                    logger.info('step [%d/%d], loss:%.6f, peak_cuda_gib:%.3f', step, args.total_iters,
                                loss.item(), torch.cuda.max_memory_allocated() / 1024 ** 3)
                    state(step=step, loss=loss.item())
                if args.smoke_steps and step >= args.smoke_steps:
                    indices = [next(i for i, row in enumerate(val.rows) if row['category'] == kind)
                               for kind in ['normal', 'annotated']]
                    values, calibrated = evaluate(model, DataLoader(Subset(val, indices), batch_size=2), args.device,
                                                   folder / 'smoke_val_scores.csv')
                    record_metrics(args, step, 'smoke_val_subset', values)
                    save_checkpoint(args, model, optimizer, scheduler, step, 'smoke.pth')
                    checkpoint = torch.load(folder / 'smoke.pth', map_location='cpu')
                    if checkpoint['step'] != step:
                        raise RuntimeError('Checkpoint roundtrip failed')
                    del checkpoint
                    logger.info('Smoke passed: steps=%d, validation pipeline and checkpoint verified', step)
                    state(state='completed', step=step, smoke_calibration=calibrated)
                    return
                if step % args.checkpoint_interval == 0:
                    save_checkpoint(args, model, optimizer, scheduler, step)
                if step % args.eval_interval == 0 or step == args.total_iters:
                    values, calibrated = evaluate(model, DataLoader(val, batch_size=args.batch_size, num_workers=4),
                                                   args.device, folder / ('val_scores_{}.csv'.format(step)))
                    record_metrics(args, step, 'val', values)
                    last_threshold = calibrated['threshold']
                    key = (values[0], values[4])
                    if key > best_key:
                        best_key, best_step, best_threshold = key, step, last_threshold
                        save_checkpoint(args, model, optimizer, scheduler, step, 'best_val.pth')
                        (folder / 'best_selection.json').write_text(json.dumps(dict(step=step, metrics=dict(zip(METRIC_NAMES,values)),
                                                                    calibration=calibrated), indent=2))
                    logger.info('val step %d: %s; best step=%d', step, dict(zip(METRIC_NAMES,values)), best_step)
                    state(step=step, best_step=best_step, validation=dict(zip(METRIC_NAMES, values)))
                    model.train()
                if step == args.total_iters:
                    break
        save_checkpoint(args, model, optimizer, scheduler, step, 'final.pth')
        test = DatasetClass(args.data_path, 'test', image_size=args.image_size, exclude_samples=exclusions['test'])
        test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=4)
        training_finished_at = time.time()
        summary = dict(best_validation_step=best_step, final_step=step, model_info=model_info,
                       training_seconds=training_finished_at - status['started_at'],
                       peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                       peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
        final_values, final_cal = evaluate(model, test_loader, args.device, folder / 'test_final_scores.csv', last_threshold)
        record_metrics(args, step, 'test_final', final_values)
        summary['test_final'] = dict(metrics=dict(zip(METRIC_NAMES,final_values)), calibrated=final_cal)
        if best_step != step:
            checkpoint = torch.load(folder / 'best_val.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            del checkpoint
            best_values, best_cal = evaluate(model, test_loader, args.device, folder / 'test_best_val_scores.csv', best_threshold)
        else:
            best_values, best_cal = final_values, final_cal
        record_metrics(args, best_step, 'test_best_val', best_values)
        summary['test_best_val'] = dict(metrics=dict(zip(METRIC_NAMES,best_values)), calibrated=best_cal)
        summary.update(total_seconds=time.time() - status['started_at'],
                       peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                       peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
        logger.info('Completed: %s', summary)
        state(state='completed', step=step, finished_at=time.time(), summary=summary)
    except BaseException as exc:
        state(state='failed', error=str(exc))
        logger.exception('Bridge baseline failed')
        raise


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='/home/skye/data/Skye/databases/Bridge2893_split_seed42')
    parser.add_argument('--save_dir', default='runs')
    parser.add_argument('--save_name', required=True)
    parser.add_argument('--model_variant', choices=['v0', 'v1_amcm'], default='v0')
    parser.add_argument('--image_size', type=int, choices=[392], default=392)
    parser.add_argument('--dataset_profile', choices=['bridge2893','bridge512_balance','bridge512_balance_repaired','dacl_v3_seed42'], default='bridge2893')
    parser.add_argument('--split_view', choices=['original','deduplicated'], default='original')
    add_arguments(parser)
    parser.set_defaults(eval_interval=2000)
    main(parser.parse_args())
