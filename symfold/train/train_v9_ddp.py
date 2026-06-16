# -*- coding: utf-8 -*-
"""PriFold v9 DDP training: DensityNet-Pro+ (2x H20 GPUs).

Fully utilizes two H20 GPUs (97GB each) with:
  - DistributedDataParallel for data-parallel training
  - Per-GPU batch: effective batch = per_gpu_batch × 2 GPUs
  - torch.compile for additional throughput
  - Gradient accumulation compatible with DDP
  - Automatic mixed precision (bf16)
  - Efficient all-reduce with NCCL

Usage:
  # Launch with torchrun (recommended):
  torchrun --nproc_per_node=2 --standalone \
    symfold/train/train_v9_ddp.py symfold/config/v9/v9_ddp.json

  # Or use the wrapper script:
  bash symfold/train/run_train_v9_ddp.sh
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

faulthandler.enable()

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Numerical safety for H20
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, LengthBucketBatchSampler, make_collate_fn
from symfold.metrics import contact_metrics
from symfold.v9.model import DensityNetProPlus


# ============================================================
# DDP Utilities
# ============================================================

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def all_reduce_mean(tensor):
    """All-reduce a tensor and divide by world size."""
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


# ============================================================
# Model Building
# ============================================================

def build_model(cfg: dict, extractor) -> DensityNetProPlus:
    """Build v9 DensityNet-Pro+ from config."""
    mcfg = cfg['model']
    v9cfg = cfg.get('v9', {})
    lcfg = cfg.get('loss', {})

    model = DensityNetProPlus(
        extractor=extractor,
        freeze_mars=mcfg.get('freeze_mars', True),
        mars_dim=mcfg.get('mars_dim', 1056),
        mars_n_attn_layers=mcfg.get('mars_n_attn_layers', 6),
        mars_n_heads=mcfg.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mcfg.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        hidden_dim=v9cfg.get('hidden_dim', 192),
        num_layers=v9cfg.get('num_layers', 8),
        num_heads=v9cfg.get('num_heads', 6),
        dim_head=v9cfg.get('dim_head', 32),
        ff_mult=v9cfg.get('ff_mult', 4),
        dropout=v9cfg.get('dropout', 0.2),
        drop_path=v9cfg.get('drop_path', 0.15),
        use_rope=v9cfg.get('use_rope', True),
        focal_gamma=lcfg.get('focal_gamma', 1.0),
        pos_weight_base=lcfg.get('pos_weight_base', 99.0),
        dice_weight=lcfg.get('dice_weight', 0.5),
        dst_weight=lcfg.get('dst_weight', 0.5),
        dst_low_threshold=lcfg.get('dst_low_threshold', 0.05),
        dst_tversky_alpha=lcfg.get('dst_tversky_alpha', 0.7),
        dst_tversky_beta=lcfg.get('dst_tversky_beta', 0.3),
        pair_count_weight=lcfg.get('pair_count_weight', 0.3),
        ratio_penalty_weight=lcfg.get('ratio_penalty_weight', 0.2),
        ratio_penalty_threshold=lcfg.get('ratio_penalty_threshold', 1.20),
        density_loss_weight=lcfg.get('density_loss_weight', 0.3),
        ohem_enabled=lcfg.get('ohem_enabled', True),
        ohem_neg_ratio=lcfg.get('ohem_neg_ratio', 3),
        fp_penalty_enabled=lcfg.get('fp_penalty_enabled', True),
        fp_penalty_weight=lcfg.get('fp_penalty_weight', 2.0),
        bp_compat_enabled=lcfg.get('bp_compat_enabled', False),
        bp_compat_weight=lcfg.get('bp_compat_weight', 0.0),
        bp_compat_in_inference=lcfg.get('bp_compat_in_inference', False),
        shift_loss_enabled=lcfg.get('shift_loss_enabled', True),
        shift_loss_weight=lcfg.get('shift_loss_weight', 0.8),
        shift_radius=lcfg.get('shift_radius', 2),
    )
    return model


# ============================================================
# Data Loading (DDP-aware with dynamic batch sizing)
# ============================================================

class DDPLengthBucketBatchSampler(torch.utils.data.Sampler):
    """Dynamic batch sampler for DDP: each rank gets non-overlapping subsets,
    with batch size determined by sequence length (memory ~ O(L^2)).
    
    This ensures GPU memory usage stays stable regardless of sequence length,
    while still splitting data across DDP ranks.
    """
    def __init__(self, lengths, batch_size: int, num_replicas: int, rank: int,
                 shuffle: bool = True, seed: int = 0, max_sq_tokens: int | None = None):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        if max_sq_tokens is None:
            median_len = sorted(self.lengths)[len(self.lengths) // 2]
            self.max_sq_tokens = batch_size * median_len * median_len
        else:
            self.max_sq_tokens = max_sq_tokens

    def _get_dynamic_batch_size(self, length: int) -> int:
        bs = max(1, self.max_sq_tokens // (length * length))
        return min(bs, self.batch_size * 4)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        # Sort by length for bucketing
        order.sort(key=lambda i: self.lengths[i])
        
        # Build all batches with dynamic size
        all_batches = []
        i = 0
        while i < len(order):
            cur_len = self.lengths[order[i]]
            bs = self._get_dynamic_batch_size(cur_len)
            batch = order[i:i + bs]
            all_batches.append(batch)
            i += len(batch)
        
        # Shuffle batches
        if self.shuffle:
            rng.shuffle(all_batches)
        
        # Split batches across ranks (round-robin)
        my_batches = all_batches[self.rank::self.num_replicas]
        yield from my_batches

    def __len__(self):
        # Estimate
        sorted_lengths = sorted(self.lengths)
        n_batches = 0
        i = 0
        while i < len(sorted_lengths):
            cur_len = sorted_lengths[i]
            bs = self._get_dynamic_batch_size(cur_len)
            i += bs
            n_batches += 1
        # Each rank gets ~1/num_replicas of total batches
        return max(1, n_batches // self.num_replicas)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def build_ddp_loaders(config, tokenizer):
    """Build data loaders with dynamic batch sizing for DDP."""
    tcfg = config['training']
    dataset_mode = tcfg.get('dataset_mode', 'bprna')
    data_dir = config['paths']['data_dir']
    max_len = tcfg.get('max_len_filter', 490)
    
    # Train dataset
    train_records = build_records(data_dir, f'{dataset_mode}-train', max_len=max_len)
    train_dataset = PriFoldSymFlowDataset(
        train_records,
        augment=tcfg.get('augmentation', {}).get('enabled', False),
        select=tcfg.get('augmentation', {}).get('select', 0.15),
        replace=tcfg.get('augmentation', {}).get('replace', 0.35),
    )
    
    # Dynamic batch sampler (DDP-aware)
    lengths = [len(r.seq) for r in train_records]
    train_sampler = DDPLengthBucketBatchSampler(
        lengths=lengths,
        batch_size=tcfg.get('batch_size', 12),
        num_replicas=get_world_size(),
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=True,
        seed=config.get('seed', 3407),
        max_sq_tokens=tcfg.get('max_sq_tokens', 800000),
    )
    
    collate_fn = make_collate_fn(tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=tcfg.get('num_workers', 8),
        pin_memory=tcfg.get('pin_memory', True),
        prefetch_factor=tcfg.get('prefetch_factor', 4),
        persistent_workers=True,
    )
    
    # Val dataset (only evaluate on rank 0)
    val_records = build_records(data_dir, f'{dataset_mode}-val', max_len=max_len)
    val_dataset = PriFoldSymFlowDataset(val_records, augment=False)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader, train_sampler


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, ddp_model, loader, optimizer, device, config, logger, epoch,
                    compiled_model=None):
    """DDP training loop for one epoch."""
    ddp_model.train()
    totals = {'loss': 0.0, 'bce': 0.0, 'density': 0.0, 'fp_penalty': 0.0, 'shift': 0.0}
    n = 0
    t0 = time.time()
    
    amp_dtype = torch.bfloat16
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)
    grad_clip = config['training'].get('grad_clip', 1.0)
    log_every = config['training'].get('log_every', 20)
    
    forward_fn = compiled_model if compiled_model is not None else ddp_model
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        
        # Use no_sync for gradient accumulation (skip all-reduce until step)
        sync_context = ddp_model.no_sync if (step + 1) % grad_accum != 0 else lambda: torch.enable_grad()
        
        with sync_context():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, loss_dict = forward_fn(batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process():
                    logger.warning(f'[Train] e{epoch} step={step} NaN/Inf, skip')
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaled_loss = loss / grad_accum
            scaled_loss.backward()
        
        # Step optimizer
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        n += 1
        totals['loss'] += float(loss.item())
        for k in ('bce', 'density', 'fp_penalty', 'shift'):
            totals[k] += float(loss_dict.get(k, torch.tensor(0.0)).item())
        
        if is_main_process() and step % log_every == 0:
            logger.info(
                f"[Train] e{epoch} step={step}/{len(loader)} L={batch['set_max_len']} "
                f"loss={loss.item():.5f} bce={float(loss_dict['bce']):.4f}")
    
    # All-reduce loss across GPUs
    avg_loss = torch.tensor(totals['loss'] / max(n, 1), device=device)
    all_reduce_mean(avg_loss)
    
    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg['time_s'] = time.time() - t0
    avg['steps_per_sec'] = n / avg['time_s']
    avg['loss'] = float(avg_loss.item())  # Use all-reduced loss
    
    if is_main_process():
        logger.info(f'[Train] e{epoch} done loss={avg["loss"]:.5f} '
                    f'time={avg["time_s"]:.1f}s steps/s={avg["steps_per_sec"]:.1f}')
    return avg


@torch.no_grad()
def evaluate_ddp(model, loader, device, config, logger, stage='val'):
    """Evaluate model (only on rank 0)."""
    model.eval()
    scfg = config.get('sampling', {})
    amp_dtype = torch.bfloat16

    results = {'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0,
               'gt_pairs': 0, 'pred_pairs': 0, 'n': 0}

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            pred, _ = model.predict(
                batch,
                budget_fraction=scfg.get('default_budget_fraction', 0.30),
                use_density_budget=scfg.get('use_density_budget', True),
                score_threshold=scfg.get('score_threshold', 0.43),
                length_decay=scfg.get('length_decay', 0.15),
                budget_floor=scfg.get('budget_floor', 0.6),
            )
        m = contact_metrics(pred, batch['contact'], batch['length'])
        bs = pred.shape[0]
        results['precision'] += m['precision'] * bs
        results['recall'] += m['recall'] * bs
        results['f1'] += m['f1'] * bs
        results['mcc'] += m['mcc'] * bs
        results['gt_pairs'] += m['gt_pairs'] * bs
        results['pred_pairs'] += m['pred_pairs'] * bs
        results['n'] += bs

    n = results['n']
    if n > 0:
        for k in ['precision', 'recall', 'f1', 'mcc', 'gt_pairs', 'pred_pairs']:
            results[k] /= n
    return results


def lr_for_epoch(config, epoch):
    """Cosine LR schedule with warmup."""
    tcfg = config['training']
    base_lr = tcfg.get('lr', 5e-4)
    warmup = max(tcfg.get('warmup_epochs', 8), 1)
    total = max(tcfg.get('epochs', 200), 1)
    
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    
    # DDP setup
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    world_size = get_world_size()
    
    # Logging (only rank 0)
    logger = None
    if is_main_process():
        log_dir = Path(config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"{config['task_name']}.log", mode='a'),
                logging.StreamHandler(sys.stdout)
            ],
        )
        logger = logging.getLogger('v9-ddp')
        logger.info('=' * 80)
        logger.info(f'PriFold v9 DDP training: {config["task_name"]}')
        logger.info(f'World size: {world_size}, Local rank: {local_rank}')
        logger.info(f'Config: {json.dumps(config, indent=2)}')
        logger.info('=' * 80)
    
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_save_dir'])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
    
    # Seed
    seed = config.get('seed', 3407)
    torch.manual_seed(seed + dist.get_rank())
    np.random.seed(seed + dist.get_rank())
    
    # Load LM extractor
    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)
    
    # Build model
    model = build_model(config, extractor).to(device)
    
    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'[Model] Trainable params: {n_params/1e6:.2f}M')
    
    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # torch.compile
    compiled_model = None
    if config['training'].get('compile_model', False):
        if is_main_process():
            logger.info('[v9] Applying torch.compile...')
        try:
            compile_mode = config['training'].get('compile_mode', 'reduce-overhead')
            compiled_model = torch.compile(ddp_model, mode=compile_mode)
            if is_main_process():
                logger.info(f'[v9] torch.compile(mode="{compile_mode}") successful!')
        except Exception as e:
            if is_main_process():
                logger.warning(f'[v9] torch.compile failed: {e}')
            compiled_model = None
    
    # Data
    train_loader, val_loader, train_sampler = build_ddp_loaders(config, tokenizer)
    if is_main_process():
        logger.info(f'[Data] train_batches={len(train_loader)} val_samples={len(val_loader.dataset)}')
    
    # Optimizer with layer-wise LR decay (optional)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training'].get('lr', 5e-4),
        weight_decay=config['training'].get('weight_decay', 0.01),
        betas=(0.9, 0.999),
    )
    
    # Training loop
    history = []
    best_f1 = -1.0
    patience_count = 0
    best_path = model_dir / 'best.pt'
    last_path = model_dir / 'last.pt'
    
    # Resume
    start_epoch = 0
    if last_path.exists() and config['training'].get('auto_resume', True):
        if is_main_process():
            logger.info('[Resume] Loading last.pt...')
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history = ckpt.get('history', [])
        best_f1 = ckpt.get('best_f1', -1.0)
        start_epoch = ckpt.get('epoch', -1) + 1
        if is_main_process():
            logger.info(f'[Resume] start_epoch={start_epoch} best_f1={best_f1:.4f}')
    
    # Sync model across processes
    dist.barrier()
    
    for epoch in range(start_epoch, config['training'].get('epochs', 200)):
        train_sampler.set_epoch(epoch)
        
        # LR schedule
        lr = lr_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        if is_main_process():
            logger.info(f'[LR] epoch={epoch} lr={lr:.6g}')
        
        # Train
        train_stats = train_one_epoch(
            model, ddp_model, train_loader, optimizer, device, config,
            logger or logging.getLogger(), epoch,
            compiled_model=compiled_model)
        
        entry = {'epoch': epoch, 'lr': lr, **train_stats}
        
        # Evaluate (only on rank 0)
        if is_main_process():
            val_stats = evaluate_ddp(model, val_loader, device, config, logger, 'val')
            entry.update({f'val_{k}': v for k, v in val_stats.items()})
            
            if val_stats['f1'] > best_f1:
                best_f1 = val_stats['f1']
                patience_count = 0
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'config': config, 'best_f1': best_f1},
                           best_path)
                logger.info(f'[Save] new best F1={best_f1:.4f}')
            else:
                patience_count += 1
                logger.info(f'[Eval] no improve {patience_count}/{config["training"].get("patience", 35)}')
            
            history.append(entry)
            with open(output_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            # Plot training curves
            try:
                from symfold.train.train_v3 import plot_curves
                plot_curves(history, output_dir, logger)
            except Exception as e:
                logger.warning(f'[Plot] failed: {e}')
            
            # Save last checkpoint
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history, 'best_f1': best_f1, 'config': config,
            }, last_path)
        
        # Broadcast best_f1 and patience to all ranks
        info_tensor = torch.tensor([best_f1, patience_count], device=device)
        dist.broadcast(info_tensor, src=0)
        best_f1 = float(info_tensor[0].item())
        patience_count = int(info_tensor[1].item())
        
        # Early stopping
        if patience_count >= config['training'].get('patience', 35):
            if is_main_process():
                logger.info('[EarlyStop] patience reached')
            break
    
    if is_main_process():
        logger.info('Training finished!')
    cleanup_distributed()


if __name__ == '__main__':
    main()
