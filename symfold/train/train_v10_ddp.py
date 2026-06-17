# -*- coding: utf-8 -*-
"""PriFold v10 DDP training: DensityNet-Ultra (2x H20 GPUs).

Key changes vs v9:
  U1. Partial MARS unfreeze (last 2 layers) — needs layered LR
  U2. Family-balanced curriculum sampling — oversample hard families
  U3. Warmup longer (15 epochs) — protect unfrozen MARS layers from early noise

Usage:
  torchrun --nproc_per_node=2 --standalone \
    symfold/train/train_v10_ddp.py symfold/config/v10/v10_ddp.json
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

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, make_collate_fn
from symfold.metrics import contact_metrics
from symfold.v10.model import DensityNetUltra


# ============================================================
# DDP Utilities
# ============================================================

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


# ============================================================
# [U2] Family-Balanced Curriculum Sampler
# ============================================================

class CurriculumBatchSampler(torch.utils.data.Sampler):
    """Dynamic batch sampler with family-balanced curriculum.
    
    Strategy:
      - Phase 1 (epoch 0-20): normal sampling, learn easy patterns first
      - Phase 2 (epoch 20+): oversample hard families (RFAM) by 2x
      - Dynamic batch sizing based on sequence length (same as v9)
    """
    def __init__(self, lengths, datasets, batch_size: int,
                 num_replicas: int, rank: int, shuffle: bool = True,
                 seed: int = 0, max_sq_tokens: int = 600000,
                 hard_oversample: float = 2.0, curriculum_start: int = 20):
        self.lengths = list(lengths)
        self.datasets = list(datasets)  # 'bpRNA' etc.
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.max_sq_tokens = max_sq_tokens
        self.hard_oversample = hard_oversample
        self.curriculum_start = curriculum_start
        
        # Pre-compute indices by source
        self.hard_indices = [i for i, d in enumerate(self.datasets)
                           if 'RFAM' in str(d).upper() or 'rfam' in str(d).lower()]
        self.easy_indices = [i for i in range(len(self.lengths)) 
                           if i not in set(self.hard_indices)]

    def _get_dynamic_batch_size(self, length: int) -> int:
        bs = max(1, self.max_sq_tokens // (length * length))
        return min(bs, self.batch_size * 4)

    def _build_order(self, rng):
        """Build sample order with optional curriculum oversample."""
        if self.epoch >= self.curriculum_start and self.hard_indices:
            n_extra = int(len(self.hard_indices) * (self.hard_oversample - 1))
            extra = rng.choice(self.hard_indices, size=n_extra, replace=True).tolist()
            order = list(range(len(self.lengths))) + extra
        else:
            order = list(range(len(self.lengths)))
        return order

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = self._build_order(rng)
        
        if self.shuffle:
            rng.shuffle(order)
        order.sort(key=lambda i: self.lengths[i % len(self.lengths)])
        
        # Build batches
        all_batches = []
        i = 0
        while i < len(order):
            idx = order[i] % len(self.lengths)
            cur_len = self.lengths[idx]
            bs = self._get_dynamic_batch_size(cur_len)
            batch = order[i:i + bs]
            batch = [x % len(self.lengths) for x in batch]
            all_batches.append(batch)
            i += bs
        
        if self.shuffle:
            rng.shuffle(all_batches)
        
        my_batches = all_batches[self.rank::self.num_replicas]
        yield from my_batches

    def __len__(self):
        # Accurate estimate considering curriculum
        rng = np.random.default_rng(self.seed + self.epoch)
        order = self._build_order(rng)
        order.sort(key=lambda i: self.lengths[i % len(self.lengths)])
        n_batches = 0
        i = 0
        while i < len(order):
            idx = order[i] % len(self.lengths)
            cur_len = self.lengths[idx]
            bs = self._get_dynamic_batch_size(cur_len)
            i += bs
            n_batches += 1
        return max(1, n_batches // self.num_replicas)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# ============================================================
# Model Building
# ============================================================

def build_model(cfg: dict, extractor) -> DensityNetUltra:
    """Build v10 DensityNet-Ultra from config."""
    mcfg = cfg['model']
    v10cfg = cfg.get('v10', {})
    lcfg = cfg.get('loss', {})

    model = DensityNetUltra(
        extractor=extractor,
        freeze_mars=v10cfg.get('freeze_mars', 'partial'),
        unfreeze_last_n=v10cfg.get('unfreeze_last_n', 2),
        mars_dim=mcfg.get('mars_dim', 1056),
        mars_n_attn_layers=mcfg.get('mars_n_attn_layers', 6),
        mars_n_heads=mcfg.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mcfg.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        hidden_dim=v10cfg.get('hidden_dim', 192),
        num_layers=v10cfg.get('num_layers', 8),
        num_heads=v10cfg.get('num_heads', 6),
        dim_head=v10cfg.get('dim_head', 32),
        ff_mult=v10cfg.get('ff_mult', 4),
        dropout=v10cfg.get('dropout', 0.2),
        drop_path=v10cfg.get('drop_path', 0.15),
        use_rope=v10cfg.get('use_rope', True),
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
# Data Loading
# ============================================================

def build_ddp_loaders(config, tokenizer):
    """Build data loaders with curriculum sampling for DDP."""
    tcfg = config['training']
    dataset_mode = tcfg.get('dataset_mode', 'bprna')
    data_dir = config['paths']['data_dir']
    max_len = tcfg.get('max_len_filter', 490)
    
    train_records = build_records(data_dir, f'{dataset_mode}-train', max_len=max_len)
    train_dataset = PriFoldSymFlowDataset(
        train_records,
        augment=tcfg.get('augmentation', {}).get('enabled', False),
        select=tcfg.get('augmentation', {}).get('select', 0.20),
        replace=tcfg.get('augmentation', {}).get('replace', 0.40),
    )
    
    lengths = [len(r.seq) for r in train_records]
    datasets = [r.file_name for r in train_records]  # contains RFAM in name if from RFAM
    
    # [U2] Curriculum batch sampler
    curriculum_cfg = tcfg.get('curriculum', {})
    train_sampler = CurriculumBatchSampler(
        lengths=lengths,
        datasets=datasets,
        batch_size=tcfg.get('batch_size', 8),
        num_replicas=get_world_size(),
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=True,
        seed=config.get('seed', 3407),
        max_sq_tokens=tcfg.get('max_sq_tokens', 400000),
        hard_oversample=curriculum_cfg.get('hard_oversample', 2.0),
        curriculum_start=curriculum_cfg.get('start_epoch', 20),
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

def train_one_epoch(model, ddp_model, loader, optimizer, device, config, logger, epoch):
    """DDP training loop for one epoch."""
    ddp_model.train()
    # [U1] MARS unfrozen layers should stay in train mode
    totals = {'loss': 0.0, 'bce': 0.0, 'density': 0.0, 'fp_penalty': 0.0, 'shift': 0.0}
    n = 0
    t0 = time.time()
    
    amp_dtype = torch.bfloat16
    grad_accum = config['training'].get('gradient_accumulation_steps', 2)
    grad_clip = config['training'].get('grad_clip', 0.5)  # Lower for MARS stability
    log_every = config['training'].get('log_every', 20)
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, loss_dict = ddp_model(batch)
            loss = loss / grad_accum
        
        loss.backward()
        
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        bs = batch['contact'].shape[0]
        totals['loss'] += loss_dict['total'].item() * bs
        totals['bce'] += loss_dict['bce'].item() * bs
        totals['density'] += loss_dict['density'].item() * bs
        totals['fp_penalty'] += loss_dict['fp_penalty'].item() * bs
        totals['shift'] += loss_dict['shift'].item() * bs
        n += bs
        
        if is_main_process() and step % log_every == 0:
            L = int(batch['length'].max().item())
            logger.info(f'[Train] e{epoch} step={step}/{len(loader)} L={L} '
                       f'loss={loss_dict["total"].item():.5f} bce={loss_dict["bce"].item():.4f}')
    
    elapsed = time.time() - t0
    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg['time_s'] = elapsed
    avg['steps_per_sec'] = len(loader) / elapsed
    
    if is_main_process():
        logger.info(f'[Train] e{epoch} done loss={avg["loss"]:.5f} '
                    f'time={elapsed:.1f}s steps/s={avg["steps_per_sec"]:.1f}')
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
    """Cosine LR schedule with longer warmup for v10."""
    tcfg = config['training']
    base_lr = 1.0  # We use per-group LR, scheduler returns multiplier
    warmup = max(tcfg.get('warmup_epochs', 15), 1)
    total = max(tcfg.get('epochs', 200), 1)
    
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================================================
# Main
# ============================================================

def main():
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    world_size = get_world_size()
    
    # Parse config
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    # Logging
    log_dir = Path(config['paths']['log_dir'])
    if is_main_process():
        log_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    
    logger = logging.getLogger('v10_ddp')
    logger.setLevel(logging.INFO)
    if is_main_process():
        fh = logging.FileHandler(log_dir / 'v10_ddp.log')
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(sh)
        logger.info('=' * 80)
        logger.info(f'v10 DensityNet-Ultra DDP Training')
        logger.info(f'World size: {world_size}, Local rank: {local_rank}')
        logger.info(f'Config: {json.dumps(config, indent=2)}')
        logger.info('=' * 80)
    
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_save_dir'])
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
    
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
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'[Model] Total params: {n_total/1e6:.2f}M, Trainable: {n_train/1e6:.2f}M')
    
    # [U1] Layered LR optimizer
    v10cfg = config.get('v10', {})
    tcfg = config['training']
    param_groups = model.get_param_groups(
        mars_lr=v10cfg.get('mars_lr', 1e-5),
        head_lr=tcfg.get('lr', 5e-4),
        weight_decay=tcfg.get('weight_decay', 0.02),
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
    
    # Wrap with DDP
    # [U1] find_unused_parameters=True because frozen MARS layers don't participate
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Data
    train_loader, val_loader, train_sampler = build_ddp_loaders(config, tokenizer)
    if is_main_process():
        logger.info(f'[Data] train_batches={len(train_loader)} val_samples={len(val_loader.dataset)}')
    
    # Training state
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0
    history = []
    
    # Auto-resume
    last_ckpt = model_dir / 'last.pt'
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f1 = ckpt.get('best_f1', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        if 'history' in ckpt:
            history = ckpt['history']
        if is_main_process():
            logger.info(f'[Resume] from epoch {start_epoch}, best_f1={best_f1:.4f}')
    
    # Training loop
    epochs = tcfg.get('epochs', 200)
    patience = tcfg.get('patience', 40)
    
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        
        # LR schedule
        lr_mult = lr_for_epoch(config, epoch)
        for g in optimizer.param_groups:
            base_lr = g.get('_base_lr', g['lr'])
            if '_base_lr' not in g:
                g['_base_lr'] = g['lr']
            g['lr'] = base_lr * lr_mult
        
        if is_main_process():
            logger.info(f'[LR] epoch={epoch} mult={lr_mult:.6f} '
                       f'mars_lr={optimizer.param_groups[0]["lr"]:.2e} '
                       f'head_lr={optimizer.param_groups[2]["lr"]:.2e}')
        
        # Train
        train_avg = train_one_epoch(model, ddp_model, train_loader, optimizer,
                                     device, config, logger, epoch)
        
        # Eval (rank 0 only)
        torch.cuda.empty_cache()
        if is_main_process():
            val_results = evaluate_ddp(model, val_loader, device, config, logger)
            val_f1 = val_results['f1']
            
            logger.info(f'[Eval] e{epoch} val_f1={val_f1:.4f} '
                       f'prec={val_results["precision"]:.4f} rec={val_results["recall"]:.4f}')
            
            # History
            entry = {
                'epoch': epoch,
                'lr_mult': lr_mult,
                'loss': train_avg['loss'],
                'bce': train_avg['bce'],
                'density': train_avg['density'],
                'fp_penalty': train_avg['fp_penalty'],
                'shift': train_avg['shift'],
                'time_s': train_avg['time_s'],
                'steps_per_sec': train_avg['steps_per_sec'],
                'val_precision': val_results['precision'],
                'val_recall': val_results['recall'],
                'val_f1': val_f1,
                'val_mcc': val_results['mcc'],
                'val_gt_pairs': val_results['gt_pairs'],
                'val_pred_pairs': val_results['pred_pairs'],
                'val_n': val_results['n'],
            }
            history.append(entry)
            
            # Save history
            with open(output_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            # Plot training curves
            try:
                from symfold.train.train_v3 import plot_curves
                plot_curves(history, output_dir, logger)
            except Exception as e:
                logger.warning(f'[Plot] failed: {e}')
            
            # Best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_f1': best_f1,
                    'config': config,
                }, model_dir / 'best.pt')
                logger.info(f'[Save] New best! F1={best_f1:.4f} @ epoch {epoch}')
            else:
                patience_counter += 1
                logger.info(f'[Eval] no improve {patience_counter}/{patience}')
            
            # Periodic test eval (every 20 epochs)
            test_every = tcfg.get('test_eval_every', 20)
            if (epoch + 1) % test_every == 0 or epoch == epochs - 1:
                try:
                    test_records = build_records(
                        config['paths']['data_dir'], 'bprna-test',
                        max_len=tcfg.get('max_len_filter', 490))
                    test_dataset = PriFoldSymFlowDataset(test_records, augment=False)
                    test_loader = DataLoader(
                        test_dataset, batch_size=1, shuffle=False,
                        collate_fn=make_collate_fn(tokenizer),
                        num_workers=4, pin_memory=True)
                    test_results = evaluate_ddp(model, test_loader, device, config, logger, 'test')
                    logger.info(f'[Test] e{epoch} test_f1={test_results["f1"]:.4f} '
                               f'prec={test_results["precision"]:.4f} rec={test_results["recall"]:.4f}')
                    # Save to history entry
                    history[-1]['test_f1'] = test_results['f1']
                    history[-1]['test_precision'] = test_results['precision']
                    history[-1]['test_recall'] = test_results['recall']
                    with open(output_dir / 'history.json', 'w') as f:
                        json.dump(history, f, indent=2)
                except Exception as e:
                    logger.warning(f'[Test] eval failed: {e}')
            
            # Last checkpoint (for resume)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_f1,
                'patience_counter': patience_counter,
                'history': history,
                'config': config,
            }, model_dir / 'last.pt')
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f'[Stop] Early stopping at epoch {epoch}')
                break
        
        dist.barrier()
    
    if is_main_process():
        logger.info(f'[Done] Training complete. Best F1={best_f1:.4f}')
    
    cleanup_distributed()


if __name__ == '__main__':
    main()
