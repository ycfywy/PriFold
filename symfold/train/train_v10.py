# -*- coding: utf-8 -*-
"""PriFold v10 training: v9 + MARS unfreeze, single GPU, warm-start from v9 best.

v10 和 v9 唯一的区别：freeze_mars=false，MARS 全部参数可训练。
使用分层 LR：MARS 用 mars_lr(5e-6)，下游 head 用 lr(5e-4)。
从 v9 best.pt warm-start。

Usage:
  CUDA_VISIBLE_DEVICES=0 python symfold/train/train_v10.py symfold/config/v10/v10_ddp.json
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

faulthandler.enable()

import numpy as np
import torch
import torch.nn as nn

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


def build_model(cfg, extractor):
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


def warm_start(model, ckpt_path, logger):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    src = ckpt.get('model', ckpt)
    tgt = model.state_dict()
    loaded, skipped = 0, 0
    for k in tgt:
        if k in src and src[k].shape == tgt[k].shape:
            tgt[k] = src[k]
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(tgt)
    logger.info(f'[WarmStart] {ckpt_path}: loaded={loaded} skipped={skipped}')


def build_param_groups(model, head_lr, mars_lr, weight_decay):
    mars_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('extractor.'):
            mars_params.append(p)
        else:
            head_params.append(p)
    groups = [
        {'params': mars_params, 'lr': mars_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay},
    ]
    n_mars = sum(p.numel() for p in mars_params)
    n_head = sum(p.numel() for p in head_params)
    return groups, n_mars, n_head


def lr_schedule(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, device, config, logger, epoch):
    model.train()
    tcfg = config['training']
    amp_dtype = torch.bfloat16
    grad_accum = tcfg.get('gradient_accumulation_steps', 1)
    grad_clip = tcfg.get('grad_clip', 0.5)

    totals = {'loss': 0.0, 'bce': 0.0, 'fp_penalty': 0.0, 'shift': 0.0}
    n = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, ld = model(batch)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'e{epoch} step={step} NaN/Inf skip')
            optimizer.zero_grad(set_to_none=True)
            continue

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        n += 1
        totals['loss'] += float(loss)
        totals['bce'] += float(ld['bce'])
        totals['fp_penalty'] += float(ld['fp_penalty'])
        totals['shift'] += float(ld['shift'])

        if step % 20 == 0:
            logger.info(f'[Train] e{epoch} step={step}/{len(loader)} L={batch["set_max_len"]} '
                        f'loss={loss.item():.4f} bce={ld["bce"].item():.4f}')

    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg['time_s'] = time.time() - t0
    avg['steps_per_sec'] = n / avg['time_s']
    logger.info(f'[Train] e{epoch} done {avg}')
    return avg


def evaluate(model, loader, device, config):
    model.eval()
    scfg = config.get('sampling', {})
    amp_dtype = torch.bfloat16
    results = {'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0,
               'gt_pairs': 0, 'pred_pairs': 0, 'n': 0}
    with torch.no_grad():
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
            for k in results:
                if k != 'n':
                    results[k] += m[k] * bs
            results['n'] += bs
    n = results['n']
    if n > 0:
        for k in ['precision', 'recall', 'f1', 'mcc', 'gt_pairs', 'pred_pairs']:
            results[k] /= n
    return results


def plot_curves(history, output_dir, logger):
    """参考 v8 格式的 6 子图可视化：loss, val F1/MCC, val P/R/F1, LR, test F1, test MCC。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = [h['epoch'] for h in history]
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))

        # 1. Training Loss
        ax = axes[0, 0]
        ax.plot(epochs, [h['loss'] for h in history], 'r-', label='train loss')
        ax.plot(epochs, [h['bce'] for h in history], 'r--', alpha=0.5, label='bce')
        ax.set_title('Training Loss'); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
        ax.legend(); ax.grid(True)

        # 2. Validation F1 / MCC
        ax = axes[0, 1]
        val_f1 = [h['val_f1'] for h in history]
        val_mcc = [h['val_mcc'] for h in history]
        ax.plot(epochs, val_f1, 'b-o', markersize=2, label='val F1')
        ax.plot(epochs, val_mcc, 'g-o', markersize=2, label='val MCC')
        best_idx = max(range(len(val_f1)), key=lambda i: val_f1[i])
        ax.plot(epochs[best_idx], val_f1[best_idx], 'o', color='gold', markersize=10,
                label=f'best F1={val_f1[best_idx]:.4f}@e{epochs[best_idx]}')
        ax.set_title('Validation F1 / MCC'); ax.set_xlabel('epoch'); ax.set_ylabel('score')
        ax.legend(); ax.grid(True)

        # 3. Validation P / R / F1
        ax = axes[1, 0]
        ax.plot(epochs, [h['val_precision'] for h in history], 'm-o', markersize=2, label='val precision')
        ax.plot(epochs, [h['val_recall'] for h in history], color='brown', marker='o', markersize=2, label='val recall')
        ax.plot(epochs, val_f1, 'b-o', markersize=2, label='val F1')
        ax.set_title('Validation P / R / F1'); ax.set_xlabel('epoch'); ax.set_ylabel('score')
        ax.legend(); ax.grid(True)

        # 4. Learning Rate
        ax = axes[1, 1]
        lr_epochs = [h['epoch'] for h in history if 'lr_mars' in h]
        lr_mars_vals = [h['lr_mars'] for h in history if 'lr_mars' in h]
        lr_head_vals = [h['lr_head'] for h in history if 'lr_head' in h]
        if lr_mars_vals:
            ax.plot(lr_epochs, lr_mars_vals, 'gray', label='mars lr')
            ax.plot(lr_epochs, lr_head_vals, 'k-', label='head lr')
            ax.set_yscale('log')
        elif any('lr_mult' in h for h in history):
            mult_epochs = [h['epoch'] for h in history if 'lr_mult' in h]
            mult_vals = [h['lr_mult'] for h in history if 'lr_mult' in h]
            ax.plot(mult_epochs, mult_vals, 'gray', label='lr multiplier')
        ax.set_title('Learning Rate'); ax.set_xlabel('epoch'); ax.set_ylabel('lr')
        ax.legend(); ax.grid(True)

        # 5. Test F1 (periodic eval)
        ax = axes[2, 0]
        test_epochs = [h['epoch'] for h in history if 'test_f1' in h]
        test_f1s = [h['test_f1'] for h in history if 'test_f1' in h]
        if test_f1s:
            ax.plot(test_epochs, test_f1s, 'b-o', label='bprna-test F1')
        ax.set_title('Test F1 (periodic eval)'); ax.set_xlabel('epoch'); ax.set_ylabel('F1')
        ax.legend(); ax.grid(True)

        # 6. Test MCC (periodic eval)
        ax = axes[2, 1]
        test_mccs = [h.get('test_mcc', h.get('test_f1', None)) for h in history if 'test_f1' in h]
        if test_mccs and test_mccs[0] is not None:
            ax.plot(test_epochs, test_mccs, 'b-o', label='bprna-test MCC')
        ax.set_title('Test MCC (periodic eval)'); ax.set_xlabel('epoch'); ax.set_ylabel('MCC')
        ax.legend(); ax.grid(True)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'training_curves.png', dpi=120)
        plt.close()
    except Exception as e:
        logger.warning(f'[Plot] {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    tcfg = config['training']

    device = torch.device(config.get('device', 'cuda:0'))
    torch.manual_seed(config.get('seed', 3407))
    np.random.seed(config.get('seed', 3407))

    # dirs
    log_dir = Path(config['paths']['log_dir'])
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_dir / 'v10.log', mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('v10')
    logger.info(f'Config: {json.dumps(config, indent=2)}')

    # LM
    class A: pass
    a = A(); a.pretrained_lm_dir = config['paths']['pretrained_lm_dir']; a.model_scale = config['model']['mars_scale']
    extractor, tokenizer = get_extractor(a)

    # Model
    model = build_model(config, extractor).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[Model] Total={n_total/1e6:.1f}M Trainable={n_train/1e6:.1f}M')

    # Warm-start
    warm_path = tcfg.get('warm_start_from')
    last_path = model_dir / 'last.pt'
    if warm_path and not last_path.exists():
        warm_start(model, warm_path, logger)

    # Optimizer with layered LR
    head_lr = tcfg.get('lr', 5e-4)
    mars_lr = tcfg.get('mars_lr', 5e-6)
    wd = tcfg.get('weight_decay', 0.02)
    param_groups, n_mars, n_head = build_param_groups(model, head_lr, mars_lr, wd)
    logger.info(f'[Optim] mars={n_mars/1e6:.1f}M lr={mars_lr}, head={n_head/1e6:.1f}M lr={head_lr}')
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    # Data
    data_dir = config['paths']['data_dir']
    max_len = tcfg.get('max_len_filter', 490)
    train_recs = build_records(data_dir, f'{tcfg["dataset_mode"]}-train', max_len=max_len)
    val_recs = build_records(data_dir, f'{tcfg["dataset_mode"]}-val', max_len=max_len)
    train_ds = PriFoldSymFlowDataset(train_recs,
        augment=tcfg.get('augmentation', {}).get('enabled', False),
        select=tcfg.get('augmentation', {}).get('select', 0.2),
        replace=tcfg.get('augmentation', {}).get('replace', 0.4))
    val_ds = PriFoldSymFlowDataset(val_recs, augment=False)
    collate = make_collate_fn(tokenizer)

    lengths = [len(r.seq) for r in train_recs]
    train_sampler = LengthBucketBatchSampler(lengths, batch_size=tcfg.get('batch_size', 4),
        shuffle=True, seed=config.get('seed', 3407))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
        num_workers=tcfg.get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)

    # Resume or finetune
    history = []
    best_f1 = 0.0
    patience_cnt = 0
    start_epoch = 0
    finetune_from = tcfg.get('finetune_from')

    # Load existing history from output_dir if exists (for continuation on same experiment)
    history_path = output_dir / 'history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text())
        logger.info(f'[History] loaded {len(history)} entries from {history_path}')

    if last_path.exists() and tcfg.get('auto_resume', True):
        # Standard resume: load model + optimizer + epoch
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_f1 = ckpt.get('best_f1', 0.0)
        start_epoch = ckpt.get('epoch', -1) + 1
        patience_cnt = ckpt.get('patience_cnt', 0)
        logger.info(f'[Resume] epoch={start_epoch} best_f1={best_f1:.4f}')
    elif finetune_from:
        # Finetune mode: load model weights only, fresh optimizer, continue epoch numbering
        ckpt = torch.load(finetune_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        best_f1 = ckpt.get('best_f1', 0.0)
        # Continue from the last epoch in history
        if history:
            start_epoch = history[-1]['epoch'] + 1
        logger.info(f'[Finetune] from {finetune_from}, best_f1={best_f1:.4f}, start_epoch={start_epoch}, fresh optimizer')

    # Train
    n_epochs = tcfg.get('epochs', 100)
    end_epoch = start_epoch + n_epochs
    warmup = tcfg.get('warmup_epochs', 10)
    patience = tcfg.get('patience', 30)
    test_every = tcfg.get('test_eval_every', 20)

    # Set base LR for each param group (needed for cosine schedule after resume)
    base_lrs = [mars_lr, head_lr]
    for i, g in enumerate(optimizer.param_groups):
        g['_base_lr'] = base_lrs[i]

    logger.info(f'[Train] start_epoch={start_epoch} end_epoch={end_epoch} n_epochs={n_epochs}')

    for epoch in range(start_epoch, end_epoch):
        train_sampler.set_epoch(epoch)
        # LR schedule relative to this training phase
        local_epoch = epoch - start_epoch
        mult = lr_schedule(local_epoch, warmup, n_epochs)
        for g in optimizer.param_groups:
            g['lr'] = g['_base_lr'] * mult
        logger.info(f'[LR] e{epoch} mult={mult:.4f} mars_lr={optimizer.param_groups[0]["lr"]:.2e} head_lr={optimizer.param_groups[1]["lr"]:.2e}')

        train_avg = train_one_epoch(model, train_loader, optimizer, device, config, logger, epoch)
        val_res = evaluate(model, val_loader, device, config)
        logger.info(f'[Val] e{epoch} F1={val_res["f1"]:.4f} P={val_res["precision"]:.4f} R={val_res["recall"]:.4f}')

        entry = {'epoch': epoch, **train_avg,
                 'lr_mars': optimizer.param_groups[0]['lr'],
                 'lr_head': optimizer.param_groups[1]['lr'],
                 'val_f1': val_res['f1'], 'val_precision': val_res['precision'],
                 'val_recall': val_res['recall'], 'val_mcc': val_res['mcc'],
                 'val_gt_pairs': val_res['gt_pairs'], 'val_pred_pairs': val_res['pred_pairs'],
                 'val_n': val_res['n']}

        # test eval
        if (epoch + 1) % test_every == 0:
            test_recs = build_records(data_dir, 'bprna-test', max_len=max_len)
            test_ds = PriFoldSymFlowDataset(test_recs, augment=False)
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)
            test_res = evaluate(model, test_loader, device, config)
            entry['test_f1'] = test_res['f1']
            entry['test_precision'] = test_res['precision']
            entry['test_recall'] = test_res['recall']
            entry['test_mcc'] = test_res['mcc']
            logger.info(f'[Test] e{epoch} F1={test_res["f1"]:.4f} MCC={test_res["mcc"]:.4f}')

        history.append(entry)
        json.dump(history, open(output_dir / 'history.json', 'w'), indent=2)
        plot_curves(history, output_dir, logger)

        if val_res['f1'] > best_f1:
            best_f1 = val_res['f1']
            patience_cnt = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_f1': best_f1, 'config': config},
                       model_dir / 'best.pt')
            logger.info(f'[Save] best F1={best_f1:.4f}')
        else:
            patience_cnt += 1
            logger.info(f'[Eval] no improve {patience_cnt}/{patience}')

        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'history': history, 'best_f1': best_f1, 'patience_cnt': patience_cnt}, model_dir / 'last.pt')

        if patience_cnt >= patience:
            logger.info('[Stop] early stopping')
            break

    logger.info(f'[Done] best F1={best_f1:.4f}')


if __name__ == '__main__':
    main()
