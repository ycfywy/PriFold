# -*- coding: utf-8 -*-
"""Train v12: Flow Matching + DiT for RNA Secondary Structure.

Usage:
  CUDA_VISIBLE_DEVICES=0 python symfold/train/train_v12.py symfold/config/v12/v12_flow_dit.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, LengthBucketBatchSampler, make_collate_fn
from symfold.metrics import contact_metrics
from symfold.v12.model import RNAFlowDiT


def build_model(cfg, extractor):
    mcfg = cfg['model']
    model = RNAFlowDiT(
        extractor=extractor,
        freeze_mars=mcfg.get('freeze_mars', True),
        mars_dim=mcfg.get('mars_dim', 1056),
        mars_n_attn_layers=mcfg.get('mars_n_attn_layers', 6),
        mars_n_heads=mcfg.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mcfg.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        hidden_dim=mcfg.get('hidden_dim', 256),
        num_heads=mcfg.get('num_heads', 8),
        dim_head=mcfg.get('dim_head', 32),
        num_layers=mcfg.get('num_layers', 8),
        ff_mult=mcfg.get('ff_mult', 4),
        dropout=mcfg.get('dropout', 0.2),
        drop_path=mcfg.get('drop_path', 0.15),
        use_rope=mcfg.get('use_rope', True),
        use_seq_oh=mcfg.get('use_seq_oh', True),
        max_len=mcfg.get('max_len', 512),
        use_gradient_checkpoint=mcfg.get('use_gradient_checkpoint', False),
        # Discrete flow
        rho_0=mcfg.get('rho_0', 0.005),
        loss_config=cfg.get('loss', None),
    )
    return model


def train_one_epoch(model, loader, optimizer, device, config, logger, epoch):
    model.train()
    tcfg = config['training']
    amp_dtype = torch.bfloat16
    grad_accum = tcfg.get('gradient_accumulation_steps', 1)
    grad_clip = tcfg.get('grad_clip', 1.0)

    # 离散 flow 损失分项
    comp_keys = ['bce', 'dice', 'pair_count', 'ratio_pen', 'stack', 'nc']
    totals = {'loss': 0.0}
    totals.update({k: 0.0 for k in comp_keys})
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
        for k in comp_keys:
            if k in ld:
                totals[k] += float(ld[k])

        if step % 50 == 0:
            logger.info(f'[Train] e{epoch} step={step}/{len(loader)} '
                        f'loss={loss.item():.4f} '
                        f'bce={float(ld.get("bce", 0)):.4f} '
                        f'dice={float(ld.get("dice", 0)):.4f} '
                        f'pc={float(ld.get("pair_count", 0)):.4f} '
                        f'nc={float(ld.get("nc", 0)):.4f}')

    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg['time_s'] = time.time() - t0
    avg['steps_per_sec'] = n / avg['time_s']
    logger.info(f'[Train] e{epoch} done: loss={avg["loss"]:.4f} '
                f'bce={avg["bce"]:.4f} dice={avg["dice"]:.4f} '
                f'time={avg["time_s"]:.0f}s')
    return avg


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    amp_dtype = torch.bfloat16
    scfg = config.get('sampling', {})
    # τ-leap 采样步数 (eval 可用较少步数提速)
    num_steps = scfg.get('eval_num_steps', scfg.get('num_steps', 20))
    threshold = scfg.get('threshold', 0.5)

    results = {'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0,
               'gt_pairs': 0, 'pred_pairs': 0, 'n': 0}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            pred, _ = model.sample(batch, num_steps=num_steps, threshold=threshold)

        m = contact_metrics(pred, batch['contact'], batch['length'])
        bs = pred.shape[0]
        for k in results:
            if k != 'n':
                results[k] += m[k] * bs
        results['n'] += bs
        # Free memory between samples
        del pred, batch
        torch.cuda.empty_cache()

    n = results['n']
    if n > 0:
        for k in ['precision', 'recall', 'f1', 'mcc', 'gt_pairs', 'pred_pairs']:
            results[k] /= n
    return results


def plot_curves(history, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = [h['epoch'] for h in history]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # (0,0) total train loss
        ax = axes[0, 0]
        ax.plot(epochs, [h.get('loss', 0) for h in history], 'r-', label='train total')
        ax.set_title('Training Loss (total)'); ax.set_xlabel('epoch'); ax.legend(); ax.grid(True)

        # (0,1) loss components
        ax = axes[0, 1]
        for key, color in [('bce', 'tab:blue'), ('dice', 'tab:green'),
                           ('pair_count', 'tab:orange'), ('ratio_pen', 'tab:red'),
                           ('stack', 'tab:purple'), ('nc', 'tab:brown')]:
            vals = [h.get(key, 0) for h in history]
            if any(abs(v) > 1e-9 for v in vals):
                ax.plot(epochs, vals, label=key, color=color)
        ax.set_title('Loss Components'); ax.set_xlabel('epoch'); ax.legend(fontsize=8); ax.grid(True)

        # (0,2) val F1 with best marker
        ax = axes[0, 2]
        val_f1 = [h.get('val_f1', 0) for h in history]
        ax.plot(epochs, val_f1, 'b-o', markersize=2, label='val F1')
        if any(v > 0 for v in val_f1):
            best_idx = max(range(len(val_f1)), key=lambda i: val_f1[i])
            ax.plot(epochs[best_idx], val_f1[best_idx], 'o', color='gold', markersize=10,
                    label=f'best={val_f1[best_idx]:.4f}@e{epochs[best_idx]}')
        ax.set_title('Validation F1'); ax.set_xlabel('epoch'); ax.legend(); ax.grid(True)

        # (1,0) val precision / recall
        ax = axes[1, 0]
        ax.plot(epochs, [h.get('val_precision', 0) for h in history], 'g-', label='precision')
        ax.plot(epochs, [h.get('val_recall', 0) for h in history], 'm-', label='recall')
        ax.set_title('Validation Precision / Recall'); ax.set_xlabel('epoch'); ax.legend(); ax.grid(True)

        # (1,1) learning rate
        ax = axes[1, 1]
        ax.plot(epochs, [h.get('lr', 0) for h in history], 'k-')
        ax.set_title('Learning Rate'); ax.set_xlabel('epoch'); ax.grid(True)

        # (1,2) test F1 (periodic) + pred/gt pairs ratio
        ax = axes[1, 2]
        test_epochs = [h['epoch'] for h in history if 'test_f1' in h]
        test_f1s = [h['test_f1'] for h in history if 'test_f1' in h]
        if test_f1s:
            ax.plot(test_epochs, test_f1s, 'b-o', label='test F1')
        ax.set_title('Test F1 (periodic)'); ax.set_xlabel('epoch'); ax.legend(); ax.grid(True)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'training_curves.png', dpi=120)
        plt.close()
    except Exception as e:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    tcfg = config['training']

    device = torch.device(config.get('device', 'cuda:0'))
    torch.manual_seed(config.get('seed', 3407))
    np.random.seed(config.get('seed', 3407))

    # Dirs
    log_dir = Path(config['paths']['log_dir'])
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_dir / 'v12.log', mode='a'), logging.StreamHandler()])
    logger = logging.getLogger('v12')
    logger.info(f'Config: {json.dumps(config, indent=2)}')

    # LM
    class A: pass
    a = A()
    a.pretrained_lm_dir = config['paths']['pretrained_lm_dir']
    a.model_scale = 'lx'
    extractor, tokenizer = get_extractor(a)

    # Model
    model = build_model(config, extractor).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[Model] Total={n_total/1e6:.1f}M Trainable={n_train/1e6:.1f}M')

    # Optimizer
    lr = tcfg.get('lr', 3e-4)
    wd = tcfg.get('weight_decay', 0.01)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=wd, betas=(0.9, 0.999))

    # Data
    data_dir = config['paths']['data_dir']
    max_len = tcfg.get('max_len_filter', 490)
    train_recs = build_records(data_dir, f'{tcfg["dataset_mode"]}-train', max_len=max_len)
    val_recs = build_records(data_dir, f'{tcfg["dataset_mode"]}-val', max_len=max_len)
    train_ds = PriFoldSymFlowDataset(train_recs, augment=False)
    val_ds = PriFoldSymFlowDataset(val_recs, augment=False)
    collate = make_collate_fn(tokenizer)

    lengths = [len(r.seq) for r in train_recs]
    train_sampler = LengthBucketBatchSampler(
        lengths, batch_size=tcfg.get('batch_size', 4),
        shuffle=True, seed=config.get('seed', 3407))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
        num_workers=tcfg.get('num_workers', 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)

    # Scheduler: cosine with warmup
    n_epochs = tcfg.get('epochs', 100)
    warmup = tcfg.get('warmup_epochs', 5)

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(n_epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    history = []
    best_f1 = 0.0
    patience_cnt = 0
    start_epoch = 0
    history_path = output_dir / 'history.json'

    if history_path.exists():
        history = json.loads(history_path.read_text())
        logger.info(f'[History] loaded {len(history)} entries')

    last_path = model_dir / 'last.pt'
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        best_f1 = ckpt.get('best_f1', 0.0)
        start_epoch = ckpt.get('epoch', -1) + 1
        patience_cnt = ckpt.get('patience_cnt', 0)
        logger.info(f'[Resume] epoch={start_epoch} best_f1={best_f1:.4f}')

    patience = tcfg.get('patience', 30)
    test_every = tcfg.get('test_eval_every', 10)

    logger.info(f'[Train] start_epoch={start_epoch} end_epoch={n_epochs}')

    for epoch in range(start_epoch, n_epochs):
        train_sampler.set_epoch(epoch)
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info(f'[LR] e{epoch} lr={cur_lr:.2e}')

        train_avg = train_one_epoch(model, train_loader, optimizer, device, config, logger, epoch)
        torch.cuda.empty_cache()

        # Validation
        val_res = evaluate(model, val_loader, device, config)
        torch.cuda.empty_cache()
        logger.info(f'[Val] e{epoch} F1={val_res["f1"]:.4f} P={val_res["precision"]:.4f} R={val_res["recall"]:.4f}')

        entry = {
            'epoch': epoch, **train_avg,
            'lr': cur_lr,
            'val_f1': val_res['f1'], 'val_precision': val_res['precision'],
            'val_recall': val_res['recall'], 'val_mcc': val_res['mcc'],
            'val_gt_pairs': val_res['gt_pairs'], 'val_pred_pairs': val_res['pred_pairs'],
            'val_n': val_res['n'],
        }

        # Test eval
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
        plot_curves(history, output_dir)

        # Save best
        if val_res['f1'] > best_f1:
            best_f1 = val_res['f1']
            patience_cnt = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'best_f1': best_f1, 'config': config},
                       model_dir / 'best.pt')
            logger.info(f'[Save] best F1={best_f1:.4f}')
        else:
            patience_cnt += 1
            logger.info(f'[Eval] no improve {patience_cnt}/{patience}')

        scheduler.step()

        # Save last
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'history': history, 'best_f1': best_f1, 'patience_cnt': patience_cnt},
                   model_dir / 'last.pt')

        if patience_cnt >= patience:
            logger.info('[Stop] early stopping')
            break

    logger.info(f'[Done] best F1={best_f1:.4f}')


if __name__ == '__main__':
    main()
