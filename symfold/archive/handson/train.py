# -*- coding: utf-8 -*-
"""PriFold-SymFlow v4 训练脚本 (self-contained handson version).

用法:
    python symfold/handson/train.py symfold/handson/config.json
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

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path(__file__).resolve().parents[2]  # PriFold/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.handson.data import build_loader
from symfold.handson.metrics import contact_metrics
from symfold.handson.model import PriFoldSymFlow_v4


# ============================================================
# Utilities
# ============================================================

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def setup_logging(config: dict):
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{config['task_name']}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger('SymFlow-v4-handson')


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def lr_for_epoch(config: dict, epoch: int) -> float:
    """Linear warmup + cosine annealing."""
    tcfg = config['training']
    base_lr = tcfg.get('lr', 8e-5)
    warmup = max(tcfg.get('warmup_epochs', 1), 1)
    total = max(tcfg.get('epochs', 1), 1)
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def plot_curves(history: list, output_dir: Path, logger=None):
    """Plot training curves to PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not history:
        return
    epochs = [h['epoch'] for h in history]
    train_loss = [h.get('loss') for h in history]
    lr = [h.get('lr') for h in history]
    eval_epochs, val_f1, val_mcc = [], [], []
    for h in history:
        if 'val_f1' in h:
            eval_epochs.append(h['epoch'])
            val_f1.append(h['val_f1'])
            val_mcc.append(h['val_mcc'])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(epochs, train_loss, '-o', color='#d62728', ms=3)
    axes[0, 0].set_title('Training Loss'); axes[0, 0].grid(alpha=0.3)
    if eval_epochs:
        axes[0, 1].plot(eval_epochs, val_f1, '-o', color='#1f77b4', label='F1', ms=4)
        axes[0, 1].plot(eval_epochs, val_mcc, '-s', color='#2ca02c', label='MCC', ms=3)
        best_i = int(max(range(len(val_f1)), key=lambda i: val_f1[i]))
        axes[0, 1].scatter([eval_epochs[best_i]], [val_f1[best_i]], color='gold',
                           edgecolor='black', zorder=5, s=120,
                           label=f"best={val_f1[best_i]:.4f}")
        axes[0, 1].legend()
    axes[0, 1].set_title('Validation F1/MCC'); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(epochs, lr, '-o', color='#7f7f7f', ms=3)
    axes[1, 0].set_title('Learning Rate'); axes[1, 0].grid(alpha=0.3)
    # Test curves
    test_stages = []
    for h in history:
        for k in h:
            if k.startswith('test_') and k.endswith('_f1'):
                s = k[5:-3]
                if s not in test_stages:
                    test_stages.append(s)
    if test_stages:
        colors = ['#1f77b4', '#d62728', '#2ca02c']
        for ti, stage in enumerate(test_stages):
            xs = [h['epoch'] for h in history if f'test_{stage}_f1' in h]
            ys = [h[f'test_{stage}_f1'] for h in history if f'test_{stage}_f1' in h]
            axes[1, 1].plot(xs, ys, '-o', color=colors[ti % 3], label=stage, ms=4)
        axes[1, 1].legend()
    axes[1, 1].set_title('Test F1'); axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'training_curves.png', dpi=130)
    plt.close(fig)
    if logger:
        logger.info(f'[Plot] -> {output_dir}/training_curves.png')


# ============================================================
# Build model
# ============================================================

def build_model(config: dict, extractor) -> PriFoldSymFlow_v4:
    mc = config['model']
    return PriFoldSymFlow_v4(
        extractor=extractor,
        freeze_mars=mc.get('freeze_mars', True),
        mars_dim=mc.get('mars_dim', 1056),
        mars_n_attn_layers=mc.get('mars_n_attn_layers', 6),
        mars_n_heads=mc.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mc.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        mars_hidden_fusion_dim=mc.get('mars_hidden_fusion_dim', 64),
        use_seq_oh=mc.get('use_seq_oh', True),
        hidden_dim=mc.get('hidden_dim', 256),
        num_heads=mc.get('num_heads', 4),
        dim_head=mc.get('dim_head', 64),
        num_layers=mc.get('num_layers', 9),
        patch_size=mc.get('patch_size', 4),
        mars_emb_proj_dim=mc.get('mars_emb_proj_dim', 32),
        mars_attn_proj_dim=mc.get('mars_attn_proj_dim', 16),
        xt_emb_dim=mc.get('xt_emb_dim', 8),
        mlp_ratio=mc.get('mlp_ratio', 4),
        dropout=mc.get('dropout', 0.1),
        dilation_pattern=mc.get('dilation_pattern', None),
        tri_start_layer=mc.get('tri_start_layer', 6),
        tri_dim=mc.get('tri_dim', 64),
        refine_mid_ch=mc.get('refine_mid_ch', 16),
        cond_bias_zero_init=mc.get('cond_bias_zero_init', True),
        control_every=mc.get('control_every', 2),
        rho_0=mc.get('rho_0', 0.005),
        pos_weight_base=mc.get('pos_weight_base', 199.0),
        pos_weight_min=mc.get('pos_weight_min', 10.0),
        focal_gamma=mc.get('focal_gamma', 2.0),
        stack_weight=mc.get('stack_weight', 0.05),
        nc_weight=mc.get('nc_weight', 0.02),
        density_weight=mc.get('density_weight', 0.2),
        direct_weight=mc.get('direct_weight', 0.3),
        pair_count_weight=mc.get('pair_count_weight', 0.05),
        density_hint_dropout=mc.get('density_hint_dropout', 1.0),
        direct_score_weight=mc.get('direct_score_weight', 0.5),
    )


# ============================================================
# Train / Eval one epoch
# ============================================================

def train_one_epoch(model, loader, optimizer, device, config, logger, epoch):
    model.train()
    totals = {'loss': 0.0, 'bce': 0.0, 'stack': 0.0, 'nc': 0.0, 'density': 0.0}
    n = 0
    t0 = time.time()
    amp_on = config['training'].get('amp_dtype', 'fp32') in ('bf16', 'bfloat16')
    amp_dtype = torch.bfloat16 if amp_on else torch.float32

    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if amp_on:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, loss_dict = model(batch)
        else:
            loss, loss_dict = model(batch)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'[Train] e{epoch} step={step} NaN/Inf, skip')
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       config['training'].get('grad_clip', 1.0))
        optimizer.step()
        n += 1
        totals['loss'] += float(loss.item())
        for k in ('bce', 'stack', 'nc', 'density'):
            totals[k] += float(loss_dict.get(k, torch.tensor(0.0)).item())
        if step % config['training'].get('log_every', 50) == 0:
            logger.info(f"[Train] e{epoch} step={step}/{len(loader)} "
                        f"loss={loss.item():.6f} bce={float(loss_dict['bce']):.5f}")

    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg['time_s'] = time.time() - t0
    logger.info(f'[Train] e{epoch} done {avg}')
    return avg


@torch.no_grad()
def evaluate(model, loader, device, config, logger, split_name: str):
    model.eval()
    scfg = config.get('sampling', {})
    amp_on = config['training'].get('amp_dtype', 'fp32') in ('bf16', 'bfloat16')
    amp_dtype = torch.bfloat16 if amp_on else torch.float32
    merged = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0,
              'gt_pairs': 0.0, 'pred_pairs': 0.0}
    n_samples = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        kwargs = dict(
            num_steps=scfg.get('num_steps', 20),
            num_samples_per_input=scfg.get('num_samples_per_input', 1),
            density_guided=scfg.get('density_guided', False),
            projection_mode=scfg.get('projection_mode', 'score'),
            score_threshold=scfg.get('score_threshold', 0.5),
            default_budget_fraction=scfg.get('default_budget_fraction', 0.35),
        )
        if amp_on:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred, _ = model.sample(batch, **kwargs)
        else:
            pred, _ = model.sample(batch, **kwargs)
        m = contact_metrics(pred, batch['contact'], batch['length'])
        bs = m['n']
        n_samples += bs
        for k in merged:
            merged[k] += m[k] * bs
        if step % 20 == 0:
            logger.info(f"[Eval:{split_name}] step={step}/{len(loader)} F1={m['f1']:.4f}")
    out = {k: v / max(n_samples, 1) for k, v in merged.items()}
    out['n'] = n_samples
    out['time_s'] = time.time() - t0
    logger.info(f"[Eval:{split_name}] N={out['n']} F1={out['f1']:.4f} "
                f"P={out['precision']:.4f} R={out['recall']:.4f}")
    return out


# ============================================================
# Main
# ============================================================

DEFAULT_TEST_STAGES = {
    'bprna': ['bprna-test'],
    'rnastralign': ['rnastralign-test', 'archiveii-test'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)

    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 60)
    logger.info(f"SymFlow v4 handson training: {config['task_name']}")
    logger.info(json.dumps(config, indent=2, ensure_ascii=False))
    logger.info('=' * 60)

    torch.manual_seed(config.get('seed', 3407))
    np.random.seed(config.get('seed', 3407))
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')

    # ---- LM ----
    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    # ---- Data ----
    dataset_mode = config['training'].get('dataset_mode', 'rnastralign')
    train_loader = build_loader(f'{dataset_mode}-train', config, tokenizer, shuffle=True)
    val_loader = build_loader(f'{dataset_mode}-val', config, tokenizer, shuffle=False)
    logger.info(f'[Data] train={len(train_loader)} val={len(val_loader)} batches')

    # Test loaders
    test_eval_every = int(config['training'].get('test_eval_every', 10))
    test_stages = list(DEFAULT_TEST_STAGES.get(dataset_mode, []))
    test_loaders = {}
    for stage in test_stages:
        test_loaders[stage] = build_loader(stage, config, tokenizer, shuffle=False)

    # ---- Model ----
    model = build_model(config, extractor).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[Model] trainable params: {trainable:,}')

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training'].get('lr', 8e-5),
        weight_decay=config['training'].get('weight_decay', 0.01))

    # ---- Resume ----
    history = []
    best_f1 = -1.0
    start_epoch = 0
    last_path = model_dir / 'last.pt'
    if last_path.exists() and config['training'].get('auto_resume', True):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history = ckpt.get('history', [])
        best_f1 = ckpt.get('best_f1', best_f1)
        start_epoch = ckpt.get('epoch', -1) + 1
        logger.info(f'[Resume] epoch={start_epoch} best_f1={best_f1:.4f}')

    # ---- Training loop ----
    patience_count = 0
    for epoch in range(start_epoch, config['training'].get('epochs', 60)):
        lr = lr_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr

        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, device, config, logger, epoch)
        entry = {'epoch': epoch, 'lr': lr, **train_stats}

        # Val eval
        if (epoch + 1) % config['training'].get('eval_every', 2) == 0:
            val_stats = evaluate(model, val_loader, device, config, logger, 'val')
            entry.update({f'val_{k}': v for k, v in val_stats.items()})
            if val_stats['f1'] > best_f1:
                best_f1 = val_stats['f1']
                patience_count = 0
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'config': config, 'best_f1': best_f1}, model_dir / 'best.pt')
                logger.info(f'[Save] new best F1={best_f1:.4f}')
            else:
                patience_count += 1
                logger.info(f"[Eval] no improve {patience_count}/{config['training'].get('patience', 20)}")

        # Test eval
        if test_loaders and test_eval_every > 0 and (epoch + 1) % test_eval_every == 0:
            for stage, tloader in test_loaders.items():
                tstats = evaluate(model, tloader, device, config, logger, f'test:{stage}')
                entry[f'test_{stage}_f1'] = tstats['f1']
                entry[f'test_{stage}_precision'] = tstats['precision']
                entry[f'test_{stage}_recall'] = tstats['recall']

        # Save
        history.append(entry)
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        plot_curves(history, output_dir, logger)
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'history': history, 'best_f1': best_f1, 'config': config}, last_path)

        if patience_count >= config['training'].get('patience', 20):
            logger.info('[EarlyStop] patience reached')
            break

    logger.info('Training finished')


if __name__ == '__main__':
    main()
