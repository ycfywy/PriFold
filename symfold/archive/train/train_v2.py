# -*- coding: utf-8 -*-
"""PriFold-SymFlow v2 训练脚本。

支持两种数据集模式：
  - dataset_mode = "bprna"        : train=bprna-train, val=bprna-val, test=[bprna-test]
  - dataset_mode = "rnastralign"  : train=rnastralign-train, val=rnastralign-val,
                                    test=[rnastralign-test, archiveii-test]
  与 PriFold 主线 train.sh / inference.sh 的设置完全对齐。

每 epoch 自动绘制 training_curves.png，每 eval_every 评估一次。
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

# H20 + PyTorch 2.x bf16 SIGFPE 防御 ——
# 1) faulthandler.enable()：进程崩溃时打印 Python traceback 到 stderr
# 2) 显式关 bf16 cuBLAS reduced precision，强制 fp32 路径
# 3) 关 TF32 reduced precision（保 fp32 GEMM）
faulthandler.enable()

import numpy as np
import torch

# Numerical safety: 阻断 bf16 cuBLAS 路径（H20 + libcublasLt.so.12 已知会触发 SIGFPE）
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
# TF32 仍可保留（速度收益大、数值无 SIGFPE 风险）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor                                # noqa: E402
from symfold.data import build_loader                             # noqa: E402
from symfold.metrics import contact_metrics                       # noqa: E402
from symfold.v2.model import PriFoldSymFlow_v2                    # noqa: E402


# Default test stages per dataset_mode（与 eval_v2.py DEFAULT_TEST_SETS 对齐）
DEFAULT_TEST_STAGES = {
    'bprna': ['bprna-test'],
    'rnastralign': ['rnastralign-test', 'archiveii-test'],
}


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
    return logging.getLogger('PriFold-SymFlow-v2')


def write_heartbeat(path: Path, payload: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(payload, f, default=str)
    except Exception:
        pass


def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def plot_curves(history: list, output_dir: Path, logger=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        if logger:
            logger.warning(f'[Plot] matplotlib unavailable: {exc}')
        return
    if not history:
        return
    epochs = [h['epoch'] for h in history]
    train_loss = [h.get('loss') for h in history]
    bce = [h.get('bce') for h in history]
    lr = [h.get('lr') for h in history]
    eval_epochs, val_f1, val_p, val_r, val_mcc = [], [], [], [], []
    for h in history:
        if 'val_f1' in h:
            eval_epochs.append(h['epoch'])
            val_f1.append(h['val_f1'])
            val_p.append(h['val_precision'])
            val_r.append(h['val_recall'])
            val_mcc.append(h['val_mcc'])

    # Discover test stages from history (every entry with `test_<stage>_f1`)
    test_stages: list[str] = []
    for h in history:
        for k in h.keys():
            if k.startswith('test_') and k.endswith('_f1'):
                stage = k[len('test_'):-len('_f1')]
                if stage not in test_stages:
                    test_stages.append(stage)

    have_test = len(test_stages) > 0
    n_rows = 3 if have_test else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows))

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, '-o', color='#d62728', label='train loss', ms=3)
    if any(b is not None for b in bce):
        ax.plot(epochs, bce, '--', color='#ff9896', label='bce', lw=1)
    ax.set_title('Training Loss'); ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[0, 1]
    if eval_epochs:
        ax.plot(eval_epochs, val_f1, '-o', color='#1f77b4', label='val F1', ms=4)
        ax.plot(eval_epochs, val_mcc, '-s', color='#2ca02c', label='val MCC', ms=3)
        best_i = int(max(range(len(val_f1)), key=lambda i: val_f1[i]))
        ax.scatter([eval_epochs[best_i]], [val_f1[best_i]], color='gold',
                   edgecolor='black', zorder=5, s=120,
                   label=f"best F1={val_f1[best_i]:.4f}@e{eval_epochs[best_i]}")
    ax.set_title('Validation F1 / MCC'); ax.set_xlabel('epoch'); ax.set_ylabel('score')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 0]
    if eval_epochs:
        ax.plot(eval_epochs, val_p, '-o', color='#9467bd', label='val precision', ms=3)
        ax.plot(eval_epochs, val_r, '-o', color='#8c564b', label='val recall', ms=3)
        ax.plot(eval_epochs, val_f1, '-o', color='#1f77b4', label='val F1', ms=3)
    ax.set_title('Validation P / R / F1'); ax.set_xlabel('epoch'); ax.set_ylabel('score')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, lr, '-o', color='#7f7f7f', ms=3)
    ax.set_title('Learning Rate'); ax.set_xlabel('epoch'); ax.set_ylabel('lr')
    ax.grid(alpha=0.3)

    # ---- Test curves（每个 test stage 一条线，F1 / MCC）----
    if have_test:
        ax = axes[2, 0]
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
        for ti, stage in enumerate(test_stages):
            xs, ys = [], []
            for h in history:
                v = h.get(f'test_{stage}_f1')
                if v is not None:
                    xs.append(h['epoch'])
                    ys.append(v)
            ax.plot(xs, ys, '-o', color=colors[ti % len(colors)],
                    label=f'{stage} F1', ms=5)
        ax.set_title('Test F1 (periodic eval)')
        ax.set_xlabel('epoch'); ax.set_ylabel('F1')
        ax.grid(alpha=0.3); ax.legend()

        ax = axes[2, 1]
        for ti, stage in enumerate(test_stages):
            xs, ys = [], []
            for h in history:
                v = h.get(f'test_{stage}_mcc')
                if v is not None:
                    xs.append(h['epoch'])
                    ys.append(v)
            ax.plot(xs, ys, '-s', color=colors[ti % len(colors)],
                    label=f'{stage} MCC', ms=4)
        ax.set_title('Test MCC (periodic eval)')
        ax.set_xlabel('epoch'); ax.set_ylabel('MCC')
        ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    out_path = output_dir / 'training_curves.png'
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    if logger:
        logger.info(f'[Plot] curves -> {out_path}')


# ============================================================
# Build / train / eval
# ============================================================

def build_model(config: dict, extractor) -> PriFoldSymFlow_v2:
    mc = config['model']
    return PriFoldSymFlow_v2(
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
        rho_0=mc.get('rho_0', 0.005),
        pos_weight_base=mc.get('pos_weight_base', 199.0),
        pos_weight_min=mc.get('pos_weight_min', 20.0),
        focal_gamma=mc.get('focal_gamma', 1.5),
        stack_weight=mc.get('stack_weight', 0.05),
        nc_weight=mc.get('nc_weight', 0.02),
        density_weight=mc.get('density_weight', 0.2),
        density_hint_dropout=mc.get('density_hint_dropout', 0.5),
    )


def lr_for_epoch(config: dict, epoch: int) -> float:
    tcfg = config['training']
    base_lr = tcfg.get('lr', 8e-5)
    warmup = max(tcfg.get('warmup_epochs', 1), 1)
    total = max(tcfg.get('epochs', 1), 1)
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _resolve_amp_dtype(config: dict):
    """Read amp dtype from config['training']['amp_dtype'] in {'fp32','bf16','fp16'}.
    Returns (enabled, dtype). bf16/fp16 → enabled True; fp32/missing → False.
    """
    name = str(config.get('training', {}).get('amp_dtype', 'fp32')).lower()
    if name in ('bf16', 'bfloat16'):
        return True, torch.bfloat16
    if name in ('fp16', 'half', 'float16'):
        return True, torch.float16
    return False, torch.float32


def train_one_epoch(model, loader, optimizer, device, config, logger, epoch,
                    heartbeat_path):
    model.train()
    totals = {'loss': 0.0, 'bce': 0.0, 'stack': 0.0, 'nc': 0.0, 'density': 0.0}
    n = 0
    t0 = time.time()
    amp_on, amp_dtype = _resolve_amp_dtype(config)
    # fp16 需要 GradScaler，bf16 不需要；fp32 完全不进 autocast
    scaler = torch.amp.GradScaler('cuda') if (amp_on and amp_dtype == torch.float16) else None
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if amp_on:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, loss_dict = model(batch)
        else:
            loss, loss_dict = model(batch)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'[Train] e{epoch} step={step} got NaN/Inf, skip')
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config['training'].get('grad_clip', 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config['training'].get('grad_clip', 1.0))
            optimizer.step()

        n += 1
        totals['loss'] += float(loss.item())
        for k in ('bce', 'stack', 'nc', 'density'):
            totals[k] += float(loss_dict.get(k, torch.tensor(0.0)).item())

        if step % config['training'].get('log_every', 20) == 0:
            logger.info(
                f"[Train] e{epoch} step={step}/{len(loader)} L={batch['set_max_len']} "
                f"loss={loss.item():.6f} bce={float(loss_dict['bce']):.5f} "
                f"den={float(loss_dict['density']):.5f}")
        if step % config['training'].get('heartbeat_every', 20) == 0:
            write_heartbeat(heartbeat_path, {
                'time': time.asctime(),
                'epoch': epoch, 'step': step,
                'loss': float(loss.item()),
                'gpu_mb': torch.cuda.memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0,
                'pid': os.getpid(),
            })

    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg['time_s'] = time.time() - t0
    logger.info(f'[Train] e{epoch} done {avg}')
    return avg


@torch.no_grad()
def evaluate(model, loader, device, config, logger, split_name: str):
    model.eval()
    scfg = config.get('sampling', {})
    amp_on, amp_dtype = _resolve_amp_dtype(config)
    merged = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0,
              'gt_pairs': 0.0, 'pred_pairs': 0.0}
    n_samples = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        if amp_on:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred, _ = model.sample(
                    batch,
                    num_steps=scfg.get('num_steps', 20),
                    num_samples_per_input=scfg.get('num_samples_per_input', 1),
                    density_guided=scfg.get('density_guided', True))
        else:
            pred, _ = model.sample(
                batch,
                num_steps=scfg.get('num_steps', 20),
                num_samples_per_input=scfg.get('num_samples_per_input', 1),
                density_guided=scfg.get('density_guided', True))
        m = contact_metrics(pred, batch['contact'], batch['length'])
        bs = m['n']
        n_samples += bs
        for k in merged:
            merged[k] += m[k] * bs
        if step % 20 == 0:
            logger.info(f"[Eval:{split_name}] step={step}/{len(loader)} "
                        f"L={batch['set_max_len']} F1={m['f1']:.4f}")
    out = {k: v / max(n_samples, 1) for k, v in merged.items()}
    out['n'] = n_samples
    out['time_s'] = time.time() - t0
    logger.info(
        f"[Eval:{split_name}] N={out['n']} F1={out['f1']:.4f} "
        f"P={out['precision']:.4f} R={out['recall']:.4f} MCC={out['mcc']:.4f} "
        f"time={out['time_s']:.1f}s")
    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)

    log_dir = Path(config['paths']['log_dir'])
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = log_dir / f"{config['task_name']}.heartbeat"

    def handle_signal(sig, frame):
        write_heartbeat(heartbeat_path.with_suffix('.signal'),
                        {'signal': sig, 'time': time.asctime()})
        raise SystemExit(128 + sig)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, handle_signal)

    logger.info('=' * 80)
    logger.info(f"PriFold-SymFlow v2 training: {config['task_name']}")
    amp_on, amp_dtype = _resolve_amp_dtype(config)
    logger.info(f"AMP: {'ON' if amp_on else 'OFF'} dtype={amp_dtype}")
    logger.info(json.dumps(config, indent=2, ensure_ascii=False))
    logger.info('=' * 80)

    torch.manual_seed(config.get('seed', 3407))
    np.random.seed(config.get('seed', 3407))
    torch.backends.cudnn.benchmark = False  # 变长输入下 benchmark 反而慢，且可能引入不稳
    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')

    # ---- LM ----
    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    # ---- Data ----
    dataset_mode = config['training'].get('dataset_mode', 'rnastralign')  # bprna | rnastralign
    train_stage = f'{dataset_mode}-train'
    val_stage = f'{dataset_mode}-val'

    train_loader = build_loader(train_stage, config, tokenizer, shuffle=True)
    val_loader = build_loader(val_stage, config, tokenizer, shuffle=False)
    logger.info(f'[Data] mode={dataset_mode} '
                f'train_batches={len(train_loader)} val_batches={len(val_loader)}')

    # ---- Periodic test eval setup ----
    test_eval_every = int(config['training'].get('test_eval_every', 10))
    test_stages_cfg = config['training'].get('test_stages')
    if test_stages_cfg:
        test_stages = [s.strip() for s in test_stages_cfg.split(',') if s.strip()] \
            if isinstance(test_stages_cfg, str) else list(test_stages_cfg)
    else:
        test_stages = list(DEFAULT_TEST_STAGES.get(dataset_mode, []))
    test_loaders = {}
    if test_eval_every > 0 and test_stages:
        for stage in test_stages:
            test_loaders[stage] = build_loader(stage, config, tokenizer, shuffle=False)
            logger.info(f'[Data] test_stage={stage} batches={len(test_loaders[stage])}')
        logger.info(f'[TestEval] every {test_eval_every} epochs on {list(test_loaders.keys())}')
    else:
        logger.info('[TestEval] disabled')

    # ---- Model ----
    model = build_model(config, extractor).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[Model] params total={total:,} trainable={trainable:,}')

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training'].get('lr', 8e-5),
        weight_decay=config['training'].get('weight_decay', 0.01))

    history = []
    best_f1 = -1.0
    last_path = model_dir / 'last.pt'
    if last_path.exists() and config['training'].get('auto_resume', True):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        history = ckpt.get('history', [])
        best_f1 = ckpt.get('best_f1', best_f1)
        start_epoch = ckpt.get('epoch', -1) + 1
        logger.info(f'[Resume] start_epoch={start_epoch} best_f1={best_f1:.4f}')
    else:
        start_epoch = 0

    patience_count = 0
    for epoch in range(start_epoch, config['training'].get('epochs', 60)):
        lr = lr_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        logger.info(f'[LR] epoch={epoch} lr={lr:.6g}')

        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, config, logger, epoch, heartbeat_path)
        entry = {'epoch': epoch, 'lr': lr, **train_stats}

        if (epoch + 1) % config['training'].get('eval_every', 1) == 0:
            val_stats = evaluate(model, val_loader, device, config, logger, 'val')
            entry.update({f'val_{k}': v for k, v in val_stats.items()})
            if val_stats['f1'] > best_f1:
                best_f1 = val_stats['f1']
                patience_count = 0
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'config': config, 'best_f1': best_f1},
                           model_dir / 'best.pt')
                logger.info(f'[Save] new best F1={best_f1:.4f}')
            else:
                patience_count += 1
                logger.info(f"[Eval] no improve {patience_count}/{config['training'].get('patience', 20)}")

        # ---- Periodic test eval (default: every 10 epochs) ----
        if test_loaders and test_eval_every > 0 and (epoch + 1) % test_eval_every == 0:
            logger.info(f'[TestEval] e{epoch} running test on {list(test_loaders.keys())}')
            test_results_this_epoch = {}
            for stage, tloader in test_loaders.items():
                tstats = evaluate(model, tloader, device, config, logger, f'test:{stage}')
                test_results_this_epoch[stage] = tstats
                # flatten into history entry for plot_curves discovery
                entry[f'test_{stage}_f1'] = tstats['f1']
                entry[f'test_{stage}_precision'] = tstats['precision']
                entry[f'test_{stage}_recall'] = tstats['recall']
                entry[f'test_{stage}_mcc'] = tstats['mcc']
                entry[f'test_{stage}_n'] = tstats['n']
            # 同时写入独立 test_eval_history.json，方便后续溯源
            try:
                ev_path = output_dir / 'test_eval_history.json'
                if ev_path.exists():
                    test_eval_history = json.loads(ev_path.read_text())
                else:
                    test_eval_history = []
                test_eval_history.append({
                    'epoch': epoch,
                    'lr': lr,
                    'results': test_results_this_epoch,
                })
                with open(ev_path, 'w') as f:
                    json.dump(test_eval_history, f, indent=2)
                logger.info(f'[TestEval] saved -> {ev_path}')
            except Exception as exc:
                logger.warning(f'[TestEval] save failed: {exc}')

        history.append(entry)
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        plot_curves(history, output_dir, logger)
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history, 'best_f1': best_f1, 'config': config,
        }, last_path)
        if (epoch + 1) % config['training'].get('save_every', 5) == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'config': config}, model_dir / f'epoch_{epoch+1:03d}.pt')
        if patience_count >= config['training'].get('patience', 20):
            logger.info('[EarlyStop] patience reached')
            break

    logger.info('Training finished')
    plot_curves(history, output_dir, logger)


if __name__ == '__main__':
    main()
