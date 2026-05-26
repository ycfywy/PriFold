from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.metrics import contact_metrics
from symfold.model import PriFoldSymFlowModel


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def setup_logging(config: dict):
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{config['task_name']}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("PriFold-SymFlow")


def write_heartbeat(path: Path, payload: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, default=str)
    except Exception:
        pass


def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def build_model(config: dict, extractor) -> PriFoldSymFlowModel:
    mc = config["model"]
    return PriFoldSymFlowModel(
        extractor=extractor,
        freeze_mars=mc.get("freeze_mars", True),
        d_pair=mc.get("d_pair", 64),
        hidden_dim=mc.get("hidden_dim", 256),
        num_heads=mc.get("num_heads", 4),
        num_layers=mc.get("num_layers", 6),
        patch_size=mc.get("patch_size", 4),
        dropout=mc.get("dropout", 0.1),
        rho_0=mc.get("rho_0", 0.005),
        use_pos_bias=mc.get("use_pos_bias", True),
        output_refine=mc.get("output_refine", True),
        pos_weight_base=mc.get("pos_weight_base", 199.0),
        pos_weight_min=mc.get("pos_weight_min", 20.0),
        focal_gamma=mc.get("focal_gamma", 1.5),
        stack_weight=mc.get("stack_weight", 0.0),
        nc_weight=mc.get("nc_weight", 0.0),
        density_weight=mc.get("density_weight", 0.2),
    )


def lr_for_epoch(config: dict, epoch: int) -> float:
    tcfg = config["training"]
    base_lr = tcfg.get("lr", 8e-5)
    warmup = max(tcfg.get("warmup_epochs", 1), 1)
    total = max(tcfg.get("epochs", 1), 1)
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, device, config, logger, epoch, heartbeat_path):
    model.train()
    totals = {"loss": 0.0, "bce": 0.0, "stack": 0.0, "nc": 0.0, "density": 0.0}
    n = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = model(batch)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"[Train] e{epoch} step={step} got NaN/Inf, skip")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("grad_clip", 1.0))
        optimizer.step()

        n += 1
        totals["loss"] += float(loss.item())
        for key in ("bce", "stack", "nc", "density"):
            totals[key] += float(loss_dict.get(key, torch.tensor(0.0)).item())

        if step % config["training"].get("log_every", 10) == 0:
            logger.info(
                f"[Train] e{epoch} step={step}/{len(loader)} L={batch['set_max_len']} "
                f"loss={loss.item():.6f} bce={float(loss_dict['bce']):.5f} "
                f"den={float(loss_dict['density']):.5f}"
            )
        if step % config["training"].get("heartbeat_every", 10) == 0:
            write_heartbeat(heartbeat_path, {
                "time": time.asctime(),
                "epoch": epoch,
                "step": step,
                "loss": float(loss.item()),
                "gpu_mb": torch.cuda.memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0,
                "pid": os.getpid(),
            })
    avg = {key: val / max(n, 1) for key, val in totals.items()}
    avg["time_s"] = time.time() - t0
    logger.info(f"[Train] e{epoch} done {avg}")
    return avg


@torch.no_grad()
def evaluate(model, loader, device, config, logger, split_name: str):
    model.eval()
    scfg = config.get("sampling", {})
    merged = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "gt_pairs": 0.0, "pred_pairs": 0.0}
    n_batches = 0
    n_samples = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        pred, _ = model.sample(
            batch,
            num_steps=scfg.get("num_steps", 10),
            num_samples_per_input=scfg.get("num_samples_per_input", 1),
        )
        metrics = contact_metrics(pred, batch["contact"], batch["length"])
        bs = metrics["n"]
        n_samples += bs
        n_batches += 1
        for key in merged:
            merged[key] += metrics[key] * bs
        if step % 20 == 0:
            logger.info(f"[Eval:{split_name}] step={step}/{len(loader)} L={batch['set_max_len']} F1={metrics['f1']:.4f}")
    out = {key: val / max(n_samples, 1) for key, val in merged.items()}
    out["n"] = n_samples
    out["time_s"] = time.time() - t0
    logger.info(
        f"[Eval:{split_name}] N={out['n']} F1={out['f1']:.4f} "
        f"P={out['precision']:.4f} R={out['recall']:.4f} MCC={out['mcc']:.4f} "
        f"GTpairs={out['gt_pairs']:.1f} Predpairs={out['pred_pairs']:.1f} time={out['time_s']:.1f}s"
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=str(Path(__file__).resolve().parent / "config" / "prifold_symflow_v0.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    logger = setup_logging(config)

    log_dir = Path(config["paths"]["log_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    model_dir = Path(config["paths"]["model_save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = log_dir / f"{config['task_name']}.heartbeat"

    def handle_signal(sig, frame):
        write_heartbeat(heartbeat_path.with_suffix(".signal"), {"signal": sig, "time": time.asctime()})
        raise SystemExit(128 + sig)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, handle_signal)

    logger.info("=" * 80)
    logger.info(f"PriFold-SymFlow training: {config['task_name']}")
    logger.info(json.dumps(config, indent=2, ensure_ascii=False))
    logger.info("=" * 80)

    torch.manual_seed(config.get("seed", 3407))
    np.random.seed(config.get("seed", 3407))
    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config["paths"].get("pretrained_lm_dir", str(ROOT / "model"))
    lm_args.model_scale = config["model"].get("mars_scale", "lx")
    extractor, tokenizer = get_extractor(lm_args)

    train_loader = build_loader("train", config, tokenizer, shuffle=True)
    val_loader = build_loader("val", config, tokenizer, shuffle=False)
    logger.info(f"[Data] train_batches={len(train_loader)} val_batches={len(val_loader)}")

    model = build_model(config, extractor).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Model] params total={total:,} trainable={trainable:,}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["training"].get("lr", 8e-5), weight_decay=0.01)
    history = []
    best_f1 = -1.0
    last_path = model_dir / "last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt.get("history", [])
        best_f1 = ckpt.get("best_f1", best_f1)
        start_epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"[Resume] start_epoch={start_epoch} best_f1={best_f1:.4f}")
    else:
        start_epoch = 0

    patience_count = 0
    for epoch in range(start_epoch, config["training"].get("epochs", 5)):
        lr = lr_for_epoch(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        logger.info(f"[LR] epoch={epoch} lr={lr:.6g}")

        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, device, config, logger, epoch, heartbeat_path)
        entry = {"epoch": epoch, "lr": lr, **train_stats}

        if (epoch + 1) % config["training"].get("eval_every", 1) == 0:
            val_stats = evaluate(model, val_loader, device, config, logger, "val")
            entry.update({f"val_{k}": v for k, v in val_stats.items()})
            if val_stats["f1"] > best_f1:
                best_f1 = val_stats["f1"]
                patience_count = 0
                torch.save({"epoch": epoch, "model": model.state_dict(), "config": config, "best_f1": best_f1}, model_dir / "best.pt")
                logger.info(f"[Save] new best F1={best_f1:.4f}")
            else:
                patience_count += 1
                logger.info(f"[Eval] no improve {patience_count}/{config['training'].get('patience', 5)}")

        history.append(entry)
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_f1": best_f1,
            "config": config,
        }, last_path)
        if (epoch + 1) % config["training"].get("save_every", 1) == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": config}, model_dir / f"epoch_{epoch+1:03d}.pt")
        if patience_count >= config["training"].get("patience", 5):
            logger.info("[EarlyStop] patience reached")
            break

    logger.info("Training finished")


if __name__ == "__main__":
    main()
