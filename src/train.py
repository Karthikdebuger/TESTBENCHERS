"""
SemiCon AI Hackathon â€” Training Script
=======================================
AMP, EMA, cosine LR with warmup, seeded, config-hashed.
Per architecture spec Â§6: reproducible, OOM-safe, fully configurable.
"""

import sys
import os
import argparse
import yaml
import logging
import gc
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure imports work from both `python src/train.py` and `python -m src.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.dataset import get_dataloader
from src.losses import CompositeLoss
from src.augment import CurriculumScheduler, TrainAugmentation
from src.utils import (set_seed, get_device, compute_psnr, compute_ssim,
                       save_checkpoint, load_checkpoint, hash_config,
                       AverageMeter, EMA, setup_logging)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SemiCon Image Restoration â€” Training Script"
    )
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, or cpu')
    return parser.parse_args()


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, ema,
                    epoch, config, device):
    """
    Train for one epoch with OOM auto-halving, AMP, gradient clipping.

    Returns:
        dict of average metrics for the epoch
    """
    model.train()
    training_cfg = config.get('training', {})
    use_amp = training_cfg.get('amp', True) and device.type == 'cuda'
    clip_norm = training_cfg.get('gradient_clip_norm', 1.0)
    log_every = training_cfg.get('log_every', 50)

    loss_meter = AverageMeter()
    component_meters = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for step, batch in enumerate(pbar):
        try:
            noisy = batch['noisy'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            
            # Extract degradation params if available
            deg_params = None
            if 'deg_params' in batch and len(batch['deg_params']) > 0:
                deg_params = {k: v.to(device) for k, v in batch['deg_params'].items()}
                
        except Exception as e:
            logger.error(f"Error loading batch at step {step}: {e}")
            continue

        try:
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP (bfloat16 prevents NAFNet squaring overflow)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                output = model(noisy)
                restored = output['restored']
                # Forward full output and deg_params for auxiliary losses
                total_loss, loss_dict = criterion(
                    restored, gt, model_output=output, deg_params=deg_params
                )

            # Check for NaN / Inf loss before backward
            loss_val = total_loss.item()
            if not (loss_val == loss_val) or loss_val == float('inf'):  # NaN / Inf check
                logger.warning(f"NaN/Inf loss at step {step+1}, skipping batch completely")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            # EMA update
            if ema is not None:
                ema.update(model)

            # Track metrics
            loss_meter.update(loss_val, noisy.size(0))
            for k, v in loss_dict.items():
                if k not in component_meters:
                    component_meters[k] = AverageMeter()
                component_meters[k].update(v, noisy.size(0))

            # Periodic logging
            if (step + 1) % log_every == 0:
                comp_str = " ".join(
                    f"{k}={m.avg:.4f}" for k, m in component_meters.items()
                )
                logger.info(
                    f"Epoch {epoch + 1} Step [{step + 1}/{len(train_loader)}] "
                    f"Loss={loss_meter.avg:.4f} {comp_str} "
                    f"LR={optimizer.param_groups[0]['lr']:.2e}"
                )

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(
                    f"OOM at step {step}, batch_size={noisy.size(0)}. "
                    f"Clearing cache and skipping batch."
                )
                optimizer.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    return {
        'loss': loss_meter.avg,
        **{k: m.avg for k, m in component_meters.items()}
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    """
    Run validation and compute PSNR/SSIM metrics.
    """
    model.eval()
    training_cfg = config.get('training', {})
    use_amp = training_cfg.get('amp', True) and device.type == 'cuda'

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        try:
            noisy = batch['noisy'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                output = model(noisy)
                restored = output['restored']
                total_loss, _ = criterion(restored, gt)

            loss_meter.update(total_loss.item(), noisy.size(0))

            # Metrics (on clamped outputs)
            restored_clamped = torch.clamp(restored, 0.0, 1.0)
            gt_clamped = torch.clamp(gt, 0.0, 1.0)

            psnr_val = compute_psnr(restored_clamped, gt_clamped)
            ssim_val = compute_ssim(restored_clamped, gt_clamped)

            psnr_meter.update(psnr_val, noisy.size(0))
            ssim_meter.update(ssim_val, noisy.size(0))

        except Exception as e:
            logger.error(f"Error during validation: {e}")
            continue

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


def main():
    args = parse_args()

    # --- Setup logging ---
    os.makedirs('weights', exist_ok=True)
    setup_logging(log_file='weights/train.log')
    logger.info("=" * 60)
    logger.info("SemiCon Image Restoration â€” Training")
    logger.info("=" * 60)

    # --- Load config ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {args.config}: {e}")
        return

    cfg_hash = hash_config(config)
    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Config hash: {cfg_hash}")

    # --- Seed & device ---
    seed = config.get('seed', 42)
    set_seed(seed)
    device = get_device(args.device if args.device != 'auto'
                        else config.get('device', 'auto'))
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")

    # --- Build model ---
    model = build_model(config, device)

    # --- EMA ---
    ema_obj = None
    ema_cfg = config.get('training', {}).get('ema', {})
    if ema_cfg.get('enabled', False):
        ema_decay = ema_cfg.get('decay', 0.999)
        ema_obj = EMA(model, decay=ema_decay)
        logger.info(f"EMA enabled with decay={ema_decay}")

    # --- Optimizer ---
    opt_cfg = config.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg.get('lr', 1e-3),
        weight_decay=opt_cfg.get('weight_decay', 1e-4),
        betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
    )

    # --- LR Scheduler ---
    sched_cfg = config.get('scheduler', {})
    training_cfg = config.get('training', {})
    total_epochs = training_cfg.get('epochs', 200)
    warmup_epochs = sched_cfg.get('warmup_epochs', 5)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=sched_cfg.get('min_lr', 1e-6),
    )

    # --- Loss ---
    criterion = CompositeLoss(config).to(device)

    # --- AMP scaler ---
    use_amp = training_cfg.get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    logger.info(f"AMP: {'enabled' if use_amp else 'disabled'}")

    # --- Resume ---
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from {args.resume}")
        start_epoch = load_checkpoint(
            args.resume, model, optimizer,
            device=str(device)
        )
        logger.info(f"Resumed at epoch {start_epoch}")
        
        # FIX: Re-initialize EMA shadow weights using the loaded checkpoint!
        if ema_obj is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_obj.shadow[name] = param.data.clone()
                    
        # FIX: Re-initialize the scheduler so it decays perfectly in the remaining epochs
        remaining_epochs = total_epochs - start_epoch
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, remaining_epochs),
            eta_min=sched_cfg.get('min_lr', 1e-6),
        )

    # --- Data loaders ---
    data_cfg = config.get('data', {})
    aug_cfg = config.get('augmentation', {})

    # Build augmentation pipeline
    augment_fn = None
    if aug_cfg.get('synthetic', {}).get('enabled', False):
        augment_fn = TrainAugmentation(aug_cfg)

    train_loader = get_dataloader(config, split='train', augment_fn=augment_fn)
    val_loader = get_dataloader(config, split='val', augment_fn=None)

    if len(train_loader) == 0:
        logger.error("Training dataloader is empty! Check data directories.")
        return

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Curriculum scheduler ---
    curriculum_cfg = aug_cfg.get('curriculum', {})
    curriculum = None
    if curriculum_cfg.get('enabled', False):
        curriculum = CurriculumScheduler(
            warmup_epochs=curriculum_cfg.get('warmup_epochs', 20),
            full_epochs=curriculum_cfg.get('full_epochs', 50),
        )
        logger.info("Curriculum scheduling enabled")

    # --- Training loop ---
    save_every = training_cfg.get('save_every', 10)
    checkpoint_dir = training_cfg.get('checkpoint_dir', 'weights')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Training for {total_epochs} epochs starting from epoch {start_epoch}")
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        # Warmup LR (linear warmup for first N epochs)
        if epoch < warmup_epochs:
            warmup_lr = sched_cfg.get('min_lr', 1e-6) + \
                (opt_cfg.get('lr', 1e-3) - sched_cfg.get('min_lr', 1e-6)) * \
                (epoch / max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, ema_obj,
            epoch, config, device
        )

        # Step scheduler (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        # Validate
        if ema_obj is not None:
            ema_obj.apply_shadow(model)

        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device, config
        )

        if ema_obj is not None:
            ema_obj.restore(model)

        # Log
        logger.info(
            f"Epoch {epoch + 1}/{total_epochs} â€” "
            f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
            logger.info(f"â˜… New best PSNR: {best_psnr:.2f} dB")
            save_checkpoint(
                model, optimizer, epoch + 1, config,
                os.path.join(checkpoint_dir, 'model.pt'),
                ema_model=None,
            )

        # Periodic save
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, config,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'),
                ema_model=None,
            )

    # --- Final summary ---
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"  Total time:  {elapsed / 3600:.2f} hours")
    logger.info(f"  Best PSNR:   {best_psnr:.2f} dB")
    logger.info(f"  Config hash: {cfg_hash}")
    logger.info(f"  Best model:  {os.path.join(checkpoint_dir, 'model.pt')}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
