"""
Diagnostic script to verify loss computation correctness and check for data leakage.
"""

import sys
import os
# Add parent directory FIRST to prioritize local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add src directory for shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
import logging as log

from config import Config, set_seed
from train import Trainer

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def verify_loss_computation():
    """
    Verify that loss computation is correct and consistent.
    """
    set_seed()
    cfg = Config()
    
    log.info(f"\n{'='*60}")
    log.info("Verifying Loss Computation")
    log.info(f"{'='*60}")
    
    # Initialize trainer
    trainer = Trainer(
        cfg=cfg,
        data_path="../../../data/ecuador_training_data.csv",
    )
    
    # Get a single batch
    trainer.model.eval()
    batch = next(iter(trainer.train_loader))
    
    device = trainer.device
    x = batch["input"].to(device)
    targets = batch["target"].to(device)
    bin_idx = batch["bin_idx"].to(device)
    mask = batch.get("mask")
    
    B, max_bins, n_feat = x.shape
    x_flat = x.view(B * max_bins, n_feat)
    bin_idx_flat = bin_idx.view(B * max_bins)
    
    # Forward pass
    with torch.no_grad():
        output_flat = trainer.model(x_flat, bin_idx_flat)
        output = output_flat.view(B, max_bins)
        
        if mask is not None:
            output = output.masked_fill(mask == 0, float('-inf'))
    
    # Method 1: Using trainer's criterion
    loss1 = trainer.criterion(output, targets, mask).item()
    log.info(f"Loss (trainer criterion): {loss1:.6f}")
    
    # Method 2: Manual CE computation per sample
    loss2 = 0.0
    for b in range(B):
        loss2 += F.cross_entropy(output[b:b+1], targets[b:b+1]).item()
    loss2 = loss2 / B
    log.info(f"Loss (manual per-sample CE): {loss2:.6f}")
    
    # Method 3: Direct CE on flattened (with masking)
    if mask is not None:
        mask_flat = mask.view(-1).bool()
        output_masked = output.view(-1)[mask_flat]
        targets_masked = targets.view(-1)[mask_flat]
    else:
        output_masked = output.view(-1)
        targets_masked = targets.view(-1)
    
    loss3 = F.cross_entropy(output_masked.unsqueeze(0), targets_masked.unsqueeze(0)).item()
    log.info(f"Loss (flattened CE): {loss3:.6f}")
    
    # Check consistency
    if abs(loss1 - loss2) < 1e-6:
        log.info("✓ Trainer criterion matches manual computation")
    else:
        log.warning(f"✗ Loss mismatch! Difference: {abs(loss1 - loss2):.8f}")
    
    # Verify predictions sum to ~1 per sample (after softmax)
    probs = F.softmax(output, dim=-1)
    log.info(f"\nPrediction checks:")
    log.info(f"  Output shape: {output.shape}")
    log.info(f"  Targets shape: {targets.shape}")
    log.info(f"  Probs shape: {probs.shape}")
    
    for b in range(min(3, B)):
        if mask is not None:
            valid_mask = mask[b].bool()
            prob_sum = probs[b][valid_mask].sum().item()
        else:
            prob_sum = probs[b].sum().item()
        
        log.info(f"  Sample {b}: prob sum = {prob_sum:.6f} (should be ~1.0)")
    
    # Verify targets sum to 1 per sample
    log.info(f"\nTarget checks:")
    for b in range(min(3, B)):
        if mask is not None:
            valid_mask = mask[b].bool()
            target_sum = targets[b][valid_mask].sum().item()
        else:
            target_sum = targets[b].sum().item()
        
        log.info(f"  Sample {b}: target sum = {target_sum:.6f} (should be ~1.0)")
    
    return True


def check_data_leakage():
    """
    Check for potential data leakage between train/val/test splits.
    """
    set_seed()
    cfg = Config()
    
    log.info(f"\n{'='*60}")
    log.info("Checking for Data Leakage")
    log.info(f"{'='*60}")
    
    # Initialize trainer
    trainer = Trainer(
        cfg=cfg,
        data_path="../../../data/ecuador_training_data.csv",
    )
    
    # Extract sample indices from each split
    train_samples = set()
    val_samples = set()
    test_samples = set()
    
    for batch in trainer.train_loader:
        sample_idx = batch["sample_idx"]
        if isinstance(sample_idx, torch.Tensor):
            train_samples.update(sample_idx.numpy().tolist())
        else:
            train_samples.update(sample_idx.tolist())
    
    for batch in trainer.val_loader:
        sample_idx = batch["sample_idx"]
        if isinstance(sample_idx, torch.Tensor):
            val_samples.update(sample_idx.numpy().tolist())
        else:
            val_samples.update(sample_idx.tolist())
    
    for batch in trainer.test_loader:
        sample_idx = batch["sample_idx"]
        if isinstance(sample_idx, torch.Tensor):
            test_samples.update(sample_idx.numpy().tolist())
        else:
            test_samples.update(sample_idx.tolist())
    
    log.info(f"\nSplit sizes:")
    log.info(f"  Train: {len(train_samples)} unique samples")
    log.info(f"  Val: {len(val_samples)} unique samples")
    log.info(f"  Test: {len(test_samples)} unique samples")
    
    # Check overlaps
    train_val_overlap = train_samples & val_samples
    train_test_overlap = train_samples & test_samples
    val_test_overlap = val_samples & test_samples
    
    log.info(f"\nOverlap checks:")
    if len(train_val_overlap) == 0:
        log.info("  ✓ No overlap between train and val")
    else:
        log.warning(f"  ✗ Train-Val overlap: {len(train_val_overlap)} samples")
    
    if len(train_test_overlap) == 0:
        log.info("  ✓ No overlap between train and test")
    else:
        log.warning(f"  ✗ Train-Test overlap: {len(train_test_overlap)} samples")
    
    if len(val_test_overlap) == 0:
        log.info("  ✓ No overlap between val and test")
    else:
        log.warning(f"  ✗ Val-Test overlap: {len(val_test_overlap)} samples")
    
    # Check bin distributions
    train_bins = set()
    val_bins = set()
    test_bins = set()
    
    for batch in trainer.train_loader:
        bin_idx = batch["bin_idx"]
        if isinstance(bin_idx, torch.Tensor):
            bin_idx = bin_idx.numpy()
        train_bins.update(bin_idx.flatten().tolist())
    
    for batch in trainer.val_loader:
        bin_idx = batch["bin_idx"]
        if isinstance(bin_idx, torch.Tensor):
            bin_idx = bin_idx.numpy()
        val_bins.update(bin_idx.flatten().tolist())
    
    for batch in trainer.test_loader:
        bin_idx = batch["bin_idx"]
        if isinstance(bin_idx, torch.Tensor):
            bin_idx = bin_idx.numpy()
        test_bins.update(bin_idx.flatten().tolist())
    
    log.info(f"\nBIN distribution:")
    log.info(f"  Train: {len(train_bins)} unique BINs")
    log.info(f"  Val: {len(val_bins)} unique BINs")
    log.info(f"  Test: {len(test_bins)} unique BINs")
    
    # Check if val/test bins exist in training (this is expected and OK)
    val_bins_in_train = val_bins & train_bins
    test_bins_in_train = test_bins & train_bins
    
    log.info(f"\nBIN overlap (expected for latent factors):")
    log.info(f"  Val BINs in train: {len(val_bins_in_train)}/{len(val_bins)} ({100*len(val_bins_in_train)/max(1,len(val_bins)):.1f}%)")
    log.info(f"  Test BINs in train: {len(test_bins_in_train)}/{len(test_bins)} ({100*len(test_bins_in_train)/max(1,len(test_bins)):.1f}%)")
    
    return train_val_overlap, train_test_overlap, val_test_overlap


def check_gradient_flow():
    """
    Check if gradients are flowing properly through the model.
    """
    set_seed()
    cfg = Config()
    
    log.info(f"\n{'='*60}")
    log.info("Checking Gradient Flow")
    log.info(f"{'='*60}")
    
    # Initialize trainer
    trainer = Trainer(
        cfg=cfg,
        data_path="../../../data/ecuador_training_data.csv",
    )
    
    # Get a single batch
    trainer.model.train()
    batch = next(iter(trainer.train_loader))
    
    device = trainer.device
    x = batch["input"].to(device)
    targets = batch["target"].to(device)
    bin_idx = batch["bin_idx"].to(device)
    mask = batch.get("mask")
    
    B, max_bins, n_feat = x.shape
    x_flat = x.view(B * max_bins, n_feat)
    bin_idx_flat = bin_idx.view(B * max_bins)
    
    # Enable gradients for latent
    trainer.model.latent_embedding.weight.requires_grad = True
    
    # Forward pass
    output_flat = trainer.model(x_flat, bin_idx_flat)
    output = output_flat.view(B, max_bins)
    
    if mask is not None:
        output = output.masked_fill(mask == 0, float('-inf'))
    
    # Compute loss
    loss = trainer.criterion(output, targets, mask)
    
    # Backward
    loss.backward()
    
    # Check gradients
    log.info(f"\nGradient statistics:")
    
    # MLP gradients
    mlp_grads = []
    for name, param in trainer.model.mlp.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()
            grad_mean = param.grad.abs().mean().item()
            mlp_grads.append(grad_norm)
            log.info(f"  MLP {name}: norm={grad_norm:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}")
    
    # Latent gradients
    latent_grad = trainer.model.latent_embedding.weight.grad
    if latent_grad is not None:
        grad_norm = latent_grad.norm().item()
        grad_max = latent_grad.abs().max().item()
        grad_mean = latent_grad.abs().mean().item()
        nonzero = (latent_grad.abs() > 1e-10).sum().item()
        total = latent_grad.numel()
        
        log.info(f"\n  Latent: norm={grad_norm:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f}")
        log.info(f"  Latent nonzero: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
        
        # Check which bins have gradients
        bin_grads = latent_grad.abs().sum(dim=1)  # Sum across latent_dim
        bins_with_grad = (bin_grads > 1e-10).sum().item()
        log.info(f"  BINs with gradients: {bins_with_grad}/{len(bin_grads)} ({100*bins_with_grad/len(bin_grads):.1f}%)")
        
        # Compare to bins in batch
        unique_bins = torch.unique(bin_idx_flat)
        bins_in_batch = len(unique_bins)
        log.info(f"  BINs in batch: {bins_in_batch}")
        
        # Check if gradient exists for bins in batch
        for b in unique_bins[:10]:  # Check first 10
            bin_grad_norm = bin_grads[b].item()
            log.info(f"    BIN {b}: grad_norm={bin_grad_norm:.6f}")
    else:
        log.warning("  ✗ No latent gradients!")
    
    trainer.model.latent_embedding.weight.requires_grad = False
    
    return True


def run_all_diagnostics():
    """
    Run all diagnostic checks.
    """
    log.info("="*80)
    log.info("COMPREHENSIVE DIAGNOSTIC SUITE")
    log.info("="*80)
    
    try:
        verify_loss_computation()
    except Exception as e:
        log.error(f"Loss verification failed: {e}")
    
    try:
        check_data_leakage()
    except Exception as e:
        log.error(f"Data leakage check failed: {e}")
    
    try:
        check_gradient_flow()
    except Exception as e:
        log.error(f"Gradient flow check failed: {e}")
    
    log.info("\n" + "="*80)
    log.info("DIAGNOSTIC SUITE COMPLETE")
    log.info("="*80)


if __name__ == "__main__":
    run_all_diagnostics()
