"""
Diagnostic script to compare learning rates for latent optimization.
Tests different learning rates to find optimal value.
"""

import sys
import os
# Add parent directory FIRST to prioritize local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add src directory for shared utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging as log

from config import Config, set_seed
from train import Trainer

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_learning_rates(lr_options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]):
    """
    Test different learning rates for latent optimization.
    """
    results = {}
    
    for lr in lr_options:
        log.info(f"\n{'='*60}")
        log.info(f"Testing learning rate: {lr}")
        log.info(f"{'='*60}")
        
        set_seed()
        cfg = Config()
        cfg.latent_lr = lr
        
        # Initialize trainer
    data_path = "/Users/emmaboehly/Documents/Vector/Metabarcoding/data/ecuador_training_data.csv"
    trainer = Trainer(
        cfg=cfg,
        data_path=data_path,
        )
        
        # Train initial MLP
        log.info("Training initial MLP...")
        for epoch in tqdm(range(min(50, cfg.epochs_init)), desc="Init training", leave=False):
            train_loss = trainer.train_epoch()
        
        # Perform latent optimization with this learning rate
        model = trainer.model
        model.train()
        device = model.device
        model.latent_embedding.weight.requires_grad = True
        
        # Re-initialize optimizer with new learning rate
        latent_optimizer = torch.optim.AdamW(
            [model.latent_embedding.weight], lr=lr
        )
        
        # Prepare H matrix
        H_coo = trainer.model.latent_solver.H.tocoo()
        H_indices = torch.LongTensor(np.vstack([H_coo.row, H_coo.col]))
        H_values = torch.FloatTensor(H_coo.data)
        H_torch = torch.sparse_coo_tensor(
            H_indices, H_values, size=H_coo.shape, device=device
        )
        
        lambda_smooth = float(cfg.latent_smooth_reg)
        lambda_norm = float(cfg.latent_norm_reg)
        
        # Track metrics
        losses = []
        latent_norms = []
        latent_changes = []
        
        prev_latent = model.latent_embedding.weight.data.clone()
        
        n_steps = 50
        for step in range(n_steps):
            total_loss = 0.0
            n_batches = 0
            
            for batch in trainer.train_loader:
                latent_optimizer.zero_grad()
                
                # Forward pass
                x = batch["input"].to(device)
                targets = batch["target"].to(device)
                bin_idx = batch["bin_idx"].to(device)
                mask = batch.get("mask")
                
                B, max_bins, n_feat = x.shape
                x_flat = x.view(B * max_bins, n_feat)
                bin_idx_flat = bin_idx.view(B * max_bins)
                
                output_flat = model(x_flat, bin_idx_flat)
                output = output_flat.view(B, max_bins)
                
                if mask is not None:
                    output = output.masked_fill(mask == 0, float('-inf'))
                
                # Compute CE loss per sample
                ce_loss = 0.0
                for b in range(B):
                    ce_loss += torch.nn.functional.cross_entropy(
                        output[b:b+1], targets[b:b+1]
                    )
                ce_loss = ce_loss / B
                
                # Smoothness regularization
                Z = model.latent_embedding.weight
                HZ = torch.sparse.mm(H_torch, Z)
                smooth_loss = lambda_smooth * torch.sum((Z - HZ) ** 2)
                
                # Norm regularization
                norm_loss = lambda_norm * torch.sum(Z ** 2)
                
                # Total loss
                loss = ce_loss + smooth_loss + norm_loss
                
                # Backward
                loss.backward()
                latent_optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Track metrics
            avg_loss = total_loss / max(1, n_batches)
            losses.append(avg_loss)
            
            curr_latent = model.latent_embedding.weight.data.clone()
            latent_norm = torch.norm(curr_latent).item()
            latent_change = torch.norm(curr_latent - prev_latent).item()
            
            latent_norms.append(latent_norm)
            latent_changes.append(latent_change)
            prev_latent = curr_latent
            
            if step % 10 == 0:
                log.info(f"  Step {step}: loss={avg_loss:.6f}, norm={latent_norm:.6f}, change={latent_change:.6f}")
        
        model.latent_embedding.weight.requires_grad = False
        
        # Evaluate validation loss
        val_loss = trainer.validate(split="val")
        
        results[lr] = {
            'losses': losses,
            'latent_norms': latent_norms,
            'latent_changes': latent_changes,
            'val_loss': val_loss,
        }
        
        log.info(f"Final validation loss: {val_loss:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss convergence
    ax = axes[0, 0]
    for lr, data in results.items():
        ax.plot(data['losses'], label=f'lr={lr:.0e}', marker='o', markersize=2)
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Loss Convergence by Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Latent norm progression
    ax = axes[0, 1]
    for lr, data in results.items():
        ax.plot(data['latent_norms'], label=f'lr={lr:.0e}', marker='o', markersize=2)
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Latent Norm ||Z||')
    ax.set_title('Latent Norm by Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Latent change per step
    ax = axes[1, 0]
    for lr, data in results.items():
        ax.plot(data['latent_changes'], label=f'lr={lr:.0e}', marker='o', markersize=2)
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Latent Change ||Z_t - Z_{t-1}||')
    ax.set_title('Latent Change by Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Final validation loss vs. learning rate
    ax = axes[1, 1]
    lrs = sorted(results.keys())
    val_losses = [results[lr]['val_loss'] for lr in lrs]
    ax.semilogx(lrs, val_losses, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs. Learning Rate')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1e-3, color='red', linestyle='--', label='Current default (1e-3)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    log.info(f"\nSaved learning rate analysis to figures/learning_rate_analysis.png")
    
    # Print summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY: Validation Loss by Learning Rate")
    log.info(f"{'='*60}")
    for lr in lrs:
        log.info(f"  lr={lr:.0e}: val_loss = {results[lr]['val_loss']:.6f}")
    
    # Find best learning rate
    best_lr = min(results.keys(), key=lambda k: results[k]['val_loss'])
    log.info(f"\nBest learning rate: {best_lr:.0e} (val_loss={results[best_lr]['val_loss']:.6f})")
    
    return results


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    results = test_learning_rates()
