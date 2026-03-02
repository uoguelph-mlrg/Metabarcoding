"""
Diagnostic script to check the effect of gradient masking.
Tests whether limiting gradient updates to batch bins + neighbors restricts learning.
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


def compare_gradient_masking():
    """
    Compare latent optimization with and without gradient masking.
    """
    set_seed()
    cfg = Config()
    
    # Test configurations
    configs = {
        'with_masking': {'use_masking': True},
        'without_masking': {'use_masking': False},
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        log.info(f"\n{'='*60}")
        log.info(f"Testing: {config_name}")
        log.info(f"{'='*60}")
        
        set_seed()  # Reset seed for fair comparison
        
        # Initialize trainer
    data_path = "/Users/emmaboehly/Documents/Vector/Metabarcoding/data/ecuador_training_data.csv"
    trainer = Trainer(
        cfg=cfg,
        data_path=data_path,
        )
        
        # Train initial MLP
        log.info("Training initial MLP...")
        for epoch in tqdm(range(min(50, cfg.epochs_init)), desc="Init training"):
            train_loss = trainer.train_epoch()
        
        # Perform one latent update with/without masking
        model = trainer.model
        model.train()
        device = model.device
        model.latent_embedding.weight.requires_grad = True
        
        # Prepare H matrix
        H_coo = trainer.model.latent_solver.H.tocoo()
        H_indices = torch.LongTensor(np.vstack([H_coo.row, H_coo.col]))
        H_values = torch.FloatTensor(H_coo.data)
        H_torch = torch.sparse_coo_tensor(
            H_indices, H_values, size=H_coo.shape, device=device
        )
        
        lambda_smooth = float(cfg.latent_smooth_reg)
        lambda_norm = float(cfg.latent_norm_reg)
        
        # Track gradient statistics
        grad_stats = {
            'sparsity': [],  # Fraction of zero gradients
            'max_grad': [],
            'mean_grad': [],
            'bins_updated': [],
        }
        
        n_steps = 20
        for step in range(n_steps):
            total_loss = 0.0
            n_batches = 0
            
            for batch in trainer.train_loader:
                trainer.latent_optimizer.zero_grad()
                
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
                
                # Track gradient before masking
                grad_before = model.latent_embedding.weight.grad.clone()
                
                # Apply masking if enabled
                if config_params['use_masking']:
                    if mask is not None:
                        mask_flat = mask.view(B * max_bins).bool()
                        bin_idx_masked = bin_idx_flat[mask_flat]
                    else:
                        bin_idx_masked = bin_idx_flat
                    
                    bins_in_batch = set(bin_idx_masked.detach().cpu().numpy().tolist())
                    bins_to_update = set(bins_in_batch)
                    
                    # Add neighbors
                    for b in bins_in_batch:
                        if 0 <= b < len(trainer.model.latent_solver.ng.neighbours):
                            bins_to_update.update(trainer.model.latent_solver.ng.neighbours[b])
                    
                    if bins_to_update:
                        grad = model.latent_embedding.weight.grad
                        mask_update = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
                        mask_update[list(bins_to_update)] = True
                        grad[~mask_update] = 0.0
                
                # Track gradient statistics
                grad_after = model.latent_embedding.weight.grad
                nonzero_grads = (grad_after.abs() > 1e-10).sum().item()
                total_grads = grad_after.numel()
                
                grad_stats['sparsity'].append(1.0 - nonzero_grads / total_grads)
                grad_stats['max_grad'].append(grad_after.abs().max().item())
                grad_stats['mean_grad'].append(grad_after.abs().mean().item())
                grad_stats['bins_updated'].append(nonzero_grads / cfg.latent_dim)
                
                trainer.latent_optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / max(1, n_batches)
            if step % 5 == 0:
                log.info(f"  Step {step}: loss={avg_loss:.6f}, "
                        f"grad_sparsity={np.mean(grad_stats['sparsity'][-n_batches:]):.3f}, "
                        f"bins_updated={np.mean(grad_stats['bins_updated'][-n_batches:]):.1f}")
        
        model.latent_embedding.weight.requires_grad = False
        
        # Evaluate validation loss
        val_loss = trainer.validate(split="val")
        results[config_name] = {
            'val_loss': val_loss,
            'grad_stats': grad_stats,
        }
        log.info(f"Final validation loss: {val_loss:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'with_masking': 'red', 'without_masking': 'blue'}
    
    # Plot 1: Gradient sparsity
    ax = axes[0, 0]
    for config_name, data in results.items():
        sparsity = data['grad_stats']['sparsity']
        ax.plot(sparsity, label=config_name, color=colors[config_name], alpha=0.3)
        # Smooth with moving average
        window = 10
        smooth = np.convolve(sparsity, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(sparsity)), smooth, color=colors[config_name], linewidth=2)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Gradient Sparsity (fraction of zeros)')
    ax.set_title('Gradient Sparsity Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Max gradient magnitude
    ax = axes[0, 1]
    for config_name, data in results.items():
        max_grad = data['grad_stats']['max_grad']
        ax.plot(max_grad, label=config_name, color=colors[config_name], alpha=0.3)
        # Smooth
        window = 10
        smooth = np.convolve(max_grad, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(max_grad)), smooth, color=colors[config_name], linewidth=2)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Max Gradient Magnitude')
    ax.set_title('Maximum Gradient Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Number of bins updated per batch
    ax = axes[1, 0]
    for config_name, data in results.items():
        bins_updated = data['grad_stats']['bins_updated']
        ax.plot(bins_updated, label=config_name, color=colors[config_name], alpha=0.3)
        # Smooth
        window = 10
        smooth = np.convolve(bins_updated, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(bins_updated)), smooth, color=colors[config_name], linewidth=2)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Number of Bins Updated')
    ax.set_title('Bins Updated per Batch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation loss comparison
    ax = axes[1, 1]
    config_names = list(results.keys())
    val_losses = [results[name]['val_loss'] for name in config_names]
    bars = ax.bar(config_names, val_losses, color=[colors[name] for name in config_names])
    ax.set_ylabel('Validation Loss')
    ax.set_title('Final Validation Loss Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/gradient_masking_analysis.png', dpi=300, bbox_inches='tight')
    log.info(f"\nSaved gradient masking analysis to figures/gradient_masking_analysis.png")
    
    # Print summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY: Effect of Gradient Masking")
    log.info(f"{'='*60}")
    for config_name in config_names:
        val_loss = results[config_name]['val_loss']
        avg_sparsity = np.mean(results[config_name]['grad_stats']['sparsity'])
        avg_bins_updated = np.mean(results[config_name]['grad_stats']['bins_updated'])
        log.info(f"{config_name}:")
        log.info(f"  Validation Loss: {val_loss:.6f}")
        log.info(f"  Avg Gradient Sparsity: {avg_sparsity:.3f}")
        log.info(f"  Avg Bins Updated: {avg_bins_updated:.1f}")
    
    return results


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    results = compare_gradient_masking()
