"""
Diagnostic script to check latent convergence in gradient-based solver.
Tests whether 5 gradient steps is sufficient for latent optimization.
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

def test_latent_convergence(n_steps_options=[5, 10, 20, 50, 100]):
    """
    Test how many gradient steps are needed for latent to converge.
    """
    set_seed()
    cfg = Config()
    
    # Initialize trainer
    data_path = "/Users/emmaboehly/Documents/Vector/Metabarcoding/data/ecuador_training_data.csv"
    trainer = Trainer(
        cfg=cfg,
        data_path=data_path,
    )
    
    # Train initial MLP (Phase 0)
    log.info("Training initial MLP...")
    for epoch in tqdm(range(cfg.epochs_init), desc="Init training"):
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate(split="val")
        if epoch % 10 == 0:
            log.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    # Test different numbers of gradient steps
    results = {}
    
    for n_steps in n_steps_options:
        log.info(f"\n{'='*60}")
        log.info(f"Testing with {n_steps} gradient steps")
        log.info(f"{'='*60}")
        
        # Reset latent to initial state (zeros to match baseline)
        trainer.model.latent_embedding.weight.data.zero_()
        
        # Store latent norms after each step
        latent_norms = []
        latent_changes = []
        losses = []
        
        prev_latent = trainer.model.latent_embedding.weight.data.clone()
        
        # Modified solve with tracking
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
        
        criterion = torch.nn.CrossEntropyLoss()
        
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
                trainer.latent_optimizer.step()
                
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
            
            if step % max(1, n_steps // 10) == 0:
                log.info(f"  Step {step}: loss={avg_loss:.6f}, norm={latent_norm:.6f}, change={latent_change:.6f}")
        
        model.latent_embedding.weight.requires_grad = False
        
        results[n_steps] = {
            'losses': losses,
            'latent_norms': latent_norms,
            'latent_changes': latent_changes,
        }
        
        # Evaluate validation loss after this latent update
        val_loss = trainer.validate(split="val")
        results[n_steps]['val_loss'] = val_loss
        log.info(f"Final validation loss: {val_loss:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss convergence
    ax = axes[0, 0]
    for n_steps, data in results.items():
        ax.plot(data['losses'], label=f'{n_steps} steps', marker='o', markersize=3)
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Latent norm progression
    ax = axes[0, 1]
    for n_steps, data in results.items():
        ax.plot(data['latent_norms'], label=f'{n_steps} steps', marker='o', markersize=3)
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Latent Norm ||Z||')
    ax.set_title('Latent Norm Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Latent change per step
    ax = axes[1, 0]
    for n_steps, data in results.items():
        ax.plot(data['latent_changes'], label=f'{n_steps} steps', marker='o', markersize=3)
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Latent Change ||Z_t - Z_{t-1}||')
    ax.set_title('Latent Change per Step (Convergence Indicator)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Final validation loss vs. n_steps
    ax = axes[1, 1]
    n_steps_list = sorted(results.keys())
    val_losses = [results[n]['val_loss'] for n in n_steps_list]
    ax.plot(n_steps_list, val_losses, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Gradient Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs. Number of Gradient Steps')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=5, color='red', linestyle='--', label='Current default (5)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/latent_convergence_analysis.png', dpi=300, bbox_inches='tight')
    log.info(f"\nSaved convergence analysis to figures/latent_convergence_analysis.png")
    
    # Print summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY: Validation Loss by Number of Gradient Steps")
    log.info(f"{'='*60}")
    for n_steps in n_steps_list:
        log.info(f"  {n_steps:3d} steps: val_loss = {results[n_steps]['val_loss']:.6f}")
    
    return results


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    results = test_latent_convergence(n_steps_options=[5, 10, 20, 50])
