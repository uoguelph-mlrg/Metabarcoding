"""
Final diagnostic: Why does latent hurt correlation while improving MAE?
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config, set_seed
from train import Trainer

def final_diagnostic():
    set_seed(14)
    cfg = Config()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../data/ecuador_training_data.csv")
    
    print("="*70)
    print("FINAL DIAGNOSTIC: Why Latent Hurts Correlation")
    print("="*70)
    
    # Create trainer and train MLP
    trainer = Trainer(cfg, data_path)
    for _ in range(50):
        trainer.train_epoch()
    
    # Get predictions
    intrinsic_vec = trainer.model.predict_MLP_only(trainer.train_loader_ordered, loss_mode="bin")
    y = trainer.data["train"]["y"].values
    
    # Convert to probability
    pred_before = torch.sigmoid(torch.tensor(intrinsic_vec)).numpy()
    
    # Solve latent
    latent_D = trainer.model.latent_solver.solve(y=y, intrinsic_vec=intrinsic_vec, loss_type="logistic")
    
    # Combined prediction
    bin_indices = trainer.data["train"]["X"].index.get_level_values("bin_uri").map(trainer.bin_index).values
    combined_logit = intrinsic_vec + latent_D[bin_indices]
    pred_after = torch.sigmoid(torch.tensor(combined_logit)).numpy()
    
    print("\n1. Looking at predictions for non-zero targets only:")
    nonzero_mask = y > 0
    print(f"   Non-zero observations: {nonzero_mask.sum()}")
    
    y_nz = y[nonzero_mask]
    pred_before_nz = pred_before[nonzero_mask]
    pred_after_nz = pred_after[nonzero_mask]
    
    print(f"\n   Target (non-zero): mean={y_nz.mean():.6f}, std={y_nz.std():.6f}")
    print(f"   Before latent:     mean={pred_before_nz.mean():.6f}, std={pred_before_nz.std():.6f}")
    print(f"   After latent:      mean={pred_after_nz.mean():.6f}, std={pred_after_nz.std():.6f}")
    
    corr_nz_before = np.corrcoef(pred_before_nz, y_nz)[0, 1]
    corr_nz_after = np.corrcoef(pred_after_nz, y_nz)[0, 1]
    print(f"\n   Correlation (non-zero only):")
    print(f"   Before latent: {corr_nz_before:.4f}")
    print(f"   After latent:  {corr_nz_after:.4f}")
    
    print("\n2. Looking at predictions for zero targets only:")
    zero_mask = y == 0
    print(f"   Zero observations: {zero_mask.sum()}")
    
    y_z = y[zero_mask]
    pred_before_z = pred_before[zero_mask]
    pred_after_z = pred_after[zero_mask]
    
    print(f"\n   Target (zero):     mean={y_z.mean():.6f}")
    print(f"   Before latent:     mean={pred_before_z.mean():.6f}")
    print(f"   After latent:      mean={pred_after_z.mean():.6f}")
    
    print("\n3. THE ISSUE - Variance compression:")
    print(f"   Target variance:       {y.var():.8f}")
    print(f"   Before latent var:     {pred_before.var():.8f}")
    print(f"   After latent var:      {pred_after.var():.8f}")
    print(f"   Variance ratio (after/before): {pred_after.var() / pred_before.var():.4f}")
    
    print("\n4. Checking if latent is bin-level aggregation issue:")
    # The latent is solving for a SINGLE value per bin
    # But within each bin, there can be multiple samples with different targets
    
    # Check variance within bins
    within_bin_var_target = []
    within_bin_var_pred_before = []
    within_bin_var_pred_after = []
    
    for b_uri, b_idx in trainer.bin_index.items():
        mask = bin_indices == b_idx
        if mask.sum() > 1:  # Need at least 2 observations
            within_bin_var_target.append(y[mask].var())
            within_bin_var_pred_before.append(pred_before[mask].var())
            within_bin_var_pred_after.append(pred_after[mask].var())
    
    print(f"\n   Within-bin variance analysis (bins with 2+ observations):")
    print(f"   Target:        mean within-bin var = {np.mean(within_bin_var_target):.8f}")
    print(f"   Before latent: mean within-bin var = {np.mean(within_bin_var_pred_before):.8f}")
    print(f"   After latent:  mean within-bin var = {np.mean(within_bin_var_pred_after):.8f}")
    
    print("\n5. Key insight - The latent adds the SAME offset to all samples in a bin:")
    # Example with a bin that has high within-bin target variance
    high_var_bins = [(b_uri, y[bin_indices == b_idx].var(), (bin_indices == b_idx).sum()) 
                     for b_uri, b_idx in trainer.bin_index.items() 
                     if (bin_indices == b_idx).sum() > 5]
    high_var_bins.sort(key=lambda x: -x[1])
    
    if high_var_bins:
        example_bin = high_var_bins[0]
        b_idx = trainer.bin_index[example_bin[0]]
        mask = bin_indices == b_idx
        
        print(f"\n   Example bin: {example_bin[0]} ({mask.sum()} observations)")
        print(f"   Targets:       {y[mask][:10]}")
        print(f"   Before latent: {pred_before[mask][:10]}")
        print(f"   After latent:  {pred_after[mask][:10]}")
        print(f"   Latent value:  {latent_D[b_idx]:.4f}")
        print(f"\n   ⚠️ Notice: After latent, all predictions in this bin are shifted by the SAME amount!")
        print(f"   This destroys the sample-specific variation that the MLP learned!")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The latent solver computes ONE value per bin (d_b) that is added to ALL 
observations in that bin. This means:

1. The latent can shift the mean prediction for a bin to match the mean target
2. BUT it cannot capture sample-specific variation within a bin

The MLP learns sample-specific patterns (e.g., "this sample has high reads, 
so predict higher abundance"). When we add a bin-level latent offset, we're 
adding the same value to all samples in that bin, which:

- Helps MAE because the mean is better
- HURTS correlation because we're adding noise to the sample-specific predictions

The MLP-only model with bin embeddings doesn't have this problem because:
- The embedding is an INPUT to the MLP
- The MLP can still learn sample-specific patterns ON TOP of the bin embedding
- There's no forced "same offset for all samples in bin" constraint

FIX: The latent should NOT be added after the MLP output. Instead, it should 
be an input (like the bin embedding) so the MLP can modulate it per-sample.
""")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Before vs After correlation
    ax = axes[0]
    ax.scatter(y, pred_before, alpha=0.1, s=1, label=f'Before (r={corr_nz_before:.3f})')
    ax.scatter(y, pred_after, alpha=0.1, s=1, label=f'After (r={corr_nz_after:.3f})')
    ax.set_xlabel('Target')
    ax.set_ylabel('Predicted')
    ax.set_title('Predictions vs Target')
    ax.legend()
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 0.05)
    
    # Plot 2: Histogram of predictions
    ax = axes[1]
    ax.hist(pred_before, bins=50, alpha=0.5, label='Before latent', density=True)
    ax.hist(pred_after, bins=50, alpha=0.5, label='After latent', density=True)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Predictions')
    ax.legend()
    ax.set_xlim(0, 0.01)
    
    # Plot 3: Latent values histogram
    ax = axes[2]
    ax.hist(latent_D, bins=50)
    ax.set_xlabel('Latent D value')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Latent Values')
    ax.axvline(0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'figures/latent_diagnostic.png'), dpi=150)
    print(f"\nSaved diagnostic plot to figures/latent_diagnostic.png")


if __name__ == "__main__":
    final_diagnostic()
