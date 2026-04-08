"""
Refined Analysis: Understanding the Latent as a Bin-Specific Amplification Factor

The latent d_b is added in LOGIT space:
    logit(p) = m(s,b) + d_b
    
This means:
    log(p / (1-p)) = m(s,b) + d_b
    p / (1-p) = exp(m(s,b)) * exp(d_b)
    odds = base_odds * amplification_factor
    
So exp(d_b) is a MULTIPLICATIVE factor on the ODDS.

For relative abundance (which is close to 0), odds ≈ p, so:
    p ≈ base_p * exp(d_b)
    
exp(d_b) represents the bin-specific amplification factor:
- exp(d_b) > 1: This bin tends to amplify more (appears more abundant)
- exp(d_b) < 1: This bin tends to amplify less (appears less abundant)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config, set_seed
from train import Trainer

def analyze_amplification():
    set_seed(14)
    cfg = Config()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.data_path = os.path.join(script_dir, "../../data/ecuador_training_data.csv")
    
    print("="*70)
    print("REFINED ANALYSIS: Latent as Bin-Specific Amplification Factor")
    print("="*70)
    
    # Create trainer with the UPDATED architecture [128, 64]
    print("\n1. Creating Trainer (now with [128, 64] hidden dims)...")
    trainer = Trainer(cfg)
    
    print(f"\n2. Model Architecture Comparison:")
    print(f"   MLP + Latent:")
    print(f"     - Input: {trainer.data['train']['X'].shape[1]} features")
    print(f"     - Hidden dims: {[128, 64]} (UPDATED to match MLP-only)")
    print(f"     - Latent: 1 scalar per bin, added in logit space")
    print(f"   MLP-only:")
    print(f"     - Input: 15 features + 32-dim embedding = 47")
    print(f"     - Hidden dims: [128, 64]")
    print(f"     - No additive latent")
    
    # Train MLP for a while
    print("\n3. Training MLP for 50 epochs...")
    for epoch in range(50):
        loss = trainer.train_epoch()
        if epoch % 10 == 0:
            val_loss = trainer.validate("val")
            print(f"   Epoch {epoch}: train={loss:.6f}, val={val_loss:.6f}")
    
    val_before = trainer.validate("val")
    print(f"\n   Final val loss (before latent): {val_before:.6f}")
    
    # Get MLP predictions
    intrinsic_vec = trainer.model.predict_MLP_only(trainer.train_loader_ordered, loss_mode="bin")
    y = trainer.data["train"]["y"].values
    
    # Solve for latent
    print("\n4. Solving for latent D (bin-specific amplification in logit space)...")
    latent_D = trainer.model.latent_solver.solve(y=y, intrinsic_vec=intrinsic_vec, loss_type="logistic")
    
    # Interpret as amplification factors
    amplification = np.exp(latent_D)
    print(f"\n5. Amplification factors exp(d_b):")
    print(f"   Mean: {amplification.mean():.4f}")
    print(f"   Median: {np.median(amplification):.4f}")
    print(f"   Min: {amplification.min():.6f} (bin amplifies {amplification.min():.4f}x)")
    print(f"   Max: {amplification.max():.4f} (bin amplifies {amplification.max():.1f}x)")
    print(f"   Std: {amplification.std():.4f}")
    
    # Distribution of amplification factors
    print(f"\n   Distribution of amplification factors:")
    print(f"   < 0.1x (very low amplification): {(amplification < 0.1).sum()} bins")
    print(f"   0.1x - 0.5x (low amplification): {((amplification >= 0.1) & (amplification < 0.5)).sum()} bins")
    print(f"   0.5x - 2x (normal): {((amplification >= 0.5) & (amplification < 2)).sum()} bins")
    print(f"   2x - 10x (high amplification): {((amplification >= 2) & (amplification < 10)).sum()} bins")
    print(f"   > 10x (very high amplification): {(amplification >= 10).sum()} bins")
    
    # Check what happens after adding latent
    trainer.model.set_latent(latent_D)
    val_after = trainer.validate("val")
    print(f"\n6. Val loss after adding latent: {val_after:.6f}")
    print(f"   Change: {val_after - val_before:+.6f} ({'WORSE' if val_after > val_before else 'BETTER'})")
    
    # Get predictions before and after
    bin_indices = trainer.data["train"]["X"].index.get_level_values("bin_uri").map(trainer.bin_index).values
    
    pred_logit_before = intrinsic_vec
    pred_logit_after = intrinsic_vec + latent_D[bin_indices]
    
    pred_prob_before = torch.sigmoid(torch.tensor(pred_logit_before)).numpy()
    pred_prob_after = torch.sigmoid(torch.tensor(pred_logit_after)).numpy()
    
    print("\n7. THE CORE ISSUE - Sample-specific vs Bin-level adjustment:")
    
    # Look at bins with high within-bin variance in targets
    print("\n   Analyzing bins with multiple samples and varying targets...")
    
    example_count = 0
    for b_uri, b_idx in trainer.bin_index.items():
        mask = bin_indices == b_idx
        n_obs = mask.sum()
        if n_obs < 10:
            continue
        
        y_bin = y[mask]
        if y_bin.std() < 0.001:  # Skip bins with low variance
            continue
        
        if example_count >= 2:
            break
        example_count += 1
        
        pred_before_bin = pred_prob_before[mask]
        pred_after_bin = pred_prob_after[mask]
        amp_factor = amplification[b_idx]
        
        print(f"\n   Bin: {b_uri} ({n_obs} observations, amplification={amp_factor:.4f}x)")
        print(f"   Target variance: {y_bin.var():.8f}")
        
        # Show a few examples
        sorted_idx = np.argsort(y_bin)[::-1][:5]  # Top 5 by target
        print(f"   Top 5 by target:")
        print(f"   {'Target':<12} {'Before':<12} {'After':<12} {'Change':<12}")
        for i in sorted_idx:
            change = pred_after_bin[i] - pred_before_bin[i]
            print(f"   {y_bin[i]:<12.6f} {pred_before_bin[i]:<12.6f} {pred_after_bin[i]:<12.6f} {change:+.6f}")
        
        # The key issue: all predictions shifted by same multiplicative factor
        print(f"\n   ⚠️ Issue: All predictions in this bin are multiplied by ~{amp_factor:.4f}")
        print(f"   This shifts the mean correctly but doesn't help with sample-specific variation!")
    
    print("\n" + "="*70)
    print("8. MATHEMATICAL INTERPRETATION")
    print("="*70)
    print("""
    The model assumes:
        p_{s,b} = σ(m(s,b) + d_b)
        
    In terms of odds:
        odds_{s,b} = exp(m(s,b)) × exp(d_b)
                   = base_odds × amplification_factor
    
    This is CORRECT if the true data generating process is:
        "Each bin has a species-specific amplification bias exp(d_b)
         that multiplies the 'true' relative abundance"
    
    HOWEVER, the issue is that this forces:
        - The SAME amplification factor for ALL samples in a bin
        - No sample-specific modulation of the amplification
    
    In reality, amplification might depend on:
        - Sample conditions (collection day, read depth, etc.)
        - Interactions between species and sample
    
    The MLP-only model with bin embeddings can learn:
        p_{s,b} = σ(MLP([features, bin_embedding]))
        
    This allows the MLP to learn sample-specific modulation of bin effects,
    which is why it achieves better correlation despite having no explicit
    "amplification factor" interpretation.
    """)
    
    # Correlation analysis
    print("9. Correlation Analysis:")
    corr_before = np.corrcoef(pred_prob_before, y)[0, 1]
    corr_after = np.corrcoef(pred_prob_after, y)[0, 1]
    print(f"   Before latent: {corr_before:.4f}")
    print(f"   After latent:  {corr_after:.4f}")
    
    # Non-zero only
    nz = y > 0
    corr_before_nz = np.corrcoef(pred_prob_before[nz], y[nz])[0, 1]
    corr_after_nz = np.corrcoef(pred_prob_after[nz], y[nz])[0, 1]
    print(f"   Before latent (non-zero): {corr_before_nz:.4f}")
    print(f"   After latent (non-zero):  {corr_after_nz:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Distribution of amplification factors
    ax = axes[0, 0]
    ax.hist(np.log10(amplification + 1e-10), bins=50, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='No amplification (1x)')
    ax.set_xlabel('log10(amplification factor)')
    ax.set_ylabel('Number of bins')
    ax.set_title('Distribution of Bin Amplification Factors')
    ax.legend()
    
    # Plot 2: Scatter of predictions before vs after
    ax = axes[0, 1]
    sample_idx = np.random.choice(len(y), min(5000, len(y)), replace=False)
    ax.scatter(pred_prob_before[sample_idx], pred_prob_after[sample_idx], 
               alpha=0.3, s=5, c=y[sample_idx], cmap='viridis')
    ax.plot([0, 0.05], [0, 0.05], 'r--', label='No change')
    ax.set_xlabel('Prediction before latent')
    ax.set_ylabel('Prediction after latent')
    ax.set_title('How Latent Changes Predictions')
    ax.set_xlim(0, 0.02)
    ax.set_ylim(0, 0.02)
    ax.legend()
    
    # Plot 3: Correlation visualization (non-zero targets)
    ax = axes[1, 0]
    nz_sample = np.random.choice(np.where(nz)[0], min(2000, nz.sum()), replace=False)
    ax.scatter(y[nz_sample], pred_prob_before[nz_sample], alpha=0.3, s=10, label=f'Before (r={corr_before_nz:.3f})')
    ax.scatter(y[nz_sample], pred_prob_after[nz_sample], alpha=0.3, s=10, label=f'After (r={corr_after_nz:.3f})')
    ax.plot([0, 0.1], [0, 0.1], 'k--', alpha=0.5)
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title('Predictions vs Target (non-zero only)')
    ax.legend()
    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 0.02)
    
    # Plot 4: Amplification vs mean target per bin
    ax = axes[1, 1]
    bin_mean_target = []
    bin_amp = []
    for b_uri, b_idx in trainer.bin_index.items():
        mask = bin_indices == b_idx
        if mask.sum() > 0:
            bin_mean_target.append(y[mask].mean())
            bin_amp.append(amplification[b_idx])
    ax.scatter(bin_mean_target, bin_amp, alpha=0.3, s=5)
    ax.set_xlabel('Mean target per bin')
    ax.set_ylabel('Amplification factor exp(d_b)')
    ax.set_title('Amplification vs Mean Abundance')
    ax.set_xlim(0, 0.02)
    ax.axhline(1, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'figures/amplification_analysis.png'), dpi=150)
    print(f"\nSaved analysis plot to figures/amplification_analysis.png")
    
    print("\n" + "="*70)
    print("10. CONCLUSION")
    print("="*70)
    print("""
    The latent factor exp(d_b) as a bin-specific amplification factor is a
    VALID biological interpretation. However, the implementation has a problem:
    
    ✅ CORRECT: Modeling bin-specific amplification bias
    ❌ PROBLEM: Using the SAME factor for ALL samples ignores:
       - Sample-specific conditions affecting amplification
       - Feature-dependent modulation of the amplification
    
    The MLP-only model implicitly handles this by allowing the neural network
    to learn sample-specific weights for the bin embedding, effectively
    computing a sample-dependent amplification.
    
    POTENTIAL FIX: Instead of adding d_b after MLP, make d_b an input:
        p = σ(MLP([features, d_b]))  or
        p = σ(MLP([features]) + f(features) × d_b)  # sample-dependent scaling
    """)


if __name__ == "__main__":
    analyze_amplification()
