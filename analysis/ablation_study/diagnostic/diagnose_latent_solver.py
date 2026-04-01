"""
Deep diagnostic of the latent solver issue.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch
from config import Config, set_seed
from train import Trainer
from utils import load

def diagnose_latent_solver():
    set_seed(14)
    cfg = Config()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../data/ecuador_training_data.csv")
    
    print("="*70)
    print("DEEP DIAGNOSTIC: Latent Solver Analysis")
    print("="*70)
    
    # Create trainer
    trainer = Trainer(cfg, data_path)
    
    # Train MLP only for a bit
    print("\n1. Training MLP for 50 epochs...")
    for epoch in range(50):
        trainer.train_epoch()
    
    # Get MLP predictions
    print("\n2. Getting MLP predictions (intrinsic_vec)...")
    intrinsic_vec = trainer.model.predict_MLP_only(trainer.train_loader_ordered, loss_mode="bin")
    print(f"   intrinsic_vec: mean={intrinsic_vec.mean():.4f}, std={intrinsic_vec.std():.4f}, "
          f"min={intrinsic_vec.min():.4f}, max={intrinsic_vec.max():.4f}")
    
    # Get targets
    y = trainer.data["train"]["y"].values
    print(f"\n3. Target probabilities (y):")
    print(f"   y: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")
    print(f"   Number of zeros: {(y == 0).sum()} / {len(y)} ({100*(y==0).mean():.1f}%)")
    
    # Convert to logit space (what latent solver does)
    y_clipped = np.clip(y, 1e-7, 1 - 1e-7)
    y_logit = np.log(y_clipped / (1 - y_clipped))
    print(f"\n4. Target logits (y_logit = log(y/(1-y))):")
    print(f"   y_logit: mean={y_logit.mean():.4f}, std={y_logit.std():.4f}, "
          f"min={y_logit.min():.4f}, max={y_logit.max():.4f}")
    
    # The residual that latent solver tries to explain
    residual = y_logit - intrinsic_vec
    print(f"\n5. Residual (y_logit - intrinsic_vec):")
    print(f"   residual: mean={residual.mean():.4f}, std={residual.std():.4f}, "
          f"min={residual.min():.4f}, max={residual.max():.4f}")
    
    # Now solve for latent
    print("\n6. Solving for latent D...")
    print(f"   Regularization: L2={cfg.latent_l2_reg}, smooth={cfg.latent_smooth_reg}")
    
    latent_D = trainer.model.latent_solver.solve(y=y, intrinsic_vec=intrinsic_vec, loss_type="logistic")
    print(f"   Latent D: mean={latent_D.mean():.4f}, std={latent_D.std():.4f}, "
          f"min={latent_D.min():.4f}, max={latent_D.max():.4f}")
    
    # Check what prediction becomes with latent
    # For each observation, get its bin index and add latent
    bin_indices = trainer.data["train"]["X"].index.get_level_values("bin_uri").map(trainer.bin_index).values
    latent_contribution = latent_D[bin_indices]
    combined_logit = intrinsic_vec + latent_contribution
    
    print(f"\n7. Combined prediction (logit space):")
    print(f"   combined: mean={combined_logit.mean():.4f}, std={combined_logit.std():.4f}")
    
    # Convert both to probability space for comparison
    pred_prob_before = torch.sigmoid(torch.tensor(intrinsic_vec)).numpy()
    pred_prob_after = torch.sigmoid(torch.tensor(combined_logit)).numpy()
    
    print(f"\n8. Predictions in probability space:")
    print(f"   Before latent: mean={pred_prob_before.mean():.6f}, std={pred_prob_before.std():.6f}")
    print(f"   After latent:  mean={pred_prob_after.mean():.6f}, std={pred_prob_after.std():.6f}")
    print(f"   Target:        mean={y.mean():.6f}, std={y.std():.6f}")
    
    # Compute MAE in probability space
    mae_before = np.mean(np.abs(pred_prob_before - y))
    mae_after = np.mean(np.abs(pred_prob_after - y))
    print(f"\n9. MAE comparison:")
    print(f"   MAE before latent: {mae_before:.6f}")
    print(f"   MAE after latent:  {mae_after:.6f}")
    print(f"   Change: {mae_after - mae_before:+.6f} ({'WORSE' if mae_after > mae_before else 'BETTER'})")
    
    # THE KEY ISSUE: Check correlation in probability space
    corr_before = np.corrcoef(pred_prob_before, y)[0, 1]
    corr_after = np.corrcoef(pred_prob_after, y)[0, 1]
    print(f"\n10. Correlation comparison:")
    print(f"   Corr before latent: {corr_before:.4f}")
    print(f"   Corr after latent:  {corr_after:.4f}")
    print(f"   Change: {corr_after - corr_before:+.4f} ({'WORSE' if corr_after < corr_before else 'BETTER'})")
    
    # Analyze per-bin statistics
    print("\n11. Per-bin analysis:")
    print(f"    Number of bins: {len(trainer.bin_index)}")
    
    # How many observations per bin?
    from collections import Counter
    bin_counts = Counter(bin_indices)
    counts = list(bin_counts.values())
    print(f"    Observations per bin: mean={np.mean(counts):.1f}, min={min(counts)}, max={max(counts)}")
    print(f"    Bins with 1 obs: {sum(1 for c in counts if c == 1)}")
    print(f"    Bins with 2+ obs: {sum(1 for c in counts if c >= 2)}")
    
    # Check if latent is making correct adjustments
    print("\n12. Is latent adjusting in the right direction?")
    # For bins where mean(y_logit) > mean(intrinsic), latent should be positive
    correct_direction = 0
    wrong_direction = 0
    for b_uri, b_idx in trainer.bin_index.items():
        mask = bin_indices == b_idx
        if mask.sum() == 0:
            continue
        mean_residual = residual[mask].mean()
        latent_val = latent_D[b_idx]
        
        # They should have same sign
        if (mean_residual > 0 and latent_val > 0) or (mean_residual < 0 and latent_val < 0):
            correct_direction += 1
        elif abs(mean_residual) < 0.1 or abs(latent_val) < 0.1:
            pass  # negligible
        else:
            wrong_direction += 1
    
    print(f"    Correct direction: {correct_direction}")
    print(f"    Wrong direction: {wrong_direction}")
    
    # Check the magnitude of latent vs what's needed
    print("\n13. Magnitude analysis:")
    print(f"    Mean |residual|: {np.mean(np.abs(residual)):.4f}")
    print(f"    Mean |latent|: {np.mean(np.abs(latent_D)):.4f}")
    print(f"    Ratio (latent/residual): {np.mean(np.abs(latent_D)) / np.mean(np.abs(residual)):.4f}")


if __name__ == "__main__":
    diagnose_latent_solver()
