"""
Diagnostic script to understand why MLP + Latent performs worse than MLP-only.

This script:
1. Compares the MLP architectures between both models
2. Traces what happens when latent is added
3. Checks if the latent solver is causing issues
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import torch
import torch.nn.functional as F
from config import Config, set_seed
from train import Trainer
from utils import load

def diagnose():
    set_seed(14)
    cfg = Config()
    
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../../data/ecuador_training_data.csv")
    
    print("="*70)
    print("DIAGNOSTIC: Understanding MLP + Latent Performance")
    print("="*70)
    
    # Create trainer with logistic loss
    print("\n1. Creating Trainer with logistic loss...")
    trainer = Trainer(cfg, data_path, loss_type="logistic")
    
    print(f"\n2. Model Architecture:")
    print(f"   MLP input dim: {trainer.data['train']['X'].shape[1]}")
    print(f"   MLP architecture: {trainer.model.mlp}")
    print(f"   Latent vector size: {trainer.model.n_bins}")
    print(f"   Latent initial values: mean={trainer.model.latent_vec.mean():.4f}, std={trainer.model.latent_vec.std():.4f}")
    
    # Run a few training epochs WITHOUT latent updates
    print("\n3. Training MLP only (no latent updates) for 50 epochs...")
    trainer.model.train()
    losses_no_latent = []
    for epoch in range(50):
        loss = trainer.train_epoch()
        losses_no_latent.append(loss)
        if epoch % 10 == 0:
            val_loss = trainer.validate("val")
            print(f"   Epoch {epoch}: train_loss={loss:.6f}, val_loss={val_loss:.6f}")
    
    val_loss_before_latent = trainer.validate("val")
    print(f"\n   Final val loss (before latent): {val_loss_before_latent:.6f}")
    
    # Get predictions BEFORE latent
    preds_before, targets, *_ = trainer.get_predictions("test")
    mae_before = np.mean(np.abs(preds_before - targets))
    corr_before = np.corrcoef(preds_before, targets)[0, 1] if np.std(preds_before) > 0 else 0
    print(f"   Test MAE (before latent): {mae_before:.6f}")
    print(f"   Test Correlation (before latent): {corr_before:.4f}")
    
    # Check MLP output statistics
    print("\n4. MLP output statistics (before latent):")
    all_outputs = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in trainer.train_loader:
            inputs = batch["input"].to(trainer.device)
            bin_idx = batch["bin_idx"].to(trainer.device)
            mlp_out = trainer.model.mlp(inputs).squeeze(-1)
            all_outputs.extend(mlp_out.cpu().numpy())
    all_outputs = np.array(all_outputs)
    print(f"   MLP outputs: mean={all_outputs.mean():.4f}, std={all_outputs.std():.4f}, "
          f"min={all_outputs.min():.4f}, max={all_outputs.max():.4f}")
    
    # Now solve for latent
    print("\n5. Solving for latent vector...")
    latent_vec = trainer.solve_latent()
    print(f"   Latent D: mean={latent_vec.mean():.4f}, std={latent_vec.std():.4f}, "
          f"min={latent_vec.min():.4f}, max={latent_vec.max():.4f}")
    
    # Check loss after adding latent
    val_loss_after_latent = trainer.validate("val")
    print(f"\n6. Val loss AFTER adding latent: {val_loss_after_latent:.6f}")
    print(f"   Change: {val_loss_after_latent - val_loss_before_latent:+.6f} "
          f"({'WORSE' if val_loss_after_latent > val_loss_before_latent else 'BETTER'})")
    
    # Get predictions AFTER latent
    preds_after, targets, *_ = trainer.get_predictions("test")
    mae_after = np.mean(np.abs(preds_after - targets))
    corr_after = np.corrcoef(preds_after, targets)[0, 1] if np.std(preds_after) > 0 else 0
    print(f"   Test MAE (after latent): {mae_after:.6f}")
    print(f"   Test Correlation (after latent): {corr_after:.4f}")
    
    # Check the combined output (MLP + latent)
    print("\n7. Combined output statistics (MLP + latent):")
    all_combined = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in trainer.train_loader:
            inputs = batch["input"].to(trainer.device)
            bin_idx = batch["bin_idx"].to(trainer.device)
            combined = trainer.model(inputs, bin_idx)
            all_combined.extend(combined.cpu().numpy())
    all_combined = np.array(all_combined)
    print(f"   Combined outputs: mean={all_combined.mean():.4f}, std={all_combined.std():.4f}, "
          f"min={all_combined.min():.4f}, max={all_combined.max():.4f}")
    
    # Check what the latent is actually doing
    print("\n8. Latent contribution analysis:")
    latent_contributions = latent_vec[np.array([trainer.bin_index[b] for b in 
                                                 trainer.data["train"]["X"].index.get_level_values("bin_uri")])]
    print(f"   Latent contributions: mean={latent_contributions.mean():.4f}, "
          f"std={latent_contributions.std():.4f}")
    
    # Compare to targets
    y_train = trainer.data["train"]["y"].values
    y_clipped = np.clip(y_train, 1e-7, 1 - 1e-7)
    y_logit = np.log(y_clipped / (1 - y_clipped))
    print(f"   Target logits: mean={y_logit.mean():.4f}, std={y_logit.std():.4f}")
    
    # Now continue training for a few more epochs
    print("\n9. Training 10 more epochs AFTER latent update...")
    trainer.model.train()
    for epoch in range(10):
        loss = trainer.train_epoch()
        if epoch % 2 == 0:
            val_loss = trainer.validate("val")
            print(f"   Epoch {epoch}: train_loss={loss:.6f}, val_loss={val_loss:.6f}")
    
    val_loss_after_more_training = trainer.validate("val")
    print(f"\n   Final val loss (after more training): {val_loss_after_more_training:.6f}")
    
    # Final predictions
    preds_final, targets, *_ = trainer.get_predictions("test")
    mae_final = np.mean(np.abs(preds_final - targets))
    corr_final = np.corrcoef(preds_final, targets)[0, 1] if np.std(preds_final) > 0 else 0
    print(f"   Test MAE (final): {mae_final:.6f}")
    print(f"   Test Correlation (final): {corr_final:.4f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Before latent:  MAE={mae_before:.6f}, Corr={corr_before:.4f}")
    print(f"After latent:   MAE={mae_after:.6f}, Corr={corr_after:.4f}")
    print(f"After training: MAE={mae_final:.6f}, Corr={corr_final:.4f}")
    
    # Key diagnostic: Is the latent hurting?
    if mae_after > mae_before:
        print("\n⚠️  ISSUE DETECTED: Adding latent INCREASES error!")
        print("   The latent solver is producing values that hurt predictions.")
    
    if val_loss_after_latent > val_loss_before_latent:
        print("\n⚠️  ISSUE DETECTED: Adding latent INCREASES validation loss!")
        print("   This explains why early stopping triggers quickly.")


if __name__ == "__main__":
    diagnose()
