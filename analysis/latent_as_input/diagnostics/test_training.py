"""
Quick test to verify the latent-as-input model trains without NaN with updated hyperparameters.
"""

import sys
import os
# Add parent directory FIRST to prioritize local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add src directory for shared utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import torch
import logging as log
from train import Trainer
from config import Config, set_seed

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_training():
    """Test training with updated hyperparameters."""
    set_seed()
    cfg = Config()
    
    # Print current configuration
    log.info("=" * 70)
    log.info("CONFIGURATION")
    log.info("=" * 70)
    log.info(f"latent_input_dim: {cfg.latent_input_dim}")
    log.info(f"latent_steps: {cfg.latent_steps}")
    log.info(f"latent_lr: {cfg.latent_lr}")
    log.info(f"latent_init_std: {cfg.latent_init_std}")
    log.info(f"batch_size_sample: {cfg.batch_size_sample}")
    log.info(f"epochs_init: {cfg.epochs_init}")
    log.info(f"latent_norm_reg: {cfg.latent_norm_reg}")
    log.info(f"latent_smooth_reg: {cfg.latent_smooth_reg}")
    log.info(f"gradient_masking: DISABLED (aggressive fix)")
    log.info("=" * 70)
    
    # Initialize trainer
    data_path = "/Users/emmaboehly/Documents/Vector/Metabarcoding/data/ecuador_training_data.csv"
    trainer = Trainer(
        cfg=cfg,
        data_path=data_path,
    )
    
    # Train initial MLP (Phase 0)
    log.info("\n" + "=" * 70)
    log.info("PHASE 0: Training initial MLP")
    log.info("=" * 70)
    for epoch in range(min(10, cfg.epochs_init)):
        train_loss = trainer.train_epoch()
        if torch.isnan(torch.tensor(train_loss)):
            log.error(f"NaN loss detected at epoch {epoch}!")
            return False
        if epoch % 5 == 0:
            val_loss = trainer.validate(split="val")
            log.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    # Run one EM cycle
    log.info("\n" + "=" * 70)
    log.info("TESTING ONE EM CYCLE")
    log.info("=" * 70)
    
    # Phase A: Update latent
    log.info("Phase A: Updating latent embeddings")
    trainer.model.latent_solver.solve_gradient_based(
        model=trainer.model,
        data_loader=trainer.train_loader,
        optimizer=trainer.latent_optimizer,
        n_steps=cfg.latent_steps,
        loss_mode="sample",
    )
    
    # Check latent stats
    latent = trainer.model.get_latent()
    latent_norm = float(torch.tensor(latent).norm())
    latent_mean = float(latent.mean())
    latent_std = float(latent.std())
    log.info(f"Latent stats: norm={latent_norm:.6f}, mean={latent_mean:.6f}, std={latent_std:.6f}")
    
    # Phase B: Update MLP
    log.info("Phase B: Updating MLP")
    for epoch in range(min(5, cfg.epochs)):
        train_loss = trainer.train_epoch()
        if torch.isnan(torch.tensor(train_loss)):
            log.error(f"NaN loss detected during MLP training!")
            return False
        if epoch == 0 or epoch == cfg.epochs - 1:
            val_loss = trainer.validate(split="val")
            log.info(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    log.info("\n" + "=" * 70)
    log.info("✓ TEST PASSED: No NaN losses detected!")
    log.info("✓ Configuration appears stable")
    log.info("=" * 70)
    return True

if __name__ == "__main__":
    success = test_training()
    exit(0 if success else 1)
