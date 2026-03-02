#!/usr/bin/env python3
"""
Example usage of the multiplicative gating architecture.

This script demonstrates how to train the model with the new architecture.
"""
import sys
import os
from pathlib import Path

# Add parent directories to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root / "src"))

import logging as log
from config import Config, set_seed
from train import Trainer

# Setup logging
log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """Train model with multiplicative gating."""
    # Set random seed for reproducibility
    set_seed(14)
    
    # Configure model
    cfg = Config(
        # Architecture
        embed_dim=8,              # Vector dimension
        gating_fn="sigmoid",      # Gating function: sigmoid, softplus, or exp
        
        # Regularization
        latent_l2_reg=1e-2,       # L2 penalty
        latent_smooth_reg=1e-3,   # Smoothness penalty
        
        # Training
        epochs_init=50,           # Initial MLP-only epochs
        epochs=10,                # Epochs per phase B
        max_cycles=50,            # Maximum alternation cycles
        patience=5,               # Early stopping patience
        
        # Optimization
        lr=5e-4,
        batch_size_bin=1024,
        batch_size_sample=8,
    )
    
    log.info(f"Configuration:")
    log.info(f"  embed_dim: {cfg.embed_dim}")
    log.info(f"  gating_fn: {cfg.gating_fn}")
    log.info(f"  latent_l2_reg: {cfg.latent_l2_reg}")
    log.info(f"  latent_smooth_reg: {cfg.latent_smooth_reg}")
    
    # Initialize trainer
    data_path = root / "data" / "ecuador_training_data.csv"
    if not data_path.exists():
        log.error(f"Data file not found: {data_path}")
        log.error("Please provide the correct path to your data file.")
        return
    
    trainer = Trainer(
        cfg,
        data_path=str(data_path),
        loss_type="cross_entropy",  # or "logistic"
    )
    
    # Train model
    log.info("\nStarting training...")
    results = trainer.run(use_wandb=False)
    
    # Print results
    log.info("\n" + "="*70)
    log.info("Training complete!")
    log.info("="*70)
    log.info(f"Test loss: {results['test_loss']:.4f}")
    log.info(f"Test metrics:")
    for key, value in results['test_metrics'].items():
        log.info(f"  {key}: {value:.4f}")
    log.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    log.info(f"Model saved to: {trainer.save_path}")


if __name__ == "__main__":
    main()
