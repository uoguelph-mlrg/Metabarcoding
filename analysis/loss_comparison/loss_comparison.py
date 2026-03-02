"""
Loss Function Comparison: Cross-Entropy vs Logistic (BCE) Loss

This script trains the MLP + Latent model twice using the existing Trainer class:
1. Cross-Entropy Loss - sample-level training with softmax normalization
2. Logistic Loss (BCEWithLogitsLoss) - bin-level training

Results are saved to pickle for visualization by loss_comparison_visualize.py

Usage:
    python loss_comparison.py --data_path ../../data/ecuador_training_data.csv
    python loss_comparison.py --data_path ../../data/ecuador_training_data.csv --no_wandb
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, Any
import logging as log

import sys
sys.path.insert(0, '../../src')

from config import Config, set_seed
from train import Trainer

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def run_comparison(data_path: str, cfg: Config, use_wandb: bool = True) -> Dict[str, Any]:
    """
    Run the loss function comparison by training with both loss types.
    
    Args:
        data_path: Path to the training data CSV
        cfg: Configuration object
        use_wandb: Whether to log to Weights & Biases
    
    Returns:
        Dictionary with results for both loss types
    """
    results = {}
    
    # Train with Cross-Entropy Loss
    log.info("\n" + "="*70)
    log.info("TRAINING WITH CROSS-ENTROPY LOSS")
    log.info("="*70)
    
    set_seed(14)
    ce_trainer = Trainer(cfg, data_path, loss_type="cross_entropy")
    ce_results = ce_trainer.run(use_wandb=use_wandb)
    results["cross_entropy"] = ce_results
    
    # Train with Logistic Loss
    log.info("\n" + "="*70)
    log.info("TRAINING WITH LOGISTIC (BCE) LOSS")
    log.info("="*70)
    
    set_seed(14)  # Reset seed for fair comparison
    log_trainer = Trainer(cfg, data_path, loss_type="logistic")
    log_results = log_trainer.run(use_wandb=use_wandb)
    results["logistic"] = log_results
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loss Function Comparison: Cross-Entropy vs Logistic (BCE)"
    )
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to data CSV file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", 
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory for results pickle")
    args = parser.parse_args()
    
    # Setup
    set_seed(14)
    cfg = Config()
    
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    
    if use_wandb:
        wandb.init(
            project="metabarcoding-loss-comparison",
            name=f"loss_comparison_{time.strftime('%Y-%m-%d_%H-%M')}",
            config=cfg.__dict__,
            reinit=True,
        )
    
    # Run comparison
    results = run_comparison(args.data_path, cfg, use_wandb=use_wandb)
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    results_path = os.path.join(output_dir, "loss_comparison_results.pkl")
    save_results(results, results_path)
    
    log.info(f"\n{'='*70}")
    log.info("COMPARISON COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {results_path}")
    log.info(f"Run visualization: python loss_comparison_visualize.py --results_path {results_path}")
    
    if use_wandb:
        wandb.finish()
