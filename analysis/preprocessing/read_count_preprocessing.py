"""
Read Count Preprocessing Comparison

This script trains the MLP + Latent model with three different preprocessing methods:
1. Original: Normalize by sample + log transform (current method)
2. Normalized: Only normalize by sample (no log)
3. Logarithm: Only log transform (no sample normalization)

Results are saved to pickle for visualization by preprocessing_visualization.py

Usage:
    # First, generate the datasets
    python utils_test.py --data_path data/ecuador_training_data.csv
    
    # Then run the comparison
    python read_count_preprocessing.py
    python read_count_preprocessing.py --no_wandb
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


def run_comparison(cfg: Config, use_wandb: bool = True) -> Dict[str, Any]:
    """
    Run the preprocessing comparison by training with all three methods.
    
    Args:
        cfg: Configuration object
        use_wandb: Whether to log to Weights & Biases
    
    Returns:
        Dictionary with results for all preprocessing methods
    """
    results = {}
    preprocessing_methods = ["original", "normalized", "logarithm"]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for method in preprocessing_methods:
        log.info("\n" + "="*70)
        log.info(f"TRAINING WITH {method.upper()} PREPROCESSING")
        log.info("="*70)

        # Path to the preprocessed data
        data_dir = os.path.join(script_dir, "data", method)

        if not os.path.exists(data_dir):
            log.error(f"Data directory not found: {data_dir}")
            log.error("Please run: python utils_test.py --data_path data/ecuador_training_data.csv")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Reset seed for fair comparison
        set_seed(14)

        # Create trainer with preprocessed data
        trainer = Trainer(cfg, data_dir=data_dir, loss_type="cross_entropy")

        # Run training
        method_results = trainer.run(use_wandb=use_wandb)

        results[method] = method_results

        log.info(f"✓ Completed {method} preprocessing")
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read Count Preprocessing Comparison"
    )
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
            project="metabarcoding-preprocessing-comparison",
            name=f"preprocessing_comparison_{time.strftime('%Y-%m-%d_%H-%M')}",
            config=cfg.__dict__,
            reinit=True,
        )
    
    # Run comparison
    results = run_comparison(cfg, use_wandb=use_wandb)
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    results_path = os.path.join(output_dir, "preprocessing_results.pkl")
    save_results(results, results_path)
    
    log.info(f"\n{'='*70}")
    log.info("COMPARISON COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {results_path}")
    log.info(f"Run visualization: python preprocessing_visualization.py --results_path {results_path}")
    
    if use_wandb:
        wandb.finish()
