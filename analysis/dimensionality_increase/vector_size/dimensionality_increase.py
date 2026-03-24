"""
Latent Dimensionality Analysis: Testing Impact of Embedding/Output Dimensions


This script trains models with different latent vector and MLP output dimensions:
- Baseline (dim=1): Original additive architecture: ŷ = sigmoid(m(x) + d_bin)
- Dimensions 2, 5, 10, 20, 50: Multiplicative gating with softplus gating function
    - m̃ = m(x) ⊙ g(h[bin_id]) where g(h) = 1 + softplus(h) - ε

All models use the same softplus gating function to isolate the effect of dimensionality.

Results are saved to pickle for visualization by dimensionality_increase_visualize.py

Usage:
    python dimensionality_increase.py --data_path ../../data/ecuador_training_data.csv
    python dimensionality_increase.py --data_path ../../data/ecuador_training_data.csv --no_wandb
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, Any
import logging as log

import sys
import os
from pathlib import Path

# Setup paths
root_dir = Path(__file__).parent
src_path = str(root_dir.parent.parent.parent / "src")  # Metabarcoding/src

# Import from src (which now contains both baseline and gating variants)
sys.path.insert(0, src_path)
import train as train_module
import config as config_module
sys.path.pop(0)

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def run_comparison(data_path: str, use_wandb: bool = True, dimensions: list = None) -> Dict[str, Any]:
    """
    Run dimensionality analysis by training models with different embed dimensions.
    
    Args:
        data_path: Path to the training data CSV
        use_wandb: Whether to log to Weights & Biases
        dimensions: List of embedding dimensions to test. If None, tests [1, 2, 5, 10, 20, 50]
    
    Returns:
        Dictionary with results for each dimension
    """
    if dimensions is None:
        dimensions = [1, 2, 5, 10, 20, 50] #[1, 5, 6, 8, 10, 12, 15, 20]
    
    results = {}
    
    # Train with Baseline (Additive) Architecture (dim=1)
    log.info("\n" + "="*70)
    log.info("TRAINING BASELINE (DIMENSION=1, ADDITIVE) ARCHITECTURE")
    log.info("="*70)
    log.info(f"train module: {train_module.__file__}")
    log.info(f"config module: {config_module.__file__}")
    
    config_module.set_seed(14)
    baseline_cfg = config_module.Config(embed_dim=1)  # scalar additive mode (original baseline)
    
    baseline_trainer = train_module.Trainer(baseline_cfg, data_path, loss_type="cross_entropy")
    log.info(f"Baseline model type: {type(baseline_trainer.model).__name__}")
    
    baseline_results = baseline_trainer.run(use_wandb=use_wandb)
    results["dim_1"] = baseline_results
    
    # Train with Multiplicative Gating Architecture for each dimension
    for embed_dim in dimensions:
        if embed_dim == 1:
            log.info(f"\nSkipping dim=1 (already trained as baseline)")
            continue
            
        log.info("\n" + "="*70)
        log.info(f"TRAINING MULTIPLICATIVE GATING: DIMENSION={embed_dim}")
        log.info("="*70)
        
        config_module.set_seed(14)  # Reset seed for fair comparison
        cfg = config_module.Config(
            embed_dim=embed_dim,
            gating_fn="softplus",  # Always use softplus
        )
        log.info(f"Config: embed_dim={cfg.embed_dim}, gating_fn={cfg.gating_fn}")
        
        trainer = train_module.Trainer(cfg, data_path, loss_type="cross_entropy")
        log.info(f"Model type: {type(trainer.model).__name__}")
        log.info(f"Gating function: {trainer.model.gating_fn}")
        
        results[f"dim_{embed_dim}"] = trainer.run(use_wandb=use_wandb)
        
        log.info(f"Completed training for dimension {embed_dim}")
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latent Dimensionality Analysis: Testing Impact of Embedding/Output Dimensions"
    )
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to data CSV file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", 
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory for results pickle")
    parser.add_argument("--dimensions", type=int, nargs='+', default=None,
                        help="Specific dimensions to test (default: 1 2 5 10 20 50)")
    args = parser.parse_args()
    
    # Setup logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    
    if use_wandb:
        wandb.init(
            project="metabarcoding-dimensionality-analysis",
            name=f"dimensionality_analysis_{time.strftime('%Y-%m-%d_%H-%M')}",
            reinit=True,
        )
    
    # Run comparison
    results = run_comparison(args.data_path, use_wandb=use_wandb, dimensions=args.dimensions)
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    results_path = os.path.join(output_dir, "dimensionality_analysis_results.pkl")
    save_results(results, results_path)
    
    log.info(f"\n{'='*70}")
    log.info("ANALYSIS COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {results_path}")
    log.info(f"Run visualization: python dimensionality_increase_visualize.py --results_path {results_path}")
    
    if use_wandb:
        wandb.finish()

