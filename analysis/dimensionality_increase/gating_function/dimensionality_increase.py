"""
Gating Function Comparison: Baseline vs Multiple Gating Architectures

This script trains multiple model architectures:
1. Baseline - Original additive architecture: ŷ = sigmoid(m(x) + d_bin)
2-7. Multiplicative Gating with different gating functions:
   - exp: m̃ = m ⊙ exp(h)
   - scaled_exp: m̃ = m ⊙ exp(α·h)
   - additive: m̃ = m ⊙ (1 + h)
    - softplus: m̃ = m ⊙ (1 + softplus(h) - ε)
    - tanh: m̃ = m ⊙ (1 + tanh(h)·κ)
    - sigmoid: m̃ = m ⊙ (2·σ(h))
8. Dot Product: z = w^T (m(x) ⊙ h)  (gate_torch returns h, goes through final linear)

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
from typing import Dict, Any, Optional, List
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


def run_comparison(data_path: str, use_wandb: bool = True, gating_functions: Optional[list] = None) -> Dict[str, Any]:
    """
    Run the architecture comparison by training baseline and all gating functions.
    
    Args:
        data_path: Path to the training data CSV
        use_wandb: Whether to log to Weights & Biases
        gating_functions: List of gating functions to test. If None, tests all.
    
    Returns:
        Dictionary with results for baseline and all gating architectures
    """
    if gating_functions is None:
        gating_functions = ["dot_product", "exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid"]
    
    results = {}
    
    # Train with Baseline (Additive) Architecture
    log.info("\n" + "="*70)
    log.info("TRAINING BASELINE (ADDITIVE) ARCHITECTURE")
    log.info("="*70)
    log.info(f"train module: {train_module.__file__}")
    log.info(f"config module: {config_module.__file__}")
    
    config_module.set_seed(14)
    baseline_cfg = config_module.Config(embed_dim=1)  # scalar additive mode (original baseline)
    
    baseline_trainer = train_module.Trainer(baseline_cfg, data_path, loss_type="cross_entropy")
    log.info(f"Baseline model type: {type(baseline_trainer.model).__name__}")
    
    baseline_results = baseline_trainer.run(use_wandb=use_wandb)
    results["baseline"] = baseline_results
    
    # Train with each Multiplicative Gating Architecture
    for gating_fn in gating_functions:
        log.info("\n" + "="*70)
        log.info(f"TRAINING MULTIPLICATIVE GATING: {gating_fn.upper()}")
        log.info("="*70)
        
        config_module.set_seed(14)  # Reset seed for fair comparison
        cfg = config_module.Config(
            embed_dim=10,
            gating_fn=gating_fn,
        )
        log.info(f"Config: embed_dim={cfg.embed_dim}, gating_fn={cfg.gating_fn}")
        
        trainer = train_module.Trainer(cfg, data_path, loss_type="cross_entropy")
        log.info(f"Model type: {type(trainer.model).__name__}")
        log.info(f"Gating function: {trainer.model.gating_fn}")
        
        results[gating_fn] = trainer.run(use_wandb=use_wandb)
        
        log.info(f"Completed training for {gating_fn}")
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gating Function Comparison: Baseline vs Multiple Gating Architectures"
    )
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to data CSV file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", 
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory for results pickle")
    parser.add_argument("--gating_functions", type=str, nargs='+', default=None,
                        help="Specific gating functions to test (default: all)")
    args = parser.parse_args()
    
    # Setup logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    
    if use_wandb:
        wandb.init(
            project="metabarcoding-gating-comparison",
            name=f"gating_comparison_{time.strftime('%Y-%m-%d_%H-%M')}",
            reinit=True,
        )
    
    # Run comparison
    results = run_comparison(args.data_path, use_wandb=use_wandb, gating_functions=args.gating_functions)
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    results_path = os.path.join(output_dir, "gating_comparison_results.pkl")
    save_results(results, results_path)
    
    log.info(f"\n{'='*70}")
    log.info("COMPARISON COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {results_path}")
    log.info(f"Run visualization: python dimensionality_increase_visualize.py --results_path {results_path}")
    
    if use_wandb:
        wandb.finish()
