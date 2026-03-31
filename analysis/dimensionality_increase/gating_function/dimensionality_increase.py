"""
Gating Function Variant Runner

This script trains only gating-function variants (no baseline retraining):
1-7. Multiplicative Gating with different gating functions:
   - exp: m̃ = m ⊙ exp(h)
   - scaled_exp: m̃ = m ⊙ exp(α·h)
   - additive: m̃ = m ⊙ (1 + h)
    - softplus: m̃ = m ⊙ (1 + softplus(h) - ε)
    - tanh: m̃ = m ⊙ (1 + tanh(h)·κ)
    - sigmoid: m̃ = m ⊙ (2·σ(h))
8. Dot Product: z = w^T (m(x) ⊙ h)  (gate_torch returns h, goes through final linear)

Each variant is saved to its own pickle file for later comparison.

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

analysis_root = str(root_dir.parent.parent)
if analysis_root not in sys.path:
    sys.path.insert(0, analysis_root)
from variant_helpers import (
    make_output_dir,
    make_run_group,
    save_variant_result,
    variant_wandb_run,
)

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def run_comparison(
    data_path: str,
    use_wandb: bool = True,
    gating_functions: Optional[List[str]] = None,
    run_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train selected gating-function variants.
    
    Args:
        data_path: Path to the training data CSV
        use_wandb: Whether to log to Weights & Biases
        gating_functions: List of gating functions to test. If None, tests all.
    
    Returns:
        Dictionary with results for selected gating variants
    """
    if gating_functions is None:
        gating_functions = ["dot_product", "exp", "scaled_exp", "additive", "softplus", "tanh", "sigmoid"]
    
    results = {}
    log.info(f"train module: {train_module.__file__}")
    log.info(f"config module: {config_module.__file__}")
    
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

        with variant_wandb_run(
            use_wandb=use_wandb,
            wandb_module=wandb,
            project="metabarcoding-gating-comparison",
            analysis_name="gating_comparison",
            variant_name=gating_fn,
            run_group=run_group,
            tags=["gating_comparison", gating_fn, "variant_only"],
            config={"embed_dim": cfg.embed_dim, "gating_fn": cfg.gating_fn},
        ):
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
        description="Train gating-function variants without retraining baseline"
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
    run_group = make_run_group("gating_comparison")
    
    # Run comparison
    results = run_comparison(
        args.data_path,
        use_wandb=use_wandb,
        gating_functions=args.gating_functions,
        run_group=run_group,
    )
    
    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    for variant, variant_results in results.items():
        save_variant_result(output_dir, "gating_comparison", variant, variant_results)
    
    log.info(f"\n{'='*70}")
    log.info("VARIANT TRAINING COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {output_dir}")
    log.info(f"Run visualization with one or more files from: {output_dir}")
