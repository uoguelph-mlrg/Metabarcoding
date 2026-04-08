"""
Latent Dimensionality Variant Runner

This script trains only dimensionality variants (no baseline retraining):
- Dimensions 2, 5, 10, 20, 50: Multiplicative gating with softplus gating function
    - m̃ = m(x) ⊙ g(h[bin_id]) where g(h) = 1 + softplus(h) - ε

All models use the same softplus gating function to isolate the effect of dimensionality.

Each variant is saved to its own pickle file for later comparison.

Usage:
    python dimensionality_increase.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, Any, List, Optional
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
    use_wandb: bool = True,
    dimensions: Optional[List[int]] = None,
    run_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train selected dimensionality variants.
    
    Args:
        use_wandb: Whether to log to Weights & Biases
        dimensions: List of embedding dimensions to test. If None, tests [2, 5, 10, 20, 50]
    
    Returns:
        Dictionary with results for each dimension
    """
    if dimensions is None:
        dimensions = [2, 5, 10, 20, 50]
    
    results = {}
    log.info(f"train module: {train_module.__file__}")
    log.info(f"config module: {config_module.__file__}")
    
    for embed_dim in dimensions:
        log.info("\n" + "="*70)
        log.info(f"TRAINING MULTIPLICATIVE GATING: DIMENSION={embed_dim}")
        log.info("="*70)
        
        config_module.set_seed(14)  # Reset seed for fair comparison
        cfg = config_module.Config(
            embed_dim=embed_dim,
            gating_fn="softplus",  # Always use softplus
        )
        log.info(f"Config: embed_dim={cfg.embed_dim}, gating_fn={cfg.gating_fn}")

        variant = f"dim_{embed_dim}"
        
        with variant_wandb_run(
            use_wandb=use_wandb,
            wandb_module=wandb,
            analysis_name="dimensionality_analysis",
            variant_name=variant,
            run_group=run_group,
            tags=["dimensionality_analysis", variant, "variant_only"],
            config={"embed_dim": cfg.embed_dim, "gating_fn": cfg.gating_fn},
        ):
            trainer = train_module.Trainer(cfg)
            log.info(f"Model type: {type(trainer.model).__name__}")
            log.info(f"Gating function: {trainer.model.gating_fn}")
            
            results[variant] = trainer.run(use_wandb=use_wandb)
        
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
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", 
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory for results pickle")
    parser.add_argument("--dimensions", type=int, nargs='+', default=None,
                        help="Specific dimensions to test (default: 2 5 10 20 50)")
    args = parser.parse_args()
    
    # Setup logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = make_run_group("dimensionality_analysis")
    
    # Run comparison
    results = run_comparison(
        use_wandb=use_wandb,
        dimensions=args.dimensions,
        run_group=run_group,
    )
    
    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    for variant, variant_results in results.items():
        save_variant_result(output_dir, "dimensionality_analysis", variant, variant_results)
    
    log.info(f"\n{'='*70}")
    log.info("VARIANT TRAINING COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {output_dir}")
    log.info(f"Run visualization with one or more files from: {output_dir}")

