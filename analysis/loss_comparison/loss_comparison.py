"""
Loss Variant Runner

This script trains only loss-function variants (no baseline retraining).
Each variant is saved to its own pickle file for later comparison.

Usage:
    python loss_comparison.py --data_path ../../data/ecuador_training_data.csv
    python loss_comparison.py --data_path ../../data/ecuador_training_data.csv --variants logistic --no_wandb
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, Any, List
import logging as log

import sys
sys.path.insert(0, '../../src')
analysis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if analysis_root not in sys.path:
    sys.path.insert(0, analysis_root)

from config import Config, set_seed
from train import Trainer
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
    cfg: Config,
    variants: List[str],
    use_wandb: bool = True,
    run_group: str | None = None,
) -> Dict[str, Any]:
    """
    Train selected loss variants.
    
    Args:
        data_path: Path to the training data CSV
        cfg: Configuration object
        use_wandb: Whether to log to Weights & Biases
    
    Returns:
        Dictionary keyed by variant name
    """
    results = {}
    for variant in variants:
        log.info("\n" + "="*70)
        log.info(f"TRAINING LOSS VARIANT: {variant.upper()}")
        log.info("="*70)

        set_seed(14)
        with variant_wandb_run(
            use_wandb=use_wandb,
            wandb_module=wandb,
            project="metabarcoding-loss-comparison",
            analysis_name="loss_comparison",
            variant_name=variant,
            run_group=run_group,
            tags=["loss_comparison", variant, "variant_only"],
            config={**cfg.__dict__, "variant": variant},
        ):
            trainer = Trainer(cfg, data_path, loss_type=variant)
            results[variant] = trainer.run(use_wandb=use_wandb)
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train loss-function variants without retraining baseline"
    )
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to data CSV file")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", 
                        help="Disable Weights & Biases logging")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Output directory for results pickle")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for quick dry-runs",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["logistic"],
        choices=["cross_entropy", "logistic"],
        help="Loss variants to train (default: logistic)",
    )
    args = parser.parse_args()
    
    # Setup
    set_seed(14)
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = make_run_group("loss_comparison")
    
    # Run comparison
    results = run_comparison(
        args.data_path,
        cfg,
        variants=args.variants,
        use_wandb=use_wandb,
        run_group=run_group,
    )
    
    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    for variant, variant_results in results.items():
        save_variant_result(output_dir, "loss_comparison", variant, variant_results)
    
    log.info(f"\n{'='*70}")
    log.info("VARIANT TRAINING COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {output_dir}")
    log.info(f"Run visualization with one or more files from: {output_dir}")
