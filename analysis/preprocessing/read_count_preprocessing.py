"""
Read Count Preprocessing Variant Runner

This script trains only preprocessing variants (no baseline retraining).
Each variant is saved to its own pickle file for later comparison.

Usage:
    python read_count_preprocessing.py
    python read_count_preprocessing.py --methods normalized logarithm --no_wandb
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Any, List
import logging as log

import sys
sys.path.insert(0, '../../src')
sys.path.insert(0, '..')

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
    cfg: Config,
    methods: List[str],
    use_wandb: bool = True,
    run_group: str | None = None,
) -> Dict[str, Any]:
    """
    Train selected preprocessing variants.
    
    Args:
        cfg: Configuration object
        use_wandb: Whether to log to Weights & Biases
    
    Returns:
        Dictionary keyed by preprocessing variant
    """
    results = {}
    preprocessing_methods = methods
    
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

        with variant_wandb_run(
            use_wandb=use_wandb,
            wandb_module=wandb,
            project="metabarcoding-preprocessing-comparison",
            analysis_name="preprocessing",
            variant_name=method,
            run_group=run_group,
            tags=["preprocessing", method, "variant_only"],
            config={**cfg.__dict__, "variant": method},
        ):
            trainer = Trainer(cfg, data_dir=data_dir, loss_type="cross_entropy")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for quick dry-runs",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["normalized", "logarithm"],
        choices=["original", "normalized", "logarithm"],
        help="Preprocessing variants to train (default: normalized logarithm)",
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
    run_group = make_run_group("preprocessing_comparison")
    
    # Run comparison
    results = run_comparison(
        cfg,
        methods=args.methods,
        use_wandb=use_wandb,
        run_group=run_group,
    )
    
    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    for variant, variant_results in results.items():
        save_variant_result(output_dir, "preprocessing", variant, variant_results)
    
    log.info(f"\n{'='*70}")
    log.info("VARIANT TRAINING COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {output_dir}")
    log.info(f"Run visualization with one or more files from: {output_dir}")
