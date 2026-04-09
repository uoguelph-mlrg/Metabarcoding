"""
Interpolated Latent Variant Runner

This script trains interpolation-focused variants and saves each run to its own
pickle file for later comparison.

Usage:
    python interpolated_latent.py --no_wandb
"""
from __future__ import annotations

import argparse
import copy
import os
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
    base_cfg: Config,
    variants: List[str],
    variant_overrides: Dict[str, Dict[str, Any]],
    use_wandb: bool = True,
    run_group: str | None = None,
) -> Dict[str, Any]:
    """
    Train selected interpolation variants.
    
    Args:
        base_cfg: Base configuration object
        variants: Variant names to run
        variant_overrides: Mapping of variant -> config overrides
        use_wandb: Whether to log to Weights & Biases
    
    Returns:
        Dictionary keyed by variant name
    """
    results = {}
    for variant in variants:
        if variant not in variant_overrides:
            raise ValueError(f"Unknown variant: {variant}")

        cfg = copy.deepcopy(base_cfg)
        for key, value in variant_overrides[variant].items():
            setattr(cfg, key, value)

        log.info("\n" + "="*70)
        log.info(f"TRAINING INTERPOLATION VARIANT: {variant.upper()}")
        log.info("="*70)
        log.info(
            "settings | interpolated_sample_fraction=%.3f | "
            "train_MLP_with_interpolation=%s | "
            "inference_with_interpolation=%s | "
            "include_self_in_interpolation=%s",
            cfg.interpolated_sample_fraction,
            cfg.train_MLP_with_interpolation,
            cfg.inference_with_interpolation,
            cfg.include_self_in_interpolation,
        )

        set_seed(14)
        with variant_wandb_run(
            use_wandb=use_wandb,
            wandb_module=wandb,
            analysis_name="interpolated_latent",
            variant_name=variant,
            run_group=run_group,
            tags=["interpolated_latent", variant, "variant_only"],
            config={**cfg.__dict__, "variant": variant},
        ):
            trainer = Trainer(cfg)
            results[variant] = trainer.run(use_wandb=use_wandb)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train interpolation-focused variants"
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
        "--variants",
        nargs="+",
        default=None,
        help="Optional subset of variants to run",
    )
    args = parser.parse_args()

    variant_overrides: Dict[str, Dict[str, Any]] = {
        # Default profile requested for all non-targeted checks.
        "default_with_interpolation": {
            "interpolated_sample_fraction": 0.2,
            "train_MLP_with_interpolation": True,
            "inference_with_interpolation": False,
            "include_self_in_interpolation": True,
        },
        "include_self_false": {
            "interpolated_sample_fraction": 0.2,
            "train_MLP_with_interpolation": True,
            "inference_with_interpolation": False,
            "include_self_in_interpolation": False,
        },
        "inference_true": {
            "interpolated_sample_fraction": 0.2,
            "train_MLP_with_interpolation": True,
            "inference_with_interpolation": True,
            "include_self_in_interpolation": True,
        },
        "train_mlp_false": {
            "interpolated_sample_fraction": 0.2,
            "train_MLP_with_interpolation": False,
            "inference_with_interpolation": False,
            "include_self_in_interpolation": True,
        },
        "fraction_0p1": {
            "interpolated_sample_fraction": 0.1,
            "train_MLP_with_interpolation": True,
            "inference_with_interpolation": False,
            "include_self_in_interpolation": True,
        },
        "fraction_0p5": {
            "interpolated_sample_fraction": 0.5,
            "train_MLP_with_interpolation": True,
            "inference_with_interpolation": False,
            "include_self_in_interpolation": True,
        },
        "fraction_1p0": {
            "interpolated_sample_fraction": 1.0,
            "train_MLP_with_interpolation": True,
            "inference_with_interpolation": False,
            "include_self_in_interpolation": True,
        },
    }
    
    # Setup
    set_seed(14)
    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.variants is None:
        variants = list(variant_overrides.keys())
    else:
        invalid_variants = [name for name in args.variants if name not in variant_overrides]
        if invalid_variants:
            raise ValueError(
                f"Invalid variant(s): {invalid_variants}. "
                f"Valid choices: {list(variant_overrides.keys())}"
            )
        variants = args.variants
    
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = make_run_group("interpolated_latent")
    
    # Run comparison
    results = run_comparison(
        cfg,
        variants=variants,
        variant_overrides=variant_overrides,
        use_wandb=use_wandb,
        run_group=run_group,
    )
    
    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    for variant, variant_results in results.items():
        save_variant_result(output_dir, "interpolated_latent", variant, variant_results)
    
    log.info(f"\n{'='*70}")
    log.info("VARIANT TRAINING COMPLETE")
    log.info(f"{'='*70}")
    log.info(f"Results saved to: {output_dir}")
    log.info(f"Run visualization with one or more files from: {output_dir}")
