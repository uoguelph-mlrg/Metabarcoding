"""
BarcodeBERT Variant Runner

This script trains only the BarcodeBERT embedding-neighbor variant
(no baseline retraining in this analysis script).
Results are saved as one variant pickle for later comparison.

Usage:
    python barcodebert.py 
"""
from __future__ import annotations

import argparse
import logging as log
import os
import pickle
import sys
from typing import Any, Dict, Optional

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
    use_wandb: bool = True,
    run_group: Optional[str] = None,
) -> Dict[str, Any]:
    """Train BarcodeBERT variant and return result dict keyed by variant name."""
    results: Dict[str, Any] = {}

    cfg = Config()
    
    if cfg.data_path is None:
        raise ValueError("No data path provided.")
    log.info("\n" + "=" * 78)
    log.info("TRAINING TAXONOMIC DISTANCE MODEL")
    log.info("=" * 78)

    set_seed(14)

    cfg.use_taxonomy = True
    cfg.use_embedding = False

    with variant_wandb_run(
        use_wandb=use_wandb,
        wandb_module=wandb,
        analysis_name="barcodebert",
        variant_name="taxonomy",
        run_group=run_group,
        tags=["taxonomic_distances", "taxonomy", "variant_only"],
    ):
        barcode_trainer = Trainer(
            cfg=cfg,
            model_name="taxonomy",
        )
        results = barcode_trainer.run(use_wandb=use_wandb)
        results["taxonomy"] = results

    return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Comparison: Taxonomy Neighbours vs BarcodeBERT Embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results pickle",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    # Setup logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = make_run_group("barcodebert_comparison")
    if not use_wandb and not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")

    # Run comparison
    results = run_comparison(
        use_wandb=use_wandb,
        run_group=run_group,
    )

    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    results_path = save_variant_result(output_dir, "taxonomy", "taxonomy", results["taxonomy"])

    log.info("\n" + "=" * 78)
    log.info("VARIANT TRAINING COMPLETE")
    log.info("=" * 78)
    log.info(f"Results saved to: {os.path.abspath(results_path)}")
    log.info(
        "To visualize, run:\n"
        f"  python ../visualize_results.py --results_path {os.path.relpath(results_path)}"
    )
