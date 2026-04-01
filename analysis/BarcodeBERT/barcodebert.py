"""
BarcodeBERT Variant Runner

This script trains only the BarcodeBERT embedding-neighbor variant
(no baseline retraining in this analysis script).
Results are saved as one variant pickle for later comparison.

Usage:
    python barcodebert.py --data_path ../../data/ecuador_training_data.csv \\
        --embedding_path ../../data/barcodebert_embeddings.npy
    python barcodebert.py --data_dir ../../data \\
        --embedding_path ../../data/barcodebert_embeddings.npy --no_wandb
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
    data_path: Optional[str],
    data_dir: Optional[str],
    embedding_path: Optional[str],
    barcode_data_path: Optional[str],
    emb_distance_metric: str,
    use_wandb: bool = True,
    run_group: Optional[str] = None,
) -> Dict[str, Any]:
    """Train BarcodeBERT variant and return result dict keyed by variant name."""
    results: Dict[str, Any] = {}

    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # =========================================================================
    # DATA PATH
    # =========================================================================
    if data_path is None and data_dir is None:
        data_path = os.path.abspath(
            os.path.join(root_dir, "..", "..", "data", "data_merged.csv")
        )
        log.info(f"No data path provided. Using default: {data_path}")

    # =========================================================================
    # BARCODEBERT MODEL (Embedding Neighbours)
    # =========================================================================
    log.info("\n" + "=" * 78)
    log.info("TRAINING BARCODEBERT MODEL (EMBEDDING NEIGHBOURS)")
    log.info("=" * 78)

    set_seed(14)

    barcode_cfg = Config()
    barcode_cfg.use_taxonomy = False
    barcode_cfg.use_embedding = True
    barcode_cfg.emb_distance_metric = emb_distance_metric
    barcode_cfg.embedding_path = embedding_path
    barcode_cfg.barcode_data_path = barcode_data_path
    if embedding_path is not None and barcode_data_path is not None:
        barcode_cfg.barcode_data_path = "../../data/data_merged.csv"

    with variant_wandb_run(
        use_wandb=use_wandb,
        wandb_module=wandb,
        analysis_name="barcodebert",
        variant_name="barcodebert",
        run_group=run_group,
        tags=["barcodebert", "embedding-neighbours", "variant_only"],
        config={
            "emb_distance_metric": emb_distance_metric,
            "embedding_path": embedding_path,
            "barcode_data_path": barcode_data_path,
            "variant": "barcodebert",
        },
    ):
        barcode_trainer = Trainer(
            barcode_cfg,
            data_path=data_path,
            data_dir=data_dir,
        )
        barcodebert_results = barcode_trainer.run(use_wandb=use_wandb)
        results["barcodebert"] = barcodebert_results

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
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to raw data CSV file (e.g. ../../data/ecuador_training_data.csv)",
    )
    group.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing processed CSV files (X_*.csv, y_*.csv, taxonomic_data.csv)",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="Path to .npy embedding dict {bin_uri: vector}. "
        "If not provided, embeddings will be computed on-the-fly from --barcode_data.",
    )
    parser.add_argument(
        "--barcode_data",
        type=str,
        default=None,
        help="Path to TSV/CSV with 'bin_uri' and 'seq' columns. "
        "Used for fallback embedding inference if --embedding_path is not provided.",
    )
    parser.add_argument(
        "--emb_distance_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric for embedding-based neighbor graph.",
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

    # Decide data source
    data_path = args.data_path
    data_dir = args.data_dir
    if data_path is None and data_dir is None:
        data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "data_merged.csv")
        )
        log.info(f"No --data_path or --data_dir provided. Using default: {data_path}")

    # Handle embedding paths
    embedding_path = args.embedding_path
    if embedding_path is not None and not os.path.isabs(embedding_path):
        embedding_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), embedding_path)
        )

    barcode_data_path = args.barcode_data
    if barcode_data_path is not None and not os.path.isabs(barcode_data_path):
        barcode_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), barcode_data_path)
        )

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = make_run_group("barcodebert_comparison")
    if not use_wandb and not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")

    # Run comparison
    results = run_comparison(
        data_path=data_path,
        data_dir=data_dir,
        embedding_path=embedding_path,
        barcode_data_path=barcode_data_path,
        emb_distance_metric=args.emb_distance_metric,
        use_wandb=use_wandb,
        run_group=run_group,
    )

    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    results_path = save_variant_result(output_dir, "barcodebert", "barcodebert", results["barcodebert"])

    log.info("\n" + "=" * 78)
    log.info("VARIANT TRAINING COMPLETE")
    log.info("=" * 78)
    log.info(f"Results saved to: {os.path.abspath(results_path)}")
    log.info(
        "To visualize, run:\n"
        f"  python ../visualize_results.py --results_path {os.path.relpath(results_path)}"
    )
