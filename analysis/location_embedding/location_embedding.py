"""
Model Comparison: Baseline vs Location Embedding Variants

This script trains and compares models using different location embedding backends:
1. Baseline model - no location embeddings
2. SatCLIP variant - satellite imagery + location CLIP embeddings (256D)
3. RANGE variant - retrieval-augmented neural fields (1280D)
4. GeoCLIP variant - image-GPS alignment embeddings (512D)
5. AlphaEarth variant - Google Earth Engine embeddings (64D)

All models use the same training data and random seed for fair comparison.
Results are saved to pickle for analysis and visualization.

Usage:
    python location_embedding.py --data_path ../../data/data_merged.csv
    python location_embedding.py --data_dir ../../data --no_wandb
    python location_embedding.py --data_dir ../../data \\
        --satclip_ckpt_path /path/to/satclip.pth \\
        --range_db_path /path/to/range_db.npz
"""
from __future__ import annotations

import argparse
import logging as log
import os
import pickle
import sys
import time
from typing import Any, Dict, Optional

# Local directory first so local config.py and utils.py shadow src versions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.insert(2, os.path.join(os.path.dirname(__file__), '..'))

from config import Config, set_seed          # local config.py (has location-embedding fields)
from train import Trainer as BaseTrainer     # src/train.py
from utils import load, load_processed       # local utils.py (has location-embedding support)
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

import numpy as np


def _make_cfg(
    use_location_embedding: bool = False,
    location_embedder_model: str = "satclip",
    keep_raw_gps: bool = False,
    location_embedder_device: str = "cpu",
    location_embedder_batch_size: int = 2048,
    satclip_ckpt_path: Optional[str] = None,
    range_db_path: Optional[str] = None,
    range_model_name: str = "RANGE+",
    range_beta: float = 0.5,
    alphaearth_year: int = 2024,
    alphaearth_scale_meters: int = 10,
    alphaearth_project: Optional[str] = 'metabarcoding-491221',
) -> Config:
    """Build a Config with taxonomy enabled and optional location embedding fields."""
    cfg = Config()
    cfg.use_taxonomy = True
    cfg.use_embedding = False
    cfg.use_location_embedding = use_location_embedding
    cfg.location_embedder_model = location_embedder_model
    cfg.keep_raw_gps_features = keep_raw_gps
    cfg.location_embedding_prefix = "loc_emb"
    cfg.location_embedder_device = location_embedder_device
    cfg.location_embedder_batch_size = location_embedder_batch_size
    cfg.satclip_ckpt_path = satclip_ckpt_path
    cfg.range_db_path = range_db_path
    cfg.range_model_name = range_model_name
    cfg.range_beta = range_beta
    cfg.alphaearth_year = alphaearth_year
    cfg.alphaearth_scale_meters = alphaearth_scale_meters
    cfg.alphaearth_project = alphaearth_project
    return cfg


# ---------------------------------------------------------------------------
# Removed: TrainerWithEmbeddings subclass
# Config-driven approach: set fields on Config, pass to BaseTrainer directly.
# BaseTrainer.__init__ calls load(data_path, self.cfg, ...) which reads the
# location embedding fields off the config object via the local utils.py.
# ---------------------------------------------------------------------------


def run_comparison(
    data_path: Optional[str],
    data_dir: Optional[str],
    use_wandb: bool = True,
    satclip_ckpt_path: Optional[str] = None,
    range_db_path: Optional[str] = None,
    range_model_name: str = "RANGE+",
    range_beta: float = 0.5,
    alphaearth_year: int = 2024,
    alphaearth_scale_meters: int = 10,
    alphaearth_project: Optional[str] = 'metabarcoding-491221',
    keep_raw_gps: bool = False,
    location_embedder_device: str = "cpu",
    location_embedder_batch_size: int = 2048,
    embedders: Optional[list[str]] = None,
    run_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train location-embedding variants (no baseline retraining) and return results.
    
    Args:
        data_path: Path to CSV data file
        data_dir: Path to directory with processed CSVs
        use_wandb: Whether to log to Weights & Biases
        satclip_ckpt_path: Optional path to SatCLIP checkpoint
        range_db_path: Optional RANGE database (.npz), required for RANGE/RANGE+
        range_model_name: RANGE variant name ('RANGE' or 'RANGE+')
        range_beta: Interpolation parameter for RANGE+ (0.0-1.0)
        alphaearth_year: Year for AlphaEarth satellite data
        alphaearth_scale_meters: Scale for AlphaEarth sampling
        alphaearth_project: GCP project ID for Earth Engine initialization
        keep_raw_gps: Whether to keep raw GPS features alongside embeddings
        location_embedder_device: Device for embedding inference ('cpu', 'cuda', 'mps')
        location_embedder_batch_size: Batch size for embedder inference
    """
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
    # LOCATION EMBEDDING VARIANTS
    # =========================================================================
    if embedders is None:
        embedders = ["satclip", "geoclip", "range", "alphaearth"]
    
    for embedder_name in embedders:
        log.info("\n" + "=" * 78)
        log.info(f"TRAINING MODEL WITH {embedder_name.upper()} LOCATION EMBEDDINGS")
        log.info("=" * 78)

        set_seed(14)

        # Build a config with this embedder's settings
        model_kwargs: Dict[str, Any] = {}
        if embedder_name == "satclip":
            model_kwargs["satclip_ckpt_path"] = satclip_ckpt_path
        elif embedder_name == "range":
            model_kwargs["range_db_path"] = range_db_path
            model_kwargs["range_model_name"] = range_model_name
            model_kwargs["range_beta"] = range_beta
        elif embedder_name == "alphaearth":
            model_kwargs["alphaearth_year"] = alphaearth_year
            model_kwargs["alphaearth_scale_meters"] = alphaearth_scale_meters
            model_kwargs["alphaearth_project"] = alphaearth_project

        cfg = _make_cfg(
            use_location_embedding=True,
            location_embedder_model=embedder_name,
            keep_raw_gps=keep_raw_gps,
            location_embedder_device=location_embedder_device,
            location_embedder_batch_size=location_embedder_batch_size,
            **model_kwargs,
        )
        if data_path is not None:
            cfg.data_path = data_path

        try:
            with variant_wandb_run(
                use_wandb=use_wandb,
                wandb_module=wandb,
                project="metabarcoding-location-embeddings",
                analysis_name="location_embedding",
                variant_name=embedder_name,
                run_group=run_group,
                tags=["location_embedding", embedder_name, "variant_only"],
                config={"embedder": embedder_name, **cfg.__dict__},
            ):
                trainer = BaseTrainer(
                    cfg,
                    data_path=data_path,
                    data_dir=data_dir,
                )
                embedder_results = trainer.run(use_wandb=use_wandb)
                results[embedder_name] = embedder_results
                log.info(f"✓ {embedder_name.upper()} training completed successfully")
        except Exception as e:
            log.error(f"✗ {embedder_name.upper()} training failed: {e}")
            import traceback
            traceback.print_exc()
            log.info(f"Skipping {embedder_name.upper()} variant")
            results[embedder_name] = {"error": str(e)}

    return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Comparison: Baseline vs Location Embedding Variants"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to raw data CSV file (e.g. ../../data/data_merged.csv)",
    )
    group.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to directory containing processed CSV files (X_*.csv, y_*.csv)",
    )
    
    # Location embedder parameters
    parser.add_argument(
        "--satclip_ckpt_path",
        type=str,
        default=None,
        help="Optional path to SatCLIP checkpoint file (.pth)",
    )
    parser.add_argument(
        "--range_db_path",
        type=str,
        default='third_party/RANGE/pretrained/range_db_med.npz',
        help="Path to RANGE database (.npz) - required for RANGE/RANGE+ embeddings",
    )
    parser.add_argument(
        "--range_model_name",
        type=str,
        default="RANGE+",
        choices=["RANGE", "RANGE+"],
        help="RANGE model variant to use",
    )
    parser.add_argument(
        "--range_beta",
        type=float,
        default=0.5,
        help="Interpolation parameter for RANGE+ (0.0=RANGE, 1.0=RANGE+)",
    )
    parser.add_argument(
        "--alphaearth_year",
        type=int,
        default=2024,
        help="Year for AlphaEarth satellite imagery",
    )
    parser.add_argument(
        "--alphaearth_scale_meters",
        type=int,
        default=10,
        help="Scale in meters for AlphaEarth sampling",
    )
    parser.add_argument(
        "--alphaearth_project",
        type=str,
        default='metabarcoding-491221',
        help="GCP project ID for Earth Engine (required for AlphaEarth)",
    )
    parser.add_argument(
        "--keep_raw_gps",
        action="store_true",
        help="Keep raw latitude/longitude features alongside location embeddings",
    )
    parser.add_argument(
        "--location_embedder_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for location embedding model inference",
    )
    parser.add_argument(
        "--location_embedder_batch_size",
        type=int,
        default=2048,
        help="Batch size for location embedding inference",
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=["satclip", "range", "geoclip", "alphaearth"],
        choices=["satclip", "range", "geoclip", "alphaearth"],
        help="Location embedding variants to train (default: alphaearth)",
    )
    
    # General parameters
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

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    run_group = make_run_group("location_embedding")
    if not use_wandb and not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")

    # Run comparison
    results = run_comparison(
        data_path=data_path,
        data_dir=data_dir,
        use_wandb=use_wandb,
        satclip_ckpt_path=args.satclip_ckpt_path,
        range_db_path=args.range_db_path,
        range_model_name=args.range_model_name,
        range_beta=args.range_beta,
        alphaearth_year=args.alphaearth_year,
        alphaearth_scale_meters=args.alphaearth_scale_meters,
        alphaearth_project=args.alphaearth_project,
        keep_raw_gps=args.keep_raw_gps,
        location_embedder_device=args.location_embedder_device,
        location_embedder_batch_size=args.location_embedder_batch_size,
        embedders=args.embedders,
        run_group=run_group,
    )

    # Save results
    output_dir = make_output_dir(__file__, args.output_dir)
    for variant, variant_results in results.items():
        save_variant_result(output_dir, "location_embedding", variant, variant_results)

    log.info("\n" + "=" * 78)
    log.info("VARIANT TRAINING COMPLETE")
    log.info("=" * 78)
    log.info(f"Results saved to: {os.path.abspath(output_dir)}")
    log.info(f"Models trained: {', '.join(args.embedders)}")
    log.info(
        "Results contain training metrics for each variant and can be compared "
        "with visualize_results.py using --results_paths or a folder path"
    )
