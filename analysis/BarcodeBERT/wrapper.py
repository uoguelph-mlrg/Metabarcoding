"""
Model comparison runner: Taxonomy baseline vs BarcodeBERT embedding neighbours.

This script trains both variants and saves:
1) Per-model result pickles (taxonomy_results.pkl, barcodebert_results.pkl)
2) A combined comparison pickle (model_comparison_results.pkl)

The combined file is the one consumed by barcodebert_visualize.py.
"""

from __future__ import annotations

import argparse
import importlib
import logging as log
import os
import pickle
import sys
import time
from typing import Any, Dict, Optional, Tuple


try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _reset_cached_modules() -> None:
    """Clear modules that can conflict when switching between src and local imports."""
    for mod in [
        "config",
        "train",
        "model",
        "latent_solver",
        "neighbor_graph",
        "utils",
        "dataset",
        "loss",
        "mlp",
    ]:
        if mod in sys.modules:
            del sys.modules[mod]


def _import_baseline(src_dir: str):
    """Import src Trainer/Config with a clean module namespace."""
    local_dir = os.path.dirname(os.path.abspath(__file__))

    if local_dir in sys.path:
        sys.path.remove(local_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    _reset_cached_modules()
    src_train = importlib.import_module("train")
    src_config = importlib.import_module("config")
    return src_train.Trainer, src_config.Config, src_config.set_seed


def _import_barcodebert(local_dir: str, src_dir: str):
    """Import BarcodeBERT analysis Trainer (from this folder) and shared Config."""
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    _reset_cached_modules()
    local_train = importlib.import_module("train")
    local_config = importlib.import_module("config")
    return local_train.Trainer, local_config.Config, local_config.set_seed


def _save_pickle(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def run_comparison(
    data_path: Optional[str],
    data_dir: Optional[str],
    embedding_path: Optional[str],
    barcode_data_path: Optional[str],
    emb_distance_metric: str,
    use_wandb: bool,
) -> Dict[str, Any]:
    """Train taxonomy and BarcodeBERT variants with the same split/seed."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(root_dir, "..", "..", "src"))

    # ------------------ Baseline taxonomy model ------------------
    log.info("\n" + "=" * 72)
    log.info("TRAINING BASELINE MODEL (TAXONOMY NEIGHBOURS)")
    log.info("=" * 72)

    BaseTrainer, BaseConfig, set_seed = _import_baseline(src_dir)
    set_seed(14)

    base_cfg = BaseConfig()
    base_cfg.use_taxonomy = True
    base_cfg.use_embedding = False
    base_cfg.embedding_path = None
    base_cfg.barcode_data_path = None

    base_trainer = BaseTrainer(
        base_cfg,
        data_path=data_path,
        data_dir=data_dir,
    )
    baseline_results = base_trainer.run(use_wandb=use_wandb)

    # ------------------ BarcodeBERT embedding model ------------------
    log.info("\n" + "=" * 72)
    log.info("TRAINING BARCODEBERT VARIANT (EMBEDDING NEIGHBOURS)")
    log.info("=" * 72)

    LocalTrainer, LocalConfig, set_seed = _import_barcodebert(root_dir, src_dir)
    set_seed(14)

    emb_cfg = LocalConfig()
    emb_cfg.use_taxonomy = False
    emb_cfg.use_embedding = True
    emb_cfg.emb_distance_metric = emb_distance_metric
    emb_cfg.embedding_path = embedding_path
    emb_cfg.barcode_data_path = barcode_data_path

    emb_trainer = LocalTrainer(
        emb_cfg,
        data_path=data_path,
        data_dir=data_dir,
        fixed_split_indices=base_trainer.split_indices,
    )
    barcodebert_results = emb_trainer.run(use_wandb=use_wandb)

    return {
        "taxonomy": baseline_results,
        "barcodebert": barcodebert_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare taxonomy baseline vs BarcodeBERT embedding neighbours"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to raw CSV data (e.g. ../../data/ecuador_training_data.csv)",
    )
    group.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to processed data dir (X_*.csv, y_*.csv, taxonomic_data.csv)",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="Path to .npy embedding dict. If missing, --barcode_data can be used for on-the-fly inference.",
    )
    parser.add_argument(
        "--barcode_data",
        type=str,
        default=None,
        help="Path to TSV/CSV containing bin_uri and seq columns for fallback embedding inference.",
    )
    parser.add_argument(
        "--emb_distance_metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric used for embedding neighbours.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for comparison result pickles.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    log.basicConfig(
        level=log.DEBUG if args.verbose else log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data_path
    data_dir = args.data_dir
    if data_path is None and data_dir is None:
        # Keep default aligned with current BarcodeBERT train.py fallback.
        data_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "data_merged.csv"))
        log.info(f"No --data_path or --data_dir provided. Using default: {data_path}")

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)

    embedding_path = args.embedding_path
    if embedding_path is not None and not os.path.isabs(embedding_path):
        embedding_path = os.path.join(script_dir, embedding_path)

    barcode_data_path = args.barcode_data
    if barcode_data_path is not None and not os.path.isabs(barcode_data_path):
        barcode_data_path = os.path.join(script_dir, barcode_data_path)

    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="metabarcoding-barcodebert-comparison",
            name=f"taxonomy_vs_barcodebert_{time.strftime('%Y-%m-%d_%H-%M')}",
            reinit=True,
        )
    elif not WANDB_AVAILABLE:
        log.warning("wandb is not installed; continuing without wandb logging")

    results = run_comparison(
        data_path=data_path,
        data_dir=data_dir,
        embedding_path=embedding_path,
        barcode_data_path=barcode_data_path,
        emb_distance_metric=args.emb_distance_metric,
        use_wandb=use_wandb,
    )

    taxonomy_path = os.path.join(output_dir, "taxonomy_results.pkl")
    barcodebert_path = os.path.join(output_dir, "barcodebert_results.pkl")
    combined_path = os.path.join(output_dir, "model_comparison_results.pkl")

    _save_pickle(taxonomy_path, results["taxonomy"])
    _save_pickle(barcodebert_path, results["barcodebert"])
    _save_pickle(combined_path, results)

    log.info("\n" + "=" * 72)
    log.info("COMPARISON COMPLETE")
    log.info("=" * 72)
    log.info(f"Saved taxonomy results:   {taxonomy_path}")
    log.info(f"Saved BarcodeBERT results: {barcodebert_path}")
    log.info(f"Saved combined results:    {combined_path}")
    log.info(
        "Run visualization: python barcodebert_visualize.py "
        f"--results_path {combined_path}"
    )

    if use_wandb:
        wandb.finish()
