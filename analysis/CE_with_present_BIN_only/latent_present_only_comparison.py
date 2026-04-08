"""
Latent Present-Only vs All-BINs Comparison

Trains the model twice using cross-entropy loss:
1. latent_present_only=True  – latent is updated only on observations where y > 0
2. latent_present_only=False – latent is updated on all observations (present + absent)

Usage:
    python latent_present_only_comparison.py
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time
from typing import Any, Dict
import logging as log

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config import Config, set_seed
from train import Trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================

VARIANTS = {
    "present_only": {"latent_present_only": True,  "label": "Present BINs only"},
    "all_bins":     {"latent_present_only": False, "label": "All BINs"},
}

SEED = 42


def run_comparison(
    cfg: Config,
    use_wandb: bool,
) -> Dict[str, Any]:
    results = {}

    for key, variant in VARIANTS.items():
        log.info("\n" + "=" * 70)
        log.info(f"TRAINING: {variant['label']}  (latent_present_only={variant['latent_present_only']})")
        log.info("=" * 70)

        # Deep-copy config so each run is independent
        run_cfg = copy.deepcopy(cfg)
        run_cfg.latent_present_only = variant["latent_present_only"]

        set_seed(SEED)
        trainer = Trainer(run_cfg)

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="metabarcoding",
                name=f"{key}_{time.strftime('%Y-%m-%d_%H-%M')}",
                config={**run_cfg.__dict__, "variant": key},
                reinit=True,
            )

        run_results = trainer.run(use_wandb=(use_wandb and WANDB_AVAILABLE))
        run_results["label"] = variant["label"]
        run_results["latent_present_only"] = variant["latent_present_only"]
        results[key] = run_results

        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

    return results


def save_results(results: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "latent_present_only_results.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"Results saved to {path}")
    return path


# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent present-only vs all-BINs comparison")
    group = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"),
                        help="Directory in which to save results pickle")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug-level logging")
    args = parser.parse_args()

    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = Config()
    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    results = run_comparison(cfg, use_wandb=use_wandb)

    out_path = save_results(results, args.output_dir)

    log.info("\n" + "=" * 70)
    log.info("COMPARISON COMPLETE")
    log.info("=" * 70)
    log.info(f"Results: {out_path}")
    log.info(f"Visualise: python latent_present_only_visualize.py --results_path {out_path}")
